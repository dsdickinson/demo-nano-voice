"""
Modern Decoder-Only Transformer LLM Architecture from Scratch.
Implements:
  - RMSNorm (Root Mean Square Normalization)
  - Rotary Position Embeddings (RoPE)
  - Grouped-Query Attention (GQA) with KV-Caching
  - SwiGLU Feed-Forward Network
  - Pure PyTorch SDPA (FlashAttention native backend)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Generator
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for the DecoderLLM architecture."""
    vocab_size: int = 4096
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4        # Grouped-Query Attention (n_heads / n_kv_heads = 2)
    hidden_dim: Optional[int] = None  # SwiGLU intermediate dimension
    head_dim: Optional[int] = None
    max_seq_len: int = 512
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.head_dim = self.dim // self.n_heads
        if self.hidden_dim is None:
            # Standard LLaMA-style 8/3 expansion rounded to nearest multiple of 64
            hidden = int(2 * (4 * self.dim) / 3)
            self.hidden_dim = ((hidden + 63) // 64) * 64


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Formula: y = (x / sqrt(mean(x^2) + eps)) * weight
    Avoids mean-centering compute while maintaining training stability.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_rope_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex exponential frequencies for Rotary Position Embeddings (RoPE).
    Returns complex tensor of shape (max_seq_len, dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^(i * theta)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.
    xq: (batch, seq_len, n_heads, head_dim)
    xk: (batch, seq_len, n_kv_heads, head_dim)
    freqs_cis: (seq_len, head_dim // 2)
    """
    # Reshape into complex numbers: (..., head_dim // 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Match broadcast shape (1, seq_len, 1, head_dim // 2)
    seqlen = xq_.shape[1]
    freqs_cis_slice = freqs_cis[:seqlen].view(1, seqlen, 1, xq_.shape[-1])
    
    # Complex multiplication rotates vectors by position angle
    xq_out = torch.view_as_real(xq_ * freqs_cis_slice).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_slice).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat Key/Value heads to match Query heads in Grouped-Query Attention (GQA).
    x: (batch, seq_len, n_kv_heads, head_dim) -> (batch, seq_len, n_heads, head_dim)
    """
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with optional KV Cache.
    n_heads query heads share n_kv_heads key/value heads.
    When n_kv_heads == 1 -> Multi-Query Attention (MQA)
    When n_kv_heads == n_heads -> Standard Multi-Head Attention (MHA)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim

        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seqlen, _ = x.shape

        # Linear projections
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # Reshape to (bsz, seqlen, n_heads/n_kv_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply Rotary Position Embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # KV-Cache update for autoregressive generation
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            # Concatenate along sequence dimension
            xk = torch.cat([prev_k, xk], dim=1)
            xv = torch.cat([prev_v, xv], dim=1)
            updated_kv_cache = (xk, xv)
        else:
            updated_kv_cache = None

        # Repeat KV heads for GQA
        key = repeat_kv(xk, self.n_rep)
        val = repeat_kv(xv, self.n_rep)

        # Transpose for PyTorch scaled_dot_product_attention: (bsz, n_heads, seqlen, head_dim)
        query = xq.transpose(1, 2)
        key = key.transpose(1, 2)
        val = val.transpose(1, 2)

        # Determine causal mask (only needed if generating without past KV cache or initial prompt)
        is_causal = (seqlen > 1) and (kv_cache is None)
        
        # Native PyTorch SDPA (uses FlashAttention on GPU/CUDA when available)
        attn_out = F.scaled_dot_product_attention(
            query, key, val,
            is_causal=is_causal
        )

        # Re-assemble heads: (bsz, seqlen, n_heads * head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(attn_out), updated_kv_cache


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    Formula: output = w_down(SiLU(w_gate(x)) * w_up(x))
    Standard in modern models (LLaMA-3, Mistral, PaLM, DeepSeek).
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    """Single modern Transformer Decoder Layer with Pre-RMSNorm."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = SwiGLUFFN(config.dim, config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm Attention with residual
        h_attn, new_kv = self.attention(
            self.attention_norm(x),
            freqs_cis=freqs_cis,
            kv_cache=kv_cache,
            start_pos=start_pos
        )
        x = x + h_attn
        # Pre-norm FFN with residual
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv


class DecoderLLM(nn.Module):
    """
    Full Decoder-Only LLM.
    Pure PyTorch, self-contained, educational and optimized for edge devices.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE complex frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs_cis(
                config.head_dim,
                config.max_seq_len,
                config.rope_theta
            ),
            persistent=False
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with standard normal distribution scaled by sqrt(dim)."""
        std = 0.02
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=std)

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        kv_caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass for training or generation.
        tokens: (bsz, seqlen) of integer token IDs
        """
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # Dynamically extend RoPE frequencies if sequence exceeds precomputed length
        needed_pos = start_pos + seqlen
        if needed_pos > self.freqs_cis.shape[0]:
            new_len = max(needed_pos, self.freqs_cis.shape[0] * 2)
            self.freqs_cis = precompute_rope_freqs_cis(
                self.config.head_dim,
                new_len,
                self.config.rope_theta
            ).to(tokens.device)

        # Slice RoPE frequencies for current positions
        freqs_cis = self.freqs_cis[start_pos : needed_pos]

        new_kv_caches = [] if kv_caches is not None else None

        for idx, layer in enumerate(self.layers):
            layer_cache = kv_caches[idx] if kv_caches is not None else None
            h, updated_cache = layer(
                h,
                freqs_cis=freqs_cis,
                kv_cache=layer_cache,
                start_pos=start_pos
            )
            if new_kv_caches is not None:
                new_kv_caches.append(updated_cache)

        h = self.norm(h)
        logits = self.output(h)
        return logits, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> Generator[int, None, None]:
        """
        Autoregressive token generator supporting streaming and KV-cache acceleration.
        Yields newly generated token IDs one by one.
        """
        self.eval()
        bsz, prompt_len = prompt_tokens.shape
        assert bsz == 1, "Streaming generation currently configured for batch_size=1"

        kv_caches = [None] * len(self.layers) if use_cache else None
        
        # Prefill stage: process prompt tokens
        logits, kv_caches = self(prompt_tokens, start_pos=0, kv_caches=kv_caches)
        next_token_logits = logits[:, -1, :]

        curr_pos = prompt_len
        curr_token = self._sample_next_token(next_token_logits, temperature, top_k, top_p)
        yield curr_token.item()

        # Generation loop: process one token at a time
        for _ in range(max_new_tokens - 1):
            if eos_id is not None and curr_token.item() == eos_id:
                break

            if use_cache:
                # With KV cache, only pass single current token
                logits, kv_caches = self(
                    curr_token.unsqueeze(0),
                    start_pos=curr_pos,
                    kv_caches=kv_caches
                )
                next_token_logits = logits[:, -1, :]
            else:
                prompt_tokens = torch.cat([prompt_tokens, curr_token.unsqueeze(0)], dim=1)
                logits, _ = self(prompt_tokens)
                next_token_logits = logits[:, -1, :]

            curr_pos += 1
            curr_token = self._sample_next_token(next_token_logits, temperature, top_k, top_p)
            yield curr_token.item()

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Sample next token from logits using temperature, top-k, and nucleus (top-p) filtering."""
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1)

        logits = logits / temperature

        # Top-K filtering
        if top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k_val)
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Top-p (Nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
