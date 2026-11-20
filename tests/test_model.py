"""
Unit tests for the custom modern Transformer core and BPE tokenizer.
"""

import os
import tempfile
import pytest
import torch
from core.model import (
    ModelConfig,
    RMSNorm,
    precompute_rope_freqs_cis,
    apply_rotary_emb,
    GroupedQueryAttention,
    SwiGLUFFN,
    DecoderLLM,
)
from core.tokenizer import BPETokenizer


def test_rmsnorm():
    dim = 64
    norm = RMSNorm(dim=dim, eps=1e-5)
    x = torch.randn(2, 10, dim) * 5.0 + 3.0
    out = norm(x)

    assert out.shape == x.shape
    # Check that root mean square across last dimension is approximately 1.0 (before scaling)
    rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_rope_rotation():
    dim = 32
    seq_len = 16
    freqs_cis = precompute_rope_freqs_cis(dim=dim, max_seq_len=seq_len)

    xq = torch.randn(2, seq_len, 4, dim)
    xk = torch.randn(2, seq_len, 2, dim)

    rot_q, rot_k = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    assert rot_q.shape == xq.shape
    assert rot_k.shape == xk.shape
    # Norm of vectors is preserved under rotation
    assert torch.allclose(torch.norm(xq, dim=-1), torch.norm(rot_q, dim=-1), atol=1e-3)


def test_gqa_forward_and_kv_cache():
    config = ModelConfig(
        dim=128,
        n_heads=4,
        n_kv_heads=2,  # 4 query heads share 2 kv heads
        max_seq_len=64
    )
    gqa = GroupedQueryAttention(config)
    freqs_cis = precompute_rope_freqs_cis(config.head_dim, config.max_seq_len)

    # 1. Full sequence forward pass (no cache)
    x = torch.randn(2, 10, config.dim)
    out, cache = gqa(x, freqs_cis=freqs_cis[:10])
    assert out.shape == (2, 10, config.dim)
    assert cache is None

    # 2. Step-by-step with KV cache
    # Initial prompt step (length 5)
    x_prompt = torch.randn(1, 5, config.dim)
    out_prompt, kv_cache = gqa(x_prompt, freqs_cis=freqs_cis[:5], kv_cache=(torch.empty(1, 0, 2, config.head_dim), torch.empty(1, 0, 2, config.head_dim)))
    assert out_prompt.shape == (1, 5, config.dim)
    assert kv_cache[0].shape == (1, 5, 2, config.head_dim)

    # Incremental token step (length 1)
    x_next = torch.randn(1, 1, config.dim)
    out_next, kv_cache = gqa(x_next, freqs_cis=freqs_cis[5:6], kv_cache=kv_cache)
    assert out_next.shape == (1, 1, config.dim)
    assert kv_cache[0].shape == (1, 6, 2, config.head_dim)


def test_swiglu_ffn():
    dim = 64
    hidden_dim = 128
    ffn = SwiGLUFFN(dim=dim, hidden_dim=hidden_dim)
    x = torch.randn(2, 8, dim)
    out = ffn(x)
    assert out.shape == (2, 8, dim)


def test_decoder_llm_forward_and_generate():
    config = ModelConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64
    )
    model = DecoderLLM(config)
    counts = model.count_parameters()
    assert counts["total"] > 0

    # Forward pass
    tokens = torch.randint(0, 256, (2, 8))
    logits, _ = model(tokens)
    assert logits.shape == (2, 8, 256)

    # Autoregressive generation
    prompt = torch.tensor([[10, 20, 30]])
    gen_tokens = list(model.generate(prompt, max_new_tokens=5, temperature=0.7, use_cache=True))
    assert len(gen_tokens) == 5
    for t in gen_tokens:
        assert isinstance(t, int)
        assert 0 <= t < 256


def test_bpe_tokenizer():
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Edge AI voice assistant running on Jetson Orin Nano.",
        "Deep learning and transformer architectures from scratch."
    ]
    # Train BPE tokenizer
    tokenizer = BPETokenizer.train_from_texts(sample_texts, target_vocab_size=300, min_freq=1)
    assert tokenizer.vocab_size >= 256

    test_sentence = "Edge AI on Jetson"
    encoded = tokenizer.encode(test_sentence, add_bos=True, add_eos=True)
    assert len(encoded) > 0
    assert encoded[0] == tokenizer.bos_id
    assert encoded[-1] == tokenizer.eos_id

    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert decoded == test_sentence

    # Test save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tok.json")
        tokenizer.save(path)
        loaded = BPETokenizer.load(path)
        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.decode(loaded.encode(test_sentence)) == test_sentence
