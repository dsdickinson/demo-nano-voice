# Nano-Voice Architecture & Mathematical Foundations

This document provides a comprehensive technical walkthrough of the **Nano-Voice** system: from the mathematical formulation of its from-scratch decoder-only Transformer to the real-time edge voice pipeline running on the NVIDIA Jetson Orin Nano.

---

## 1. High-Level System Architecture

The project is structured around two complementary subsystems:
1. **Deep Learning Core**: Pure PyTorch implementation of modern LLM primitives (LLaMA-3 / Mistral / DeepSeek style).
2. **Physical Voice Loop**: Real-time edge streaming pipeline (VAD $\to$ STT $\to$ Local RAG $\to$ LLM $\to$ Neural TTS).

```
[🎤 Microphone]
       │ (16kHz PCM audio stream)
       ▼
┌──────────────┐
│  Silero VAD  │ ──> Detects speech onset & terminal silence
└──────┬───────┘
       │ (Audio segment tensor)
       ▼
┌──────────────┐
│Faster-Whisper│ ──> Transcribes speech to text
└──────┬───────┘
       │ (Transcribed query text)
       ▼
┌──────────────┐
│  Local RAG   │ <── Searches local vector store (FAISS/dense embeddings)
└──────┬───────┘
       │ (Context-augmented prompt)
       ▼
┌──────────────┐
│  DecoderLLM  │ ──> Modern Transformer core with autoregressive KV-cache
└──────┬───────┘
       │ (Streaming token stream)
       ▼
┌──────────────┐
│Streaming TTS │ ──> Piper / Kokoro neural female speech synthesis
└──────┬───────┘
       │ (Echo ducking mutes microphone during playback)
       ▼
[🔊 Speaker]
```

---

## 2. Transformer Mathematical Formulations

### 2.1 Root Mean Square Layer Normalization (RMSNorm)
Standard LayerNorm centers activations by subtracting their mean and dividing by standard deviation:
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

RMSNorm hypothesizes that the scaling property of LayerNorm is what stabilizes training, not mean centering. By omitting mean calculation, compute cost is reduced by ~7% while maintaining identical training dynamics:

$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma$$

Where $\gamma \in \mathbb{R}^d$ is a learnable scaling parameter initialized to ones.

---

### 2.2 Rotary Position Embeddings (RoPE)
Instead of adding absolute position embeddings to input tokens (which fail to generalize across extended sequence lengths), Rotary Position Embeddings (RoPE) encode relative position by rotating queries and keys in the complex plane.

For a 2D vector $(x_1, x_2)$ at sequence index $m$:

$$R_{\Theta, m}^2 = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$$

Where frequencies $\theta_i$ are precomputed as:
$$\theta_i = 10000^{-2(i-1)/d}, \quad i \in \left[1, \dots, \frac{d}{2}\right]$$

By pairing adjacent elements of the query $q$ and key $k$ as complex numbers $z = x_{2i} + i \cdot x_{2i+1}$, rotation becomes a complex multiplication:

$$q_m = q \cdot e^{i \cdot m\theta}, \quad k_n = k \cdot e^{i \cdot n\theta}$$

When calculating attention scores $\langle q_m, k_n \rangle$:
$$\text{Re}\left(q_m k_n^*\right) = \text{Re}\left(q k^* e^{i(m-n)\theta}\right)$$

The attention score depends **only on the relative distance $(m - n)$**, conferring seamless length extrapolation.

---

### 2.3 Grouped-Query Attention (GQA) & KV Cache
In standard Multi-Head Attention (MHA), each query head has its own key and value head:
$$\text{Memory}_{KV} = 2 \times b \times s \times n_{heads} \times d_{head}$$

On edge hardware like the Jetson Orin Nano, memory bandwidth is the primary bottleneck during autoregressive token generation. Grouped-Query Attention (GQA) allows $N$ query heads to share $M$ key-value heads ($N > M$):

```
       MHA (Standard)                      GQA (Nano-Voice)
Query Heads      KV Heads            Query Heads       KV Heads
   Q1 ───────────── K1/V1               Q1 ──┬─────────── K1/V1
   Q2 ───────────── K2/V2               Q2 ──┘
   Q3 ───────────── K3/V3               Q3 ──┬─────────── K2/V2
   Q4 ───────────── K4/V4               Q4 ──┘
```

With 8 query heads and 4 key-value heads, KV cache memory traffic is **halved (50% bandwidth reduction)** with negligible loss in modeling capacity.

During step-by-step autoregressive generation:
1. At step $t=0$ (prompt prefill), full key and value tensors are computed and stored in cache.
2. At step $t > 0$, only the single new token is projected:
   $$K_{new} = W_k x_t, \quad V_{new} = W_v x_t$$
3. $K_{new}$ and $V_{new}$ are concatenated to the cache tensor along the sequence dimension.
4. Scaled dot-product attention is calculated using native PyTorch SDPA (`F.scaled_dot_product_attention`), automatically routing to optimized FlashAttention kernels on CUDA.

---

### 2.4 SwiGLU Activation Function
Modern models replace standard GELU / ReLU feedforward networks with SwiGLU (Swish Gated Linear Unit).

Standard FFN:
$$\text{FFN}(x) = \text{GELU}(x W_1) W_2$$

SwiGLU FFN:
$$\text{FFN}_{SwiGLU}(x) = \left(\text{SiLU}(x W_{gate}) \otimes x W_{up}\right) W_{down}$$

Where $\text{SiLU}(z) = z \cdot \sigma(z)$. To maintain parameter parity with standard FFNs with hidden dimension $4d$, SwiGLU uses an intermediate dimension of approximately $\frac{8}{3}d$ rounded to a multiple of 64.

---

## 3. Tensor Shape Pipeline

```
Input Token IDs: (batch_size, seq_len)
       │
       ▼  Embedding Matrix (vocab_size, dim)
Hidden States:   (batch_size, seq_len, dim)
       │
       ├──────────────────────────────────────────┐
       │ Pre-RMSNorm                              │ Residual Connection
       ▼                                          │
       │ wq: (dim, n_heads * head_dim)            │
       │ wk: (dim, n_kv_heads * head_dim)         │
       │ wv: (dim, n_kv_heads * head_dim)         │
       │ RoPE rotation on Q and K                 │
       │ Repeat KV heads: n_kv_heads -> n_heads   │
       │ SDPA Attention: Q @ K.T / sqrt(d) @ V    │
       │ wo: (n_heads * head_dim, dim)            │
       ▼                                          │
       + <────────────────────────────────────────┘
       │
       ├──────────────────────────────────────────┐
       │ Pre-RMSNorm                              │ Residual Connection
       ▼                                          │
       │ SwiGLU FFN:                              │
       │ w_gate(x) * SiLU * w_up(x) @ w_down      │
       ▼                                          │
       + <────────────────────────────────────────┘
       │
       ▼  Repeat across n_layers
Final RMSNorm:   (batch_size, seq_len, dim)
       │
       ▼  LM Head (dim, vocab_size)
Output Logits:   (batch_size, seq_len, vocab_size)
```

---

## 4. Local RAG Pipeline

1. **Ingestion**: Documents placed in `profiles/public/knowledge/` are segmented into 300-character chunks with a 50-character stride.
2. **Dense Indexing**: Chunks are embedded into normalized dense vectors and indexed in memory.
3. **Sub-20ms Retrieval**: When a query arrives, it is embedded and scored against all stored chunks using cosine similarity ($S = A \cdot B$).
4. **Context Injection**: High-scoring snippets are wrapped in `<context>...</context>` XML tags and prepended to the user's turn in the prompt.
