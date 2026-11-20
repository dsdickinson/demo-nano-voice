# 🎙️ Nano-Voice: Edge AI Conversational Assistant & From-Scratch LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware: NVIDIA Jetson Orin Nano](https://img.shields.io/badge/Hardware-NVIDIA%20Jetson%20Orin%20Nano-76b900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

**Nano-Voice** is an end-to-end edge AI voice assistant and modern decoder-only Large Language Model (LLM) built from scratch in pure PyTorch, optimized to run fully offline on the **NVIDIA Jetson Orin Nano (8GB)**.

It pairs deep learning fundamentals (zero-black-box Transformer architecture with RoPE, GQA, SwiGLU, RMSNorm, and custom BPE tokenization) with real-time physical streaming systems (Voice Activity Detection, Speech-to-Text, local RAG vector search, and neural female voice synthesis).

---

## 🌟 Key Features

- **🧠 Modern Transformer from Scratch (`core/model.py`)**:
  - **RMSNorm**: Compute-efficient pre-normalization without mean shifting.
  - **Rotary Position Embeddings (RoPE)**: Dynamic 2D complex rotations preserving relative distance.
  - **Grouped-Query Attention (GQA)**: 8 query heads sharing 4 key-value heads, cutting KV-cache memory traffic by 50%.
  - **SwiGLU FFN**: LLaMA-3 / Mistral standard gated activation.
  - **Autoregressive KV Cache**: High-speed token-by-token generation with native PyTorch SDPA (`scaled_dot_product_attention`).
- **🔤 Byte-Level BPE Tokenizer (`core/tokenizer.py`)**:
  - Self-contained, zero-dependency BPE trainer guaranteeing 0% out-of-vocabulary rate on any UTF-8 string.
- **📚 Local Retrieval-Augmented Generation (`rag/`)**:
  - Sub-20ms vector similarity search over local documents (`profiles/public/knowledge/`) using dense embeddings and cosine similarity.
- **🎙️ Real-Time Edge Voice Pipeline (`assistant/`)**:
  - **VAD**: Low-latency Voice Activity Detection (Silero VAD) to segment spoken utterances.
  - **STT**: High-accuracy local speech-to-text with Faster-Whisper.
  - **TTS**: Natural female voice synthesis with Piper TTS / Kokoro.
  - **Sentence-Chunked Streaming**: Punctuation-boundary streaming begins speech playback while subsequent tokens are still being generated.
  - **Echo Ducking**: Mutes microphone input during speaker playback to eliminate acoustic feedback.
- **⚡ Dual-Engine Flexibility**:
  - `--model scratch`: Runs your custom PyTorch weights trained from scratch.
  - `--model instruct`: Plugs in compact instruction models (e.g. SmolLM2-360M-Instruct) for witty open-domain conversational chat.
- **🛡️ Public Showcase vs. Private Personal Profile Isolation**:
  - Default `profiles/public/` is completely sanitized and safe for GitHub portfolios.
  - `profiles/private/` is git-ignored for private memories, personal data, and custom behavioral instructions.

---

## 📐 System Pipeline

```
[🎤 Microphone]
       │ (16kHz PCM audio stream)
       ▼
┌──────────────┐
│  Silero VAD  │ ──> Detects speech onset & silence endpoint
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Faster-Whisper│ ──> Transcribes speech to text in ~150ms
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Local RAG   │ <── Searches local knowledge chunks in <20ms
└──────┬───────┘
       │ (Context-augmented prompt)
       ▼
┌──────────────┐
│  DecoderLLM  │ ──> Autoregressive KV-cache streaming generation
└──────┬───────┘
       │ (Sentence boundary tokens)
       ▼
┌──────────────┐
│Streaming TTS │ ──> Natural neural female voice synthesis
└──────┬───────┘
       │ (Echo ducking mutes mic)
       ▼
[🔊 Speaker]
```

---

## 📂 Repository Structure

```
demo_llm_training/
├── core/                    # Core deep learning modules from scratch
│   ├── model.py             # RMSNorm, RoPE, GQA, SwiGLU, DecoderLLM, KV-Cache
│   ├── tokenizer.py         # Byte-level BPE tokenizer (train, encode, decode)
│   └── dataset.py           # Sequence batcher & dynamic padding
├── rag/                     # Local Retrieval-Augmented Generation
│   ├── ingest.py            # Document parsing and text chunking
│   ├── retriever.py         # Dense vector store & cosine similarity search
│   └── augment.py           # Prompt context augmentation
├── assistant/               # Edge voice streaming loop
│   ├── audio.py             # Sounddevice capture & VAD energy segmentation
│   ├── stt.py               # Faster-Whisper transcription wrapper
│   ├── tts.py               # Piper neural female voice synthesis
│   ├── llm_runner.py        # Dual-engine runner (scratch vs. instruct)
│   └── pipeline.py          # Real-time pipeline coordinator
├── profiles/                # Profile configuration & knowledge bases
│   └── public/              # Sanitized public showcase (persona & demo knowledge)
├── docs/                    # Architectural & hardware guides
│   ├── architecture.md      # Mathematical formulations & tensor diagrams
│   └── jetson_setup.md      # JetPack, power modes, swap, and audio config
├── tools/                   # Developer & verification scripts
│   ├── check_env.py         # Hardware & audio device diagnostic tool
│   └── train_toy.py         # 30-second rapid training demo script
├── tests/                   # Pytest test suite
│   ├── test_model.py        # Unit tests for RMSNorm, RoPE, GQA, BPE, LLM
│   └── test_rag.py          # Unit tests for RAG chunking and vector search
├── train.py                 # Full training loop (AdamW, AMP, Cosine Warmup)
├── run_voice.py             # Interactive assistant entry point
└── requirements.txt         # Project dependencies
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/nano-voice.git
cd nano-voice

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Hardware & Audio Devices

```bash
python tools/check_env.py
```
This utility checks PyTorch compute devices (CUDA / Apple MPS / CPU) and lists all connected microphone and speaker hardware.

### 3. Run 30-Second Rapid Training Demo

Train a micro-LLM (~340K parameters) from scratch in under 30 seconds on your local machine:

```bash
python tools/train_toy.py
```

This script:
1. Generates a curated dialogue dataset.
2. Trains a custom Byte-Pair Encoding (BPE) tokenizer.
3. Trains the `DecoderLLM` with AdamW and cosine decay.
4. Saves checkpoints to `checkpoints/model_final.pt` and verifies generation.

### 4. Run the Voice Assistant

#### Real-Time Voice Mode (with Microphone):
```bash
python run_voice.py --model scratch
```

#### Interactive Terminal Text Mode (No Mic Required):
```bash
python run_voice.py --model scratch --text-only
```

#### Dual-Engine Instruct Mode:
```bash
python run_voice.py --model instruct --text-only
```

---

## 🧪 Running Unit Tests

Run the comprehensive pytest suite verifying model shapes, RoPE rotations, GQA caching, and vector retrieval:

```bash
pytest
```

---

## 📚 Technical Documentation

- **[Architecture & Mathematical Foundations](docs/architecture.md)**: Deep dive into RMSNorm, RoPE complex frequency rotations, Grouped-Query Attention formulas, and SwiGLU activation math.
- **[Jetson Orin Nano Deployment Guide](docs/jetson_setup.md)**: JetPack setup, 15W MAXN clock optimization, NVMe swapfile setup, and headless memory optimization.

---

## 📄 License

Released under the [MIT License](LICENSE).
