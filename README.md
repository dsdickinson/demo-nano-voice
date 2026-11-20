# 🎙️ demo-nano-voice: Edge AI Voice Assistant & Transformer from Scratch

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
demo-nano-voice/
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
git clone https://github.com/your-username/demo-nano-voice.git
cd demo-nano-voice

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

#### Example Interaction:
```
[👤 User]: How are you today?
[🧠 Thinking & Generating...]
[🔊 Nova speaking]: "I am doing wonderful!"
[🔊 Nova speaking]: "All systems and neural engines are running smoothly."

[👤 User]: You know my name?
[🧠 Thinking & Generating...]
[🔊 Nova speaking]: "Honeybees device."
[🔊 Nova speaking]: "How can I help you today?"
```

---

## 📈 Scaling Beyond the Toy Model: From 340K to 25M+ Parameters

While `tools/train_toy.py` provides an immediate ~30-second verification run for KV caching and tensor flow, the custom Transformer architecture (`core/model.py`) is designed to scale seamlessly into a production-grade micro-LLM (15M to 45M parameters).

### 1. Scale Model Capacity
In `train.py`, increase the architectural parameters to scale from base settings up to ~25M parameters (notably expanding `vocab_size` to 4096 from the initial 1024, and increasing `n_layers` to 8 from the initial 4):

```python
config = ModelConfig(
    vocab_size=4096,     # Expanded subword vocabulary (originally set to 1024)
    dim=384,             # Hidden representation dimension (vs 96 in toy)
    n_layers=8,          # 8 Transformer blocks (originally set to 4)
    n_heads=6,           # 6 Query attention heads
    n_kv_heads=2,        # 2 Key-Value heads (3:1 GQA head sharing ratio)
    max_seq_len=256      # Extended autoregressive context window
)
```

### 2. Download the TinyStories Dataset
To achieve grammatical fluency and coherent conversational logic without needing cluster-scale compute, we use the **TinyStories V2** dataset (synthetic English stories by Microsoft Research).

Download the dataset using the Hugging Face CLI:

```bash
# 1. Install Hugging Face CLI (if not already installed)
curl -LsSf https://hf.co/cli/install.sh | bash

# 2. Download the TinyStories V2 dataset to data/tinystories/
mkdir -p data/tinystories
hf download roneneldan/TinyStories TinyStoriesV2-GPT4-train.txt --repo-type=dataset --local-dir ./data/tinystories
```

### 3. Launching Device-Agnostic Training
Train on your primary workstation (NVIDIA GPU, Apple Silicon Mac via MPS, or cloud Colab):

```bash
python train.py \
    --data data/tinystories/TinyStoriesV2-GPT4-train.txt \
    --epochs 3 \
    --batch-size 32 \
    --lr 3e-4
```

#### Hyperparameter Breakdown:
- **`--data` (TinyStoriesV2-GPT4)**: Synthetic high-density narrative corpus by Microsoft Research, designed specifically for micro-models (10M–30M parameters) to acquire fluent English grammar, common sense, and reasoning without trillions of web tokens.
- **`--epochs 3`**: Three complete passes through the dataset. One epoch is insufficient for convergence; three provides the optimal balance between language generalization and avoiding rote memorization.
- **`--batch-size 32`**: Computes gradients across 32 sequences in parallel per step. Stabilizes the AdamW gradient trajectory while fitting comfortably in consumer GPU memory.
- **`--lr 3e-4` ($3 \times 10^{-4}$ or $0.0003$)**: The frontier standard peak learning rate for Transformer pretraining with AdamW. It warms up linearly and decays along a half-cosine curve to maintain stable backpropagation.

#### Hardware & Training Benchmarks:
| Hardware | Accelerator | Estimated Time (3 Epochs) | Expected Final Loss / PPL |
| :--- | :--- | :--- | :--- |
| **NVIDIA RTX 3090 / 4090** | CUDA (FP16 AMP) | ~15 – 35 minutes | Loss: ~1.3 – 1.6 (PPL: ~3.7 – 5.0) |
| **Apple Silicon (M1/M2/M3 Max)** | MPS | ~45 – 90 minutes | Loss: ~1.4 – 1.7 (PPL: ~4.0 – 5.5) |
| **Google Colab (Free Tier)** | T4 GPU | ~1 – 2 hours | Loss: ~1.4 – 1.8 (PPL: ~4.0 – 6.0) |

### 4. Deploying Trained Checkpoints to the Jetson Orin Nano

When pretraining finishes, your training run outputs two essential companion files:
- **`model_final.pt` (The Brain's Neurons)**: Serialized PyTorch weights containing all learned attention matrices ($W_q, W_k, W_v, W_{gate}, W_{up}, W_{down}$), RMSNorm parameters, and architectural configuration metadata.
- **`tokenizer.json` (The Language Dictionary)**: Learned Byte-Pair Encoding (BPE) vocabulary and merge tables used to encode spoken words into token IDs and decode generated output IDs back into text.

#### Checkpoint Placement on the Jetson Orin Nano:
Place both exported files directly into the **`checkpoints/`** folder of your repository on the Jetson Orin Nano:

```
demo-nano-voice/                       <-- Root project directory on Jetson
├── assistant/
├── core/
├── docs/
├── profiles/
├── rag/
├── tools/
├── checkpoints/                       <-- 📁 Place exported checkpoint files here
│   ├── model_final.pt                 <-- Your trained PyTorch weights
│   └── tokenizer.json                 <-- Your trained BPE tokenizer
├── run_voice.py
└── train.py
```

#### Running the Scaled Model:
When you launch the assistant with your custom weights:
```bash
python run_voice.py --model scratch
```

The runtime automatically:
1. Loads `checkpoints/model_final.pt` and reads the model dimensions (`dim`, `n_layers`, `n_heads`).
2. Allocates the Transformer directly in the Jetson Orin Nano's unified memory (LPDDR5 RAM/GPU pool).
3. Initializes `checkpoints/tokenizer.json` for token translation.
4. Links the microphone, local RAG vector store, and neural female TTS into the live conversational loop!

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
