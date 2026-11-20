## Why

Creating a production-grade, end-to-end edge AI voice assistant that combines deep learning fundamentals (training a modern LLM from scratch) with physical edge hardware deployment (NVIDIA Jetson Orin Nano) creates a premier portfolio project. This demonstrates mastery across the complete AI lifecycle: from raw tensor math (RoPE, GQA, SwiGLU, RMSNorm) to real-time physical streaming systems (VAD, STT, RAG, and natural neural TTS).

## What Changes

Build a complete open-source repository showcasing:
- A custom, modern PyTorch decoder-only LLM architecture implemented from scratch with zero high-level framework wrappers.
- A portable BPE tokenizer and training pipeline runnable on Mac (MPS), CUDA, and cloud GPUs.
- A lightweight, local Retrieval-Augmented Generation (RAG) engine enabling external knowledge ingestion with sub-20ms latency.
- A real-time voice-to-voice assistant pipeline optimized for the Jetson Orin Nano (8GB) featuring natural female speech synthesis, streaming audio I/O, and a dual-engine LLM backend (scratch model or compact instruct model).
- A dual-profile architecture separating a sanitized **Public Showcase Profile** (safe for GitHub/demos) from an isolated, git-ignored **Private Personal Profile** (custom persona, personal knowledge base, behavioral instructions, and private memory).
- Comprehensive technical documentation including architecture diagrams, tensor shape flowcharts, and Jetson deployment guides.

## Capabilities

### New Capabilities
- `transformer-core`: Implementation of modern decoder-only LLM primitives (RMSNorm, RoPE, Grouped-Query Attention, SwiGLU FFN, KV Cache) and a custom BPE tokenizer from scratch in pure PyTorch.
- `model-training`: Training loop with learning rate warmup, cosine decay, AdamW optimizer, mixed precision (FP16/BF16), and checkpoint management.
- `local-rag`: Ingestion, chunking, embedding, and vector similarity search over local documents (`knowledge/`) to ground the assistant with zero-latency personal memory.
- `voice-pipeline`: Real-time audio streaming loop integrating Voice Activity Detection (Silero VAD), Speech-to-Text (Faster-Whisper), selectable LLM backend, and Text-to-Speech (Piper/Kokoro natural female voice) within the Jetson Orin Nano memory budget.

### Modified Capabilities
*(None - brand new project)*

## Impact
- **Codebase**: Creates a clean, modular repository with `core/` (architecture & training), `rag/` (retrieval engine), `assistant/` (voice pipeline), and `docs/` (hardware & math specs).
- **Dependencies**: Adds PyTorch, torchaudio, tiktoken/tokenizers, faiss-cpu/sqlite-vec, faster-whisper, piper-tts / kokoro-onnx, and silero-vad.
- **Hardware Targets**: Primary target is NVIDIA Jetson Orin Nano 8GB (JetPack 5/6), with full cross-platform training support on macOS (Apple Silicon MPS), Linux (NVIDIA CUDA), and Windows.
