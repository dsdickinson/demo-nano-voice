## Context

The goal is to build a modern, production-grade edge AI voice assistant that runs on the NVIDIA Jetson Orin Nano (8GB) and showcases deep learning engineering from first principles. The project features a dual-engine architecture:
1. A modern decoder-only LLM (RoPE, GQA, SwiGLU, RMSNorm) written from scratch in pure PyTorch, with a custom BPE tokenizer and device-agnostic training loop.
2. A low-latency local voice loop (Silero VAD + Faster-Whisper + Local RAG + Piper/Kokoro neural TTS) running within the 8GB unified memory envelope of the Jetson Orin Nano.

The system supports selecting between the custom from-scratch model (proving deep ML fundamentals) and an optimized compact instruction-tuned model (providing witty, broad conversational intelligence) with a natural female voice.

## Goals / Non-Goals

**Goals:**
- **Zero-Black-Box Transformer Core**: Build an educational, readable, yet state-of-the-art transformer architecture in pure PyTorch (`core/model.py`) matching 2024-2026 conventions (RoPE, GQA, SwiGLU, RMSNorm, KV Cache).
- **Self-Contained Tokenizer & Training**: Build custom BPE tokenization and a clean training script supporting mixed precision (FP16/BF16) and cosine learning rate schedules on Mac (MPS), CUDA, and Colab.
- **Sub-100ms Local RAG Engine**: Embed and index documents in `knowledge/` into a vector database (FAISS/sqlite-vec) with an embedding footprint under 80MB RAM.
- **Physical Voice-to-Voice Loop on Jetson Orin Nano**: Integrate real-time audio capture, Voice Activity Detection (Silero VAD), Speech-to-Text (Whisper), prompt assembly, token streaming, and neural female TTS (Piper/Kokoro) with an end-to-end response time under 1.5 seconds.
- **Clean CLI & Dual-Engine Mode**: Provide `python run_voice.py --model scratch` and `--model instruct` modes.
- **Portfolio Documentation**: Visual tensor flows, mathematical explanations, and Jetson setup guides.

**Non-Goals:**
- Pretraining a multi-billion parameter model on hundreds of GPUs (this project focuses on edge-scale ~15M–30M params for scratch training, with plug-and-play inference for existing 350M–1B instruction models).
- Cloud API dependencies (the entire pipeline runs 100% locally and offline on the Jetson Orin Nano).
- Video / vision processing (purely focused on speech, text, retrieval, and LLM reasoning).

## Decisions

### 1. Transformer Core: LLaMA-3 / Mistral Primitives
- **RMSNorm over LayerNorm**: Reduces compute overhead by removing mean-centering while maintaining training stability.
- **Rotary Position Embeddings (RoPE)**: Applied to query and key projections instead of absolute learned positional embeddings. Enables relative position awareness and context length extrapolation.
- **Grouped-Query Attention (GQA)**: $N$ query heads share $M$ key-value heads ($N > M$). Dramatically reduces KV-cache memory bandwidth during autoregressive token generation on memory-constrained hardware like the Jetson Orin Nano.
- **SwiGLU Activation**: Used in the Feed-Forward Network ($FFN_{SwiGLU}(x) = (xW_1 \cdot \text{SiLU}(xW_{gate}))W_2$), matching LLaMA-3 and DeepSeek standards.
- **Native PyTorch SDPA**: Uses `torch.nn.functional.scaled_dot_product_attention` for FlashAttention acceleration with zero external C++ build requirements.

### 2. Tokenizer: Byte-Level BPE
- Clean Byte-Pair Encoding implementation capable of training on local text corpora with configurable vocabulary size (e.g. 4096 or 8192 tokens for fast local convergence), with fallback/compatibility with OpenAI `tiktoken`.

### 3. Local RAG: Micro-Embeddings + FAISS
- Uses `all-MiniLM-L6-v2` or `bge-micro-v2` (~60MB RAM footprint).
- FAISS (IndexFlatIP with normalized vectors for cosine similarity) provides sub-5ms vector lookups over thousands of text chunks.
- A simple drop directory `knowledge/` auto-indexes `.md` and `.txt` files on startup.

### 4. Audio Pipeline: Silero VAD + Faster-Whisper + Piper TTS
- **VAD**: Silero VAD running on CPU/GPU buffers detects when the user begins and finishes speaking without requiring push-to-talk.
- **STT**: `faster-whisper` (`tiny.en` or `base.en`) running in INT8/FP16 executes transcription in ~150ms on Orin Nano.
- **TTS**: `piper-tts` with high-quality female ONNX voice models (e.g. `en_US-lessac-medium` or `en_US-amy-medium`) or Kokoro-82M delivers fluid natural speech with audio generation latency under 50ms per sentence chunk.
- **Sentence-Chunked Streaming**: As the LLM streams output tokens, a sentence boundary detector flushes complete clauses to the TTS engine so speech playback starts before the LLM finishes generating the entire response.

### 5. Memory Budget (8GB Jetson Orin Nano)
```
Component                 Memory Target
OS + JetPack runtime      ~1.2 GB
VAD + Audio Buffers       ~100 MB
Faster-Whisper (base.en)  ~300 MB
Embedding Model + FAISS   ~100 MB
LLM (Scratch or Instruct) ~150 MB (25M) to 1.5 GB (4-bit 1B)
Piper TTS                 ~150 MB
Headroom / Dynamic Cache  ~4.0+ GB free
```

### 6. Profile Isolation: Public Showcase vs. Private Personal Profile
To enable sharing the project publicly on GitHub without exposing personal data or identity:
- **Directory Layout**:
  - `profiles/public/`: Contains default, sanitized configuration, generic persona prompt ("Nova/Aria: an articulate edge AI female assistant"), and generic demo knowledge files (technical docs, open-domain facts). Fully tracked in Git.
  - `profiles/private/`: Ignored via `.gitignore`. Contains user personal information, customized behavioral instructions ("how you want it to treat you, personal memory, relationship to user"), and private knowledge documents.
  - `profiles/private.example/`: Tracked in Git as a template for users setting up their personal profile.
- **Runtime Flag**:
  - `python run_voice.py --profile public` (default, completely clean and shareable).
  - `python run_voice.py --profile private` (loads personal persona, memories, private RAG vector index).
- **Safety Enforcement**:
  - The vector index and raw files for `profiles/private/` are stored separately to prevent accidental embedding leakage into the public repository.

## Risks / Trade-offs

- **[Risk: Scratch Model Hallucinations / Coherence]** → *Mitigation*: The RAG engine feeds factual context directly into the prompt; training corpus uses clean conversational pairs (e.g., TinyStories or curated dialogues); and the dual-engine switch allows demoing the instruct model for general intelligence.
- **[Risk: Audio Latency on Jetson Orin Nano]** → *Mitigation*: Stream tokens from the LLM directly into sentence-level TTS synthesis; use INT8/FP16 quantization for Whisper; pre-allocate KV cache.
- **[Risk: Audio Feedback / Echo Loop]** → *Mitigation*: Implement audio ducking and mute VAD during TTS playback to prevent the assistant from hearing its own voice.
- **[Risk: Cross-Platform Training Dependency Hell]** → *Mitigation*: Keep core transformer and training code pure PyTorch with zero non-standard C++ extensions so it runs seamlessly on Mac MPS, NVIDIA CUDA, and Linux CPU.

## Open Questions

- Default sample rate and mic device selection utility for Jetson USB audio peripherals (provide a handy `tools/test_audio.py` script).
