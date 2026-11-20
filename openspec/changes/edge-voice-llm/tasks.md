## 1. Project Scaffolding & Environment Setup

- [x] 1.1 Create project directory structure (`core/`, `rag/`, `assistant/`, `docs/`, `profiles/public/`, `tools/`, `tests/`) with `.gitignore` configured to keep all personal data safe
- [x] 1.2 Create `requirements.txt` and `pyproject.toml` with versioned dependencies for PyTorch, torchaudio, faster-whisper, silero-vad, piper-tts, faiss-cpu, and sounddevice
- [x] 1.3 Add environment verification script `tools/check_env.py` validating CUDA/MPS availability and audio devices

## 2. Modern Transformer Core from Scratch

- [x] 2.1 Implement RMSNorm in `core/model.py` with numerical stability epsilon and learnable gain
- [x] 2.2 Implement Rotary Position Embeddings (RoPE) with frequency precomputation and complex rotation in `core/model.py`
- [x] 2.3 Implement Grouped-Query Attention (GQA) with KV-cache support and native PyTorch scaled dot-product attention in `core/model.py`
- [x] 2.4 Implement SwiGLU Feed-Forward Network and assemble full `TransformerBlock` and `DecoderLLM` in `core/model.py`
- [x] 2.5 Implement Byte-Pair Encoding (BPE) tokenizer in `core/tokenizer.py` with save/load and encode/decode routines
- [x] 2.6 Add unit tests for transformer forward pass, KV-cache autoregressive generation, and tensor shapes in `tests/test_model.py`

## 3. Training Pipeline & Sample Data

- [x] 3.1 Implement dataset reader and token sequence batcher with dynamic padding in `core/dataset.py`
- [x] 3.2 Build training script `train.py` with AdamW optimizer, cosine learning rate scheduler with warmup, and mixed precision (AMP)
- [x] 3.3 Add checkpoint save and resume functionality preserving model weights, hyperparameter configuration, and optimizer state
- [x] 3.4 Add sample conversational dataset loader and quick sanity-check training run script `tools/train_toy.py`

## 4. Local RAG Engine

- [x] 4.1 Implement document ingestion and text chunking in `rag/ingest.py`
- [x] 4.2 Implement vector store and semantic search using micro-embeddings and FAISS in `rag/retriever.py`
- [x] 4.3 Implement context prompt augmentation formatting retrieved chunks for the LLM in `rag/augment.py`
- [x] 4.4 Add public showcase knowledge documents in `profiles/public/knowledge/` and unit tests in `tests/test_rag.py`

## 5. Jetson Voice Pipeline (Audio I/O, VAD, STT, TTS)

- [x] 5.1 Implement microphone capture and Silero VAD speech segmentation in `assistant/audio.py`
- [x] 5.2 Implement Speech-to-Text with Faster-Whisper in `assistant/stt.py`
- [x] 5.3 Implement neural female voice synthesis using Piper TTS / Kokoro with streaming sentence chunking in `assistant/tts.py`
- [x] 5.4 Implement echo suppression / audio ducking during playback to prevent acoustic feedback
- [x] 5.5 Implement dual-engine LLM runner in `assistant/llm_runner.py` supporting both custom scratch weights and compact instruct models with public sanitized persona ("Nova")
- [x] 5.6 Build the real-time coordinator `assistant/pipeline.py` and interactive entrypoint `run_voice.py`

## 6. Documentation & Portfolio Polish

- [x] 6.1 Write `docs/architecture.md` detailing the mathematical formulations, RoPE, GQA, SwiGLU, and tensor flowcharts
- [x] 6.2 Write `docs/jetson_setup.md` covering JetPack configuration, USB audio setup, and memory optimization
- [x] 6.3 Write high-impact `README.md` with visual architecture diagrams, installation guide, quickstart commands, and demo walkthrough
