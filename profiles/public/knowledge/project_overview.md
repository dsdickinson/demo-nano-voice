# Nano-Voice Architecture Overview

Nano-Voice is an edge AI conversational assistant designed specifically for embedded silicon like the NVIDIA Jetson Orin Nano (8GB).

## Hardware Specifications
- Board: NVIDIA Jetson Orin Nano Developer Kit
- Memory: 8GB 128-bit LPDDR5 unified memory with 68 GB/s bandwidth
- GPU: Ampere architecture with 1024 CUDA cores and 32 Tensor Cores
- Power envelope: 7W to 15W configurable operating modes

## Neural Network Core
The model architecture uses a decoder-only transformer implemented in pure PyTorch with modern conventions:
- Rotary Position Embeddings (RoPE) for relative sequence position encoding
- Grouped-Query Attention (GQA) with 8 query heads and 4 key-value heads to reduce memory consumption
- RMSNorm for scale invariance without mean shifting
- SwiGLU activation in the feed-forward projection layers
