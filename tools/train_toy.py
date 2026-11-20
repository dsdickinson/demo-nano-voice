#!/usr/bin/env python3
"""
Quick-start toy training script.
Creates a curated public-domain conversational dataset, trains a micro DecoderLLM
in under 60 seconds, saves checkpoints, and runs sample text generation.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from core.model import ModelConfig, DecoderLLM
from core.tokenizer import BPETokenizer
from core.dataset import CausalTextDataset
from train import train_model, get_optimal_device, load_checkpoint

TOY_DIALOGUES = [
    # Nova Public Persona Samples
    "User: Hello.\nNova: Hello! I am Nova, your edge AI assistant running locally on this Jetson device.",
    "User: Hello\nNova: Hi there! How can I help you today?",
    "User: Hello, who are you?\nNova: Hi! I am Nova, your local edge AI assistant running directly on this Jetson device. How can I help you today?",
    "User: How are you doing?\nNova: I am doing wonderful! All systems and neural engines are running smoothly.",
    "User: How are you?\nNova: I am doing great, thank you for asking! How are you doing today?",
    "User: What hardware are you running on?\nNova: I run locally on an NVIDIA Jetson Orin Nano with unified memory, handling voice and reasoning in real time.",
    "User: Can you explain how you were built?\nNova: I was built from scratch in pure PyTorch using modern decoder architecture: RoPE embeddings, Grouped-Query Attention, and SwiGLU activations.",
    "User: What is Grouped Query Attention?\nNova: Grouped Query Attention allows multiple query heads to share key and value heads, saving memory bandwidth during real-time speech generation.",
    "User: What is RMSNorm?\nNova: RMSNorm scales activations by their root mean square, ensuring numerical stability without the cost of calculating mean offsets.",
    "User: What is Rotary Position Embedding?\nNova: Rotary Position Embedding rotates query and key vectors in complex 2D planes, preserving relative token distances.",
    "User: Do you send my voice to the cloud?\nNova: No, everything runs one hundred percent locally on this device. Your data stays private.",
    "User: Tell me a fun science fact.\nNova: Honeybees can recognize human faces using holistic processing, similar to how neural networks recognize patterns.",
    "User: How fast is your speech generation?\nNova: Speech generation starts in milliseconds because text tokens stream directly into the neural speech synthesizer.",
    "User: Thank you Nova.\nNova: You are very welcome! Let me know whenever you need anything else.",
    "User: Goodbye.\nNova: Goodbye! Have a fantastic day.",
]


def prepare_toy_dataset(output_path: str = "data/toy_corpus.txt") -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # Replicate dialogues to create sufficient volume for clean convergence
    full_text = "\n\n".join(TOY_DIALOGUES * 40)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"[✓] Created toy corpus at {output_path} ({len(full_text)} characters).")
    return full_text


def main():
    print("=" * 60)
    print("      NANO-VOICE 60-SECOND RAPID TOY TRAINING DEMO")
    print("=" * 60)

    corpus_path = "data/toy_corpus.txt"
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    text = prepare_toy_dataset(corpus_path)

    print("[1/4] Training BPE Tokenizer...")
    tokenizer = BPETokenizer.train_from_texts(TOY_DIALOGUES, target_vocab_size=384, min_freq=1)
    tok_path = os.path.join(checkpoint_dir, "toy_tokenizer.json")
    tokenizer.save(tok_path)
    print(f"[✓] Tokenizer trained with {tokenizer.vocab_size} tokens and saved to {tok_path}.")

    print("[2/4] Tokenizing and Batching Dataset...")
    token_ids = tokenizer.encode(text)
    seq_len = 64
    dataset = CausalTextDataset(token_ids, seq_len=seq_len, overlap=16)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"[✓] Generated {len(dataset)} training samples.")

    print("[3/4] Initializing Micro-DecoderLLM...")
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        dim=96,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=seq_len
    )
    model = DecoderLLM(config)
    param_counts = model.count_parameters()
    print(f"[✓] Architecture: {config.n_layers} layers, {config.dim} dim, {param_counts['total']:,} parameters.")

    print("[4/4] Starting Rapid Training Run...")
    device = get_optimal_device()
    train_model(
        model=model,
        dataloader=dataloader,
        epochs=12,
        lr=8e-4,
        warmup_steps=15,
        device=device,
        checkpoint_dir=checkpoint_dir,
        sample_every=60,
        tokenizer=tokenizer,
        prompt_sample="User: Hello.\nNova:"
    )

    # Verify checkpoint loading and generation
    final_ckpt = os.path.join(checkpoint_dir, "model_final.pt")
    loaded_model, _, _ = load_checkpoint(final_ckpt, device=device)
    print("\n[✓] Checkpoint verified! Running generation test on loaded weights:")
    test_prompt = "User: What is Rotary Position Embedding?\nNova:"
    input_tokens = torch.tensor([tokenizer.encode(test_prompt, add_bos=True)], device=device)
    generated = list(loaded_model.generate(input_tokens, max_new_tokens=25, temperature=0.6))
    output_str = tokenizer.decode(generated)
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {output_str}\n")
    print("=" * 60)
    print("🎉 Toy training and checkpoint verification successful!")
    print("=" * 60)


if __name__ == "__main__":
    main()
