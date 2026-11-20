#!/usr/bin/env python3
"""
Portable, high-throughput training script for DecoderLLM.
Supports:
  - Automatic device detection (CUDA, Apple Silicon MPS, or CPU)
  - Mixed precision training (AMP)
  - AdamW optimizer with cosine learning rate warmup & decay
  - Checkpoint saving & resuming
  - Validation loss and live generation sampling
"""

import os
import sys
import math
import argparse
from typing import Optional, Dict, Any, Tuple

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.model import ModelConfig, DecoderLLM
from core.tokenizer import BPETokenizer
from core.dataset import CausalTextDataset


def get_optimal_device() -> torch.device:
    """Select optimal compute device available on host."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1
):
    """Create cosine learning rate scheduler with warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    filepath: str,
    model: DecoderLLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: ModelConfig,
    tokenizer_path: Optional[str] = None
):
    """Save model weights, configuration, and training state to checkpoint file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    checkpoint = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "tokenizer_path": tokenizer_path
    }
    torch.save(checkpoint, filepath)
    print(f"[✓] Saved checkpoint at step {step} to {filepath}")


def load_checkpoint(
    filepath: str,
    device: torch.device
) -> Tuple[DecoderLLM, ModelConfig, Dict[str, Any]]:
    """Restore model and configuration from checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    filtered_config = {k: v for k, v in checkpoint["config"].items() if k in valid_fields}
    config = ModelConfig(**filtered_config)
    model = DecoderLLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config, checkpoint


def train_model(
    model: DecoderLLM,
    dataloader: DataLoader,
    epochs: int = 5,
    lr: float = 3e-4,
    warmup_steps: int = 50,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "checkpoints",
    sample_every: int = 100,
    tokenizer: Optional[BPETokenizer] = None,
    prompt_sample: str = "The assistant",
):
    """Execute complete training loop."""
    device = device or get_optimal_device()
    print(f"[*] Training on device: {device}")
    model.to(device)

    # Filter parameters for weight decay (exclude 1D params like biases & RMSNorm scales)
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0}
    ], lr=lr, betas=(0.9, 0.95))

    total_steps = epochs * len(dataloader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Mixed precision context
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            step += 1
            epoch_loss += loss.item()

            if step % 20 == 0 or step == total_steps:
                cur_lr = scheduler.get_last_lr()[0]
                ppl = math.exp(min(loss.item(), 20.0))  # guard against overflow
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step}/{total_steps}] | Loss: {loss.item():.4f} | PPL: {ppl:.2f} | LR: {cur_lr:.6f}")

            # Sample generation
            if step % sample_every == 0 and tokenizer is not None:
                model.eval()
                input_ids = torch.tensor([tokenizer.encode(prompt_sample, add_bos=True)], device=device)
                tokens = list(model.generate(input_ids, max_new_tokens=30, temperature=0.7))
                sampled_text = tokenizer.decode(tokens)
                print(f"\n--- [Sample @ Step {step}] ---")
                print(f"Prompt: {prompt_sample}")
                print(f"Output: {sampled_text}\n" + "-" * 30)
                model.train()

        avg_loss = epoch_loss / len(dataloader)
        print(f"\n>>> Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}\n")

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "model_final.pt")
    save_checkpoint(final_path, model, optimizer, step, loss.item(), model.config)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train custom DecoderLLM from scratch.")
    parser.add_argument("--data", type=str, default="data/sample_train.txt", help="Path to training text file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Context sequence length")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    print("=" * 60)
    print("           NANO-VOICE LLM TRAINING ENGINE")
    print("=" * 60)

    # For standalone runs, ensure sample data exists or exit gracefully
    if not os.path.exists(args.data):
        print(f"[!] Data file '{args.data}' not found. Run 'tools/train_toy.py' first for quick demo.")
        return

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    print("[*] Training BPE tokenizer on corpus...")
    tokenizer = BPETokenizer.train_from_texts([text], target_vocab_size=1024)
    tok_path = os.path.join(args.checkpoint_dir, "tokenizer.json")
    tokenizer.save(tok_path)

    token_ids = tokenizer.encode(text)
    dataset = CausalTextDataset(token_ids, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        dim=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=args.seq_len
    )
    model = DecoderLLM(config)
    print(f"[*] Initialized DecoderLLM with {model.count_parameters()['total']:,} parameters.")

    train_model(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        tokenizer=tokenizer
    )


if __name__ == "__main__":
    main()
