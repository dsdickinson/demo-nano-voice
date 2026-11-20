"""
Dual-Engine LLM Runner for Edge Assistant.
Supports:
  1. 'scratch': Custom PyTorch DecoderLLM built from scratch
  2. 'instruct': Pretrained compact instruction model (e.g. SmolLM2 / Qwen via transformers/ctransformers)
"""

import os
from typing import Generator, Optional, Dict, Any
import torch

from core.model import ModelConfig, DecoderLLM
from core.tokenizer import BPETokenizer
from train import load_checkpoint, get_optimal_device


class BaseLLMRunner:
    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        raise NotImplementedError


class ScratchLLMRunner(BaseLLMRunner):
    """Runner for custom from-scratch PyTorch DecoderLLM."""
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/model_final.pt",
        tokenizer_path: str = "checkpoints/toy_tokenizer.json",
        device: Optional[torch.device] = None
    ):
        self.device = device or get_optimal_device()
        print(f"[*] Loading Scratch LLM checkpoint from {checkpoint_path} onto {self.device}...")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Run 'python tools/train_toy.py' first.")

        self.model, self.config, _ = load_checkpoint(checkpoint_path, device=self.device)
        self.model.eval()

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at '{tokenizer_path}'.")
        self.tokenizer = BPETokenizer.load(tokenizer_path)

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.2
    ) -> Generator[str, None, None]:
        input_ids = torch.tensor([self.tokenizer.encode(prompt, add_bos=True)], device=self.device)
        accumulated = ""
        for token_id in self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_id=self.tokenizer.eos_id
        ):
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
            accumulated += token_str
            if "User:" in accumulated or "\n" in accumulated:
                # Completed conversational turn
                clean_chunk = token_str.replace("User:", "").strip()
                if clean_chunk:
                    yield clean_chunk
                break
            yield token_str


class InstructLLMRunner(BaseLLMRunner):
    """
    Plug-and-play runner for compact edge instruction models (SmolLM2-360M / Qwen2.5-0.5B).
    """
    def __init__(self, model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
        self.model_id = model_id
        self.pipeline = None

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
            import threading
            print(f"[*] Initializing Instruct Model ({model_id})...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
            self.streamer_cls = TextIteratorStreamer
            self.threading = threading
        except Exception as e:
            print(f"[-] Transformers backend unavailable ({e}). Instruct mode will return simulated response.")
            self.model = None

    def stream_generate(self, prompt: str, max_new_tokens: int = 80) -> Generator[str, None, None]:
        if self.model is None:
            # Fallback simulated response
            msg = " I am running in edge instruct mode. How can I assist you today?"
            for word in msg.split():
                yield word + " "
            return

        streamer = self.streamer_cls(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        thread = self.threading.Thread(
            target=self.model.generate,
            kwargs=dict(**inputs, max_new_tokens=max_new_tokens, streamer=streamer, temperature=0.7)
        )
        thread.start()
        for new_text in streamer:
            yield new_text


def create_llm_runner(model_type: str = "scratch") -> BaseLLMRunner:
    if model_type.lower() == "instruct":
        return InstructLLMRunner()
    return ScratchLLMRunner()
