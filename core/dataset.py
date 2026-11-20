"""
Dataset loader and sequence batcher for causal language model pretraining.
"""

from typing import List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from core.tokenizer import BPETokenizer


class CausalTextDataset(Dataset):
    """
    Dataset for autoregressive causal language modeling.
    Splits token stream into chunks of (seq_len + 1) tokens,
    yielding (x, y) where y is x shifted by one token to the right.
    """
    def __init__(
        self,
        token_ids: List[int],
        seq_len: int = 128,
        overlap: int = 0
    ):
        self.seq_len = seq_len
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        step = max(1, seq_len - overlap)
        for i in range(0, len(token_ids) - seq_len, step):
            chunk = token_ids[i : i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class DialogueDataset(Dataset):
    """
    Dataset formatted for multi-turn conversational or Q&A pairs.
    Handles dynamic padding to max sequence length in batch.
    """
    def __init__(
        self,
        conversations: List[List[Tuple[str, str]]],  # List of [(role, text), ...]
        tokenizer: BPETokenizer,
        max_seq_len: int = 256,
    ):
        self.samples = []
        for convo in conversations:
            # Build conversation string
            convo_str = ""
            for role, text in convo:
                convo_str += f"{role.capitalize()}: {text.strip()}\n"

            tokens = tokenizer.encode(convo_str, add_bos=True, add_eos=True)
            if len(tokens) > 2:
                # Truncate to max_seq_len + 1
                tokens = tokens[: max_seq_len + 1]
                x = torch.tensor(tokens[:-1], dtype=torch.long)
                y = torch.tensor(tokens[1:], dtype=torch.long)
                self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def pad_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_token_id: int = 0):
    """Collate function that dynamically pads sequences in batch to max length."""
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)

    batch_x = torch.full((len(xs), max_len), pad_token_id, dtype=torch.long)
    batch_y = torch.full((len(ys), max_len), -100, dtype=torch.long)  # -100 ignored by CrossEntropyLoss

    for i, (x, y) in enumerate(zip(xs, ys)):
        batch_x[i, :x.size(0)] = x
        batch_y[i, :y.size(0)] = y

    return batch_x, batch_y
