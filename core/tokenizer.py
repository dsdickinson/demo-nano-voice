"""
Byte-Pair Encoding (BPE) Tokenizer from Scratch.
Implements byte-level BPE:
  - Guarantees 0% out-of-vocabulary (OOV) rate on any UTF-8 text
  - Supports training on custom corpora, serialization to JSON
  - Handles special tokens (<pad>, <s>, </s>, <unk>, <context>)
"""

from typing import List, Dict, Tuple, Optional
import json
import os


class BPETokenizer:
    """
    Byte-Level Byte-Pair Encoding (BPE) Tokenizer.
    Clean-room implementation for educational clarity and zero external dependencies.
    """
    SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<context>", "</context>"]

    def __init__(self, vocab: Optional[Dict[int, bytes]] = None, merges: Optional[List[Tuple[bytes, bytes]]] = None):
        self.vocab: Dict[int, bytes] = vocab or {}
        self.inverse_vocab: Dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.merges: List[Tuple[bytes, bytes]] = merges or []
        self.merge_ranks: Dict[Tuple[bytes, bytes], int] = {m: i for i, m in enumerate(self.merges)}

        # Special token IDs
        self.pad_id = self._get_or_add_special("<pad>")
        self.bos_id = self._get_or_add_special("<s>")
        self.eos_id = self._get_or_add_special("</s>")
        self.unk_id = self._get_or_add_special("<unk>")

    def _get_or_add_special(self, token_str: str) -> int:
        token_bytes = token_str.encode("utf-8")
        if token_bytes in self.inverse_vocab:
            return self.inverse_vocab[token_bytes]
        idx = len(self.vocab)
        self.vocab[idx] = token_bytes
        self.inverse_vocab[token_bytes] = idx
        return idx

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train_from_texts(cls, texts: List[str], target_vocab_size: int = 1000, min_freq: int = 2) -> "BPETokenizer":
        """
        Train a byte-level BPE tokenizer on a list of input texts.
        Initializes vocabulary with all 256 individual byte values + special tokens,
        then iteratively merges the most frequent adjacent pairs.
        """
        tokenizer = cls()

        # Add all 256 single-byte base tokens
        for b in range(256):
            b_bytes = bytes([b])
            if b_bytes not in tokenizer.inverse_vocab:
                idx = len(tokenizer.vocab)
                tokenizer.vocab[idx] = b_bytes
                tokenizer.inverse_vocab[b_bytes] = idx

        # Pre-tokenize texts into list of byte chunks
        # Each word is represented as a list of 1-byte sequences
        words: List[List[bytes]] = []
        for text in texts:
            for word in text.strip().split():
                if word:
                    words.append([bytes([b]) for b in word.encode("utf-8")])

        # Iteratively merge pairs until target_vocab_size is reached
        while len(tokenizer.vocab) < target_vocab_size:
            # Count pair frequencies
            pair_counts: Dict[Tuple[bytes, bytes], int] = {}
            for word in words:
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            # Find best pair
            best_pair, highest_freq = max(pair_counts.items(), key=lambda item: item[1])
            if highest_freq < min_freq:
                break

            # Register new merge and new vocab token
            new_token = best_pair[0] + best_pair[1]
            new_id = len(tokenizer.vocab)
            tokenizer.vocab[new_id] = new_token
            tokenizer.inverse_vocab[new_token] = new_id
            tokenizer.merges.append(best_pair)
            tokenizer.merge_ranks[best_pair] = len(tokenizer.merges) - 1

            # Update words with new merged token
            new_words = []
            for word in words:
                i = 0
                new_word = []
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words

        return tokenizer

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text string into token IDs."""
        if not text:
            tokens = []
            if add_bos: tokens.insert(0, self.bos_id)
            if add_eos: tokens.append(self.eos_id)
            return tokens

        # Break text into initial 1-byte units
        raw_bytes = [bytes([b]) for b in text.encode("utf-8")]

        # Apply learned merges in rank order
        while len(raw_bytes) >= 2:
            pairs = [(raw_bytes[i], raw_bytes[i + 1]) for i in range(len(raw_bytes) - 1)]
            # Find the lowest rank merge available
            min_rank = float("inf")
            best_pair = None
            for p in pairs:
                rank = self.merge_ranks.get(p)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    best_pair = p

            if best_pair is None:
                break

            # Merge occurrences of best_pair
            new_bytes = []
            i = 0
            while i < len(raw_bytes):
                if i < len(raw_bytes) - 1 and (raw_bytes[i], raw_bytes[i + 1]) == best_pair:
                    new_bytes.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_bytes.append(raw_bytes[i])
                    i += 1
            raw_bytes = new_bytes

        # Map merged byte chunks to IDs
        token_ids = [self.inverse_vocab.get(b, self.unk_id) for b in raw_bytes]

        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to a UTF-8 string."""
        byte_chunks = []
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}

        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            if tid in self.vocab:
                byte_chunks.append(self.vocab[tid])

        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, filepath: str):
        """Save tokenizer vocabulary and merges to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            "merges": [[list(m[0]), list(m[1])] for m in self.merges]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """Load tokenizer from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        merges = [(bytes(m[0]), bytes(m[1])) for m in data["merges"]]
        return cls(vocab=vocab, merges=merges)
