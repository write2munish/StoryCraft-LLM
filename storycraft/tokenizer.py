"""
StoryCraft Tokenizer: Canonical Subword BPE Tokenizer for StoryCraft-LLM.
"""

import pickle
import os
from collections import Counter
from typing import Dict, List, Tuple

class StoryTokenizer:
    def __init__(self, target_vocab_size: int = 512):
        self.target_vocab_size = target_vocab_size
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: Dict[int, str] = {}
        self.inv_vocab: Dict[str, int] = {}
        self.special_tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
        self._init_specials()

    def _init_specials(self):
        for idx, token in enumerate(self.special_tokens):
            self.vocab[idx] = token
            self.inv_vocab[token] = idx

    @property
    def pad_id(self) -> int:
        return self.inv_vocab["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.inv_vocab["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.inv_vocab["<|eos|>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, corpus_text: str):
        """Train Byte-Pair Encoding merges up to target_vocab_size."""
        words = corpus_text.lower().split()
        word_freqs = Counter([tuple(list(w)) for w in words])

        base_chars = set()
        for w in words:
            base_chars.update(list(w))

        all_symbols = list(self.special_tokens) + sorted(list(base_chars))
        num_merges = max(0, self.target_vocab_size - len(all_symbols))

        for _ in range(num_merges):
            pairs = Counter()
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pairs[(word_tuple[i], word_tuple[i + 1])] += freq
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]
            p1, p2 = best_pair
            new_freqs = {}
            for word_tuple, freq in word_freqs.items():
                new_tuple = []
                i = 0
                while i < len(word_tuple):
                    if i < len(word_tuple) - 1 and word_tuple[i] == p1 and word_tuple[i + 1] == p2:
                        new_tuple.append(p1 + p2)
                        i += 2
                    else:
                        new_tuple.append(word_tuple[i])
                        i += 1
                new_freqs[tuple(new_tuple)] = freq
            word_freqs = new_freqs
            self.merges[best_pair] = p1 + p2
            all_symbols.append(p1 + p2)

        self.vocab = {idx: token for idx, token in enumerate(all_symbols)}
        self.inv_vocab = {token: idx for idx, token in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        words = text.lower().split()
        token_ids = [self.bos_id] if add_special_tokens else []

        for w in words:
            w_tuple = tuple(list(w))
            for pair, merged in self.merges.items():
                p1, p2 = pair
                new_tuple = []
                i = 0
                while i < len(w_tuple):
                    if i < len(w_tuple) - 1 and w_tuple[i] == p1 and w_tuple[i + 1] == p2:
                        new_tuple.append(merged)
                        i += 2
                    else:
                        new_tuple.append(w_tuple[i])
                        i += 1
                w_tuple = tuple(new_tuple)
            for sym in w_tuple:
                token_ids.append(self.inv_vocab.get(sym, self.inv_vocab.get("<|unk|>", 3)))

        if add_special_tokens:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        symbols = [self.vocab.get(i, "") for i in token_ids]
        filtered = [s for s in symbols if not s.startswith("<|")]
        return " ".join(filtered)

    def save_pretrained(self, save_path: str):
        """Save tokenizer configuration and merges."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({
                "target_vocab_size": self.target_vocab_size,
                "merges": self.merges,
                "vocab": self.vocab,
                "inv_vocab": self.inv_vocab
            }, f)

    @classmethod
    def from_pretrained(cls, load_path: str) -> "StoryTokenizer":
        """Load tokenizer from file."""
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        tok = cls(target_vocab_size=data["target_vocab_size"])
        tok.merges = data["merges"]
        tok.vocab = data["vocab"]
        tok.inv_vocab = data["inv_vocab"]
        return tok
