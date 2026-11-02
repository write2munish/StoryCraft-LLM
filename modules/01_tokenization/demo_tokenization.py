"""
Module 01: Data Pipeline & Tokenization
=========================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. The Subword Tokenization problem (why characters are too long & words are too sparse).
2. A minimal Byte-Pair Encoding (BPE) tokenizer implemented from scratch.
3. Special control tokens: <|bos|>, <|eos|>, <|pad|>.
4. PyTorch Causal Dataset creation with shifted target sequences (x vs y).

Run this script:
    python3 modules/01_tokenization/demo_tokenization.py
"""

import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
from storycraft import set_seed

from collections import Counter
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# 1. Minimal Byte-Pair Encoding (BPE) Implementation
# ============================================================================
class MiniBPETokenizer:
    """
    Pedagogical Byte-Pair Encoding (BPE) Tokenizer.
    Builds a subword vocabulary by iteratively merging the most frequent token pairs.
    """
    def __init__(self, target_vocab_size: int = 60):
        self.target_vocab_size = target_vocab_size
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: Dict[int, str] = {}
        self.inv_vocab: Dict[str, int] = {}

    def get_stats(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """Count adjacent symbol pairs across all split words."""
        pairs = Counter()
        for word_tuple, freq in word_freqs.items():
            for i in range(len(word_tuple) - 1):
                pairs[(word_tuple[i], word_tuple[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair: Tuple[str, str], word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """Merge all occurrences of the target pair in the dictionary."""
        new_freqs = {}
        p1, p2 = pair
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
        return new_freqs

    def train(self, corpus: str):
        """
        Train BPE merges on the corpus until target vocabulary size is reached.
        """
        print(f"[Tokenizer] Training BPE up to target vocab size: {self.target_vocab_size}...")
        
        # 1. Initialize base vocabulary with single characters and special tokens
        words = corpus.lower().split()
        word_freqs = Counter([tuple(list(w)) for w in words])

        # Track base characters
        base_chars = set()
        for w in words:
            base_chars.update(list(w))

        # Special tokens
        special_tokens = ["<|pad|>", "<|bos|>", "<|eos|>"]
        all_symbols = special_tokens + sorted(list(base_chars))

        # Iteratively merge most frequent adjacent pairs
        current_vocab_size = len(all_symbols)
        num_merges = self.target_vocab_size - current_vocab_size

        for step in range(max(0, num_merges)):
            pairs = self.get_stats(word_freqs)
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]
            word_freqs = self.merge_vocab(best_pair, word_freqs)
            merged_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged_token
            all_symbols.append(merged_token)

        # Build ID lookups
        self.vocab = {idx: token for idx, token in enumerate(all_symbols)}
        self.inv_vocab = {token: idx for idx, token in self.vocab.items()}
        print(f"[Tokenizer] Finished training! Final vocabulary size: {len(self.vocab)} tokens.")

    def encode(self, text: str) -> List[int]:
        """
        Encode raw string into list of token IDs.
        """
        words = text.lower().split()
        token_ids = [self.inv_vocab.get("<|bos|>", 1)]

        for w in words:
            w_tuple = tuple(list(w))
            # Apply learned merges in order
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

            # Map symbols to IDs
            for sym in w_tuple:
                token_ids.append(self.inv_vocab.get(sym, self.inv_vocab["<|pad|>"]))

        token_ids.append(self.inv_vocab.get("<|eos|>", 2))
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to human-readable string."""
        symbols = [self.vocab.get(i, "") for i in token_ids]
        # Filter out special control tokens for presentation
        filtered = [s for s in symbols if not s.startswith("<|")]
        return " ".join(filtered)


# ============================================================================
# 2. PyTorch Causal Language Model Dataset
# ============================================================================
class StoryCausalDataset(Dataset):
    """
    Prepares fixed-length overlapping sequence chunks for causal next-token prediction.
    
    Given a stream of tokens: [t_0, t_1, t_2, t_3, t_4]
    Input  x: [t_0, t_1, t_2, t_3]
    Target y: [t_1, t_2, t_3, t_4]  <-- Exactly shifted by 1 position!
    """
    def __init__(self, token_stream: List[int], context_len: int = 8):
        self.context_len = context_len
        self.samples = []
        for i in range(0, len(token_stream) - context_len):
            chunk = token_stream[i : i + context_len + 1]
            x = chunk[:-1]
            y = chunk[1:]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ============================================================================
# 3. Demonstration Runner
# ============================================================================
def main():
    set_seed(42)
    print("=" * 70)
    print("🔤 MODULE 01: DATA PIPELINE & TOKENIZATION")
    print("=" * 70)

    sample_story = (
        "Once upon a time there was a little girl named Lily. "
        "Lily loved to play with her red ball in the green garden. "
        "One sunny afternoon, a friendly dragon visited Lily."
    )

    print("\n1. Training BPE Tokenizer on Children's Story Corpus:")
    tokenizer = MiniBPETokenizer(target_vocab_size=55)
    tokenizer.train(sample_story)

    print("\n2. Inspecting First 15 Vocabulary Items:")
    for idx in range(min(15, len(tokenizer.vocab))):
        print(f"   ID {idx:2d} -> '{tokenizer.vocab[idx]}'")

    print("\n3. Inspecting Merges Learned by BPE:")
    for pair, merged in list(tokenizer.merges.items())[:5]:
        print(f"   Merge: {pair[0]} + {pair[1]} => '{merged}'")

    print("\n4. Encoding a Test Story Prompt:")
    test_text = "Lily loved her red ball"
    encoded_ids = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"   Original text: \"{test_text}\"")
    print(f"   Encoded IDs:   {encoded_ids}")
    print(f"   Tokens:        {[tokenizer.vocab[i] for i in encoded_ids]}")
    print(f"   Decoded text:  \"{decoded_text}\"")

    print("\n5. Building PyTorch Causal Training Batches (x vs shifted y):")
    # Simulate a stream of 30 tokens
    simulated_stream = encoded_ids * 5
    dataset = StoryCausalDataset(simulated_stream, context_len=6)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for batch_idx, (x, y) in enumerate(dataloader):
        print(f"\n   --- Batch {batch_idx + 1} ---")
        print(f"   Input  x (shape {x.shape}): {x[0].tolist()}")
        print(f"   Target y (shape {y.shape}): {y[0].tolist()}")
        print("   Notice that y is x shifted by exactly 1 token to the right!")
        print(f"   x[0] token: '{tokenizer.vocab[x[0, 0].item()]}' -> predicts y[0] token: '{tokenizer.vocab[y[0, 0].item()]}'")
        break

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. Subword tokenization balances vocabulary size and sequence length.")
    print("2. The entire autoregressive training setup relies on shifted causal pairs:")
    print("   At position t, the network takes tokens [0..t] and is penalized for")
    print("   failing to predict target token [t+1].")
    print("👉 Next: Module 02 converts these discrete IDs into dense vector geometry!")
    print("=" * 70)


if __name__ == "__main__":
    main()
