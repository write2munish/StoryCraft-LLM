"""
train.py: End-to-End Pre-Training Engine on the TinyStories Dataset.
===================================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script trains StoryCraftLM directly on:
- data/raw/tinystories_train.txt (Training set)
- data/raw/tinystories_valid.txt (Validation set)

Features:
1. Pure PyTorch Causal Language Modeling with shifted target sequences (x vs y).
2. Streaming / chunked dataset loader to handle large text files with low memory.
3. AdamW with decoupled weight decay and gradient clipping.
4. Linear Warmup + Cosine Annealing learning rate schedule.
5. Periodic validation evaluating Perplexity (PPL = exp(loss)) on held-out stories.
6. Periodic generation checkpoints ("Once upon a time Lily...") to watch grammar emerge.
7. Saves pre-trained weights directly to checkpoints/storycraft_base.pt.

Usage:
    # Quick sanity check / fast training:
    python3 modules/04_pretraining/train.py --max_steps 50 --batch_size 8

    # Full pre-training run:
    python3 modules/04_pretraining/train.py --max_steps 2000 --batch_size 16 --save
"""

import os
import sys
import math
import time
import argparse
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)

from storycraft import set_seed, StoryTokenizer, StoryCraftLM


class TinyStoriesDataset(Dataset):
    """
    Loads text from TinyStories text files and chunks into fixed context lengths.
    """
    def __init__(self, file_path: str, tokenizer: StoryTokenizer, seq_len: int = 64, max_stories: int = 5000):
        self.seq_len = seq_len
        self.samples = []

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")

        print(f"   [DataLoader] Loading stories from {os.path.basename(file_path)}...")
        raw_text = []
        with open(file_path, "r", encoding="utf-8") as f:
            story_count = 0
            current_story = []
            for line in f:
                if "<|endofstory|>" in line:
                    raw_text.append(" ".join(current_story))
                    current_story = []
                    story_count += 1
                    if story_count >= max_stories:
                        break
                else:
                    line_str = line.strip()
                    if line_str:
                        current_story.append(line_str)

        combined_text = " ".join(raw_text)
        token_ids = tokenizer.encode(combined_text, add_special_tokens=False)

        # Create overlapping sequence chunks
        stride = seq_len // 2
        for i in range(0, len(token_ids) - seq_len, stride):
            chunk = token_ids[i : i + seq_len + 1]
            self.samples.append(chunk)

        print(f"   ✓ Formatted {len(self.samples)} causal sequence chunks (context_len={seq_len})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_lr_cosine_schedule(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def evaluate(model: StoryCraftLM, val_loader: DataLoader, loss_fn: nn.Module, device: torch.device, max_batches: int = 20) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = loss_fn(logits.view(-1, model.vocab_size), y.view(-1))
            total_loss += loss.item()
            batches += 1

    avg_loss = total_loss / max(batches, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def generate_sample(model: StoryCraftLM, tokenizer: StoryTokenizer, prompt: str = "Once upon a time Lily", max_tokens: int = 20, device: torch.device = "cpu") -> str:
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = model(inp)
            next_tok = torch.argmax(logits[0, -1, :]).item()
            generated.append(next_tok)
            if next_tok == tokenizer.eos_id:
                break

    return tokenizer.decode(generated)


def train():
    parser = argparse.ArgumentParser(description="Pre-train StoryCraftLM on TinyStories")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per step")
    parser.add_argument("--seq_len", type=int, default=32, help="Context sequence length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--eval_interval", type=int, default=25, help="Steps between validation evaluations")
    parser.add_argument("--max_stories", type=int, default=2000, help="Number of stories to ingest from disk")
    parser.add_argument("--save", action="store_true", help="Save final checkpoint to checkpoints/storycraft_base.pt")
    args = parser.parse_args()

    set_seed(42)

    print("=" * 70)
    print("🚀 STORYCRAFT-LLM: PRE-TRAINING ON TINYSTORIES DATASET")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"   Compute Device: {device}")

    # Paths
    chk_dir = os.path.join(repo_root, "checkpoints")
    tok_path = os.path.join(chk_dir, "tokenizer.pkl")
    base_save_path = os.path.join(chk_dir, "storycraft_base.pt")
    data_dir = os.path.join(repo_root, "data/raw")
    train_file = os.path.join(data_dir, "tinystories_train.txt")
    valid_file = os.path.join(data_dir, "tinystories_valid.txt")

    # Load Tokenizer
    if not os.path.exists(tok_path):
        print(f"   ❌ Tokenizer missing at {tok_path}. Please run Module 01 first.")
        return
    tokenizer = StoryTokenizer.from_pretrained(tok_path)
    print(f"   ✓ Loaded Tokenizer (Vocabulary: {tokenizer.vocab_size} tokens)")

    # Prepare Datasets
    print("\n1. Loading and Tokenizing TinyStories Corpus:")
    train_ds = TinyStoriesDataset(train_file, tokenizer, seq_len=args.seq_len, max_stories=args.max_stories)
    valid_ds = TinyStoriesDataset(valid_file, tokenizer, seq_len=args.seq_len, max_stories=min(500, args.max_stories // 4))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    # Initialize Model
    print("\n2. Initializing StoryCraftLM Model Architecture:")
    cfg = {"vocab_size": tokenizer.vocab_size, "d_model": 128, "n_layers": 4, "n_heads": 4}
    model = StoryCraftLM(**cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   - Layers: {cfg['n_layers']} | Dim: {cfg['d_model']} | Heads: {cfg['n_heads']}")
    print(f"   - Total Parameters: {total_params:,}")

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    warmup_steps = max(10, args.max_steps // 10)
    print(f"\n3. Pre-Training Loop ({args.max_steps} steps, batch_size={args.batch_size}):")
    print("-" * 70)
    print(f"{'Step':^6} | {'LR':^9} | {'Train Loss':^11} | {'Val Loss':^10} | {'Val PPL':^10} | {'Elapsed':^8}")
    print("-" * 70)

    model.train()
    step = 0
    t0 = time.perf_counter()
    train_iter = iter(train_loader)

    while step < args.max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Update LR
        lr = get_lr_cosine_schedule(step, warmup_steps, args.max_steps, args.lr, args.lr * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = loss_fn(logits.view(-1, tokenizer.vocab_size), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step += 1

        if step % args.eval_interval == 0 or step == args.max_steps:
            val_loss, val_ppl = evaluate(model, valid_loader, loss_fn, device)
            elapsed = time.perf_counter() - t0
            print(f"{step:^6} | {lr:^9.5f} | {loss.item():^11.4f} | {val_loss:^10.4f} | {val_ppl:^10.2f} | {elapsed:^7.1f}s")
            
            # Sample qualitative generation
            sample = generate_sample(model, tokenizer, prompt="Once upon a time Lily", max_tokens=15, device=device)
            print(f"   [Step {step} Sample]: \"{sample}\"")
            model.train()

    print("-" * 70)

    if args.save:
        print(f"\n4. Saving Pre-trained Checkpoint to {base_save_path}:")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "trained_steps": args.max_steps,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "dataset": "TinyStories"
        }, base_save_path)
        print(f"   ✓ Successfully saved pre-trained model to {base_save_path}")

    print("\n" + "=" * 70)
    print("🎉 PRE-TRAINING RUN COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    train()
