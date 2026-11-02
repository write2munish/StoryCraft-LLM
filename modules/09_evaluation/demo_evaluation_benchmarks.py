"""
Module 09: Evaluation, Benchmarking & Safety
=============================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. Loading ALL THREE persistent checkpoints (Base vs Instruct vs Aligned).
2. Comparing Perplexity (PPL) on held-out test story sequences.
3. Lexical diversity metrics: Distinct-1 and Distinct-2 n-gram ratios.
4. Comparative progression report showing the model evolving across training phases.

Run this script:
    python3 modules/09_evaluation/demo_evaluation_benchmarks.py
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
from storycraft import StoryTokenizer, StoryCraftLM


def compute_distinct_n(text: str, n: int = 1) -> float:
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def evaluate_checkpoint_loss(model, tokenizer, test_text: str) -> float:
    ids = tokenizer.encode(test_text, add_special_tokens=False)
    if len(ids) < 4:
        return 5.0
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits, _ = model(x)
        loss = loss_fn(logits.view(-1, model.vocab_size), y.view(-1))
    return loss.item()


def main():
    print("=" * 70)
    print("📊 MODULE 09: EVALUATION, BENCHMARKING & SAFETY")
    print("=" * 70)

    chk_dir = os.path.join(repo_root, "checkpoints")
    tok_path = os.path.join(chk_dir, "tokenizer.pkl")
    base_path = os.path.join(chk_dir, "storycraft_base.pt")
    instruct_path = os.path.join(chk_dir, "storycraft_instruct.pt")
    aligned_path = os.path.join(chk_dir, "storycraft_aligned.pt")

    if not (os.path.exists(tok_path) and os.path.exists(base_path)):
        print("   ⚠️ Checkpoints missing. Please run earlier modules first.")
        return

    tokenizer = StoryTokenizer.from_pretrained(tok_path)
    test_story = "Lily and Barnaby played in the sunny green garden and ate warm sweet cookies."

    print("\n1. Evaluating Model Checkpoints Across Training Stages:")
    print("-" * 70)
    print(f"{'Checkpoint Stage':<25} | {'Path':<24} | {'Loss':^8} | {'Perplexity':^10}")
    print("-" * 70)

    stages = [
        ("Base Model (Module 04)", base_path),
        ("Instruct SFT (Module 07)", instruct_path),
        ("DPO Aligned (Module 08)", aligned_path)
    ]

    for name, path in stages:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {"vocab_size": tokenizer.vocab_size, "d_model": 128, "n_layers": 4, "n_heads": 4})
            model = StoryCraftLM(**cfg)
            clean_state = {k: v for k, v in ckpt["model_state_dict"].items() if "lora_" not in k}
            model.load_state_dict(clean_state, strict=False)
            model.eval()

            loss = evaluate_checkpoint_loss(model, tokenizer, test_story)
            ppl = math.exp(min(loss, 15.0))
            filename = os.path.basename(path)
            print(f"{name:<25} | {filename:<24} | {loss:^8.4f} | {ppl:^10.2f}")

    print("-" * 70)

    # 2. Linguistic Diversity Probes
    print("\n2. Lexical Diversity Probes (Distinct-N Ratio):")
    sample_story = "Once upon a time, Lily found a red ball and shared with Barnaby the happy dog."
    print(f"   Sample: \"{sample_story}\"")
    print(f"   - Distinct-1 (Unigrams): {compute_distinct_n(sample_story, 1):.1%}")
    print(f"   - Distinct-2 (Bigrams):  {compute_distinct_n(sample_story, 2):.1%}")

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. Module 09 loads and directly evaluates all three persisted checkpoints.")
    print("2. Tracks quantitative progression from Base -> Instruct -> Aligned.")
    print("👉 Next: Module 10 optimizes inference speed using KV Caching!")
    print("=" * 70)


if __name__ == "__main__":
    main()
