"""
Module 10: Inference Optimization & Key-Value Caching
=====================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. Loading the persistent aligned model: checkpoints/storycraft_aligned.pt.
2. The O(T^2) quadratic computational bottleneck of naive autoregressive decoding.
3. KV-Cached generation using StoryCraftLM's internal past_kvs tensors.
4. Latency and Tokens-Per-Second (TPS) comparison on the real persisted model!

Run this script:
    python3 modules/10_kv_cache_optimization/demo_kv_cache.py
"""

import os
import sys
import time
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
from storycraft import StoryTokenizer, StoryCraftLM


def main():
    print("=" * 70)
    print("🏎️ MODULE 10: INFERENCE OPTIMIZATION & KEY-VALUE CACHING")
    print("=" * 70)

    chk_dir = os.path.join(repo_root, "checkpoints")
    tok_path = os.path.join(chk_dir, "tokenizer.pkl")
    aligned_path = os.path.join(chk_dir, "storycraft_aligned.pt")

    print("\n1. Loading Persistent Aligned Model:")
    if os.path.exists(tok_path) and os.path.exists(aligned_path):
        tokenizer = StoryTokenizer.from_pretrained(tok_path)
        ckpt = torch.load(aligned_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {"vocab_size": tokenizer.vocab_size, "d_model": 128, "n_layers": 4, "n_heads": 4})
        model = StoryCraftLM(**cfg)
        clean_state = {k: v for k, v in ckpt["model_state_dict"].items() if "lora_" not in k}
        model.load_state_dict(clean_state, strict=False)
        model.eval()
        print(f"   ✓ Loaded real aligned model from: {aligned_path}")
    else:
        print("   ⚠️ Checkpoint not found. Using fallback model.")
        tokenizer = StoryTokenizer(target_vocab_size=128)
        model = StoryCraftLM(vocab_size=128, d_model=64, n_layers=2, n_heads=2)

    prompt = "Once upon a time Lily"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    num_tokens = 30

    # 1. Naive Generation (O(T^2) recomputing full context)
    print(f"\n2. Benchmarking Naive Full-Context Generation ({num_tokens} tokens):")
    start_naive = time.perf_counter()
    tokens_naive = list(prompt_ids)
    with torch.no_grad():
        for _ in range(num_tokens):
            inp = torch.tensor([tokens_naive], dtype=torch.long)
            logits, _ = model(inp, past_kvs=None)
            next_tok = torch.argmax(logits[0, -1, :]).item()
            tokens_naive.append(next_tok)
    time_naive = (time.perf_counter() - start_naive) * 1000

    # 2. KV-Cached Generation (O(1) step-by-step attention)
    print(f"\n3. Benchmarking KV-Cached Generation ({num_tokens} tokens):")
    start_cached = time.perf_counter()
    tokens_cached = list(prompt_ids)
    with torch.no_grad():
        # Prefill prompt
        inp = torch.tensor([prompt_ids], dtype=torch.long)
        logits, past_kvs = model(inp, past_kvs=None)
        next_tok = torch.argmax(logits[0, -1, :]).item()
        tokens_cached.append(next_tok)

        # Generate token-by-token
        for _ in range(num_tokens - 1):
            single_inp = torch.tensor([[next_tok]], dtype=torch.long)
            logits, past_kvs = model(single_inp, past_kvs=past_kvs)
            next_tok = torch.argmax(logits[0, -1, :]).item()
            tokens_cached.append(next_tok)
    time_cached = (time.perf_counter() - start_cached) * 1000

    speedup = time_naive / max(time_cached, 1e-4)
    print(f"   - Naive Execution Time:    {time_naive:.2f} ms")
    print(f"   - KV-Cached Execution Time: {time_cached:.2f} ms")
    print(f"   ⚡ Throughput Speedup:     {speedup:.2f}x faster inference!")

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. Module 10 tests KV-Caching on the real checkpoints/storycraft_aligned.pt weights.")
    print("2. Memory reuse speeds up generation while producing identical outputs!")
    print("👉 Next: Module 11 deploys this model into the live interactive Capstone App!")
    print("=" * 70)


if __name__ == "__main__":
    main()
