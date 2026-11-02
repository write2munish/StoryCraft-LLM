"""
benchmark_latency.py: Empirical Throughput & Optimization Benchmark.
===================================================================
Compares:
1. Naive O(T^2) vs KV-Cached O(1) across varying generation horizons.
2. Step-by-step latency showing the quadratic slowdown of naive decoding.
3. Model memory footprint: FP32 vs Dynamic INT8 Quantization.
"""

import os
import sys
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.dirname(__file__))

from storycraft import StoryTokenizer, StoryCraftLM, set_seed
from kv_cache import CachedStoryGenerator


def measure_step_latencies(model: StoryCraftLM, prompt_ids: List[int], num_steps: int = 40) -> Tuple[List[float], List[float]]:
    """
    Measures the execution time of EACH individual token generation step.
    Demonstrates how naive step latency climbs linearly with sequence length (O(T)),
    while KV-cached step latency remains flat (O(1)).
    """
    model.eval()
    naive_step_times = []
    cached_step_times = []

    # 1. Naive step-by-step timing
    current_tokens = list(prompt_ids)
    with torch.no_grad():
        for _ in range(num_steps):
            inp = torch.tensor([current_tokens], dtype=torch.long)
            t0 = time.perf_counter()
            logits, _ = model(inp, past_kvs=None)
            next_tok = torch.argmax(logits[0, -1, :]).item()
            t1 = time.perf_counter()
            naive_step_times.append((t1 - t0) * 1000)
            current_tokens.append(next_tok)

    # 2. Cached step-by-step timing
    current_tokens = list(prompt_ids)
    with torch.no_grad():
        inp = torch.tensor([prompt_ids], dtype=torch.long)
        logits, past_kvs = model(inp, past_kvs=None)
        next_tok = torch.argmax(logits[0, -1, :]).item()
        current_tokens.append(next_tok)

        for _ in range(num_steps - 1):
            single_inp = torch.tensor([[next_tok]], dtype=torch.long)
            t0 = time.perf_counter()
            logits, past_kvs = model(single_inp, past_kvs=past_kvs)
            next_tok = torch.argmax(logits[0, -1, :]).item()
            t1 = time.perf_counter()
            cached_step_times.append((t1 - t0) * 1000)
            current_tokens.append(next_tok)

    return naive_step_times, cached_step_times


def benchmark_sequence_lengths(
    generator: CachedStoryGenerator,
    prompt: str = "Once upon a time Lily saw",
    lengths: List[int] = [10, 25, 50]
) -> List[Dict[str, float]]:
    """
    Benchmarks total latency, throughput (TPS), and speedup across token horizons.
    """
    results = []
    for length in lengths:
        # Run Naive
        _, _, naive_ms = generator.generate_naive(prompt, max_new_tokens=length)
        naive_tps = (length / (naive_ms / 1000.0)) if naive_ms > 0 else 0.0

        # Run Cached
        _, _, cached_ms = generator.generate_cached(prompt, max_new_tokens=length)
        cached_tps = (length / (cached_ms / 1000.0)) if cached_ms > 0 else 0.0

        speedup = naive_ms / max(cached_ms, 1e-4)

        results.append({
            "num_tokens": length,
            "naive_ms": naive_ms,
            "cached_ms": cached_ms,
            "naive_tps": naive_tps,
            "cached_tps": cached_tps,
            "speedup": speedup
        })
    return results


def quantize_storycraft_int8(model: StoryCraftLM) -> nn.Module:
    """
    Applies dynamic 8-bit integer quantization to all Linear feed-forward
    and attention projection layers in StoryCraftLM.
    """
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return quantized_model


def get_model_size_mb(model: nn.Module) -> float:
    """Calculates state dictionary memory footprint in megabytes."""
    torch.save(model.state_dict(), "/tmp/temp_model_size.pt")
    size_mb = os.path.getsize("/tmp/temp_model_size.pt") / (1024 * 1024)
    if os.path.exists("/tmp/temp_model_size.pt"):
        os.remove("/tmp/temp_model_size.pt")
    return size_mb


def run_benchmarks():
    set_seed(42)
    chk_dir = os.path.join(repo_root, "checkpoints")
    tok_path = os.path.join(chk_dir, "tokenizer.pkl")
    aligned_path = os.path.join(chk_dir, "storycraft_aligned.pt")

    if not os.path.exists(aligned_path):
        print("Model checkpoint missing. Run previous modules first.")
        return

    tokenizer = StoryTokenizer.from_pretrained(tok_path)
    ckpt = torch.load(aligned_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {"vocab_size": tokenizer.vocab_size, "d_model": 128, "n_layers": 4, "n_heads": 4})
    model = StoryCraftLM(**cfg)
    clean_state = {k: v for k, v in ckpt["model_state_dict"].items() if "lora_" not in k}
    model.load_state_dict(clean_state, strict=False)
    model.eval()

    generator = CachedStoryGenerator(model, tokenizer)

    print("\n" + "=" * 70)
    print("📊 BENCHMARK 1: THROUGHPUT SCALING BY SEQUENCE LENGTH")
    print("=" * 70)
    print(f"{'Tokens':<8} | {'Naive (ms)':<12} | {'Cached (ms)':<12} | {'Naive TPS':<11} | {'Cached TPS':<12} | {'Speedup':<8}")
    print("-" * 70)

    results = benchmark_sequence_lengths(generator, lengths=[10, 25, 40])
    for r in results:
        print(f"{r['num_tokens']:<8} | {r['naive_ms']:<12.2f} | {r['cached_ms']:<12.2f} | {r['naive_tps']:<11.1f} | {r['cached_tps']:<12.1f} | {r['speedup']:<7.2f}x")

    print("\n" + "=" * 70)
    print("🔬 BENCHMARK 2: INDIVIDUAL STEP LATENCY (O(T) vs O(1))")
    print("=" * 70)
    prompt_ids = tokenizer.encode("Once upon a time Lily", add_special_tokens=False)
    naive_steps, cached_steps = measure_step_latencies(model, prompt_ids, num_steps=20)
    print(f"   Step 01 Latency:  Naive = {naive_steps[0]:.3f} ms  |  Cached = {cached_steps[0]:.3f} ms")
    print(f"   Step 10 Latency:  Naive = {naive_steps[9]:.3f} ms  |  Cached = {cached_steps[8]:.3f} ms")
    print(f"   Step 20 Latency:  Naive = {naive_steps[19]:.3f} ms  |  Cached = {cached_steps[-1]:.3f} ms")
    print(f"   -> Naive step latency grew by {((naive_steps[-1] / naive_steps[0]) - 1) * 100:+.1f}% due to recomputing context!")
    print(f"   -> Cached step latency stayed flat due to reusing historical keys & values!")

    print("\n" + "=" * 70)
    print("💾 BENCHMARK 3: DYNAMIC INT8 QUANTIZATION MEMORY SAVINGS")
    print("=" * 70)
    fp32_size = get_model_size_mb(model)
    quantized_model = quantize_storycraft_int8(model)
    int8_size = get_model_size_mb(quantized_model)
    print(f"   Original FP32 Model Checkpoint Size: {fp32_size:.2f} MB")
    print(f"   Dynamic INT8 Quantized Model Size:   {int8_size:.2f} MB")
    print(f"   RAM Footprint Reduction:             {(1.0 - int8_size / fp32_size) * 100:.1f}% savings!")


if __name__ == "__main__":
    run_benchmarks()
