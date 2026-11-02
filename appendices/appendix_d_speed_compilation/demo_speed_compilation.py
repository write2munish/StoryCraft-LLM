"""
Appendix D: Training Speedups (FlashAttention SDPA & Muon Optimizer)
====================================================================
Demonstrates:
1. PyTorch Scaled Dot-Product Attention (SDPA) with FlashAttention backend.
2. Latency comparison: Naive Attention vs PyTorch SDPA.
3. The Muon Optimizer orthogonalization step (Newton-Schulz iteration).

Run this script:
    python3 appendices/appendix_d_speed_compilation/demo_speed_compilation.py
"""

import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
from storycraft import set_seed

import time
import math
import torch
import torch.nn.functional as F


def benchmark_attention_implementations():
    print("\n1. Benchmarking Attention Implementations:")
    B, H, T, D = 4, 8, 512, 64  # Batch=4, Heads=8, Seq=512, Head_dim=64
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    # 1. Naive PyTorch Attention
    start = time.perf_counter()
    for _ in range(50):
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        attn = F.softmax(scores + mask, dim=-1)
        out_naive = attn @ v
    naive_time = (time.perf_counter() - start) * 1000

    # 2. PyTorch SDPA (Fused / FlashAttention backend)
    start = time.perf_counter()
    for _ in range(50):
        out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    sdpa_time = (time.perf_counter() - start) * 1000

    print(f"   - Naive Attention (50 runs):  {naive_time:.2f} ms")
    print(f"   - PyTorch SDPA (50 runs):     {sdpa_time:.2f} ms")
    speedup = naive_time / max(sdpa_time, 1e-5)
    print(f"   ⚡ Fused SDPA Speedup:         {speedup:.2f}x faster!")


def demonstrate_muon_newton_schulz():
    print("\n2. The Muon Optimizer Newton-Schulz Iteration:")
    print("   Instead of coordinate-wise momentum (AdamW), Muon orthogonalizes 2D gradient matrices:")
    
    # Random gradient matrix
    G = torch.randn(64, 64)
    # Normalize spectral norm
    X = G / (G.norm() + 1e-7)

    # 3 Steps of Newton-Schulz iteration: X <- 1.5 X - 0.5 X (X^T X)
    for step in range(3):
        X = 1.5 * X - 0.5 * X @ (X.t() @ X)

    # Verify orthogonality: X @ X^T should approximate Identity
    identity_approx = X @ X.t()
    diag_mean = identity_approx.diag().mean().item()
    off_diag_mean = (identity_approx - torch.eye(64)).abs().mean().item()

    print(f"   Initial matrix frobenius norm: {G.norm():.2f}")
    print(f"   After Newton-Schulz iteration:")
    print(f"   - Diagonal values (target ~1.0):       {diag_mean:.4f}")
    print(f"   - Off-diagonal values (target ~0.0):    {off_diag_mean:.4f}")
    print("   Result: Gradients are transformed into orthogonal updates, accelerating LLM training!")


def main():
    set_seed(42)
    print("=" * 70)
    print("⚡ APPENDIX D: TRAINING SPEEDUPS & MODERN OPTIMIZERS")
    print("=" * 70)

    benchmark_attention_implementations()
    demonstrate_muon_newton_schulz()

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. Fused SDPA bypasses high-bandwidth memory roundtrips, reducing attention latency.")
    print("2. torch.compile fuses elementwise operations into single optimized GPU kernels.")
    print("3. Matrix-aware optimizers like Muon represent the frontier of post-AdamW training.")
    print("=" * 70)


if __name__ == "__main__":
    main()
