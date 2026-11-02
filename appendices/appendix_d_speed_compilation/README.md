# ⚡ Appendix D: Training Speedups & Modern Optimizers

> **Central Anchor in this Appendix:** Supercharging training performance for `StoryCraftLM` using kernel fusion (`torch.compile`), FlashAttention (SDPA), and the frontier **Muon** optimizer.

---

## 🎯 Learning Objectives
1. Understand why standard attention is memory-bandwidth bound rather than compute bound.
2. Implement and benchmark PyTorch 2.0 Scaled Dot-Product Attention (SDPA / FlashAttention).
3. Utilize `torch.compile` to fuse operations and eliminate Python interpreter overhead.
4. Understand the **Muon** optimizer (Jordan et al., 2024) and why it outperforms AdamW on 2D weight matrices.

---

## 🔬 Core Theory: The Quest for High Throughput

### 1. FlashAttention / PyTorch SDPA
In standard attention:
$$A = 	ext{softmax}(Q K^T / \sqrt{d}) V$$
Intermediate matrices $Q K^T$ ($T 	imes T$) must be written to High Bandwidth Memory (HBM) and read back for softmax.
**FlashAttention (Dao et al.):**
- Computes attention block by block in ultra-fast SRAM.
- Never materializes the full $T 	imes T$ attention matrix in HBM!
- In PyTorch 2.0+, accessible via: `torch.nn.functional.scaled_dot_product_attention`.

### 2. The Muon Optimizer
While AdamW scales gradient updates coordinate-by-coordinate using adaptive second moments, **Muon** applies matrix-level orthogonalization via Newton-Schulz iterations:
$$O_{k+1} = a O_k + b (O_k O_k^T) O_k$$
This updates weight matrices along the most orthogonal transformation directions, leading to faster pre-training convergence!

---

## 💻 Included Demonstration Script
Run the companion demo:
```bash
python3 appendices/appendix_d_speed_compilation/demo_speed_compilation.py
```
This script demonstrates:
- Benchmarking Naive Attention vs PyTorch SDPA (FlashAttention backend).
- The Newton-Schulz iteration used in the Muon optimizer.
