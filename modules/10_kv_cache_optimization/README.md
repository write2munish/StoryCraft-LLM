# 🏎️ Module 10: Inference Optimization & Key-Value Caching

> **Central Anchor in this Module:** Eliminating the $O(T^2)$ computational bottleneck during autoregressive story generation by implementing a Key-Value (KV) Cache, achieving $>3\times$ speedups and enabling smooth real-time token streaming.

---

## 🎯 Learning Objectives
1. Understand why naive autoregressive token generation recomputes redundant Key and Value projections.
2. Implement a dynamic Key-Value Cache tensor manager inside `CausalSelfAttention`.
3. Benchmark generation throughput: Tokens Per Second (TPS) with vs. without KV cache across sequence lengths.
4. Explore 8-bit dynamic quantization to reduce model footprint.

---

## 🔬 Core Theory: The KV Cache Breakthrough

### The Problem: Naive Generation is $O(T^2)$
When generating token $t+1$:
- We feed the entire sequence $x_{1:t}$ into the model.
- The model computes $Q, K, V$ for **all** tokens $1, 2, \dots, t$.
- But the Key and Value vectors for tokens $1, \dots, t-1$ never change!
- Generating a story of length $L$ requires $\sum_{t=1}^L t = \frac{L(L+1)}{2}$ attention computations!

### The Solution: KV Caching ($O(T)$ total)
1. At step $t$, only pass the single newest token $x_t$ through the network.
2. Project $q_t = x_t W_q, \quad k_t = x_t W_k, \quad v_t = x_t W_v$.
3. Concatenate $k_t$ and $v_t$ with stored historical cache:
   $$K_{\text{all}} = [K_{\text{cached}}, k_t], \quad V_{\text{all}} = [V_{\text{cached}}, v_t]$$
4. Compute attention between single Query $q_t$ and cached keys $K_{\text{all}}$:
   $$\text{Attention}(q_t, K_{\text{all}}, V_{\text{all}}) = \text{softmax}\left(\frac{q_t K_{\text{all}}^T}{\sqrt{d_k}}\right) V_{\text{all}}$$

---

## 💻 Lab Exercises
- Implement `kv_cache_model.py`:
  - Modify `CausalSelfAttention` to accept an optional `past_key_value` tuple.
  - Implement caching logic and position tracking for RoPE.
  - Run `benchmark_latency.py` comparing generation time for 200 tokens:
    - *Naive Generation:* ~1.8 seconds
    - *KV-Cached Generation:* ~0.45 seconds ($4\times$ throughput speedup!)
