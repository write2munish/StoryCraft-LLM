# 🏎️ Module 10: Inference Optimization, Key-Value Caching & Correctness Verification

> **Central Anchor in this Module:** Eliminating the $O(T^2)$ computational bottleneck during autoregressive story generation by implementing a Key-Value (KV) Cache, proving mathematical and token-level equivalence against naive generation, achieving $>3	imes$ speedups, and applying dynamic 8-bit quantization.

---

## 🎯 Learning Objectives
1. Understand why naive autoregressive token generation recomputes redundant Key and Value projections ($O(T^2)$ quadratic complexity).
2. Implement and utilize a dynamic Key-Value Cache tensor manager inside `CausalSelfAttention` and `StoryCraftLM`.
3. **Rigorous Correctness Verification**: Mathematically verify that KV-cached inference produces identical output logits ($\Delta_{	ext{max}} < 10^{-4}$), identical token choices, and identical decoded text to full-context naive generation.
4. Benchmark generation throughput: Tokens Per Second (TPS) with vs. without KV cache across sequence lengths.
5. Apply dynamic 8-bit integer quantization (`INT8`) to compress model memory footprint by $>60\%$.

---

## 🔬 Core Theory: The KV Cache Breakthrough

### 1. The Quadratic Bottleneck: Naive Generation is $O(T^2)$
When generating token $t+1$:
- We feed the entire sequence $x_{1:t}$ into the model.
- The model computes $Q, K, V$ for **all** tokens $1, 2, \dots, t$.
- But the Key and Value vectors for tokens $1, \dots, t-1$ never change!
- Generating a story of length $L$ requires $\sum_{t=1}^L t = rac{L(L+1)}{2}$ attention computations!

### 2. The Solution: KV Caching ($O(1)$ per step, $O(T)$ total)
1. **Prefill Phase**: Process the initial prompt sequence $x_{1:P}$ once and store the resulting Keys and Values in the cache:
   $$K^{(0)} = x_{1:P} W_k, \quad V^{(0)} = x_{1:P} W_v$$
2. **Decode Phase**: At step $t > P$, pass ONLY the single newest token $x_t$ through the network:
   $$q_t = x_t W_q, \quad k_t = x_t W_k, \quad v_t = x_t W_v$$
3. Concatenate $k_t$ and $v_t$ with the stored historical cache:
   $$K_{	ext{all}} = [K_{	ext{cached}}, k_t], \quad V_{	ext{all}} = [V_{	ext{cached}}, v_t]$$
4. Compute attention between the single Query $q_t$ and cached keys $K_{	ext{all}}$:
   $$	ext{Attention}(q_t, K_{	ext{all}}, V_{	ext{all}}) = 	ext{softmax}\left(rac{q_t K_{	ext{all}}^T}{\sqrt{d_k}}ight) V_{	ext{all}}$$

---

## 📁 File Structure

```
modules/10_kv_cache_optimization/
├── README.md               # Architecture documentation & mathematical theory
├── kv_cache.py             # Inference engine, CachedStoryGenerator & verify_generation_correctness
├── benchmark_latency.py    # Empirical latency, TPS scaling, and INT8 quantization
├── demo_kv_cache.py        # Central runnable demonstration script
└── tests.py                # Comprehensive 5-test unit testing suite
```

---

## 💻 Running the Code

### 1. Run the Main Demonstration
```bash
python3 modules/10_kv_cache_optimization/demo_kv_cache.py
```
This runs the full pipeline:
- Loads `checkpoints/storycraft_aligned.pt`.
- Performs **Step-by-Step Mathematical Correctness Verification** (comparing logits and token IDs).
- Generates a fluent children's story continuation with repetition penalty.
- Benchmarks TPS across sequence lengths ($10, 25, 40$ tokens).
- Compresses the model with dynamic INT8 quantization.

### 2. Run the Benchmark Suite
```bash
python3 modules/10_kv_cache_optimization/benchmark_latency.py
```

### 3. Run the Unit Tests
```bash
python3 modules/10_kv_cache_optimization/tests.py
```
