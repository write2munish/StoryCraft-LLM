# 📐 Module 02: Vector Spaces & Positional Representations

> **Central Anchor:** Projecting story token IDs into continuous semantic geometry ($d_{\text{model}} = 128$) and implementing modern Rotary Position Embeddings (RoPE) so the story generator understands word order.

---

## 🎯 What to Expect
- **What You Will Build & Run:** You will execute `demo_embeddings_rope.py`, demonstrating token embedding table lookups, exposing the permutation-invariance flaw of raw attention, and implementing Rotary Position Embeddings (RoPE) from scratch.
- **What You Will See:** The terminal displays embedding vector shapes, rotates Query and Key tensors in 2D coordinate pairs, and outputs a mathematical verification proving that dot products $\langle q_m, k_n \rangle$ depend strictly on relative distance $(m - n)$.
- **What You Will Observe:** You will observe why a model without position encoding cannot differentiate `"Dog bites boy"` from `"Boy bites dog"`, and how RoPE mathematically injects sequence position.
- **Run Command:**
  ```bash
  python3 modules/02_embeddings/demo_embeddings_rope.py
  ```

---

## 💡 The Concept Explained Simply
Think of a token embedding table as a high-dimensional GPS map of "meaning":
- In this space, words with related meanings—like `"dog"`, `"puppy"`, and `"bark"`—cluster close together, while unrelated words like `"cookie"` or `"airplane"` sit far away.
- However, standard self-attention has a major blind spot: it treats sequences as an unordered "bag of words"! To attention, `"Lily hugged Barnaby"` and `"Barnaby hugged Lily"` look completely identical because it computes pairwise matches regardless of where words sit on the page.
- Earlier models (like GPT-2) tried to fix this by adding a fixed position vector to the word vector ($x_m = e_m + p_m$). But this often distorts the word's semantic meaning.
- **Rotary Position Embeddings (RoPE)** are the modern standard (used in LLaMA 3, Mistral, and Gemma): instead of adding vectors, RoPE rotates the Query and Key vectors in 2D circles like clock hands! The angle of the hand represents position. When the model computes dot products between two words, the math naturally extracts the angle *difference* (relative distance), allowing the model to know exactly who did what to whom!

---

## 🔬 Core Theory: From Bag-of-Words to Sequences
Self-attention is fundamentally permutation-invariant:
$$\text{Attention}(P X) = P \,\text{Attention}(X) \quad \text{for any permutation matrix } P$$
Without positional signals, token order is entirely lost.

### Rotary Position Embeddings (RoPE):
RoPE groups embedding dimensions into pairs: $(x_{2i}, x_{2i+1})$ for $i \in [0, d/2)$. At sequence position $m$, each 2D vector is rotated by angle $m \theta_i$:
$$\begin{pmatrix} q_{2i}^{(m)} \\ q_{2i+1}^{(m)} \end{pmatrix} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$
Where frequencies follow:
$$\theta_i = 10000^{-2i / d}$$

The inner product of rotated Query and Key vectors becomes a function strictly of their relative distance $(m - n)$:
$$\langle \mathbf{R}_m \mathbf{q}, \mathbf{R}_n \mathbf{k} \rangle = \mathbf{q}^T \mathbf{R}_{n - m} \mathbf{k}$$

---

## 💻 File Structure & Runnable Components
```
modules/02_embeddings/
├── README.md                  # This module guide
├── demo_embeddings_rope.py    # Runnable RoPE implementation & mathematical proof
└── tests.py                   # 3-test unit suite verifying embedding lookup & RoPE
```

---

## 🔑 Key Pedagogical Takeaways
1. **Geometric Meaning:** Embedding tables translate discrete token symbols into continuous vector spaces where distance reflects semantic affinity.
2. **Permutation Invariance:** Transformers have no innate concept of left-to-right order; positional information must be explicitly engineered.
3. **Relative Invariance:** RoPE ensures that word interactions depend on their distance from each other, allowing models to generalize to longer contexts.
4. **Bridge to Module 03:** With word representations and positional geometry established, Module 03 builds the complete multi-layer Transformer decoder network.

---

## 🧪 Running the Code & Unit Tests
```bash
# Run the interactive demonstration
python3 modules/02_embeddings/demo_embeddings_rope.py

# Run the automated unit tests
python3 modules/02_embeddings/tests.py
```
