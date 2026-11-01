# 📐 Module 02: Vector Spaces & Positional Representations

> **Central Anchor in this Module:** Projecting story token IDs into continuous semantic geometry ($d_{\text{model}} = 256$) and implementing modern Rotary Position Embeddings (RoPE) so the story generator understands word order.

---

## 🎯 Learning Objectives
1. Understand token embedding tables ($W_E \in \mathbb{R}^{V \times d_{\text{model}}}$) as continuous geometric spaces.
2. Understand why self-attention without position encodings is permutation-invariant.
3. Implement modern Rotary Position Embeddings (RoPE) from scratch.
4. Visualize semantic clustering of story concepts (characters, actions, colors) in vector space.

---

## 🔬 Core Theory: From Order-Blind Sets to Sequences
Transformers compute pairwise affinities between all tokens regardless of where they sit in a sequence. Without positional signals, `"Lily hugged the dog"` and `"The dog hugged Lily"` would look identical to the self-attention mechanism!

### Rotary Position Embeddings (RoPE):
Instead of *adding* positional vectors to token embeddings:
$$x_m = e_m + p_m \quad \text{(Absolute Positional Embedding)}$$
RoPE *rotates* the Query and Key vectors in complex 2D planes according to their position indices $m$ and $n$:
$$\mathbf{q}_m = \mathbf{R}_{\Theta, m} W_q x_m, \quad \mathbf{k}_n = \mathbf{R}_{\Theta, n} W_k x_n$$

The inner product $\mathbf{q}_m^T \mathbf{k}_n$ becomes a function solely of the relative token distance $(m - n)$:
$$\langle \mathbf{R}_{\Theta, m} \mathbf{q}, \mathbf{R}_{\Theta, n} \mathbf{k} \rangle = \mathbf{q}^T \mathbf{R}_{\Theta, n - m} \mathbf{k}$$

---

## 💻 Lab Exercises
1. `rope.py`:
   - Compute precomputed frequencies $\theta_i = 10000^{-2(i-1)/d}$.
   - Apply 2D complex rotations to Query and Key tensors.
2. `embed_visualizer.py`:
   - Inspect cosine similarities between pairs: `("puppy", "dog")` vs `("puppy", "castle")`.
