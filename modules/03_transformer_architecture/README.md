# 🧩 Module 03: The Modern Transformer Architecture from Scratch

> **Central Anchor:** Assembling the complete 820K parameter `StoryCraftLM` decoder architecture in PyTorch, containing RMSNorm, SwiGLU, Rotary Embeddings, and Causal Multi-Head Attention.

---

## 🎯 What to Expect
- **What You Will Build & Run:** You will execute `demo_transformer_architecture.py`, assembling the modern decoder-only transformer architecture used in production frontier models.
- **What You Will See:** The terminal displays the full layer-by-layer breakdown, parameter counts (819,840 parameters across 4 layers, $d_{\text{model}}=128$, 4 attention heads), and verifies tensor forward-pass shapes: `[Batch, Time]` $\to$ `[Batch, Time, Vocab_Size]`.
- **What You Will Observe:** You will observe how causal attention masking enforces autoregressive ordering, and how RMSNorm and SwiGLU activations prepare the model for stable gradient flow.
- **Run Command:**
  ```bash
  python3 modules/03_transformer_architecture/demo_transformer_architecture.py
  ```

---

## 💡 The Concept Explained Simply
Think of a Transformer Block as a writers' room with two distinct, complementary stages:
1. **The Conference Table (Causal Self-Attention):** Every word in the story looks back at all previous words to gather context. When the word `"Barnaby"` is being processed, it attends to `"Lily"`, `"dog"`, and `"garden"`. It updates its understanding: *"I am not just any dog; I am Lily's dog playing in the garden!"* Crucially, the table has a strict rule: words can never look ahead into future chapters (enforced by the **Causal Mask**).
2. **The Thinking Desk (SwiGLU Feed-Forward Network):** After gathering context from the table, each word sits at its own private desk to process what it learned through a non-linear neural network. This is where facts, associations, and grammar rules are synthesized into candidate suggestions for the next word.

We stack these blocks on top of each other. Early layers learn simple grammar (plurals, verb tenses), while deeper layers learn character arcs, dialogue, and plot resolution!

---

## 🔬 Core Theory: The Modern Architecture Recipe
`StoryCraftLM` implements the contemporary open-source architecture recipe (LLaMA 3, Mistral, Gemma):
1. **Root Mean Square Normalization (RMSNorm):** Normalizes inputs by their root mean square without subtracting the mean, cutting compute overhead by ~20%:
   $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
2. **Pre-Layer Normalization (Pre-LN):** Normalizing *before* attention and MLP blocks keeps the residual stream clean, preventing vanishing/exploding gradients.
3. **SwiGLU Non-Linear Activation:** Replaces standard ReLU/GELU with a gated bilinear activation:
   $$\text{SwiGLU}(x) = (x W_{\text{gate}} \cdot \text{SiLU}(x W_{\text{up}})) W_{\text{down}}$$
4. **Weight Tying:** The output classification layer shares weights with the token embedding matrix (`lm_head.weight = tok_embed.weight`), slashing parameter memory by half.

---

## 💻 File Structure & Runnable Components
```
modules/03_transformer_architecture/
├── README.md                              # This module guide
├── demo_transformer_architecture.py       # Runnable architecture & tensor shape verification
└── tests.py                               # 4-test unit suite verifying RMSNorm, SwiGLU & Causal Masking
```

---

## 🔑 Key Pedagogical Takeaways
1. **Causal Masking is Non-Negotiable:** Setting attention weights to $-\infty$ for future positions prevents the network from "cheating" during generation.
2. **Pre-LN Gradient Highway:** The identity residual connection ($x + f(x)$) allows gradients to flow backwards through hundreds of layers without attenuation.
3. **Modern Enhancements:** RMSNorm and SwiGLU outperform the original 2017 Transformer paper's LayerNorm and ReLU in both speed and final perplexity.
4. **Bridge to Module 04:** Having built the untyped, randomly initialized architecture, Module 04 implements the pre-training optimization loop to teach it English grammar.

---

## 🧪 Running the Code & Unit Tests
```bash
# Run the interactive demonstration
python3 modules/03_transformer_architecture/demo_transformer_architecture.py

# Run the automated unit tests
python3 modules/03_transformer_architecture/tests.py
```
