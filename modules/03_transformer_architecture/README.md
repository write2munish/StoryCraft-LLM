# 🧩 Module 03: The Transformer Architecture from Scratch

> **Central Anchor in this Module:** Assembling the complete 8.5M parameter `StoryCraftLM` neural network in PyTorch, containing RMSNorm, SwiGLU, Rotary Embeddings, and Causal Multi-Head Attention.

---

## 🎯 Learning Objectives
1. Implement Scaled Dot-Product Attention:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$
2. Construct causal attention masks ($M$) to prevent the model from peeking ahead into future story sentences.
3. Build Multi-Head Attention (MHA) and Grouped-Query Attention (GQA).
4. Implement modern Transformer sub-components: RMSNorm and SwiGLU Feed-Forward Networks.
5. Wire together the full `StoryCraftLM` model and verify tensor shapes.

---

## 🔬 Core Architecture Breakdown

```
Input Tokens: [B, T]
       │
       ▼
┌───────────────────────────────┐
│ Token Embedding Table (V, d)  │ ──► [B, T, d]
└───────────────────────────────┘
       │
       ▼
 ┌─────────────── Transformer Block (x6 Layers) ───────────────┐
 │                                                             │
 │   ┌───────────────┐                                         │
 │   │    RMSNorm    │                                         │
 │   └───────┬───────┘                                         │
 │           ▼                                                 │
 │   ┌───────────────┐                                         │
 │   │ Causal MHA    │ + RoPE                                  │
 │   └───────┬───────┘                                         │
 │           ▼                                                 │
 │          (+) ◄──────── Residual Skip Connection             │
 │           │                                                 │
 │   ┌───────┴───────┐                                         │
 │   │    RMSNorm    │                                         │
 │   └───────┬───────┘                                         │
 │           ▼                                                 │
 │   ┌───────────────┐                                         │
 │   │ SwiGLU MLP    │                                         │
 │   └───────┬───────┘                                         │
 │           ▼                                                 │
 │          (+) ◄──────── Residual Skip Connection             │
 └───────────┼─────────────────────────────────────────────────┘
             │
             ▼
     ┌───────────────┐
     │ Final RMSNorm │
     └───────┬───────┘
             ▼
     ┌───────────────┐
     │ LM Head (d, V)│ ──► Logits [B, T, V]
     └───────────────┘
```

---

## 💻 Lab Exercises
- Implement `model.py` with the following clean modular classes:
  - `RMSNorm(dim, eps)`
  - `RotaryEmbedding(dim, max_seq_len)`
  - `CausalSelfAttention(d_model, n_heads)`
  - `SwiGLUFeedForward(d_model, hidden_dim)`
  - `TransformerBlock(d_model, n_heads, hidden_dim)`
  - `StoryCraftLM(vocab_size, d_model, n_layers, n_heads)`
- Verify that a dummy batch `x = torch.randint(0, 4096, (2, 256))` flows through the network and produces logits with shape `(2, 256, 4096)`.
