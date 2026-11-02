# 🔬 Appendix B: Frontier Attention Architectures (GQA, MLA, MoE)

> **Central Anchor in this Appendix:** Exploring modern attention and routing variants used in state-of-the-art LLMs (LLaMA 3, Mistral, Mixtral, DeepSeek) to reduce KV cache memory and scale model capacity.

---

## 🎯 Learning Objectives
1. Understand **Grouped-Query Attention (GQA)** and why it replaces standard Multi-Head Attention in production models.
2. Contrast Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Grouped-Query Attention (GQA).
3. Demystify **Mixture-of-Experts (MoE)**: Router softmax gating, Top-$k$ expert selection, and sparse compute activation.

---

## 🔬 Core Theory: Scaling Beyond Standard MHA

### 1. Grouped-Query Attention (GQA)
In standard MHA, every Query head has its own Key and Value head:
- $N_q = 8, N_{kv} = 8 \implies 1:1$ ratio.
In GQA (Ainslie et al., 2023), multiple Query heads share a single Key and Value head:
- For example, $N_q = 8, N_{kv} = 2 \implies 4:1$ sharing ratio!
- **Impact:** Decreases KV Cache memory footprint by $4	imes$ during autoregressive generation with negligible loss in story coherence!

### 2. Mixture-of-Experts (MoE)
Instead of passing every token through one giant dense feed-forward network, MoE splits the FFN into $E$ parallel "Expert" networks:
- A learned **Router Linear Layer** computes logits: $s = x W_{	ext{gate}}$.
- Softmax over top-$k$ experts: only $k$ experts (e.g., $k=2$ out of $E=8$) are activated per token!
- **Impact:** A model with 30M total parameters only executes 6M active parameters per token!

---

## 💻 Included Demonstration Script
Run the companion demo:
```bash
python3 appendices/appendix_b_gqa_moe/demo_gqa_moe.py
```
This script demonstrates:
- A working Grouped-Query Attention block with 4:1 head sharing.
- A sparse Top-2 Mixture-of-Experts layer with token routing.
