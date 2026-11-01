# 💬 Module 07: Post-Training Phase 1 - Instruction Tuning (SFT) & LoRA from Scratch

> **Central Anchor in this Module:** Transforming our raw text-completing base model into a polite, prompt-following story assistant (`StoryCraft-Instruct`) and implementing **LoRA (Low-Rank Adaptation)** from scratch to build modular genre adapters updating $<1\%$ of parameters.

---

## 🎯 Learning Objectives
1. Understand why Base Models struggle with conversational instructions.
2. Structure dialogues using ChatML markup (`<|im_start|>user`, `<|im_end|>`, etc.).
3. Implement prompt masking (computing loss exclusively on assistant response tokens, ignoring system/user prompts).
4. **Implement LoRA from scratch in pure PyTorch** (inspired by Raschka Appendix E).
5. Train modular story adapters (e.g., Bedtime LoRA vs Mystery LoRA) with only ~45,000 trainable parameters.

---

## 🔬 Core Theory: SFT & Parameter-Efficient Fine-Tuning (LoRA)

### 1. ChatML & Target-Only Loss Masking
```
<|im_start|>system
You are a friendly children's storyteller.<|im_end|>
<|im_start|>user
Write a story about a kitten named Whiskers who finds a bell.<|im_end|>
<|im_start|>assistant
Once upon a sunny afternoon in a cozy kitchen...<|im_end|>
```
- User tokens: `target = -100` (ignored by PyTorch `CrossEntropyLoss`).
- Assistant tokens: standard cross-entropy loss.

### 2. LoRA from Scratch:
Full fine-tuning updates all 8.5M parameters and requires storing an entire model copy per task.

LoRA decomposes weight updates into low-rank matrices:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
- $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$: Frozen base weights.
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$: Initialized with Gaussian $\mathcal{N}(0, \sigma^2)$.
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$: Initialized with zeros (so initially $\Delta W = 0$).
- $r$: Rank (e.g., $r = 4$).
- $\alpha$: Scaling hyperparameter (e.g., $\alpha = 16$).

```
Input x (d_in)
  │
  ├──► [Frozen W_0] ──────────────────────► (+) ──► Output y (d_out)
  │                                          ▲
  └──► [A: d_in -> r] ──► [B: r -> d_out] ───┘ (scaled by alpha / r)
```

During inference, weights can be merged into $W_0$ with **zero latency overhead**!

---

## 💻 Lab Exercises
1. `lora.py`:
   - Code `LoRALinear(in_features, out_features, rank=4, alpha=16)` from scratch.
   - Implement parameter freezing and adapter replacement.
2. `train_sft.py`:
   - Format 10,000 instruction-story pairs.
   - Train `StoryCraft-Instruct` using full-parameter fine-tuning and LoRA fine-tuning; compare convergence and storage savings.
