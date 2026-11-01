# 🎲 Module 05: Generation & Sampling Mechanics

> **Central Anchor in this Module:** Taking our pre-trained `StoryCraftLM` base model and building a versatile CLI storytelling engine with Temperature, Top-k, Top-p (nucleus), and repetition penalties.

---

## 🎯 Learning Objectives
1. Understand the autoregressive decoding cycle:
   $$\text{Prompt} \to \text{Forward Pass} \to \text{Logits} \to \text{Sampling} \to \text{Append Token} \to \text{Repeat}$$
2. Implement Greedy Decoding and discover its failure modes (stuck in repetitive loops, dull text).
3. Implement Temperature scaling, Top-$k$ filtering, Top-$p$ (Nucleus) sampling, and Min-$p$ sampling.
4. Implement repetition penalties to discourage infinite repetition traps.

---

## 🔬 Core Theory: Sampling Parameters Explained

Given logits vector $z \in \mathbb{R}^V$:

### 1. Temperature ($T$)
Modulates the "sharpness" of the distribution before softmax:
$$P_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
- **$T \to 0$ (e.g. 0.1):** Almost deterministic (Greedy). Reliable, but prone to loops like `"Lily went to the park and saw a ball. The ball was red. The ball was red."`
- **$T = 0.7 - 0.8$:** Sweet spot for creative storytelling.
- **$T > 1.5$:** High entropy; nonsensical vocabulary leaps.

### 2. Top-$k$ Sampling
Truncates the logits to only the $k$ highest-probability tokens (e.g., $k=40$). Restricts wild low-probability outliers.

### 3. Top-$p$ (Nucleus) Sampling
Sorts tokens descending by probability and retains the minimal subset whose cumulative sum $\ge p$ (e.g., $p=0.90$). Dynamically broadens the candidate pool when multiple words make sense, and narrows it when the next token is obvious (e.g. after `"Once upon a"` $\to$ `"time"`).

### 4. Repetition Penalty ($\theta \ge 1.0$)
Penalizes logits for tokens that have already appeared in the generated window:
$$z_i = \begin{cases} z_i / \theta & \text{if } z_i > 0 \\ z_i \cdot \theta & \text{if } z_i < 0 \end{cases}$$

---

## 💻 Lab Exercises
- Implement `generate.py`:
  - Build class `StoryGenerator(model, tokenizer)`
  - Implement `generate(prompt, max_new_tokens=150, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.1)`
  - Run side-by-side comparisons of greedy vs temperature-sampled stories.
