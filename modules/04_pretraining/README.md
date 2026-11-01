# ⚡ Module 04: The Pre-training Engine

> **Central Anchor in this Module:** Training `StoryCraftLM` on 100,000 children's stories, driving perplexity down from $>4,000$ to $<4.5$, and watching the model acquire English grammar and narrative cohesion.

---

## 🎯 Learning Objectives
1. Implement the complete end-to-end PyTorch training and validation loop.
2. Master the relationship between Cross-Entropy Loss and Perplexity ($PPL = e^{\mathcal{L}}$).
3. Configure the AdamW optimizer with decoupled weight decay.
4. Implement a Linear Warmup with Cosine Annealing learning rate schedule.
5. Utilize Mixed Precision (`torch.amp.autocast`) and Gradient Norm Clipping.

---

## 🔬 Core Theory: The Pre-training Dynamics

### Cross-Entropy & Perplexity:
At step $t$, the model outputs raw unnormalized scores (logits) $z_t \in \mathbb{R}^V$. Softmax turns logits into probabilities:
$$P(y_t = k) = \frac{e^{z_{t, k}}}{\sum_{j=1}^V e^{z_{t, j}}}$$
The loss is the negative log-likelihood of the actual next token:
$$\mathcal{L} = -\log P(y_t = w_t^*)$$
- At initial random weights ($V = 4,096$): $\mathcal{L} \approx \log(4096) \approx 8.32$, Perplexity $\approx 4,096$.
- As the model learns grammar and syntax: $\mathcal{L} \to 1.5$, Perplexity $\to 4.48$.

### The Training Recipe:
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \text{weight decay} = 0.1$).
- **Learning Rate:** Peak $6 \times 10^{-4}$ with 500-step linear warmup, decaying to $6 \times 10^{-5}$ via cosine schedule.
- **Gradient Clipping:** Max norm $1.0$ (prevents exploding gradients).
- **Precision:** `bfloat16` on modern GPUs or Apple Silicon MPS.

---

## 💻 Lab Exercises
- **File:** `train.py`
  - Implement training loop with step logging and periodic validation.
  - Implement periodic generation checkpoints (e.g., prompt `"Once upon a time, Lily"` every 500 steps) to monitor qualitative improvement.
  - Save best weights as `checkpoints/storycraft_base.pt`.
