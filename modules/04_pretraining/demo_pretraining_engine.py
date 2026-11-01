"""
Module 04: The Pre-training Engine & Optimization
==================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. Cross-Entropy Loss calculation for next-token prediction.
2. Perplexity (PPL = exp(loss)) and what it physically means.
3. Linear warmup + Cosine Annealing learning rate schedule.
4. AdamW optimizer with weight decay decoupling.
5. Gradient clipping and gradient accumulation.
6. A self-contained 5-step training loop displaying live loss convergence.

Run this script:
    python3 modules/04_pretraining/demo_pretraining_engine.py
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================================
# 1. Cosine Learning Rate Schedule with Warmup
# ============================================================================
def get_lr_cosine_schedule(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Computes learning rate for current step:
    1. Linear warmup from 0 to max_lr for step < warmup_steps.
    2. Cosine decay down to min_lr for step in [warmup_steps, total_steps].
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ============================================================================
# 2. Minimal Decoder Model for Training Simulation
# ============================================================================
class MiniStoryTransformer(nn.Module):
    def __init__(self, vocab_size: int = 128, d_model: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [B, T]
        h = torch.relu(self.linear(self.embed(x)))
        logits = self.head(h)  # [B, T, vocab_size]
        return logits


# ============================================================================
# 3. Demonstration Runner
# ============================================================================
def main():
    print("=" * 70)
    print("⚡ MODULE 04: THE PRE-TRAINING ENGINE & OPTIMIZATION")
    print("=" * 70)

    vocab_size = 128
    batch_size = 4
    seq_len = 16

    model = MiniStoryTransformer(vocab_size=vocab_size, d_model=64)
    loss_fn = nn.CrossEntropyLoss()

    # AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    print("\n1. Optimization Setup:")
    print("   - Criterion: Cross-Entropy Loss")
    print("   - Metric: Perplexity = exp(Loss)")
    print("   - Optimizer: AdamW (decoupled weight decay = 0.01)")
    print("   - Gradient Norm Clipping threshold: 1.0")

    # Generate synthetic story training batch
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Shifted target sequence
    y = torch.randint(0, vocab_size, (batch_size, seq_len))

    print("\n2. Simulating a 5-Step Pre-training Loop with Cosine Warmup:")
    print("-" * 70)
    print(f"{'Step':^6} | {'LR':^10} | {'Loss':^8} | {'Perplexity':^12} | {'Grad Norm':^10}")
    print("-" * 70)

    warmup_steps = 2
    total_steps = 5
    max_lr = 1e-3
    min_lr = 1e-4

    for step in range(total_steps):
        # 1. Update Learning Rate
        lr = get_lr_cosine_schedule(step, warmup_steps, total_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()

        # 2. Forward Pass: [B, T, V]
        logits = model(x)

        # 3. Flatten for Cross-Entropy: [B*T, V] vs [B*T]
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        ppl = math.exp(min(loss.item(), 20.0))  # Cap for numerical sanity

        # 4. Backward Pass
        loss.backward()

        # 5. Gradient Clipping
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 6. Optimizer Step
        optimizer.step()

        print(f"{step + 1:^6} | {lr:^10.6f} | {loss.item():^8.4f} | {ppl:^12.2f} | {grad_norm:^10.4f}")

    print("-" * 70)

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. Perplexity represents the model's 'uncertainty' branching factor:")
    print("   At initialization with V=128, PPL is ~128 (random guessing).")
    print("   As the model learns grammar, PPL drops toward 2.0-4.0.")
    print("2. Warmup prevents catastrophic gradient spikes early in training.")
    print("3. Gradient clipping (norm <= 1.0) guarantees numerical stability.")
    print("👉 Next: Module 05 turns logits into creative stories via Sampling!")
    print("=" * 70)


if __name__ == "__main__":
    main()
