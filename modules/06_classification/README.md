# 🏷️ Module 06: Discriminative LLMs - Fine-Tuning for Story Classification

> **Central Anchor in this Module:** Adapting the pre-trained `StoryCraftLM` backbone from generating stories into a high-accuracy **Story Genre, Safety & Age-Appropriateness Classifier** by attaching a classification head to the final token embedding.

---

## 🎯 Learning Objectives
1. Understand the dual nature of LLMs: Generative (predicting next token across vocabulary $V$) vs Discriminative (predicting class probability across $K$ classes).
2. Adapt a causal decoder architecture for sequence-level classification.
3. Compare pooling strategies: Last Token vs Mean Pooling.
4. Compare fine-tuning regimes: Head-only fine-tuning (frozen backbone) vs Full fine-tuning.
5. Build an automated moderation and tagging filter for user-submitted and model-generated stories.

---

## 🔬 Core Theory: From Generative to Discriminative

### The Architecture Transition:
```
Generative Pre-training:
Sequence Tokens ──► [Transformer Blocks] ──► Final RMSNorm ──► LM Head (d_model -> V) ──► Logits [B, T, V]

Sequence Classification:
Story Tokens     ──► [Transformer Blocks] ──► Final RMSNorm ──► Select Last Token [B, d]
                                                                     │
                                                                     ▼
                                                         Classification Head (d -> K) ──► Logits [B, K]
```

### Why the Last Token?
Because causal masking forces each token to attend only to previous tokens, the representation of the **final token** at position $T$ has aggregated information across the entire story. Feeding this vector into a linear layer $\mathbb{R}^{d_{\text{model}} \to K}$ gives an expressive sequence classifier!

### The Multi-Task Story Classifier:
- **Genre Classification:** `Adventure`, `Bedtime`, `Animal`, `Fantasy` ($K = 4$).
- **Content Moderation:** `Safe` vs `Unsafe / Scary` ($K = 2$).
- **Target Age Group:** `Toddler (2-3)`, `Preschool (4-5)`, `Early Reader (6+)` ($K = 3$).

---

## 💻 Lab Exercises
- **File:** `train_classifier.py`
  - Implement `StoryClassificationHead(d_model, num_classes)`.
  - Load pre-trained `storycraft_base.pt`.
  - Train on 5,000 labeled stories with frozen backbone for 2 epochs, then unfreeze last transformer block for 1 epoch.
  - Evaluate Accuracy, Precision, Recall, and Confusion Matrix on held-out test stories.
- **Verification:** Unit tests in `tests.py`.
