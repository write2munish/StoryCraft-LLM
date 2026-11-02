# 🏷️ Module 06: Discriminative LLMs - Story Classification & Moderation

> **Central Anchor:** Converting our pre-trained generative base model into a multi-task classifier to detect story genres (Adventure, Bedtime, Animal, Fantasy) and enforce content safety moderation.

---

## 🎯 What to Expect
- **What You Will Build & Run:** You will execute `demo_story_classifier.py`, which loads `checkpoints/storycraft_base.pt`, attaches classification heads to the frozen backbone, extracts last-token contextual vectors, and classifies story texts.
- **What You Will See:** The terminal displays classification logits, genre category predictions with confidence percentages, and a binary safety check (Safe vs Unsafe / Scary).
- **What You Will Observe:** You will observe the "Dual LLM Paradigm": how pre-trained generative layers can be repurposed for understanding and classification without modifying the core transformer weights.
- **Run Command:**
  ```bash
  python3 modules/06_classification/demo_story_classifier.py
  ```

---

## 💡 The Concept Explained Simply
Most people think LLMs are only good for writing text. But think about what a base model had to learn during pre-training: to predict words in children's stories, its internal layers had to learn what an animal is, what nighttime means, and what dangerous or scary words look like!

**The Dual LLM Paradigm (Generative vs Discriminative):**
- In **Generative Mode**, we ask: *"Given this story so far, what is the next word?"*
- In **Discriminative Mode**, we ask: *"Given this entire story, what category does it belong to? Is it safe for a 3-year-old?"*

Instead of building a separate machine learning model for content moderation or genre tagging, we reuse the exact same `StoryCraftLM` backbone! We freeze its weights, take the output vector of the very last word in the story (which has attended to the entire narrative), and pass it through a simple linear layer. This gives us instant, accurate classification with almost zero additional training compute!

---

## 🔬 Core Theory: Last-Token Pooling in Causal Decoders
In causal (decoder-only) transformers, token $t$ can only attend to tokens $\le t$.
Therefore:
- Token $0$ only knows about the first word.
- Token $1$ only knows about the first two words.
- **The Last Token $h_{[-1]}$ has attended to the entire document!**

### Sequence Classification Architecture:
$$\mathbf{h}_{[-1]} = \text{StoryCraftLM}(\text{tokens})_{[-1]} \in \mathbb{R}^{d_{\text{model}}}$$
$$\mathbf{z}_{\text{genre}} = \mathbf{h}_{[-1]} W_{\text{genre}} \in \mathbb{R}^{4}, \quad \mathbf{z}_{\text{safety}} = \mathbf{h}_{[-1]} W_{\text{safety}} \in \mathbb{R}^{2}$$
$$\hat{y} = \text{argmax}(\text{softmax}(\mathbf{z}))$$

---

## 💻 File Structure & Runnable Components
```
modules/06_classification/
├── README.md                      # This module guide
├── demo_story_classifier.py       # Runnable multi-task genre & safety classifier
└── tests.py                       # 2-test unit suite verifying classification heads & pooling
```

---

## 🔑 Key Pedagogical Takeaways
1. **The Dual LLM Paradigm:** Modern language models are both text creators (generative) and context evaluators (discriminative).
2. **Last-Token Context:** In causal autoregressive networks, sequence-level classification must pool the final token position.
3. **Backbone Freezing:** Freezing base weights prevents catastrophic forgetting while allowing rapid training of domain-specific task heads.
4. **Bridge to Module 07:** Having mastered classification, Module 07 returns to generative training, transforming the base completer into a conversational instruction assistant using LoRA.

---

## 🧪 Running the Code & Unit Tests
```bash
# Run the interactive story classification demo
python3 modules/06_classification/demo_story_classifier.py

# Run the automated unit tests
python3 modules/06_classification/tests.py
```
