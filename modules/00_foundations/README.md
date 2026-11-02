# 🏛️ Module 00: Foundations of Language Modeling

> **Central Anchor:** Building a statistical Bigram and Trigram story generator to understand the mathematical foundations of language modeling and discover where classical statistical methods break down.

> [!TIP]
> 🚀 **Front-Loaded Prerequisite:**  
> Before diving into language modeling and neural architecture, review the front-loaded [**PyTorch Primer**](../../pytorch_primer/) (also available at `modules/primer_pytorch`) to master tensor strides, broadcasting, autograd, and GPU acceleration!

---

## 🎯 What to Expect
- **What You Will Build & Run:** You will inspect and run `demo_foundations.py`, a pure Python statistical language modeling engine built without any deep learning libraries.
- **What You Will See:** The terminal will display next-word conditional probability distributions computed from a sample children's story corpus, followed by generated stories from Bigram (1-word context) and Trigram (2-word context) models.
- **What You Will Observe:** You will observe firsthand how quickly statistical models collapse into repetitive loops (*"lily played played played"*) and incoherent fragments due to their inability to remember characters or plot beyond 1 or 2 words.
- **Run Command:**
  ```bash
  python3 modules/00_foundations/demo_foundations.py
  ```

---

## 💡 The Concept Explained Simply
Imagine you are trying to write a children's story by looking **only at the single last word** written on the page:
- If the last word is `"red"`, you consult a tally sheet of every time `"red"` appeared in your books: 80% of the time it was followed by `"ball"`, 20% by `"apple"`. You pick `"ball"`.
- But now your last word is `"ball"`. You have completely forgotten who was holding the ball, where they were playing, or what the story was about!

Count-based models (N-grams) suffer from severe short-term amnesia. To remember an entire sentence of 10 words, your tally table would require $V^{10}$ entries (where $V$ is your vocabulary size). For a modest vocabulary of 10,000 words, that requires $10^{40}$ entries—far more than the number of atoms on Earth! This exponential explosion is called **The Curse of Dimensionality**.

Furthermore, statistical tables have **zero semantic understanding**: seeing `"Lily loved the red ball"` teaches the model absolutely nothing about `"Lily liked the green toy"`. Words are treated as isolated, atomic strings.

---

## 🔬 Core Theory: The Counting Paradigm
All modern language models—including billion-parameter transformers—share the same fundamental goal: calculating the joint probability of a sequence of words by decomposing it into sequential conditional probabilities:
$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$$

### Classical N-Gram Approximations:
- **Unigram:** Draws words according to raw corpus frequency with zero context:
  $$P(w_t) = rac{\text{Count}(w_t)}{\text{Total Words}}$$
- **Bigram (Markov Assumption):** Assumes the next word depends only on the single preceding word:
  $$P(w_t \mid w_{t-1}) = rac{\text{Count}(w_{t-1}, w_t)}{\text{Count}(w_{t-1})}$$
- **Trigram:** Assumes the next word depends on the previous two words:
  $$P(w_t \mid w_{t-2}, w_{t-1}) = rac{\text{Count}(w_{t-2}, w_{t-1}, w_t)}{\text{Count}(w_{t-2}, w_{t-1})}$$

---

## 💻 File Structure & Runnable Components
```
modules/00_foundations/
├── README.md               # This module guide
├── demo_foundations.py     # Pure Python Bigram & Trigram generation engine
└── tests.py                # 5-test unit suite verifying count distributions
```

---

## 🔑 Key Pedagogical Takeaways
1. **Autoregressive Factoring:** Language modeling is fundamentally next-token prediction: $P(w_t \mid w_{<t})$.
2. **The Curse of Dimensionality:** Tabular models scale exponentially ($V^N$), making long-range context mathematically impossible to store in memory.
3. **Discrete vs. Continuous:** Atomic string counting provides zero generalization across synonyms or related concepts.
4. **The Path Forward:** To build models with memory and semantic understanding, we must transition from discrete word tables to **Subword Tokenization (Module 01)** and **Continuous Vector Embeddings (Module 02)**.

---

## 🧪 Running the Code & Unit Tests
```bash
# Run the interactive demonstration
python3 modules/00_foundations/demo_foundations.py

# Run the automated unit tests
python3 modules/00_foundations/tests.py
```
