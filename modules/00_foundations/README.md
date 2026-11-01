# 🏛️ Module 00: Foundations of Language Modeling

> **Central Anchor in this Module:** Building a simple Bigram & Trigram statistical story generator to see where classical statistical methods break down.

---

## 🎯 Learning Objectives
1. Understand the mathematical formulation of autoregressive language modeling:
   $$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$$
2. Implement an N-gram transition count model in pure Python without machine learning libraries.
3. Observe firsthand why statistical count tables cannot handle context, grammar, or long-range coherence (The Curse of Dimensionality).

---

## 🔬 Core Theory: The Counting Paradigm
Before neural networks, language models operated on transition probability tables:
- A **Unigram** model draws words according to raw frequency:
  $$P(w_t) = \frac{\text{Count}(w_t)}{\text{Total Words}}$$
- A **Bigram** model predicts the next word based on only the single immediately preceding word:
  $$P(w_t \mid w_{t-1}) = \frac{\text{Count}(w_{t-1}, w_t)}{\text{Count}(w_{t-1})}$$
- An **N-gram** model considers the previous $N-1$ words.

### The Breakdown Point:
If our vocabulary has $V = 5,000$ words:
- Bigram table: $5,000^2 = 25,000,000$ entries.
- Trigram table: $5,000^3 = 125,000,000,000$ entries.
- 5-gram table: $5,000^5 = 3.125 \times 10^{18}$ entries!

Almost all word combinations in human language will have a count of $0$ (sparsity). More importantly, an N-gram cannot generalize: seeing `"The brown dog barked"` gives it zero information about `"The black puppy yelled"`.

---

## 💻 Lab Exercise: Build the Bigram Storyteller
Implement `bigram_story_generator.py`:
1. Parse a small corpus of children's stories.
2. Build a transition dictionary: `counts[word_prev][word_next]`.
3. Normalize counts into probabilities.
4. Generate 100 words starting from `"once upon a time"`.

### Expected Output & Observation
You will notice the generator immediately enters loops or drifts into incoherent sentences:
> *"Once upon a time there was a girl and she had a ball and the ball was big and big and big and the dog saw the ball and..."*

This motivates **Module 01 (Tokenization)** and **Module 02 (Distributed Embeddings)**!
