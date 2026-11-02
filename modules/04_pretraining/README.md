# ⚡ Module 04: The Pre-training Engine & Optimization

> **Central Anchor:** Training `StoryCraftLM` on children's stories, driving perplexity down from random guessing ($>256$) to fluent coherence, and saving the foundation checkpoint `checkpoints/storycraft_base.pt`.

---

## 🎯 What to Expect
- **What You Will Build & Run:** You will execute `demo_pretraining_engine.py` for a rapid interactive simulation, and `train.py` to train directly on the real `data/raw/tinystories_train.txt` and `tinystories_valid.txt` dataset.
- **What You Will See:** The terminal displays live training metrics: step count, learning rate warmup, cross-entropy loss, validation perplexity ($PPL = e^{\mathcal{L}}$), elapsed time, and periodic generation probes showing the emergence of English grammar.
- **What You Will Observe:** You will observe the loss plummet from initial random guessing ($pprox 5.54$) down to fluent convergence, and save the persistent checkpoint `checkpoints/storycraft_base.pt`.
- **Run Commands:**
  ```bash
  # Fast demonstration (5 synthetic steps)
  python3 modules/04_pretraining/demo_pretraining_engine.py

  # Real pre-training run on TinyStories dataset
  python3 modules/04_pretraining/train.py --max_steps 50 --batch_size 16 --save
  ```

---

## 💡 The Concept Explained Simply
When a neural network is first initialized, its weights are random numbers. If you give it the prompt `"Once upon a time Lily"`, it outputs random static gibberish.

**Pre-training is the foundational school for an LLM:**
- We show the model thousands of stories. At every word, the model guesses what comes next.
- When it guesses incorrectly, **Cross-Entropy Loss** calculates how wrong it was.
- The **AdamW optimizer** uses calculus (gradients) to adjust all 820,000 weights, making the correct word slightly more likely next time.
- To prevent training instability, we use a **Warmup Schedule** (starting with a tiny learning rate so early noisy gradients don't destroy weights) and **Gradient Clipping** (capping sudden gradient spikes).
- After thousands of iterations, the model naturally absorbs English grammar, character consistency, and narrative cadence!

---

## 🔬 Core Theory: Pre-training Dynamics

### 1. Cross-Entropy Loss & Perplexity
At step $t$, the model outputs raw unnormalized logit scores $z_t \in \mathbb{R}^V$. Softmax turns them into probabilities:
$$P(y_t = k) = \frac{e^{z_{t, k}}}{\sum_{j=1}^V e^{z_{t, j}}}$$
The loss is the negative log-likelihood of the actual next token $w_t^*$:
$$\mathcal{L} = -\log P(y_t = w_t^*)$$

**Perplexity** is the exponential of cross-entropy loss:
$$\text{PPL} = e^{\mathcal{L}}$$
It represents the effective number of words the model is guessing between. With $V = 256$, initial random guessing has $\mathcal{L} \approx \log(256) \approx 5.54$, and $\text{PPL} \approx 256$. As the model masters English, $\mathcal{L} \to 1.5$ and $\text{PPL} \to 4.5$.

### 2. The Training Recipe:
- **Optimizer:** AdamW with decoupled weight decay ($0.01$) to prevent over-reliance on individual weights.
- **Learning Rate Schedule:** Linear warmup for $10\%$ of steps, followed by Cosine Annealing decay down to $10\%$ of peak learning rate.
- **Gradient Clipping:** Max norm $1.0$ prevents exploding gradients.

---

## 💻 File Structure & Runnable Components
```
modules/04_pretraining/
├── README.md                      # This module guide
├── demo_pretraining_engine.py     # Rapid 5-step warmup demonstration
├── train.py                       # Full pre-training engine on data/raw/tinystories_*.txt
└── tests.py                       # 3-test unit suite verifying loss & learning rate schedules
```

---

## 🔑 Key Pedagogical Takeaways
1. **Self-Supervised Scaling:** Pre-training requires no human annotations; next-token prediction uses the natural sequence of human text as ground truth.
2. **Perplexity is the Benchmark:** Perplexity measures the branching uncertainty of the language model; lower perplexity directly correlates with better grammatical fluency.
3. **Warmup Prevents Divergence:** Without learning rate warmup, early gradient shocks at step 1 can destabilize layer weights permanently.
4. **Bridge to Module 05:** Pre-training produces raw logit probabilities; Module 05 turns these probabilities into creative, non-repetitive stories via inference sampling.

---

## 🧪 Running the Code & Unit Tests
```bash
# Run the interactive pre-training demo
python3 modules/04_pretraining/demo_pretraining_engine.py

# Run the real TinyStories pre-training script
python3 modules/04_pretraining/train.py --max_steps 50 --batch_size 16

# Run the automated unit tests
python3 modules/04_pretraining/tests.py
```
