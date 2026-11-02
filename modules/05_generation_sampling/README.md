# 🎲 Module 05: Generation, Sampling & Foundation Model Weight Loading

> **Central Anchor:** Loading the persistent foundation checkpoint `checkpoints/storycraft_base.pt`, implementing autoregressive decoding algorithms, and mastering external open-weight state-dict translation.

---

## 🎯 What to Expect
- **What You Will Build & Run:** You will execute `demo_generation_sampling.py`, connecting directly to `checkpoints/storycraft_base.pt` to explore probability distributions and compare 4 distinct generation strategies side-by-side on the same prompt.
- **What You Will See:** The terminal displays real next-token probabilities (e.g. for prompt `"Lily saw a"`, displaying probabilities for `'red'`, `'little'`, `'green'`), generates continuations across Greedy, Low-Temp, Tuned Nucleus + Repetition Penalty, and High-Temp Chaotic modes, and verifies cross-architecture weight transfer.
- **What You Will Observe:** You will observe why Greedy decoding loops indefinitely (*"lily played played played"*), how Repetition Penalty breaks loops, and how Temperature controls creativity.
- **Run Command:**
  ```bash
  python3 modules/05_generation_sampling/demo_generation_sampling.py
  ```

---

## 💡 The Concept Explained Simply
A trained base model does not produce text directly. At every step, it simply outputs a list of scores (logits) for every word in its vocabulary. How we choose the next word from those scores determines the entire personality and quality of the generated story!

- **Greedy Decoding (argmax):** Always pick the #1 most probable word. While this sounds logical, it is like a writer who always picks the safest, most obvious word. It quickly gets trapped in vicious repetitive loops: *"Lily played with the ball. The ball was red. The ball was red. The ball..."*
- **Temperature ($T$):** Divides the logits before turning them into probabilities.
  - *Low Temperature ($T = 0.2$):* Makes high scores higher and low scores zero. The story is rigid and predictable.
  - *High Temperature ($T = 1.8$):* Flattens the distribution, giving obscure words a high chance. The story becomes chaotic and nonsensical!
  - *Balanced Temperature ($T = 0.7$):* Fluent, creative, and coherent!
- **Top-p (Nucleus) Sampling:** Only consider the top words whose cumulative probability adds up to $p$ (e.g. 90%). It dynamically expands when many words make sense and contracts when only one word is obvious.
- **Repetition Penalty:** Slightly penalizes the probability of any word that has already been generated, breaking repetition traps permanently!

---

## 🔬 Core Theory: Autoregressive Sampling Mechanics

### 1. Temperature Scaling
$$\text{Softmax}(z / T)_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

### 2. Top-k and Top-p (Nucleus) Filtering
- **Top-k:** Retains only the $k$ highest logit values; sets all others to $-\infty$.
- **Top-p:** Sorts logits in descending order and finds the smallest set of tokens whose cumulative probability exceeds $p$:
  $$\sum_{i \in V^{(p)}} P(w_i) \ge p$$

### 3. Repetition Penalty
For each token $i$ already generated in the prompt or response:
$$\tilde{z}_i = \begin{cases} z_i / \text{penalty} & \text{if } z_i > 0 \\ z_i \cdot \text{penalty} & \text{if } z_i \le 0 \end{cases}$$

### 4. Cross-Architecture Weight Mapping
Foundation model weights (e.g. HuggingFace GPT-2 or LLaMA) can be mapped into custom PyTorch architectures by translating key names and transposing linear projections:
```
transformer.wte.weight           ---> tok_embed.weight
transformer.h.<i>.ln_1.weight   ---> blocks.<i>.norm1.weight
transformer.h.<i>.attn.c_attn   ---> blocks.<i>.attn.q/k/v_proj
transformer.ln_f.weight          ---> norm_final.weight
```

---

## 💻 File Structure & Runnable Components
```
modules/05_generation_sampling/
├── README.md                      # This module guide
├── generate.py                    # StoryGenerator inference & sampling engine
├── load_pretrained.py             # Cross-architecture state-dict weight importer
├── demo_generation_sampling.py    # Runnable comparison of 4 sampling strategies
└── tests.py                       # 5-test unit suite verifying sampling & weight transfer
```

---

## 🔑 Key Pedagogical Takeaways
1. **Sampling Shapes Output:** Model weights provide raw probabilities; decoding hyperparameters (Temperature, Top-p, Penalty) dictate the actual narrative quality.
2. **Nucleus Sampling Eliminates Bad Tails:** Top-p prevents improbable or offensive tail tokens without artificially restricting vocabulary breadth.
3. **Repetition Traps are Common in Small LLMs:** Incorporating repetition penalties is essential for models under 1B parameters.
4. **State-Dict Portability:** Weights are merely named PyTorch tensors; understanding tensor reshaping allows loading billion-parameter open foundation models into custom code.
5. **Bridge to Module 06:** Instead of generating text, Module 06 repurposes this base model to classify story themes and moderate safety.

---

## 🧪 Running the Code & Unit Tests
```bash
# Run the interactive generation & sampling demo
python3 modules/05_generation_sampling/demo_generation_sampling.py

# Run the automated unit tests
python3 modules/05_generation_sampling/tests.py
```
