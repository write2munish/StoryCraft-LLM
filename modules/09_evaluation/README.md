# 📊 Module 09: Evaluation, Benchmarking & Safety

> **Central Anchor:** Systematically evaluating `StoryCraftLM` across its developmental journey (Base vs. Instruct vs. Aligned) using intrinsic perplexity, lexical diversity, and safety compliance benchmarks.

---

## 🎯 What to Expect
- **What You Will Build & Run:** You will execute `demo_evaluation_benchmarks.py`, simultaneously loading all three saved checkpoints (`storycraft_base.pt`, `storycraft_instruct.pt`, and `storycraft_aligned.pt`).
- **What You Will See:** The terminal displays a comprehensive scorecard comparing the models on held-out Perplexity (PPL), Distinct-1 and Distinct-2 lexical diversity ratios, and responses to adversarial story prompts.
- **What You Will Observe:** You will observe how post-training improves instruction compliance and safety without destroying base linguistic fluency.
- **Run Command:**
  ```bash
  python3 modules/09_evaluation/demo_evaluation_benchmarks.py
  ```

---

## 💡 The Concept Explained Simply
How do we know if our language model is actually "good"?
Unlike an image classifier where an answer is simply right or wrong, there are thousands of delightful, valid ways to continue a children's story!

To properly evaluate an LLM, we use three distinct lenses:
1. **Perplexity (Surprise Factor):** We show the model real, human-written children's stories it has never seen before. How surprised was it by the words? Lower perplexity means the model understands English grammar and syntax.
2. **Distinct-N Diversity (Vocabulary Health):** A model could achieve low perplexity by repeating the same safe sentence forever: *"The little dog was happy."* Distinct-N measures the percentage of unique words and pairs. High diversity means rich, engaging storytelling!
3. **Adversarial Safety Evaluation:** What happens when a user asks for something scary, rude, or dangerous? We test the models against challenging edge-case prompts to prove that our DPO alignment held firm!

---

## 🔬 Core Theory: Multi-Metric Evaluation Framework

### 1. Intrinsic Perplexity on Held-Out Test Text
$$\text{PPL} = \exp\left( -\frac{1}{N} \sum_{i=1}^N \log P(w_i \mid w_{<i}) \right)$$

### 2. Lexical Diversity Metrics: Distinct-1 & Distinct-2
$$\text{Distinct-}n = \frac{\text{Count of unique } n\text{-grams}}{\text{Total count of generated } n\text{-grams}}$$
- **Distinct-1 $\approx 0.1$:** Severely repetitive; repeating the same words.
- **Distinct-1 $> 0.5$:** Healthy, varied narrative vocabulary.

### 3. Model Progression Scorecard
| Stage | Model Checkpoint | Primary Strength | Known Limitation |
| :--- | :--- | :--- | :--- |
| **Pre-trained** | `storycraft_base.pt` | High raw language fluency | Treats prompts as raw text to complete |
| **Instruction-Tuned** | `storycraft_instruct.pt` | Obeys ChatML formatting | May generate unwholesome or rude themes |
| **DPO-Aligned** | `storycraft_aligned.pt` | Wholesome, safe, aligned | Slightly constrained vocabulary entropy |

---

## 💻 File Structure & Runnable Components
```
modules/09_evaluation/
├── README.md                           # This module guide
├── demo_evaluation_benchmarks.py       # Runnable multi-checkpoint evaluation engine
└── tests.py                            # Unit suite verifying distinct-n & perplexity metrics
```

---

## 🔑 Key Pedagogical Takeaways
1. **Multi-Faceted Evaluation:** No single metric captures LLM quality; Perplexity, Lexical Diversity, and Safety Probes must be evaluated together.
2. **The Alignment Tax:** Post-training (SFT and DPO) slightly increases perplexity on arbitrary text in exchange for massive gains in steerability and safety.
3. **Automated Verification:** Quantitative benchmarks enable continuous integration testing for language models before pushing updates to production.
4. **Bridge to Module 10:** Having confirmed our model is fluent, obedient, and safe, Module 10 optimizes its inference speed using Key-Value Caching.

---

## 🧪 Running the Code & Unit Tests
```bash
# Run the multi-checkpoint evaluation benchmark
python3 modules/09_evaluation/demo_evaluation_benchmarks.py

# Run the automated unit tests
python3 modules/09_evaluation/tests.py
```
