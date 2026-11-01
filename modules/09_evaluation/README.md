# 📊 Module 09: Evaluation, Benchmarking & Safety

> **Central Anchor in this Module:** Systematically testing `StoryCraftLM` across its development stages (Base vs. Instruct vs. DPO) using intrinsic perplexity, lexical diversity, and automated LLM-as-a-judge scoring.

---

## 🎯 Learning Objectives
1. Measure intrinsic metrics (Perplexity on held-out story test sets).
2. Quantify linguistic diversity: Distinct-1 and Distinct-2 n-gram ratios.
3. Build an automated "LLM-as-a-Judge" grading pipeline using modern frontier models (Gemini / Claude / GPT) to score stories on a 1–10 rubric.
4. Run adversarial safety prompts to verify alignment robustness.

---

## 🔬 Core Theory: How Do We Know an LLM is "Good"?

Evaluating generative language models is notoriously challenging because multiple distinct stories can all be valid and delightful.

### 1. Intrinsic Metric: Held-out Perplexity
Measures how surprised the model is by human-written children's stories:
$$\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log P(w_i \mid w_{<i})\right)$$

### 2. Lexical Diversity: Distinct-N
$$\text{Distinct-}n = \frac{\text{Count of unique } n\text{-grams}}{\text{Total count of generated } n\text{-grams}}$$
Prevents celebrating models that score low perplexity simply by repeating the same safe phrases over and over.

### 3. LLM-as-a-Judge Rubric
We feed 100 standardized test prompts to our model, and send prompt + generated story to an automated judge with a strict JSON evaluation rubric:
```json
{
  "plot_coherence": 9,
  "grammatical_correctness": 10,
  "creativity": 8,
  "age_appropriateness": 10,
  "feedback": "Story follows Lily and her dog effectively with clear beginning, middle, and end."
}
```

---

## 💻 Lab Exercises
- Implement `benchmark.py`:
  - Run evaluation suite across 100 diverse test prompts.
  - Calculate average Distinct-1, Distinct-2, and length distributions.
  - Call judge script (`judge_stories.py`) to generate an automated scorecard report (`benchmark_results.json`).
