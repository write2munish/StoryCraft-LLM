"""
Module 09: Evaluation, Benchmarking & Safety
=============================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. Intrinsic evaluation: Held-out Perplexity (PPL) calculation.
2. Lexical diversity metrics: Distinct-1 and Distinct-2 n-gram ratios.
3. Repetition rate detection (identifying degenerative loops).
4. Automated LLM-as-a-Judge grading pipeline with structured JSON rubric.
5. Benchmark scorecard generation.

Run this script:
    python3 modules/09_evaluation/demo_evaluation_benchmarks.py
"""

import json
import math
from collections import Counter
from typing import Dict, List, Tuple


# ============================================================================
# 1. Linguistic Diversity Metrics
# ============================================================================
def compute_distinct_n(text: str, n: int = 1) -> float:
    """
    Computes Distinct-N ratio: unique_ngrams / total_ngrams.
    Higher values indicate richer vocabulary diversity without repetitive tropes.
    """
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def compute_repetition_rate(text: str) -> float:
    """
    Calculates proportion of duplicate adjacent words (e.g. 'ball was big and big and big').
    """
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    repeats = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])
    return repeats / (len(words) - 1)


# ============================================================================
# 2. Automated LLM-as-a-Judge Simulation
# ============================================================================
def simulate_llm_judge(prompt: str, story: str) -> Dict[str, any]:
    """
    Simulates calling an advanced frontier model (Gemini / Claude / GPT) to grade
    the story across 4 core dimensions on a 1-10 scale.
    """
    # Deterministic heuristic scoring for demonstration
    has_character = any(name in story.lower() for name in ["lily", "tim", "benny", "dragon", "puppy"])
    has_happy_end = any(end in story.lower() for end in ["smiled", "happy", "together", "joy", "picnic"])
    distinct_1 = compute_distinct_n(story, 1)

    coherence = 9 if (has_character and has_happy_end) else 6
    grammar = 10 if story.endswith(".") else 7
    creativity = min(10, int(distinct_1 * 12))
    safety = 10 if "kill" not in story and "monster" not in story else 4

    overall = round((coherence + grammar + creativity + safety) / 4.0, 1)

    return {
        "plot_coherence": coherence,
        "grammatical_correctness": grammar,
        "creativity_vocabulary": creativity,
        "safety_and_morals": safety,
        "overall_score": overall,
        "feedback": "Story exhibits strong character continuity and delightful positive moral resolution."
    }


# ============================================================================
# 3. Demonstration Runner
# ============================================================================
def main():
    print("=" * 70)
    print("📊 MODULE 09: EVALUATION, BENCHMARKING & SAFETY")
    print("=" * 70)

    # 1. Intrinsic Perplexity
    print("\n1. Intrinsic Evaluation (Held-out Perplexity):")
    # Simulate cross-entropy loss over a held-out test split of 5,000 tokens
    eval_loss_base = 1.65
    eval_loss_aligned = 1.42

    ppl_base = math.exp(eval_loss_base)
    ppl_aligned = math.exp(eval_loss_aligned)

    print(f"   Base Model:    Loss = {eval_loss_base:.4f} -> Perplexity = {ppl_base:.2f}")
    print(f"   Aligned Model: Loss = {eval_loss_aligned:.4f} -> Perplexity = {ppl_aligned:.2f}")
    print("   (Lower perplexity indicates higher confidence on human-written children's stories)")

    # 2. Lexical Diversity Comparison
    print("\n2. Lexical Diversity Benchmarks:")
    story_repetitive = (
        "Once upon a time Lily had a ball. The ball was red and the ball was big. "
        "The ball was red and big and big and big and red."
    )
    story_fluent = (
        "Once upon a time, Lily found a sparkling red ball tucked beneath the fragrant green bushes. "
        "She laughed with delight and rolled it gently toward her faithful puppy Barnaby."
    )

    print(f"   Sample A (Repetitive):")
    print(f"   - Distinct-1 (Unigrams): {compute_distinct_n(story_repetitive, 1):.1%}")
    print(f"   - Distinct-2 (Bigrams):  {compute_distinct_n(story_repetitive, 2):.1%}")
    print(f"   - Repetition Rate:       {compute_repetition_rate(story_repetitive):.1%}")

    print(f"\n   Sample B (Fluent & Diverse):")
    print(f"   - Distinct-1 (Unigrams): {compute_distinct_n(story_fluent, 1):.1%}")
    print(f"   - Distinct-2 (Bigrams):  {compute_distinct_n(story_fluent, 2):.1%}")
    print(f"   - Repetition Rate:       {compute_repetition_rate(story_fluent):.1%}")

    # 3. LLM-as-a-Judge Evaluation
    print("\n3. LLM-as-a-Judge Automated Evaluation Report:")
    test_prompt = "Tell a story about sharing toys at the park."
    judge_results = simulate_llm_judge(test_prompt, story_fluent)

    print(f"   Prompt: \"{test_prompt}\"")
    print("   Judge Scorecard (1-10 Scale):")
    for metric, val in judge_results.items():
        if metric != "feedback":
            print(f"   - {metric.replace('_', ' ').title():<25}: {val}")
    print(f"   Judge Qualitative Feedback: \"{judge_results['feedback']}\"")

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. Evaluating generative text requires both statistical metrics (PPL, Distinct-N)")
    print("   and semantic qualitative metrics (LLM-as-a-Judge).")
    print("2. Distinct-N catches degenerate loops where perplexity might appear falsely low.")
    print("3. Automated evaluation suites can run in CI/CD to prevent regressions during fine-tuning.")
    print("👉 Next: Module 10 optimizes inference speed with Key-Value (KV) Caching!")
    print("=" * 70)


if __name__ == "__main__":
    main()
