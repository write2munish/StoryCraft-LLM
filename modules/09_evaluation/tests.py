import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from demo_evaluation_benchmarks import compute_distinct_n, compute_repetition_rate, simulate_llm_judge

class TestEvaluation(unittest.TestCase):
    def test_distinct_n_bounds(self):
        text_all_unique = "one two three four five"
        self.assertEqual(compute_distinct_n(text_all_unique, 1), 1.0)

        text_all_same = "ball ball ball ball"
        self.assertEqual(compute_distinct_n(text_all_same, 1), 0.25)

    def test_repetition_rate(self):
        repeats = "big big big"
        rate = compute_repetition_rate(repeats)
        self.assertEqual(rate, 1.0)

        no_repeats = "big small tall"
        rate_none = compute_repetition_rate(no_repeats)
        self.assertEqual(rate_none, 0.0)

    def test_judge_rubric_structure(self):
        res = simulate_llm_judge("prompt", "Lily smiled happily.")
        self.assertIn("plot_coherence", res)
        self.assertIn("grammatical_correctness", res)
        self.assertIn("overall_score", res)
        self.assertTrue(1 <= res["overall_score"] <= 10)

if __name__ == "__main__":
    unittest.main()
