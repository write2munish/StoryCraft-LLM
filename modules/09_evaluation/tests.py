import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from demo_evaluation_benchmarks import compute_distinct_n

class TestEvaluation(unittest.TestCase):
    def test_distinct_n_bounds(self):
        text_all_unique = "one two three four five"
        self.assertEqual(compute_distinct_n(text_all_unique, 1), 1.0)

        text_all_same = "ball ball ball ball"
        self.assertEqual(compute_distinct_n(text_all_same, 1), 0.25)

if __name__ == "__main__":
    unittest.main()
