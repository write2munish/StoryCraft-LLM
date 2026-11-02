import unittest
import sys
import os
import torch
sys.path.insert(0, os.path.dirname(__file__))
from demo_generation_sampling import apply_repetition_penalty, sample_next_token

class TestGenerationSampling(unittest.TestCase):
    def test_repetition_penalty_reduces_seen_tokens(self):
        logits = torch.tensor([2.0, 5.0, 1.0])
        seen = [1]
        penalized = apply_repetition_penalty(logits.clone(), seen, penalty=2.0)
        self.assertEqual(penalized[1].item(), 2.5)
        self.assertEqual(penalized[0].item(), 2.0)

    def test_temperature_scaling_bounds(self):
        logits = torch.tensor([1.0, 10.0, 2.0])
        samples = [sample_next_token(logits.clone(), temperature=0.01, top_k=0, top_p=1.0) for _ in range(10)]
        self.assertTrue(all(s == 1 for s in samples))

    def test_top_k_restricts_candidates(self):
        logits = torch.tensor([10.0, 9.0, 1.0, 0.5])
        samples = [sample_next_token(logits.clone(), temperature=1.0, top_k=2, top_p=1.0) for _ in range(20)]
        self.assertTrue(all(s in [0, 1] for s in samples))

if __name__ == "__main__":
    unittest.main()
