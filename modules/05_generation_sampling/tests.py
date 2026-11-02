import unittest
import sys
import os
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.dirname(__file__))

from storycraft import StoryTokenizer, StoryCraftLM
from generate import StoryGenerator, apply_repetition_penalty, sample_next_token
from load_pretrained import map_and_load_state_dict


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

    def test_story_generator_greedy_and_sample(self):
        tok = StoryTokenizer(target_vocab_size=32)
        tok.train("once upon a time there was a little bird")
        model = StoryCraftLM(vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=2)
        generator = StoryGenerator(model, tok)
        
        # Test distribution query
        dist = generator.get_next_token_distribution("once upon", temperature=1.0)
        self.assertGreater(len(dist), 0)
        
        # Test greedy generation
        story = generator.generate("once upon", max_new_tokens=5, strategy="greedy")
        self.assertTrue(isinstance(story, str))
        self.assertTrue(len(story) > 0)

    def test_map_and_load_state_dict(self):
        model = StoryCraftLM(vocab_size=32, d_model=32, n_layers=1, n_heads=2)
        mock_weights = {
            "tok_embed.weight": torch.ones_like(model.tok_embed.weight) * 3.14,
            "norm_final.weight": torch.ones_like(model.norm_final.weight) * 2.71
        }
        loaded, total = map_and_load_state_dict(mock_weights, model)
        self.assertEqual(loaded, 2)
        self.assertTrue(torch.allclose(model.tok_embed.weight, mock_weights["tok_embed.weight"]))


if __name__ == "__main__":
    unittest.main()
