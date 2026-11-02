import unittest
import sys
import os
import torch
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.dirname(__file__))
from storycraft import StoryCraftLM
from demo_story_classifier import StoryClassifier

class TestStoryClassifier(unittest.TestCase):
    def setUp(self):
        base_model = StoryCraftLM(vocab_size=100, d_model=32, n_layers=2, n_heads=2)
        self.model = StoryClassifier(base_model, num_genres=4, num_safety=2)

    def test_output_shapes(self):
        x = torch.randint(0, 100, (2, 10))
        out = self.model(x)
        self.assertEqual(out["genre_logits"].shape, (2, 4))
        self.assertEqual(out["safety_logits"].shape, (2, 2))

    def test_freezing_backbone(self):
        self.model.freeze_backbone()
        for p in self.model.base_model.parameters():
            self.assertFalse(p.requires_grad)
        self.assertTrue(self.model.genre_head.weight.requires_grad)
        self.assertTrue(self.model.safety_head.weight.requires_grad)

if __name__ == "__main__":
    unittest.main()
