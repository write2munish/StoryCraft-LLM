import unittest
import sys
import os
import torch
sys.path.insert(0, os.path.dirname(__file__))
from demo_story_classifier import StoryClassifier

class TestStoryClassifier(unittest.TestCase):
    def setUp(self):
        self.model = StoryClassifier(vocab_size=100, d_model=32, num_genres=4)

    def test_output_shapes(self):
        x = torch.randint(0, 100, (2, 10))
        out = self.model(x)
        self.assertEqual(out["genre_logits"].shape, (2, 4))
        self.assertEqual(out["safety_logits"].shape, (2, 2))

    def test_freezing_backbone(self):
        self.model.freeze_backbone()
        # Embeddings and transformer layers should not require grad
        self.assertFalse(self.model.tok_embed.weight.requires_grad)
        # Classification heads should still require grad
        self.assertTrue(self.model.genre_head.weight.requires_grad)
        self.assertTrue(self.model.safety_head.weight.requires_grad)

if __name__ == "__main__":
    unittest.main()
