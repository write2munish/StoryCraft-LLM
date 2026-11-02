import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from demo_foundations import BigramStoryModel, TrigramStoryModel

class TestFoundations(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "once upon a time there was a dog .",
            "the dog saw a cat ."
        ]
        self.bigram = BigramStoryModel()
        self.bigram.train(self.corpus)

    def test_vocab_discovery(self):
        self.assertIn("once", self.bigram.vocab)
        self.assertIn("dog", self.bigram.vocab)
        self.assertIn(".", self.bigram.vocab)

    def test_probabilities_sum_to_one(self):
        probs = self.bigram.get_probabilities("a")
        total = sum(probs.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_deterministic_transitions(self):
        # 'once' is followed only by 'upon'
        probs = self.bigram.get_probabilities("once")
        self.assertEqual(probs.get("upon"), 1.0)

    def test_unseen_word_returns_empty(self):
        probs = self.bigram.get_probabilities("nonexistent_word")
        self.assertEqual(probs, {})

    def test_generation_produces_text(self):
        story = self.bigram.generate("once upon", max_words=10)
        self.assertTrue(story.startswith("once upon"))
        self.assertGreater(len(story.split()), 2)

if __name__ == "__main__":
    unittest.main()
