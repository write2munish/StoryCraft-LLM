import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from demo_capstone_app import StoryCraftProductionPipeline

class TestCapstoneApp(unittest.TestCase):
    def setUp(self):
        self.app = StoryCraftProductionPipeline()

    def test_chatml_formatting(self):
        prompt = self.app.format_prompt(
            protagonist="Barnaby",
            theme="Adventure",
            age_group="Toddler",
            moral="Sharing"
        )
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("Barnaby", prompt)
        self.assertIn("Sharing", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_streaming_yields_tokens(self):
        prompt = self.app.format_prompt("Lily", "Bedtime", "Preschool", "Friendship")
        tokens = list(self.app.stream_story(prompt, max_tokens=10))
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(isinstance(t, str) for t in tokens))

    def test_story_branching(self):
        story = "Lily saw a pond."
        branched = self.app.branch_story(story, alternate_choice="jump in the water")
        self.assertIn("jump in the water", branched)
        self.assertTrue(branched.startswith("Lily saw a pond."))

if __name__ == "__main__":
    unittest.main()
