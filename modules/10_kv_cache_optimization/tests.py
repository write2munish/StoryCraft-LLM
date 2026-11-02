import unittest
import sys
import os
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.dirname(__file__))

from storycraft import StoryTokenizer, StoryCraftLM
from kv_cache import CachedStoryGenerator, verify_generation_correctness
from benchmark_latency import quantize_storycraft_int8


class TestKVCacheOptimization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = StoryCraftLM(vocab_size=64, d_model=32, n_layers=2, n_heads=2)
        self.model.eval()
        
        self.tokenizer = StoryTokenizer(target_vocab_size=64)
        self.tokenizer.train("once upon a time lily and barnaby dog played in the sunny garden")

    def test_cache_shape_accumulation(self):
        """Verifies that key and value cache tensors grow by exactly 1 token at each step."""
        x_prefill = torch.randint(0, 64, (1, 3))
        _, past_kvs = self.model(x_prefill, past_kvs=None)
        
        # Verify initial prefill shape: [B=1, n_heads=2, T=3, head_dim=16]
        k_prefill, v_prefill = past_kvs[0]
        self.assertEqual(k_prefill.shape, (1, 2, 3, 16))
        self.assertEqual(v_prefill.shape, (1, 2, 3, 16))

        # Decode step 1
        x_step1 = torch.randint(0, 64, (1, 1))
        _, past_kvs = self.model(x_step1, past_kvs=past_kvs)
        k_step1, _ = past_kvs[0]
        self.assertEqual(k_step1.shape[2], 4)

        # Decode step 2
        x_step2 = torch.randint(0, 64, (1, 1))
        _, past_kvs = self.model(x_step2, past_kvs=past_kvs)
        k_step2, _ = past_kvs[0]
        self.assertEqual(k_step2.shape[2], 5)

    def test_mathematical_correctness_and_token_identity(self):
        """Verifies exact logit parity and token-by-token identity between naive and cached passes."""
        report = verify_generation_correctness(
            self.model,
            self.tokenizer,
            prompt="once upon a time",
            num_tokens=10,
            atol=1e-4
        )
        self.assertTrue(report.passed)
        self.assertTrue(report.all_tokens_match)
        self.assertLess(report.max_diff_across_all_steps, 1e-4)
        self.assertGreater(report.min_cosine_sim, 0.9999)
        self.assertEqual(report.story_naive, report.story_cached)

    def test_multiple_prompts_parity(self):
        """Ensures token parity holds across diverse story prompts."""
        prompts = [
            "lily played in",
            "barnaby dog saw a",
            "sunny garden with"
        ]
        for p in prompts:
            report = verify_generation_correctness(
                self.model,
                self.tokenizer,
                prompt=p,
                num_tokens=8,
                atol=1e-4
            )
            self.assertTrue(report.all_tokens_match, f"Failed match on prompt: {p}")
            self.assertEqual(report.tokens_naive, report.tokens_cached)

    def test_cached_story_generator_runs_fluently(self):
        """Tests that CachedStoryGenerator produces valid strings with repetition penalty."""
        generator = CachedStoryGenerator(self.model, self.tokenizer)
        tokens, story, elapsed = generator.generate_cached(
            "once upon a time",
            max_new_tokens=12,
            temperature=0.7,
            repetition_penalty=1.2
        )
        self.assertIsInstance(story, str)
        self.assertGreater(len(story), 0)
        self.assertEqual(len(tokens), len(self.tokenizer.encode("once upon a time", add_special_tokens=False)) + 12)

    def test_dynamic_quantization_inference(self):
        """Verifies that dynamic INT8 quantized model executes forward pass properly."""
        q_model = quantize_storycraft_int8(self.model)
        x = torch.randint(0, 64, (1, 4))
        logits, _ = q_model(x)
        self.assertEqual(logits.shape, (1, 4, 64))


if __name__ == "__main__":
    unittest.main()
