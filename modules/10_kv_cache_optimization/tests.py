import unittest
import sys
import os
import torch
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
from storycraft import StoryCraftLM

class TestKVCache(unittest.TestCase):
    def setUp(self):
        self.model = StoryCraftLM(vocab_size=64, d_model=32, n_layers=2, n_heads=2)
        self.model.eval()

    def test_cache_accumulation(self):
        x1 = torch.randint(0, 64, (1, 3))
        _, past_kvs = self.model(x1, past_kvs=None)
        # Check first block key cache length
        k_cached = past_kvs[0][0]
        self.assertEqual(k_cached.shape[2], 3)

        # Step 2: append single new token
        x2 = torch.randint(0, 64, (1, 1))
        _, past_kvs2 = self.model(x2, past_kvs=past_kvs)
        k2 = past_kvs2[0][0]
        self.assertEqual(k2.shape[2], 4)

    def test_cached_output_matches_unfused(self):
        torch.manual_seed(42)
        p = torch.randint(0, 64, (1, 3))
        _, cache = self.model(p, past_kvs=None)

        next_tok = torch.randint(0, 64, (1, 1))
        out_cached, _ = self.model(next_tok, past_kvs=cache)

        # Naive full sequence: [p, next_tok]
        full_seq = torch.cat([p, next_tok], dim=1)
        out_naive, _ = self.model(full_seq, past_kvs=None)

        self.assertTrue(torch.allclose(out_cached[:, -1:, :], out_naive[:, -1:, :], atol=1e-4))

if __name__ == "__main__":
    unittest.main()
