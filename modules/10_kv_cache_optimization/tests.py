import unittest
import sys
import os
import torch
sys.path.insert(0, os.path.dirname(__file__))
from demo_kv_cache import KVCacheSelfAttention

class TestKVCache(unittest.TestCase):
    def setUp(self):
        self.attn = KVCacheSelfAttention(d_model=32, n_heads=2)
        self.attn.eval()

    def test_cache_accumulation(self):
        x1 = torch.randn(1, 3, 32)
        out1, cache = self.attn(x1, past_kv=None)
        k_cached, v_cached = cache
        self.assertEqual(k_cached.shape[2], 3)

        # Step 2: append single new token
        x2 = torch.randn(1, 1, 32)
        out2, cache2 = self.attn(x2, past_kv=cache)
        k2, v2 = cache2
        self.assertEqual(k2.shape[2], 4)

    def test_cached_output_matches_unfused(self):
        # Initial prompt
        torch.manual_seed(42)
        p = torch.randn(1, 3, 32)
        _, cache = self.attn(p, past_kv=None)

        # Next token
        next_tok = torch.randn(1, 1, 32)
        out_cached, _ = self.attn(next_tok, past_kv=cache)

        # Naive full sequence: [p, next_tok]
        full_seq = torch.cat([p, next_tok], dim=1)
        out_naive, _ = self.attn(full_seq, past_kv=None)

        # Output for the last token must match!
        self.assertTrue(torch.allclose(out_cached, out_naive[:, -1:, :], atol=1e-5))

if __name__ == "__main__":
    unittest.main()
