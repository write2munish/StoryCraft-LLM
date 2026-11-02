import unittest
import sys
import os
import torch
sys.path.insert(0, os.path.dirname(__file__))
from demo_embeddings_rope import RotaryPositionalEmbedding

class TestEmbeddingsRoPE(unittest.TestCase):
    def setUp(self):
        self.dim = 16
        self.rope = RotaryPositionalEmbedding(dim=self.dim, max_seq_len=64)

    def test_shape_preserved(self):
        # [batch, seq_len, heads, head_dim]
        x = torch.randn(2, 8, 4, self.dim)
        x_rot = self.rope(x, seq_len=8)
        self.assertEqual(x.shape, x_rot.shape)

    def test_norm_preservation(self):
        # 2D coordinate rotation preserves vector Euclidean length
        x = torch.randn(1, 4, 1, self.dim)
        x_rot = self.rope(x, seq_len=4)
        norm_orig = torch.norm(x, dim=-1)
        norm_rot = torch.norm(x_rot, dim=-1)
        self.assertTrue(torch.allclose(norm_orig, norm_rot, atol=1e-5))

    def test_relative_distance_invariance(self):
        # <q_m, k_n> depends only on (m - n)
        torch.manual_seed(99)
        q = torch.randn(1, 4, 1, self.dim)
        k = torch.randn(1, 4, 1, self.dim)

        q_rot = self.rope(q, seq_len=4)
        k_rot = self.rope(k, seq_len=4)

        # Dot product at offset = 1 (m=0, n=1)
        dot_0_1 = torch.sum(q_rot[0, 0, 0] * k_rot[0, 1, 0]).item()

        # Shift identical vectors to (m=2, n=3) (offset is still 1)
        q_shifted = torch.zeros(1, 4, 1, self.dim)
        k_shifted = torch.zeros(1, 4, 1, self.dim)
        q_shifted[0, 2, 0] = q[0, 0, 0]
        k_shifted[0, 3, 0] = k[0, 1, 0]

        q_rot_s = self.rope(q_shifted, seq_len=4)
        k_rot_s = self.rope(k_shifted, seq_len=4)
        dot_2_3 = torch.sum(q_rot_s[0, 2, 0] * k_rot_s[0, 3, 0]).item()

        self.assertAlmostEqual(dot_0_1, dot_2_3, places=5)

if __name__ == "__main__":
    unittest.main()
