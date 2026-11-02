import unittest
import os
import sys
import torch
import torch.nn as nn

# Robust repo root discovery whether called directly or via symlink
cur = os.path.abspath(os.path.dirname(__file__))
repo_root = cur
while repo_root and repo_root != "/":
    if os.path.exists(os.path.join(repo_root, "storycraft")):
        break
    repo_root = os.path.dirname(repo_root)

sys.path.insert(0, repo_root)

from storycraft import set_seed


class TestPyTorchPrimer(unittest.TestCase):
    def setUp(self):
        set_seed(42)

    def test_tensor_strides_and_contiguity(self):
        t = torch.arange(12).reshape(3, 4)
        self.assertEqual(t.stride(), (4, 1))
        self.assertTrue(t.is_contiguous())

        trans = t.t()
        self.assertEqual(trans.stride(), (1, 4))
        self.assertFalse(trans.is_contiguous())

        contig = trans.contiguous()
        self.assertTrue(contig.is_contiguous())
        self.assertEqual(contig.shape, (4, 3))

    def test_broadcasting_rules(self):
        query = torch.randn(2, 4, 16, 32)
        rope_cache = torch.randn(1, 1, 16, 32)
        result = query * rope_cache
        self.assertEqual(result.shape, (2, 4, 16, 32))

    def test_autograd_gradients(self):
        x = torch.tensor([3.0, 5.0], requires_grad=True)
        # y = 3x^2 - 2x
        y = 3 * (x ** 2) - 2 * x
        loss = y.sum()
        loss.backward()

        # dy/dx = 6x - 2 -> for [3, 5]: expected [16.0, 28.0]
        expected_grad = [16.0, 28.0]
        self.assertTrue(torch.allclose(x.grad, torch.tensor(expected_grad)))

    def test_no_grad_disables_graph(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        with torch.no_grad():
            out = x * 3
        self.assertFalse(out.requires_grad)

    def test_custom_module_buffers_and_parameters(self):
        class ToyAttention(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.proj = nn.Linear(dim, dim, bias=False)
                # Register causal buffer
                self.register_buffer("mask", torch.triu(torch.full((4, 4), float("-inf")), diagonal=1))

            def forward(self, x):
                return self.proj(x)

        model = ToyAttention(16)
        params = list(model.parameters())
        self.assertEqual(len(params), 1)  # Only proj.weight
        buffers = dict(model.named_buffers())
        self.assertIn("mask", buffers)
        self.assertEqual(buffers["mask"].shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
