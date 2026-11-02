import unittest
import sys
import os
import torch
sys.path.insert(0, os.path.dirname(__file__))
from demo_transformer_architecture import RMSNorm, CausalSelfAttention, SwiGLUMLP, StoryCraftLM

class TestTransformerArchitecture(unittest.TestCase):
    def test_rmsnorm_forward(self):
        norm = RMSNorm(dim=32)
        x = torch.randn(2, 5, 32)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)
        # Check that RMS is approximately 1.0
        rms = torch.sqrt(out.pow(2).mean(-1))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-3))

    def test_causal_mask_strictly_zeros_future(self):
        attn = CausalSelfAttention(d_model=64, n_heads=4)
        # Ensure that altering future tokens does not change predictions for earlier tokens
        x1 = torch.randn(1, 6, 64)
        x2 = x1.clone()
        # Modify the last token
        x2[:, -1, :] = torch.randn(64)

        with torch.no_grad():
            out1 = attn(x1)
            out2 = attn(x2)

        # Output at token 0, 1, 2 must be IDENTICAL between x1 and x2
        self.assertTrue(torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5))

    def test_swiglu_shape(self):
        mlp = SwiGLUMLP(d_model=64, hidden_dim=128)
        x = torch.randn(2, 4, 64)
        out = mlp(x)
        self.assertEqual(out.shape, (2, 4, 64))

    def test_storycraft_lm_forward_and_backward(self):
        model = StoryCraftLM(vocab_size=100, d_model=32, n_layers=2, n_heads=4)
        x = torch.randint(0, 100, (2, 8))
        logits = model(x)
        self.assertEqual(logits.shape, (2, 8, 100))

        # Test gradient propagation
        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(model.tok_embed.weight.grad)
        self.assertGreater(model.tok_embed.weight.grad.abs().sum().item(), 0.0)

if __name__ == "__main__":
    unittest.main()
