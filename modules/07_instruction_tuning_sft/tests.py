import unittest
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from demo_sft_lora import LoRALinear

class TestSFTLoRA(unittest.TestCase):
    def test_lora_zero_initialization(self):
        d = 64
        layer = LoRALinear(in_features=d, out_features=d, rank=4, alpha=16.0)
        x = torch.randn(2, 5, d)
        with torch.no_grad():
            out_lora = layer(x)
            out_base = F.linear(x, layer.base_weight)
        # At start, Delta W is strictly zero because B is zero-initialized
        self.assertTrue(torch.allclose(out_lora, out_base, atol=1e-6))

    def test_lora_parameter_counts(self):
        in_f, out_f, r = 128, 128, 4
        layer = LoRALinear(in_features=in_f, out_features=out_f, rank=r)
        base_count = in_f * out_f
        lora_count = (r * in_f) + (out_f * r)
        self.assertEqual(layer.base_weight.numel(), base_count)
        self.assertEqual(layer.lora_A.numel() + layer.lora_B.numel(), lora_count)
        self.assertLess(lora_count, base_count * 0.1)

    def test_weight_merging(self):
        layer = LoRALinear(in_features=32, out_features=32, rank=4)
        # Give B non-zero values to simulate fine-tuned state
        nn.init.normal_(layer.lora_B)
        x = torch.randn(1, 4, 32)
        out_separate = layer(x)

        merged_w = layer.merge_weights()
        out_merged = F.linear(x, merged_w)
        self.assertTrue(torch.allclose(out_separate, out_merged, atol=1e-5))

if __name__ == "__main__":
    unittest.main()
