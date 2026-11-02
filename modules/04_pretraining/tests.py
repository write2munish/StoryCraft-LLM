import unittest
import sys
import os
import math
import torch
sys.path.insert(0, os.path.dirname(__file__))
from demo_pretraining_engine import get_lr_cosine_schedule, MiniStoryTransformer

class TestPretrainingEngine(unittest.TestCase):
    def test_cosine_schedule_warmup(self):
        # Step 0 should be > 0 and <= max_lr / warmup_steps
        lr_0 = get_lr_cosine_schedule(step=0, warmup_steps=10, total_steps=100, max_lr=1e-3, min_lr=1e-4)
        self.assertAlmostEqual(lr_0, 1e-4)

        # Step at warmup should equal max_lr
        lr_warmup = get_lr_cosine_schedule(step=9, warmup_steps=10, total_steps=100, max_lr=1e-3, min_lr=1e-4)
        self.assertAlmostEqual(lr_warmup, 1e-3)

        # Step at end should equal min_lr
        lr_end = get_lr_cosine_schedule(step=101, warmup_steps=10, total_steps=100, max_lr=1e-3, min_lr=1e-4)
        self.assertEqual(lr_end, 1e-4)

    def test_perplexity_calculation(self):
        # PPL = exp(loss)
        loss = 0.0
        ppl = math.exp(loss)
        self.assertEqual(ppl, 1.0)

        loss = 2.0
        ppl = math.exp(loss)
        self.assertAlmostEqual(ppl, 7.389, places=3)

    def test_training_step_reduces_loss(self):
        torch.manual_seed(42)
        model = MiniStoryTransformer(vocab_size=50, d_model=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = torch.nn.CrossEntropyLoss()

        x = torch.randint(0, 50, (4, 8))
        y = torch.randint(0, 50, (4, 8))

        # Initial loss
        init_loss = loss_fn(model(x).view(-1, 50), y.view(-1)).item()

        # 5 optimization steps
        for _ in range(5):
            optimizer.zero_grad()
            l = loss_fn(model(x).view(-1, 50), y.view(-1))
            l.backward()
            optimizer.step()

        final_loss = loss_fn(model(x).view(-1, 50), y.view(-1)).item()
        self.assertLess(final_loss, init_loss)

if __name__ == "__main__":
    unittest.main()
