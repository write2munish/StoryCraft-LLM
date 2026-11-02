import unittest
import sys
import os
import torch
sys.path.insert(0, os.path.dirname(__file__))
from demo_alignment_dpo import compute_dpo_loss

class TestAlignmentDPO(unittest.TestCase):
    def test_dpo_loss_decreases_as_chosen_preferred(self):
        ref_w = torch.tensor([0.0])
        ref_l = torch.tensor([0.0])

        # Policy prefers winning story by +2.0 log-prob
        pi_w_good = torch.tensor([2.0])
        pi_l_good = torch.tensor([0.0])
        loss_good = compute_dpo_loss(pi_w_good, pi_l_good, ref_w, ref_l, beta=0.1)

        # Policy prefers losing story
        pi_w_bad = torch.tensor([0.0])
        pi_l_bad = torch.tensor([2.0])
        loss_bad = compute_dpo_loss(pi_w_bad, pi_l_bad, ref_w, ref_l, beta=0.1)

        self.assertLess(loss_good.item(), loss_bad.item())

    def test_dpo_gradient_direction(self):
        ref_w = torch.tensor([-5.0])
        ref_l = torch.tensor([-5.0])

        pi_w = torch.nn.Parameter(torch.tensor([-5.0]))
        pi_l = torch.nn.Parameter(torch.tensor([-5.0]))

        loss = compute_dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.5)
        loss.backward()

        # Gradient on chosen should be negative (loss decreases when pi_w increases)
        self.assertLess(pi_w.grad.item(), 0.0)
        # Gradient on rejected should be positive (loss decreases when pi_l decreases)
        self.assertGreater(pi_l.grad.item(), 0.0)

if __name__ == "__main__":
    unittest.main()
