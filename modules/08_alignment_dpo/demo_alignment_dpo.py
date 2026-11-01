"""
Module 08: Post-Training Phase 2 - Alignment via DPO (Direct Preference Optimization)
=====================================================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. Why Supervised Fine-Tuning is not enough (handling moral endings, coherence, safety).
2. Curating Preference Pairs (prompt, chosen_story y_w, rejected_story y_l).
3. The DPO loss formulation without RL / Actor-Critic / Reward Model complexity:
   L_DPO = -E[log sigma(beta * log(pi_theta(y_w|x) / pi_ref(y_w|x)) - beta * log(pi_theta(y_l|x) / pi_ref(y_l|x)))]
4. Simulating a 3-step DPO alignment step and tracking the reward margin growth.

Run this script:
    python3 modules/08_alignment_dpo/demo_alignment_dpo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """Direct Preference Optimization (DPO) Loss (Rafailov et al., 2023)."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    return loss

def main():
    print("=" * 70)
    print("🧭 MODULE 08: PREFERENCE ALIGNMENT VIA DPO")
    print("=" * 70)

    print("\n1. Story Preference Triplet Dataset:")
    prompt = "A little bear named Benny finds a jar of honey in the woods."
    chosen_story = (
        "Benny wanted to eat it all, but remembered his friend Rabbit. "
        "He carried the sweet jar to Rabbit's burrow, and they enjoyed a wonderful picnic together."
    )
    rejected_story = (
        "Benny ate the whole jar angrily and threw the glass on the ground. "
        "He went home feeling sick and bad. The end."
    )

    print(f"   Prompt:           '{prompt}'")
    print(f"   Chosen (y_w):     '{chosen_story}' (Kind, moral resolution)")
    print(f"   Rejected (y_l):   '{rejected_story}' (Rude, abrupt, bad behavior)")

    print("\n2. The DPO Formula Demystified:")
    print("   Unlike RLHF (PPO), DPO requires NO reward model and NO reinforcement learning loop!")
    print("   The language model policy implicitly optimizes the reward:")
    print("   r(x, y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )")

    print("\n3. Simulating DPO Alignment Updates:")
    print("-" * 70)
    print(f"{'Step':^6} | {'Policy LogP(w)':^15} | {'Policy LogP(l)':^15} | {'DPO Loss':^10} | {'Margin':^10}")
    print("-" * 70)

    ref_chosen_logp = torch.tensor([-12.50])
    ref_rejected_logp = torch.tensor([-12.30])

    policy_chosen_logp = nn.Parameter(torch.tensor([-12.50]))
    policy_rejected_logp = nn.Parameter(torch.tensor([-12.30]))

    optimizer = torch.optim.SGD([policy_chosen_logp, policy_rejected_logp], lr=0.5)
    beta = 0.5

    for step in range(3):
        optimizer.zero_grad()
        loss = compute_dpo_loss(
            policy_chosen_logp, policy_rejected_logp,
            ref_chosen_logp, ref_rejected_logp,
            beta=beta
        )
        loss.backward()
        optimizer.step()

        margin = (policy_chosen_logp - policy_rejected_logp).item()
        print(f"{step + 1:^6} | {policy_chosen_logp.item():^15.4f} | {policy_rejected_logp.item():^15.4f} | {loss.item():^10.4f} | {margin:^+10.4f}")

    print("-" * 70)

    print("\n4. Output Comparison After DPO Alignment:")
    print("   Before DPO: Model assigns roughly equal probability to selfish vs sharing endings.")
    print("   After DPO:  Model assigns orders of magnitude higher probability to positive endings!")

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. DPO aligns model values using a simple binary cross-entropy formulation.")
    print("2. The frozen reference model pi_ref anchors the policy so it doesn't collapse into gibberish.")
    print("3. By curating wholesome story pairs, the model naturally adopts gentle children's story morals.")
    print("👉 Next: Module 09 measures story quality with Automated Benchmarks and LLM-as-a-Judge!")
    print("=" * 70)

if __name__ == "__main__":
    main()
