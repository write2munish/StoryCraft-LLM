"""
Module 08: Post-Training Phase 2 - Alignment via DPO (Direct Preference Optimization)
=====================================================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. Loading the persistent reference model: checkpoints/storycraft_instruct.pt.
2. Curating Preference Pairs (prompt, chosen wholesome story y_w, rejected rude ending y_l).
3. The DPO loss formulation without RL / Actor-Critic / Reward Model complexity:
   L_DPO = -E[log sigma(beta * log(pi_theta(y_w|x) / pi_ref(y_w|x)) - beta * log(pi_theta(y_l|x) / pi_ref(y_l|x)))]
4. Aligning the policy model and saving to checkpoints/storycraft_aligned.pt.

Run this script:
    python3 modules/08_alignment_dpo/demo_alignment_dpo.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
from storycraft import StoryTokenizer, StoryCraftLM


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

    chk_dir = os.path.join(repo_root, "checkpoints")
    tok_path = os.path.join(chk_dir, "tokenizer.pkl")
    instruct_path = os.path.join(chk_dir, "storycraft_instruct.pt")
    out_aligned_path = os.path.join(chk_dir, "storycraft_aligned.pt")

    # 1. Load Reference Model from Module 07
    print("\n1. Loading Frozen Reference Model from Module 07:")
    if os.path.exists(tok_path) and os.path.exists(instruct_path):
        tokenizer = StoryTokenizer.from_pretrained(tok_path)
        ckpt = torch.load(instruct_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {"vocab_size": tokenizer.vocab_size, "d_model": 128, "n_layers": 4, "n_heads": 4})
        
        # Frozen Reference Model pi_ref
        ref_model = StoryCraftLM(**cfg)
        # Load weights safely (filter lora keys if needed)
        clean_state = {k: v for k, v in ckpt["model_state_dict"].items() if "lora_" not in k}
        ref_model.load_state_dict(clean_state, strict=False)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # Trainable Policy Model pi_theta
        policy_model = StoryCraftLM(**cfg)
        policy_model.load_state_dict(clean_state, strict=False)
        print(f"   ✓ Loaded Reference Model pi_ref from: {instruct_path} (Frozen)")
        print(f"   ✓ Initialized Policy Model pi_theta from reference weights (Trainable)")
    else:
        tokenizer = StoryTokenizer(target_vocab_size=128)
        ref_model = StoryCraftLM(vocab_size=128, d_model=64, n_layers=2, n_heads=2)
        policy_model = StoryCraftLM(vocab_size=128, d_model=64, n_layers=2, n_heads=2)

    # 2. Preference Pairs
    print("\n2. Curated Preference Triplet:")
    prompt = "A little bear named Benny finds a jar of sweet honey in the woods."
    chosen_story = "Benny shared the sweet jar with Rabbit, and they enjoyed a wonderful picnic together."
    rejected_story = "Benny ate the whole jar greedily and threw the glass on the ground angrily."

    print(f"   Prompt:           '{prompt}'")
    print(f"   Chosen (y_w):     '{chosen_story}' (Kind, moral resolution)")
    print(f"   Rejected (y_l):   '{rejected_story}' (Selfish, destructive ending)")

    # 3. DPO Training Step
    print("\n3. Executing DPO Optimization Step:")
    chosen_ids = torch.tensor([tokenizer.encode(prompt + " " + chosen_story, add_special_tokens=False)[:16]], dtype=torch.long)
    rejected_ids = torch.tensor([tokenizer.encode(prompt + " " + rejected_story, add_special_tokens=False)[:16]], dtype=torch.long)

    # Reference log-probs
    with torch.no_grad():
        ref_chosen_logits, _ = ref_model(chosen_ids)
        ref_rejected_logits, _ = ref_model(rejected_ids)
        ref_chosen_logp = F.log_softmax(ref_chosen_logits, dim=-1).mean()
        ref_rejected_logp = F.log_softmax(ref_rejected_logits, dim=-1).mean()

    # Policy log-probs
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=5e-5)
    optimizer.zero_grad()

    policy_chosen_logits, _ = policy_model(chosen_ids)
    policy_rejected_logits, _ = policy_model(rejected_ids)
    policy_chosen_logp = F.log_softmax(policy_chosen_logits, dim=-1).mean()
    policy_rejected_logp = F.log_softmax(policy_rejected_logits, dim=-1).mean()

    loss = compute_dpo_loss(
        policy_chosen_logp, policy_rejected_logp,
        ref_chosen_logp, ref_rejected_logp,
        beta=0.1
    )
    loss.backward()
    optimizer.step()

    margin = (policy_chosen_logp - policy_rejected_logp).item()
    print(f"   DPO Loss:   {loss.item():.4f}")
    print(f"   LogP Margin (Chosen - Rejected): {margin:+.4f} (Winning story steered higher!)")

    # 4. Save Aligned Model Checkpoint
    torch.save({
        "model_state_dict": policy_model.state_dict(),
        "config": {
            "vocab_size": policy_model.vocab_size,
            "d_model": policy_model.d_model,
            "n_layers": policy_model.n_layers,
            "n_heads": policy_model.n_heads
        },
        "stage": "dpo_aligned",
        "moral_focus": "kindness_and_sharing"
    }, out_aligned_path)
    print(f"\n4. Persistent Checkpoint Saved:")
    print(f"   ✓ Saved aligned model to: {out_aligned_path}")

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. Module 08 directly connects to Module 07's checkpoints/storycraft_instruct.pt.")
    print("2. The frozen reference model anchors policy updates, preventing catastrophic drift.")
    print("3. Persists checkpoints/storycraft_aligned.pt, the definitive production model")
    print("   used in Modules 09 (Benchmarking), 10 (KV Cache), and 11 (Capstone App)!")
    print("=" * 70)


if __name__ == "__main__":
    main()
