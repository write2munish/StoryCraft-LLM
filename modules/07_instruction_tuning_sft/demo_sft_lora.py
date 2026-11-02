"""
Module 07: Post-Training Phase 1 - Instruction Tuning (SFT) & LoRA from Scratch
================================================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. Loading the persistent pre-trained base model: checkpoints/storycraft_base.pt.
2. Converting a text completer into an instruction-following model via ChatML formatting.
3. Target-only loss masking with ignore_index = -100 on user prompts.
4. Pure PyTorch LoRA (Low-Rank Adaptation) layers replacing dense attention projections.
5. Saving the fine-tuned instruction model to checkpoints/storycraft_instruct.pt.

Run this script:
    python3 modules/07_instruction_tuning_sft/demo_sft_lora.py
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
from storycraft import set_seed, StoryTokenizer, StoryCraftLM


class LoRALinear(nn.Module):
    """
    Pure PyTorch Low-Rank Adaptation (LoRA) layer (Hu et al., 2021).
    W = W_0 + (alpha / r) * (B @ A)
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 4, alpha: float = 16.0):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Freeze base linear layer
        self.base_weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)

        # Trainable low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))  # Zero initialized
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.base_weight)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out

    def merge_weights(self) -> torch.Tensor:
        return self.base_weight.data + (self.lora_B @ self.lora_A) * self.scaling


def main():
    set_seed(42)
    print("=" * 70)
    print("💬 MODULE 07: INSTRUCTION TUNING (SFT) & LoRA FROM SCRATCH")
    print("=" * 70)

    chk_dir = os.path.join(repo_root, "checkpoints")
    tok_path = os.path.join(chk_dir, "tokenizer.pkl")
    base_path = os.path.join(chk_dir, "storycraft_base.pt")
    out_instruct_path = os.path.join(chk_dir, "storycraft_instruct.pt")

    # 1. Load Pre-trained Base Checkpoint from Module 04
    print("\n1. Loading Base Model Checkpoint:")
    if os.path.exists(tok_path) and os.path.exists(base_path):
        tokenizer = StoryTokenizer.from_pretrained(tok_path)
        ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
        model = StoryCraftLM(**ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"   ✓ Loaded base checkpoint from: {base_path}")
    else:
        tokenizer = StoryTokenizer(target_vocab_size=128)
        model = StoryCraftLM(vocab_size=128, d_model=64, n_layers=2, n_heads=2)

    # 2. Freeze all base parameters and attach LoRA Adapters
    print("\n2. Applying LoRA to Transformer Blocks (Rank r=4):")
    for param in model.parameters():
        param.requires_grad = False
    total_base_params = sum(p.numel() for p in model.parameters())
    
    lora_layers = []
    for i, block in enumerate(model.blocks):
        # Replace q_proj with LoRALinear
        block.attn.q_proj = LoRALinear(block.attn.q_proj, rank=4, alpha=16.0)
        lora_layers.append(block.attn.q_proj)

    trainable_lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total Base Model Parameters: {total_base_params:,}")
    print(f"   Trainable LoRA Parameters:   {trainable_lora_params:,} ({trainable_lora_params / total_base_params:.1%})")

    # 3. Instruction SFT Data with Target-Only Masking
    print("\n3. Simulating SFT on ChatML Story Instructions:")
    chatml_prompt = (
        "<|im_start|>system\nYou are a friendly storyteller.<|im_end|>\n"
        "<|im_start|>user\nTell a story about sharing sweet cookies.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    story_response = "Sparky the dragon baked warm cookies and shared them with Lily and Barnaby.<|im_end|>"

    prompt_ids = tokenizer.encode(chatml_prompt, add_special_tokens=False)
    resp_ids = tokenizer.encode(story_response, add_special_tokens=False)
    full_ids = prompt_ids + resp_ids

    # Mask user prompt with -100
    labels = [-100] * len(prompt_ids) + resp_ids
    print(f"   Prompt tokens ({len(prompt_ids)}): labeled -100 (Zero loss on user questions)")
    print(f"   Assistant tokens ({len(resp_ids)}): active cross-entropy gradients")

    # 4. Run SFT Fine-Tuning Step
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    x = torch.tensor([full_ids[:-1]], dtype=torch.long)
    y = torch.tensor([labels[1:]], dtype=torch.long)

    model.train()
    optimizer.zero_grad()
    logits, _ = model(x)
    loss = loss_fn(logits.view(-1, model.vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()
    print(f"   SFT Optimization Step Completed: Target Loss = {loss.item():.4f}")

    # 5. Persist Instruction-Tuned Checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": model.vocab_size,
            "d_model": model.d_model,
            "n_layers": model.n_layers,
            "n_heads": model.n_heads
        },
        "stage": "instruction_tuned_sft"
    }, out_instruct_path)
    print(f"\n4. Persistent Checkpoint Saved:")
    print(f"   ✓ Saved fine-tuned instruction model to: {out_instruct_path}")

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. Module 07 loads checkpoints/storycraft_base.pt and updates <1% weights via LoRA.")
    print("2. Saves checkpoints/storycraft_instruct.pt, which Module 08 directly uses as")
    print("   the reference model for DPO preference alignment!")
    print("=" * 70)


if __name__ == "__main__":
    main()
