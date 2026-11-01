"""
Module 07: Post-Training Phase 1 - Instruction Tuning (SFT) & LoRA from Scratch
================================================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. Base Model vs Instruct Model behavior.
2. ChatML formatting and Target-Only Loss Masking (ignoring user prompts with label=-100).
3. Pure PyTorch LoRA (Low-Rank Adaptation) layer implemented from scratch.
4. Parameter efficiency: comparing full fine-tuning vs LoRA (<1% trainable parameters).
5. Weight merging for zero-latency deployment.

Run this script:
    python3 modules/07_instruction_tuning_sft/demo_sft_lora.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Pure PyTorch LoRA Layer from Scratch
# ============================================================================
class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) linear layer (Hu et al., 2021).
    W = W_0 + (alpha / r) * (B @ A)
    where:
      W_0: Frozen pre-trained weight [out_features, in_features]
      A:   Trainable low-rank matrix [rank, in_features] ~ N(0, 1/r)
      B:   Trainable low-rank matrix [out_features, rank] = 0 (so Delta W starts at 0)
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # 1. Pretrained base weight (Frozen)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02, requires_grad=False)

        # 2. Trainable Low-Rank Adapters
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))  # Initialized to zero!

        # Initialize A with standard normal
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, in_features]
        base_out = F.linear(x, self.base_weight)  # Frozen forward pass
        # Low-rank forward: (x @ A.T) @ B.T * scaling
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out

    def merge_weights(self) -> torch.Tensor:
        """
        Merge Delta W into base_weight for zero-latency inference.
        """
        merged = self.base_weight.data + (self.lora_B @ self.lora_A) * self.scaling
        return merged


# ============================================================================
# 2. ChatML & Target-Only Loss Masking Demonstration
# ============================================================================
def demonstrate_prompt_masking():
    """
    Demonstrates how target-only masking ensures the model is ONLY trained on the
    story text, not on predicting the user's prompt!
    """
    print("\n2. ChatML Format & Target-Only Loss Masking:")
    print("   In Supervised Fine-Tuning, we format conversations using role tags:")

    chatml_lines = [
        "<|im_start|>system",
        "You are a kind storyteller.<|im_end|>",
        "<|im_start|>user",
        "Write a story about a dragon who loves cookies.<|im_end|>",
        "<|im_start|>assistant",
        "Once upon a time, Barnaby the dragon baked sweet cookies...<|im_end|>"
    ]
    print("   --- Formatted ChatML Dialogue ---")
    for line in chatml_lines:
        print("   " + line)

    # Simulated token sequences
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # Say tokens 0..7 are system/user prompt, tokens 8..11 are assistant story
    prompt_len = 8

    # Create target labels with -100 for user prompt
    labels = list(tokens)
    for i in range(prompt_len):
        labels[i] = -100  # PyTorch ignore_index

    print("\n   --- Masked Training Target Vector ---")
    print(f"   Input Token IDs: {tokens}")
    print(f"   Target Labels:   {labels}")
    print("   Notice: -100 indicates PyTorch CrossEntropyLoss will ignore these tokens!")
    print("   The model receives zero gradient penalty for how the prompt was worded.")


# ============================================================================
# 3. Demonstration Runner
# ============================================================================
def main():
    print("=" * 70)
    print("💬 MODULE 07: INSTRUCTION TUNING (SFT) & LoRA FROM SCRATCH")
    print("=" * 70)

    # 1. Base Model vs Instruct Model
    print("\n1. Base Model vs. Instruct Model:")
    print("   Prompt: 'Write a story about a frog who wants to fly.'")
    print("   - Untuned Base Model: 'Write a story about a duck. 3. Write a poem.'")
    print("   - Instruct Model:     'Once upon a lily pad, Freddie the frog looked at the birds...'")

    # 2. Prompt Masking
    demonstrate_prompt_masking()

    # 3. LoRA from Scratch
    print("\n3. Testing LoRA (Low-Rank Adaptation) Layer:")
    d_model = 256
    rank = 4
    alpha = 16.0

    lora_layer = LoRALinear(in_features=d_model, out_features=d_model, rank=rank, alpha=alpha)

    # Parameter counts
    base_params = lora_layer.base_weight.numel()
    lora_params = lora_layer.lora_A.numel() + lora_layer.lora_B.numel()

    print(f"   Base Linear Layer parameters (Frozen): {base_params:,}")
    print(f"   LoRA Adapter parameters (Trainable):   {lora_params:,} (Rank r={rank})")
    print(f"   Parameter Reduction: Trainable count is only {lora_params / base_params:.1%} of full layer!")

    # 4. Verify Forward Pass & Zero-Initialization
    dummy_x = torch.randn(2, 8, d_model)
    with torch.no_grad():
        out_init = lora_layer(dummy_x)
        out_base_only = F.linear(dummy_x, lora_layer.base_weight)

    diff = (out_init - out_base_only).abs().max().item()
    print("\n4. Verifying Zero-Initialization:")
    print(f"   Max difference between LoRA output and Base output at start: {diff:.6f}")
    print("   (At initialization, LoRA matrix B is zeros, ensuring model behavior starts unchanged!)")

    # 5. Weight Merging
    print("\n5. Weight Merging for Deployment:")
    merged_weight = lora_layer.merge_weights()
    print(f"   Merged Weight Shape: {merged_weight.shape} (Standard linear layer, zero extra latency at inference!)")

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. Target-only loss masking is essential for conversational instruction following.")
    print("2. LoRA enables modular adapters: you can keep one 8.5M base model and swap")
    print("   tiny ~45k parameter LoRA weights for Bedtime Stories, Adventure Stories, etc.")
    print("3. Merged LoRA weights run at 100% full inference speed without extra layers.")
    print("👉 Next: Module 08 aligns model morals and endings using Direct Preference Optimization (DPO)!")
    print("=" * 70)


if __name__ == "__main__":
    main()
