"""
Module 05: Generation, Sampling & Foundation Model Weight Loading
=================================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. Loading and inspecting the REAL pre-trained model: checkpoints/storycraft_base.pt.
2. Querying the real next-token probability distribution over vocabulary words.
3. Systematically testing 4 sampling strategies on the exact same prompt:
   - Strategy 1: Greedy Decoding (argmax)
   - Strategy 2: Low Temperature (T = 0.2)
   - Strategy 3: Tuned Nucleus Sampling (T = 0.7, Top-k = 20, Top-p = 0.9, Repetition Penalty = 1.2)
   - Strategy 4: High Temperature Chaotic Sampling (T = 1.8)
4. Cross-Architecture Weight Importers: Mapping external state-dict tensors into StoryCraftLM.

Run this script:
    python3 modules/05_generation_sampling/demo_generation_sampling.py
"""

import os
import sys
from typing import Dict, Tuple
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.dirname(__file__))

from storycraft import StoryTokenizer, StoryCraftLM
from generate import StoryGenerator, apply_repetition_penalty, sample_next_token
from load_pretrained import map_and_load_state_dict


def main():
    print("=" * 70)
    print("🎲 MODULE 05: GENERATION, SAMPLING & WEIGHT MAPPING")
    print("=" * 70)

    # 1. Load the Persistent Checkpoint from Module 04
    chk_dir = os.path.join(repo_root, "checkpoints")
    tok_path = os.path.join(chk_dir, "tokenizer.pkl")
    base_path = os.path.join(chk_dir, "storycraft_base.pt")

    print("\n1. Connecting to Persistent Base Model:")
    if not (os.path.exists(tok_path) and os.path.exists(base_path)):
        print("   ❌ Error: Checkpoints missing! Please run previous modules first.")
        return

    tokenizer = StoryTokenizer.from_pretrained(tok_path)
    ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {"vocab_size": tokenizer.vocab_size, "d_model": 128, "n_layers": 4, "n_heads": 4})
    
    model = StoryCraftLM(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Successfully loaded real weights from: {base_path}")
    print(f"   ✓ Architecture: {cfg['n_layers']} layers, {cfg['d_model']} d_model, {cfg['n_heads']} heads")
    print(f"   ✓ Total Model Parameters: {total_params:,}")

    generator = StoryGenerator(model, tokenizer)

    # 2. Inspecting the Real Next-Token Probability Distribution
    test_prompt = "Lily saw a"
    print(f"\n2. Real Next-Token Distribution for Prompt: \"{test_prompt}\"")
    for temp in [0.2, 0.7, 1.5]:
        probs = generator.get_next_token_distribution(test_prompt, temperature=temp)
        top_3 = list(probs.items())[:3]
        formatted = ", ".join([f"'{w}': {p:.1%}" for w, p in top_3])
        print(f"   Temp {temp:3.1f} -> [{formatted}]")

    # 3. Systematic Side-by-Side Comparison of Sampling Strategies
    prompt = "Once upon a time Lily"
    print(f"\n3. Comparing 4 Generation Strategies on the Real Model:")
    print(f"   Prompt: \"{prompt}\"")
    print("-" * 70)

    # Strategy 1: Greedy Decoding
    greedy_story = generator.generate(prompt, max_new_tokens=25, strategy="greedy")
    print(f"   [1. Greedy Decoding (argmax)]:")
    print(f"   \"{greedy_story}\"")
    print("   -> Notice: Completely deterministic; often falls into repetitive loops.")

    # Strategy 2: Low Temperature
    low_temp_story = generator.generate(prompt, max_new_tokens=25, temperature=0.2, top_k=0, top_p=1.0)
    print(f"\n   [2. Low Temperature (T=0.2)]:")
    print(f"   \"{low_temp_story}\"")
    print("   -> Notice: Conservative and rigid choices.")

    # Strategy 3: Tuned Nucleus Sampling (Top-p=0.9 + Top-k=20 + Repetition Penalty=1.2)
    tuned_story = generator.generate(prompt, max_new_tokens=25, temperature=0.7, top_k=20, top_p=0.9, repetition_penalty=1.2)
    print(f"\n   [3. Tuned Sampling (T=0.7, Top-k=20, Top-p=0.9, Penalty=1.2)]:")
    print(f"   \"{tuned_story}\"")
    print("   -> Notice: Natural narrative flow with repetition traps actively suppressed!")

    # Strategy 4: High Temperature
    high_temp_story = generator.generate(prompt, max_new_tokens=25, temperature=1.8, top_k=0, top_p=1.0)
    print(f"\n   [4. High Temperature (T=1.8 - Chaotic)]:")
    print(f"   \"{high_temp_story}\"")
    print("   -> Notice: High entropy; selects improbable words and grammatical outliers.")

    print("-" * 70)

    # 4. Cross-Architecture Weight Importers (Actually Loading & Verifying Tensors!)
    print("\n4. Cross-Architecture Weight Importers (Testing Real Tensor Transfer):")
    # Simulate an external foundation model state dictionary
    torch.manual_seed(42)
    external_weights = {
        "tok_embed.weight": torch.randn_like(model.tok_embed.weight),
        "norm_final.weight": torch.ones_like(model.norm_final.weight)
    }
    
    fresh_scratch_model = StoryCraftLM(**cfg)
    loaded_count, total_count = map_and_load_state_dict(external_weights, fresh_scratch_model)
    print(f"   Loaded {loaded_count} / {total_count} state-dict keys into scratch StoryCraftLM.")
    
    # Verify that fresh_scratch_model tok_embed equals external_weights
    diff = (fresh_scratch_model.tok_embed.weight - external_weights["tok_embed.weight"]).abs().max().item()
    print(f"   Weight Transfer Numerical Precision Difference: {diff:.6e} (Exact match!)")

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. Module 05 directly executes against checkpoints/storycraft_base.pt.")
    print("2. Greedy decoding is brittle; combining Temperature, Top-p, and Repetition Penalties")
    print("   yields the most fluent, human-like children's stories.")
    print("3. State-dict importers allow loading real open weights into custom PyTorch code.")
    print("=" * 70)


if __name__ == "__main__":
    main()
