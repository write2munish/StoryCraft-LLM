"""
Module 05: Generation, Sampling & Foundation Model Weight Loading
=================================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. Loading the persistent pre-trained checkpoint: checkpoints/storycraft_base.pt.
2. The Autoregressive Generation Loop using real model weights and tokenizer.pkl.
3. Temperature scaling, Top-k, and Top-p (Nucleus) sampling.
4. Repetition penalty to prevent degenerative text looping.
5. State-Dict Weight Mapping (mapping external open weights into custom architectures).

Run this script:
    python3 modules/05_generation_sampling/demo_generation_sampling.py
"""

import os
import sys
from typing import List
import torch
import torch.nn.functional as F

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
from storycraft import StoryTokenizer, StoryCraftLM


def apply_repetition_penalty(logits: torch.Tensor, generated_tokens: List[int], penalty: float = 1.2) -> torch.Tensor:
    """Penalize logits of tokens that have already been generated."""
    for token_id in set(generated_tokens):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def sample_next_token(logits: torch.Tensor, temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9) -> int:
    """Sample next token using Temperature, Top-k, and Top-p (Nucleus) filtering."""
    logits = logits / max(temperature, 1e-5)

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k)[0][-1]
        logits[logits < threshold] = float("-inf")

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def main():
    print("=" * 70)
    print("🎲 MODULE 05: GENERATION, SAMPLING & WEIGHT MAPPING")
    print("=" * 70)

    # 1. Load Persistent Checkpoint from Module 04
    chk_dir = os.path.join(repo_root, "checkpoints")
    tok_path = os.path.join(chk_dir, "tokenizer.pkl")
    base_path = os.path.join(chk_dir, "storycraft_base.pt")

    print("\n1. Loading Persistent Model & Tokenizer:")
    if os.path.exists(tok_path) and os.path.exists(base_path):
        tokenizer = StoryTokenizer.from_pretrained(tok_path)
        ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {"vocab_size": tokenizer.vocab_size, "d_model": 128, "n_layers": 4, "n_heads": 4})
        model = StoryCraftLM(**cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"   ✓ Loaded Tokenizer from: {tok_path} (Vocab: {tokenizer.vocab_size})")
        print(f"   ✓ Loaded Base Model from: {base_path} (Layers: {cfg['n_layers']}, Dim: {cfg['d_model']})")
    else:
        print("   ⚠️ Checkpoints not found. Initializing fallback demo model...")
        tokenizer = StoryTokenizer(target_vocab_size=128)
        model = StoryCraftLM(vocab_size=128, d_model=64, n_layers=2, n_heads=2)

    # 2. Autoregressive Story Generation
    print("\n2. Generating Story with Temperature & Top-p Sampling:")
    prompt = "Once upon a time Lily"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = list(prompt_ids)
    print(f"   Initial Prompt: \"{prompt}\"")

    with torch.no_grad():
        for _ in range(25):
            input_tensor = torch.tensor([generated], dtype=torch.long)
            logits, _ = model(input_tensor)
            next_logits = logits[0, -1, :].clone()
            penalized_logits = apply_repetition_penalty(next_logits, generated, penalty=1.2)
            next_id = sample_next_token(penalized_logits, temperature=0.7, top_k=20, top_p=0.9)
            generated.append(next_id)
            if next_id == tokenizer.eos_id:
                break

    story_text = tokenizer.decode(generated)
    print(f"   Generated Output: \"{story_text}\"")

    # 3. State-Dict Mapping Concept
    print("\n3. Foundation Model State-Dict Mapping Concept:")
    print("   Open weights (HuggingFace GPT-2 or LLaMA) map into StoryCraftLM as follows:")
    sample_mapping = {
        "transformer.wte.weight": "tok_embed.weight",
        "transformer.h.0.attn.c_attn.weight": "blocks.0.attn.q_proj / k_proj / v_proj",
        "transformer.h.0.ln_1.weight": "blocks.0.norm1.weight"
    }
    for external_key, scratch_key in sample_mapping.items():
        print(f"   Map: '{external_key}' ---> '{scratch_key}'")

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. Module 05 directly consumes checkpoints/storycraft_base.pt saved by Module 04.")
    print("2. Sampling parameters (Temperature, Top-p) control the creativity of real weights.")
    print("👉 Next: Module 06 loads this same base model and adapts it for classification!")
    print("=" * 70)


if __name__ == "__main__":
    main()
