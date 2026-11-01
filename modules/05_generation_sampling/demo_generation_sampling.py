"""
Module 05: Generation, Sampling & Weight Loading
=================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. The Autoregressive Generation Loop.
2. Greedy Decoding vs Stochastic Sampling.
3. Temperature scaling (low entropy vs high entropy).
4. Top-k and Top-p (Nucleus) filtering.
5. Repetition penalty to break degenerative repetition loops.
6. State-Dict Weight Mapping (mapping open pretrained weights into custom architectures).

Run this script:
    python3 modules/05_generation_sampling/demo_generation_sampling.py
"""

from typing import List, Dict
import torch
import torch.nn.functional as F


# ============================================================================
# 1. Pedagogical Sampling Functions
# ============================================================================
def apply_repetition_penalty(logits: torch.Tensor, generated_tokens: List[int], penalty: float = 1.2) -> torch.Tensor:
    """
    Penalize logits of tokens that have already been generated to avoid loops.
    """
    for token_id in set(generated_tokens):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def top_k_top_p_sampling(logits: torch.Tensor, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> int:
    """
    Filter logits using Temperature, Top-k, and Top-p (Nucleus) sampling.
    """
    # 1. Temperature scaling
    logits = logits / max(temperature, 1e-5)

    # 2. Top-k filtering: Keep only top k tokens
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k)[0][-1]
        logits[logits < threshold] = float("-inf")

    # 3. Top-p (Nucleus) filtering: Keep cumulative probability mass <= p
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    # 4. Softmax and sample from multinomial distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token


# ============================================================================
# 2. Demonstration Runner
# ============================================================================
def main():
    print("=" * 70)
    print("🎲 MODULE 05: GENERATION, SAMPLING & WEIGHT MAPPING")
    print("=" * 70)

    # Toy Vocabulary of Story Words
    vocab = {
        0: "<|pad|>", 1: "once", 2: "upon", 3: "a", 4: "time",
        5: "there", 6: "was", 7: "little", 8: "girl", 9: "named",
        10: "lily", 11: "and", 12: "she", 13: "had", 14: "red",
        15: "ball", 16: "dog", 17: "bark", 18: "happy", 19: "."
    }
    inv_vocab = {v: k for k, v in vocab.items()}

    # Simulate model logits for the next word following: "once upon a time there was a little"
    print("\n1. Simulating Unnormalized Model Output (Logits):")
    torch.manual_seed(101)
    base_logits = torch.randn(len(vocab))
    # Give higher plausible logits to "girl", "dog", "ball"
    base_logits[inv_vocab["girl"]] += 5.0
    base_logits[inv_vocab["dog"]] += 4.5
    base_logits[inv_vocab["ball"]] += 2.0

    print("   Top 3 Model Predictions (Unscaled):")
    top3_indices = torch.topk(base_logits, 3)[1].tolist()
    for rank, idx in enumerate(top3_indices, 1):
        print(f"   {rank}. '{vocab[idx]}' (logit={base_logits[idx].item():.2f})")

    # 2. Temperature Comparison
    print("\n2. The Effect of Temperature Scaling:")
    for temp in [0.2, 0.8, 1.8]:
        scaled_probs = F.softmax(base_logits / temp, dim=-1)
        top_prob = scaled_probs[inv_vocab["girl"]].item()
        runner_up = scaled_probs[inv_vocab["dog"]].item()
        print(f"   Temp {temp:3.1f} -> P('girl')={top_prob:5.1%}, P('dog')={runner_up:5.1%} "
              f"({'Greedy-like deterministic' if temp < 0.5 else 'Creative diverse' if temp < 1.0 else 'High entropy / chaotic'})")

    # 3. Autoregressive Story Generation Simulation
    print("\n3. Simulating Autoregressive Decoding Loop:")
    prompt_ids = [inv_vocab["once"], inv_vocab["upon"], inv_vocab["a"], inv_vocab["time"]]
    generated = list(prompt_ids)

    print(f"   Initial Prompt: {' '.join([vocab[i] for i in prompt_ids])}")

    # Generate 6 subsequent tokens using Top-p sampling and repetition penalty
    for step in range(6):
        # In a real model, this is model(generated)[:, -1, :]
        step_logits = torch.randn(len(vocab))
        # Bias plausible story words
        step_logits[inv_vocab["there"]] += 3.0
        step_logits[inv_vocab["was"]] += 2.5
        step_logits[inv_vocab["lily"]] += 3.5

        # Apply repetition penalty
        penalized_logits = apply_repetition_penalty(step_logits.clone(), generated, penalty=1.3)
        next_id = top_k_top_p_sampling(penalized_logits, temperature=0.7, top_k=5, top_p=0.9)
        generated.append(next_id)

    full_story = " ".join([vocab[i] for i in generated])
    print(f"   Generated Story: \"{full_story}\"")

    # 4. State-Dict Weight Mapping Concept (Raschka Ch 05 Pattern)
    print("\n4. Foundation Model State-Dict Mapping Concept:")
    print("   Open weights (e.g. HuggingFace GPT-2 or LLaMA) use different layer names.")
    print("   We map them directly into our StoryCraftLM scratch architecture:")
    sample_mapping = {
        "transformer.wte.weight": "tok_embed.weight",
        "transformer.h.0.attn.c_attn.weight": "blocks.0.attn.qkv_proj.weight",
        "transformer.h.0.ln_1.weight": "blocks.0.norm1.weight"
    }
    for external_key, scratch_key in sample_mapping.items():
        print(f"   Map: '{external_key}' ---> '{scratch_key}'")

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. Temperature controls confidence; Top-p dynamically trims the improbable tail.")
    print("2. Repetition penalties prevent the model from getting stuck in loops.")
    print("3. State-dict mapping allows our scratch model to run real foundation weights!")
    print("👉 Next: Module 06 turns this generative model into a Story Classifier!")
    print("=" * 70)


if __name__ == "__main__":
    main()
