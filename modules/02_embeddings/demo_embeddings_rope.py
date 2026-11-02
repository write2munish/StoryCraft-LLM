"""
Module 02: Vector Spaces & Positional Representations
======================================================
Central Anchor: StoryCraft-LLM (The Tiny Story Generator)

This script demonstrates:
1. Token Embedding lookup (mapping discrete token IDs into continuous R^{d_model}).
2. The Permutation Invariance problem: Why Transformers cannot tell "Dog bites boy"
   apart from "Boy bites dog" without positional information.
3. Rotary Position Embeddings (RoPE) implemented from scratch in pure PyTorch.
4. Numerical proof that RoPE dot products depend strictly on relative token distance (m - n).

Run this script:
    python3 modules/02_embeddings/demo_embeddings_rope.py
"""

import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)
from storycraft import set_seed

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Rotary Positional Embedding (RoPE) Implementation
# ============================================================================
class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as used in modern models (LLaMA, Mistral, Gemma).
    Instead of adding position vectors to embeddings, RoPE rotates the Query and Key
    vectors in 2D coordinate pairs based on their sequence index.
    """
    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for 2D pair rotation."
        self.dim = dim

        # Compute rotation frequencies: theta_i = 10000^(-2(i-1)/dim)
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float)
        
        # Outer product: [max_seq_len, dim // 2]
        angles = torch.outer(positions, freqs)
        
        # Precompute cos and sin caches
        self.register_buffer("cos_cached", torch.cos(angles))  # [max_seq_len, dim // 2]
        self.register_buffer("sin_cached", torch.sin(angles))  # [max_seq_len, dim // 2]

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply 2D complex plane rotation to input tensor x: [batch, seq_len, num_heads, head_dim]
        """
        # Slice cached cos and sin for current sequence length
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim//2]
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)

        # Split x into pairs of adjacent coordinates [x_1, x_2, x_3, x_4, ...]
        x1 = x[..., 0::2]  # Even indices [x_1, x_3, ...]
        x2 = x[..., 1::2]  # Odd indices  [x_2, x_4, ...]

        # Apply 2D rotation matrix:
        # [cos -sin] [x1] = [x1 * cos - x2 * sin]
        # [sin  cos] [x2]   [x1 * sin + x2 * cos]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Interleave coordinates back into original shape
        out = torch.empty_like(x)
        out[..., 0::2] = rotated_x1
        out[..., 1::2] = rotated_x2
        return out


# ============================================================================
# 2. Demonstration Runner
# ============================================================================
def main():
    set_seed(42)
    print("=" * 70)
    print("📐 MODULE 02: VECTOR SPACES & ROTARY POSITIONAL EMBEDDINGS (RoPE)")
    print("=" * 70)

    # 1. Continuous Embedding Space
    vocab_size = 100
    d_model = 16
    token_embed = nn.Embedding(vocab_size, d_model)

    dog_id = 15
    puppy_id = 16
    castle_id = 89

    # Simulate semantic similarity by pulling dog and puppy closer
    with torch.no_grad():
        token_embed.weight[puppy_id] = token_embed.weight[dog_id] + 0.05 * torch.randn(d_model)

    sim_dog_puppy = F.cosine_similarity(token_embed.weight[dog_id], token_embed.weight[puppy_id], dim=0).item()
    sim_dog_castle = F.cosine_similarity(token_embed.weight[dog_id], token_embed.weight[castle_id], dim=0).item()

    print("\n1. Continuous Semantic Geometry (Token Embeddings):")
    print(f"   Cosine Sim ('dog', 'puppy'):  {sim_dog_puppy:+.4f} (High semantic overlap)")
    print(f"   Cosine Sim ('dog', 'castle'): {sim_dog_castle:+.4f} (Distant concepts)")

    # 2. The Permutation Invariance Problem
    print("\n2. The Permutation Invariance Problem:")
    print("   Consider two sentences with inverted meaning:")
    print("   Sentence A: 'Lily hugs the dog'")
    print("   Sentence B: 'The dog hugs Lily'")
    print("   Standard dot-product self-attention treats sequences as sets.")
    print("   Without positional information, both sentences generate identical attention sums!")

    # 3. Rotary Position Embeddings (RoPE)
    print("\n3. Testing Rotary Position Embeddings (RoPE):")
    head_dim = 8
    rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=64)

    # Create dummy query and key vectors
    # Shape: [batch=1, seq_len=4, num_heads=1, head_dim=8]
    torch.manual_seed(42)
    q = torch.randn(1, 4, 1, head_dim)
    k = torch.randn(1, 4, 1, head_dim)

    # Apply RoPE
    q_rot = rope(q, seq_len=4)
    k_rot = rope(k, seq_len=4)

    print(f"   Original Q shape: {q.shape} -> Rotated Q shape: {q_rot.shape}")

    # 4. Proving Relative Invariance
    print("\n4. Mathematical Proof of Relative Distance Invariance:")
    print("   RoPE guarantees that the dot product <q_m, k_n> depends ONLY on (m - n),")
    print("   not on their absolute positions m and n in the document!")

    # Compare dot product of (pos 0, pos 1) vs (pos 2, pos 3) -> both have relative distance = 1!
    dot_pos_0_1 = torch.sum(q_rot[0, 0, 0] * k_rot[0, 1, 0]).item()
    
    # Generate identical Q and K at shifted positions (pos 2 and 3)
    q_shifted = torch.zeros(1, 4, 1, head_dim)
    k_shifted = torch.zeros(1, 4, 1, head_dim)
    q_shifted[0, 2, 0] = q[0, 0, 0]  # Place same q at index 2
    k_shifted[0, 3, 0] = k[0, 1, 0]  # Place same k at index 3

    q_rot_shifted = rope(q_shifted, seq_len=4)
    k_rot_shifted = rope(k_shifted, seq_len=4)

    dot_pos_2_3 = torch.sum(q_rot_shifted[0, 2, 0] * k_rot_shifted[0, 3, 0]).item()

    print(f"   Dot product at pos (m=0, n=1) [distance = 1]: {dot_pos_0_1:+.6f}")
    print(f"   Dot product at pos (m=2, n=3) [distance = 1]: {dot_pos_2_3:+.6f}")
    print(f"   Difference: {abs(dot_pos_0_1 - dot_pos_2_3):.2e} (Numerically identical!)")

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. Word meanings live in continuous geometric vector spaces.")
    print("2. RoPE encodes relative token offsets naturally without consuming extra weights.")
    print("3. Query and Key rotations allow models to extrapolate across varying story lengths.")
    print("👉 Next: Module 03 assembles these embeddings into the full Transformer Blocks!")
    print("=" * 70)


if __name__ == "__main__":
    main()
