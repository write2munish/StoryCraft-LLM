"""
Appendix B: Frontier Attention Architectures (GQA & MoE)
=========================================================
Demonstrates:
1. Grouped-Query Attention (GQA) with shared Key/Value head groups (LLaMA 3 style).
2. Sparse Mixture-of-Experts (MoE) block with Top-k router gating (Mixtral style).

Run this script:
    python3 appendices/appendix_b_gqa_moe/demo_gqa_moe.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Grouped-Query Attention (GQA)
# ============================================================================
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    Multiple query heads share single Key and Value heads.
    Example: 8 Query heads, 2 KV heads -> 4 query heads per KV head group.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 8, n_kv_heads: int = 2):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.num_queries_per_kv = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads to match query heads via repeat_interleave
        k_expanded = torch.repeat_interleave(k, self.num_queries_per_kv, dim=1)
        v_expanded = torch.repeat_interleave(v, self.num_queries_per_kv, dim=1)

        scores = (q @ k_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)

        out = (attn @ v_expanded).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


# ============================================================================
# 2. Sparse Mixture-of-Experts (MoE) Layer
# ============================================================================
class SparseMoELayer(nn.Module):
    """
    Mixture-of-Experts Feed-Forward Layer with Top-k Gating.
    """
    def __init__(self, d_model: int = 64, hidden_dim: int = 128, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, d_model)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq_len, d_model]
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # [B*T, C]

        # 1. Router logits & Top-k expert selection
        router_logits = self.router(x_flat)  # [B*T, num_experts]
        weights, indices = torch.topk(F.softmax(router_logits, dim=-1), self.top_k, dim=-1)
        # Normalize weights so they sum to 1
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # 2. Accumulate outputs from selected experts
        out_flat = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = indices[:, i]
            expert_weight = weights[:, i].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_out = self.experts[e](x_flat[mask])
                    out_flat[mask] += expert_weight[mask] * expert_out

        return out_flat.view(B, T, C)


# ============================================================================
# 3. Demonstration Runner
# ============================================================================
def main():
    print("=" * 70)
    print("🔬 APPENDIX B: FRONTIER ATTENTION & ROUTING (GQA & MoE)")
    print("=" * 70)

    # 1. Grouped-Query Attention Test
    print("\n1. Grouped-Query Attention (GQA):")
    gqa = GroupedQueryAttention(d_model=128, n_heads=8, n_kv_heads=2)
    dummy_story = torch.randn(2, 16, 128)  # Batch=2, Seq=16

    out_gqa = gqa(dummy_story)
    print(f"   Input shape:  {dummy_story.shape}")
    print(f"   Output shape: {out_gqa.shape}")
    print("   Query heads: 8 | Key/Value heads: 2 (Sharing ratio 4:1)")
    print("   KV Cache Memory Reduction: 75% less VRAM during generation!")

    # 2. Sparse Mixture-of-Experts Test
    print("\n2. Sparse Mixture-of-Experts (MoE):")
    moe = SparseMoELayer(d_model=64, hidden_dim=128, num_experts=4, top_k=2)
    dummy_tokens = torch.randn(2, 6, 64)

    out_moe = moe(dummy_tokens)
    print(f"   Total Experts: 4 | Active Experts Per Token: 2")
    print(f"   Input shape:  {dummy_tokens.shape}")
    print(f"   Output shape: {out_moe.shape}")
    print("   Sparse Compute: 50% of expert parameters are bypassed per token!")

    print("\n" + "=" * 70)
    print("💡 KEY PEDAGOGICAL TAKEAWAYS:")
    print("=" * 70)
    print("1. GQA delivers the speed and memory efficiency of MQA with the quality of MHA.")
    print("2. MoE decouples total parameter count from inference compute cost.")
    print("=" * 70)


if __name__ == "__main__":
    main()
