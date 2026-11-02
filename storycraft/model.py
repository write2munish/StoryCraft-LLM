"""
StoryCraftLM: Canonical Modern Decoder-Only Transformer Architecture.
Contains: RMSNorm, RoPE, SwiGLU, Causal Self-Attention with KV-Cache support.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float)
        angles = torch.outer(positions, freqs)
        self.register_buffer("cos_cached", torch.cos(angles))
        self.register_buffer("sin_cached", torch.sin(angles))

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        # cos/sin shape: [1, 1, seq_len, dim // 2] to broadcast over [B, n_heads, T, head_dim]
        cos = self.cos_cached[offset : offset + seq_len].view(1, 1, seq_len, -1)
        sin = self.sin_cached[offset : offset + seq_len].view(1, 1, seq_len, -1)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(dim=self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T_new, C = x.shape
        offset = past_kv[0].shape[2] if past_kv is not None else 0

        q = self.q_proj(x).view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, seq_len=T_new, offset=offset)
        k = self.rope(k, seq_len=T_new, offset=offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        current_kv = (k, v)

        T_total = k.shape[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if T_new > 1:
            mask = torch.triu(torch.full((T_new, T_total), float("-inf"), device=x.device), diagonal=1 + offset)
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T_new, C)
        return self.out_proj(out), current_kv

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int = 128, hidden_dim: int = 344):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, hidden_dim: int = 344):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, hidden_dim)

    def forward(self, x: torch.Tensor, past_kv=None):
        attn_out, new_kv = self.attn(self.norm1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv

class StoryCraftLM(nn.Module):
    def __init__(self, vocab_size: int = 512, d_model: int = 128, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, hidden_dim=int(d_model * 8 / 3))
            for _ in range(n_layers)
        ])
        self.norm_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight  # Weight tying

    def forward(self, token_ids: torch.Tensor, past_kvs=None):
        # token_ids: [B, T]
        x = self.tok_embed(token_ids)
        new_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = block(x, past_kv=past_kv)
            new_kvs.append(new_kv)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits, new_kvs
