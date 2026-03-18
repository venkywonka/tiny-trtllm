"""Rotary Position Embedding (RoPE)."""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        # positions: (batch, seq_len)
        cos = self.cos_cached[positions]  # (batch, seq, dim//2)
        sin = self.sin_cached[positions]  # (batch, seq, dim//2)
        # Expand for heads: (batch, seq, 1, dim//2)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        q_rot = _apply_rotary(q, cos, sin)
        k_rot = _apply_rotary(k, cos, sin)
        return q_rot, k_rot


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to x. x shape: (batch, seq, heads, dim)."""
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
