"""Vanilla SDPA fallback backend — always available, uses PyTorch's scaled_dot_product_attention."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from tinytrtllm.layers.attention import (
    AttentionBackend,
    PreparedAttention,
    register_backend,
    store_kvcache,
)


class VanillaBackend(AttentionBackend):
    """Pure PyTorch SDPA backend — slow but works everywhere."""

    def plan(
        self,
        seq_lens: torch.Tensor,
        context_lens: torch.Tensor,
        block_offsets: torch.Tensor,
        slot_mapping: torch.Tensor,
        max_seq_len: int,
    ) -> PreparedAttention:
        is_context = context_lens > 0
        return PreparedAttention(
            seq_lens=seq_lens,
            context_lens=context_lens,
            max_context_len=context_lens.max().item() if len(context_lens) > 0 else 0,
            max_seq_len=max_seq_len,
            block_offsets=block_offsets,
            slot_mapping=slot_mapping,
            is_context=is_context,
            num_tokens=slot_mapping.shape[0],
        )

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        prepared: PreparedAttention,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_cache: Optional[torch.Tensor],
        layer_idx: int,
        scale: float,
    ) -> torch.Tensor:
        """Run attention using PyTorch SDPA.

        For simplicity, this processes each sequence independently.
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        """
        # Expand KV heads if GQA
        if num_kv_heads < num_heads:
            repeat = num_heads // num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Process per-sequence (naive but correct)
        outputs = []
        token_offset = 0
        for i, seq_len in enumerate(prepared.seq_lens):
            s = seq_len.item()
            if s == 0:
                continue

            q_seq = q[token_offset : token_offset + s]  # (s, heads, dim)
            k_seq = k[token_offset : token_offset + s]
            v_seq = v[token_offset : token_offset + s]

            # Reshape for SDPA: (1, heads, seq, dim)
            q_sdpa = q_seq.transpose(0, 1).unsqueeze(0)  # (1, heads, s, dim)
            k_sdpa = k_seq.transpose(0, 1).unsqueeze(0)
            v_sdpa = v_seq.transpose(0, 1).unsqueeze(0)

            # Use causal mask for context (prefill)
            is_causal = prepared.is_context[i].item() if prepared.is_context is not None else True

            out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                is_causal=is_causal,
                scale=scale,
            )  # (1, heads, s, dim)

            out = out.squeeze(0).transpose(0, 1)  # (s, heads, dim)
            outputs.append(out)
            token_offset += s

        if not outputs:
            return torch.empty(0, num_heads, head_dim, device=q.device, dtype=q.dtype)
        return torch.cat(outputs, dim=0)


register_backend("vanilla", VanillaBackend)
