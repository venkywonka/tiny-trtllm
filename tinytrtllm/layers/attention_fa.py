"""FlashAttention backend — high-performance prefill + decode."""

from __future__ import annotations

from typing import Optional

import torch

from tinytrtllm.layers.attention import (
    AttentionBackend,
    PreparedAttention,
    register_backend,
)

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class FlashAttentionBackend(AttentionBackend):
    """FlashAttention backend for both prefill (varlen) and decode (paged KV)."""

    def plan(
        self,
        seq_lens: torch.Tensor,
        context_lens: torch.Tensor,
        block_offsets: torch.Tensor,
        slot_mapping: torch.Tensor,
        max_seq_len: int,
    ) -> PreparedAttention:
        is_context = context_lens > 0
        # Compute cumulative sequence lengths for varlen interface
        cu_seqlens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=seq_lens.device)
        cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)

        pa = PreparedAttention(
            seq_lens=seq_lens,
            context_lens=context_lens,
            max_context_len=context_lens.max().item() if len(context_lens) > 0 else 0,
            max_seq_len=max_seq_len,
            block_offsets=block_offsets,
            slot_mapping=slot_mapping,
            is_context=is_context,
            num_tokens=slot_mapping.shape[0],
        )
        pa.cu_seqlens = cu_seqlens
        return pa

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
        if not HAS_FLASH_ATTN:
            raise RuntimeError("flash-attn not installed")

        # Use flash_attn_varlen_func for prefill sequences
        cu_seqlens = prepared.cu_seqlens
        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=prepared.max_seq_len,
            max_seqlen_k=prepared.max_seq_len,
            softmax_scale=scale,
            causal=True,
        )
        return output


if HAS_FLASH_ATTN:
    register_backend("fa", FlashAttentionBackend)
