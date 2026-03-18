"""FlashInfer backend — optimized for decode with paged KV cache."""

from __future__ import annotations

from typing import Optional

import torch

from tinytrtllm.layers.attention import (
    AttentionBackend,
    PreparedAttention,
    register_backend,
)

try:
    import flashinfer

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False


class FlashInferBackend(AttentionBackend):
    """FlashInfer backend for decode-optimized paged attention."""

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
        if not HAS_FLASHINFER:
            raise RuntimeError("flashinfer not installed")

        raise NotImplementedError(
            "FlashInfer decode integration pending. Use 'vanilla' or 'fa' backend."
        )


if HAS_FLASHINFER:
    register_backend("fi", FlashInferBackend)
