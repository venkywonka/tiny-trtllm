"""HybridBackend dispatcher — uses different backends for prefill vs decode.

Config-driven: TinyLlmArgs.attn_backend can be "trtllm,fi" to use TRT-LLM for
prefill and FlashInfer for decode, or just "fa" for FlashAttention for both.
"""

from __future__ import annotations

from typing import Optional

import torch

from tinytrtllm.layers.attention import (
    AttentionBackend,
    PreparedAttention,
    get_attention_backend,
    register_backend,
)


class HybridBackend(AttentionBackend):
    """Dispatches to different backends for prefill vs decode."""

    def __init__(self, prefill_name: str, decode_name: str):
        self.prefill_backend = get_attention_backend(prefill_name)
        self.decode_backend = get_attention_backend(decode_name)

    def plan(
        self,
        seq_lens: torch.Tensor,
        context_lens: torch.Tensor,
        block_offsets: torch.Tensor,
        slot_mapping: torch.Tensor,
        max_seq_len: int,
    ) -> PreparedAttention:
        # Use prefill backend's plan (they should be compatible)
        return self.prefill_backend.plan(
            seq_lens, context_lens, block_offsets, slot_mapping, max_seq_len
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
        # If all sequences are decode (no context), use decode backend
        has_context = prepared.is_context.any().item() if prepared.is_context is not None else True
        backend = self.prefill_backend if has_context else self.decode_backend
        return backend.run(
            q, k, v, prepared,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            scale=scale,
        )


def create_hybrid_backend(config_str: str) -> AttentionBackend:
    """Parse 'prefill,decode' or 'single' backend config string."""
    parts = config_str.split(",")
    if len(parts) == 1:
        return get_attention_backend(parts[0])
    elif len(parts) == 2:
        return HybridBackend(parts[0].strip(), parts[1].strip())
    else:
        raise ValueError(f"Invalid backend config: {config_str}")
