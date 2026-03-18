"""Plan/execute attention interface — TRT-LLM's crown jewel pattern.

The separation of plan() and run() enables CUDA graph capture:
- plan(): CPU work computing seq_lens, block offsets, slot mappings
- run(): GPU kernel dispatch using the pre-computed plan

Backend registry supports: trtllm, fa, fi, vanilla, with auto-detection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class PreparedAttention:
    """Output of plan() — everything the kernel needs to dispatch attention."""

    seq_lens: torch.Tensor  # (batch,) total sequence lengths
    context_lens: torch.Tensor  # (batch,) context lengths (0 for decode)
    max_context_len: int
    max_seq_len: int
    block_offsets: torch.Tensor  # (batch, max_blocks_per_seq)
    slot_mapping: torch.Tensor  # (total_tokens,) flat mapping to KV cache slots
    is_context: torch.Tensor  # (batch,) bool: True if prefill
    num_tokens: int


class AttentionBackend(ABC):
    """Base class for attention backends with plan/execute interface."""

    @abstractmethod
    def plan(
        self,
        seq_lens: torch.Tensor,
        context_lens: torch.Tensor,
        block_offsets: torch.Tensor,
        slot_mapping: torch.Tensor,
        max_seq_len: int,
    ) -> PreparedAttention:
        ...

    @abstractmethod
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
        ...


def store_kvcache(
    kv_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor,
    layer_idx: int,
    block_size: int,
) -> None:
    """Store K/V tensors into paged KV cache at specified slots.

    kv_cache shape: (num_layers, 2, num_blocks, block_size, num_heads, head_dim)
    k, v shape: (num_tokens, num_heads, head_dim)
    slot_mapping: (num_tokens,) mapping token idx → flat slot idx. -1 means skip.
    """
    for i, slot in enumerate(slot_mapping):
        slot_val = slot.item()
        if slot_val < 0:
            continue
        block_idx = slot_val // block_size
        offset = slot_val % block_size
        kv_cache[layer_idx, 0, block_idx, offset] = k[i]
        kv_cache[layer_idx, 1, block_idx, offset] = v[i]


# ---------------------------------------------------------------------------
# Backend Registry
# ---------------------------------------------------------------------------

BACKEND_REGISTRY: dict[str, type[AttentionBackend]] = {}


def register_backend(name: str, cls: type[AttentionBackend]) -> None:
    BACKEND_REGISTRY[name] = cls


def get_attention_backend(name: str) -> AttentionBackend:
    """Get backend by name, with auto-detection for 'auto'."""
    if name == "auto":
        return _auto_detect_backend()
    if name not in BACKEND_REGISTRY:
        raise KeyError(f"Unknown attention backend: {name}. Available: {list(BACKEND_REGISTRY.keys())}")
    return BACKEND_REGISTRY[name]()


def _auto_detect_backend() -> AttentionBackend:
    """Auto-detect best available backend: trtllm → fa → vanilla."""
    for name in ["trtllm", "fa", "vanilla"]:
        if name in BACKEND_REGISTRY:
            return BACKEND_REGISTRY[name]()
    raise RuntimeError("No attention backend available")


# ---------------------------------------------------------------------------
# Attention nn.Module wrapper
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Attention module wrapping a plan/execute backend."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        backend_name: str = "auto",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.backend = get_attention_backend(backend_name)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        prepared: PreparedAttention,
        kv_cache: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        return self.backend.run(
            q, k, v, prepared,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            scale=self.scale,
        )
