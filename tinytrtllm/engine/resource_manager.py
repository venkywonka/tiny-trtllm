"""Resource managers for the inference engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from tinytrtllm.engine.block_manager import BlockManager
from tinytrtllm.engine.request import LlmRequest


class BaseResourceManager(ABC):
    """Abstract interface for resource managers.

    A resource manager is responsible for allocating, updating, and freeing
    external resources (e.g. KV cache blocks) on behalf of :class:`LlmRequest`
    objects moving through the engine pipeline.
    """

    @abstractmethod
    def get_max_resource_count(self) -> int:
        """Return the total number of resource units available."""

    @abstractmethod
    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        """Return the number of resource units needed to finish *request*."""

    @abstractmethod
    def prepare_resources(self, requests: List[LlmRequest]) -> None:
        """Allocate initial resources for a batch of new requests."""

    @abstractmethod
    def update_resources(self, requests: List[LlmRequest]) -> None:
        """Update resources after one generation step (e.g. grow block table)."""

    @abstractmethod
    def free_resources(self, request: LlmRequest) -> None:
        """Release all resources held by *request*."""

    @abstractmethod
    def can_allocate(self, request: LlmRequest) -> bool:
        """Return True if resources are available to begin *request*."""


class KVCacheResourceManager(BaseResourceManager):
    """Manages KV cache blocks for a batch of requests via :class:`BlockManager`.

    Lifecycle:
        1. ``prepare_resources`` — allocate the initial block table for each
           request based on its context length plus expected max output.
        2. ``update_resources`` — after each decode step, extend the block
           table if the total token count has grown past the current capacity.
        3. ``free_resources`` — return all blocks to the free pool.
    """

    def __init__(self, block_manager: BlockManager) -> None:
        self._bm = block_manager

    # ------------------------------------------------------------------
    # BaseResourceManager interface
    # ------------------------------------------------------------------

    def get_max_resource_count(self) -> int:
        return self._bm._num_blocks

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        total_tokens = request.context_len + request.max_tokens
        return self._bm.blocks_needed(total_tokens)

    def prepare_resources(self, requests: List[LlmRequest]) -> None:
        for req in requests:
            needed = self.get_needed_resource_to_completion(req)
            blocks = self._bm.allocate(needed)
            req.block_table = blocks

    def update_resources(self, requests: List[LlmRequest]) -> None:
        for req in requests:
            total_tokens = req.total_num_tokens
            current_capacity = len(req.block_table) * self._bm.block_size
            if total_tokens > current_capacity:
                extra_blocks_needed = self._bm.blocks_needed(
                    total_tokens - current_capacity
                )
                new_blocks = self._bm.allocate(extra_blocks_needed)
                req.block_table.extend(new_blocks)

    def free_resources(self, request: LlmRequest) -> None:
        if request.block_table:
            self._bm.free(request.block_table)
            request.block_table = []

    def can_allocate(self, request: LlmRequest) -> bool:
        needed = self.get_needed_resource_to_completion(request)
        return self._bm.can_allocate(needed)
