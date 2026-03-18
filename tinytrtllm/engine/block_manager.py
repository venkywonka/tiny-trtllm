"""Paged KV cache block manager."""

from __future__ import annotations

import math
from collections import deque
from typing import List


class BlockManager:
    """Manages allocation and freeing of fixed-size KV cache blocks.

    Each block is identified by an integer ID.  Free blocks are tracked in a
    FIFO deque; allocated blocks live in a set for O(1) membership checks.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self._block_size = block_size
        self._num_blocks = num_blocks
        self._free_block_ids: deque[int] = deque(range(num_blocks))
        self._allocated: set[int] = set()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_block_ids)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def can_allocate(self, n: int) -> bool:
        """Return True if *n* blocks can be allocated."""
        return len(self._free_block_ids) >= n

    def allocate(self, n: int) -> List[int]:
        """Pop *n* block IDs from the free pool and mark them allocated.

        Raises:
            RuntimeError: If fewer than *n* blocks are available.
        """
        if not self.can_allocate(n):
            raise RuntimeError(
                f"Cannot allocate {n} blocks — only {self.num_free_blocks} free"
            )
        blocks: List[int] = []
        for _ in range(n):
            block_id = self._free_block_ids.popleft()
            self._allocated.add(block_id)
            blocks.append(block_id)
        return blocks

    def free(self, block_ids: List[int]) -> None:
        """Return *block_ids* to the free pool.

        Raises:
            ValueError: If any block is not currently allocated (double-free).
        """
        for bid in block_ids:
            if bid not in self._allocated:
                raise ValueError(
                    f"Block {bid} is not allocated — possible double-free"
                )
        for bid in block_ids:
            self._allocated.remove(bid)
            self._free_block_ids.append(bid)

    def blocks_needed(self, num_tokens: int) -> int:
        """Return the number of blocks required to store *num_tokens* tokens."""
        if num_tokens == 0:
            return 0
        return math.ceil(num_tokens / self._block_size)
