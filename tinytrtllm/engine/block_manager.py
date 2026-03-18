"""Paged KV cache block manager with optional prefix caching."""

from __future__ import annotations

import math
import struct
from collections import deque
from typing import Dict, List, Optional, Tuple

import xxhash


class BlockManager:
    """Manages allocation and freeing of fixed-size KV cache blocks.

    Each block is identified by an integer ID.  Free blocks are tracked in a
    FIFO deque; allocated blocks live in a set for O(1) membership checks.

    When *enable_prefix_cache* is True, blocks are content-addressed via
    xxhash so that identical token prefixes share physical blocks.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_cache: bool = False,
    ) -> None:
        self._block_size = block_size
        self._num_blocks = num_blocks
        self._free_block_ids: deque[int] = deque(range(num_blocks))
        self._allocated: set[int] = set()

        # Prefix-cache bookkeeping
        self._prefix_cache_enabled = enable_prefix_cache
        self._hash_to_block: Dict[int, int] = {}
        self._ref_counts: Dict[int, int] = {}

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
                f"Cannot allocate {n} blocks \u2014 only {self.num_free_blocks} free"
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
                    f"Block {bid} is not allocated \u2014 possible double-free"
                )
        for bid in block_ids:
            self._allocated.remove(bid)
            self._free_block_ids.append(bid)

    def blocks_needed(self, num_tokens: int) -> int:
        """Return the number of blocks required to store *num_tokens* tokens."""
        if num_tokens == 0:
            return 0
        return math.ceil(num_tokens / self._block_size)

    # ------------------------------------------------------------------
    # Prefix-cache API
    # ------------------------------------------------------------------

    def compute_block_hash(
        self,
        token_tuple: Tuple[int, ...],
        parent_hash: int = 0,
    ) -> int:
        """Content-address a block by hashing its tokens chained with a parent.

        The parent_hash encodes all preceding blocks so that the same token
        chunk at different positions produces a different hash.
        """
        h = xxhash.xxh64()
        # Feed parent hash as 8-byte little-endian
        h.update(struct.pack("<Q", parent_hash))
        # Feed token data
        for tok in token_tuple:
            h.update(struct.pack("<i", tok))
        return h.intdigest()

    def allocate_with_prefix_cache(self, token_ids: List[int]) -> List[int]:
        """Allocate blocks for *token_ids*, reusing cached prefix blocks.

        Tokens are split into block_size chunks.  For each full chunk, we
        compute a chained hash.  If the hash already maps to a cached block
        we bump its ref-count and reuse it; otherwise we allocate a new
        block from the free pool.

        Returns:
            List of block IDs in order.
        """
        blocks: List[int] = []
        parent_hash = 0
        num_chunks = math.ceil(len(token_ids) / self._block_size) if token_ids else 0

        for i in range(num_chunks):
            start = i * self._block_size
            end = start + self._block_size
            chunk = tuple(token_ids[start:end])

            # Only cache full blocks; partial trailing blocks get fresh alloc
            if len(chunk) == self._block_size:
                block_hash = self.compute_block_hash(chunk, parent_hash=parent_hash)
                if block_hash in self._hash_to_block:
                    # Cache hit — reuse existing block
                    block_id = self._hash_to_block[block_hash]
                    self._ref_counts[block_id] += 1
                    blocks.append(block_id)
                    parent_hash = block_hash
                    continue
                # Cache miss — allocate and register
                new_blocks = self.allocate(1)
                block_id = new_blocks[0]
                self._hash_to_block[block_hash] = block_id
                self._ref_counts[block_id] = 1
                blocks.append(block_id)
                parent_hash = block_hash
            else:
                # Partial block — no caching
                new_blocks = self.allocate(1)
                blocks.append(new_blocks[0])

        return blocks

    def free_with_prefix_cache(
        self,
        block_ids: List[int],
        token_ids: List[int],
    ) -> None:
        """Decrement ref-counts for cached blocks; lazy-evict when count hits 0.

        Blocks whose ref-count drops to zero are *not* returned to the free
        pool immediately — they remain in the hash map for potential reuse
        (lazy eviction).  They will only be reclaimed when the free pool is
        exhausted and the manager needs to evict.
        """
        parent_hash = 0
        num_chunks = math.ceil(len(token_ids) / self._block_size) if token_ids else 0

        for i, bid in enumerate(block_ids):
            if i < num_chunks:
                start = i * self._block_size
                end = start + self._block_size
                chunk = tuple(token_ids[start:end])

                if len(chunk) == self._block_size:
                    block_hash = self.compute_block_hash(chunk, parent_hash=parent_hash)
                    if bid in self._ref_counts:
                        self._ref_counts[bid] -= 1
                        # Lazy eviction: keep in cache even at ref_count == 0
                    parent_hash = block_hash
                else:
                    # Partial block — return directly to free pool
                    if bid in self._allocated:
                        self._allocated.remove(bid)
                        self._free_block_ids.append(bid)
            else:
                # Extra block beyond token range — return to free pool
                if bid in self._allocated:
                    self._allocated.remove(bid)
                    self._free_block_ids.append(bid)
