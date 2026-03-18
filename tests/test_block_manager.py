"""Tests for paged KV cache block manager."""

import pytest
from tinytrtllm.engine.block_manager import BlockManager


class TestBlockManager:
    def test_initial_free_blocks(self):
        bm = BlockManager(num_blocks=100, block_size=256)
        assert bm.num_free_blocks == 100

    def test_allocate_returns_block_ids(self):
        bm = BlockManager(num_blocks=100, block_size=256)
        blocks = bm.allocate(5)
        assert len(blocks) == 5
        assert len(set(blocks)) == 5  # unique
        assert bm.num_free_blocks == 95

    def test_free_returns_blocks(self):
        bm = BlockManager(num_blocks=100, block_size=256)
        blocks = bm.allocate(5)
        bm.free(blocks)
        assert bm.num_free_blocks == 100

    def test_can_allocate_true(self):
        bm = BlockManager(num_blocks=100, block_size=256)
        assert bm.can_allocate(50) is True

    def test_can_allocate_false(self):
        bm = BlockManager(num_blocks=10, block_size=256)
        assert bm.can_allocate(11) is False

    def test_allocate_exhaustion_raises(self):
        bm = BlockManager(num_blocks=5, block_size=256)
        bm.allocate(5)
        with pytest.raises(RuntimeError):
            bm.allocate(1)

    def test_double_free_raises(self):
        bm = BlockManager(num_blocks=10, block_size=256)
        blocks = bm.allocate(3)
        bm.free(blocks)
        with pytest.raises(ValueError):
            bm.free(blocks)

    def test_blocks_needed_for_tokens(self):
        bm = BlockManager(num_blocks=100, block_size=256)
        assert bm.blocks_needed(256) == 1
        assert bm.blocks_needed(257) == 2
        assert bm.blocks_needed(0) == 0
        assert bm.blocks_needed(512) == 2

    def test_block_size(self):
        bm = BlockManager(num_blocks=100, block_size=128)
        assert bm.block_size == 128


class TestPrefixCache:
    def test_hash_deterministic(self):
        bm = BlockManager(num_blocks=100, block_size=4, enable_prefix_cache=True)
        h1 = bm.compute_block_hash(tuple([1, 2, 3, 4]))
        h2 = bm.compute_block_hash(tuple([1, 2, 3, 4]))
        assert h1 == h2

    def test_hash_chaining(self):
        bm = BlockManager(num_blocks=100, block_size=4, enable_prefix_cache=True)
        h1 = bm.compute_block_hash(tuple([1, 2, 3, 4]))
        h2a = bm.compute_block_hash(tuple([5, 6, 7, 8]), parent_hash=h1)
        h2b = bm.compute_block_hash(tuple([5, 6, 7, 8]), parent_hash=0)
        assert h2a != h2b  # parent hash affects child

    def test_shared_prefix_reuses_blocks(self):
        bm = BlockManager(num_blocks=100, block_size=4, enable_prefix_cache=True)
        tokens_a = list(range(8)) + [100, 101, 102, 103]  # shared prefix + unique suffix
        tokens_b = list(range(8)) + [200, 201, 202, 203]

        blocks_a = bm.allocate_with_prefix_cache(tokens_a)
        blocks_b = bm.allocate_with_prefix_cache(tokens_b)

        # First 2 blocks should be shared (same prefix)
        assert blocks_a[:2] == blocks_b[:2]
        # Last block should differ (different suffix)
        assert blocks_a[2] != blocks_b[2]

    def test_ref_count_on_hit(self):
        bm = BlockManager(num_blocks=100, block_size=4, enable_prefix_cache=True)
        tokens = list(range(4))
        bm.allocate_with_prefix_cache(tokens)
        bm.allocate_with_prefix_cache(tokens)  # cache hit
        # The block for [0,1,2,3] should have ref_count=2
        h = bm.compute_block_hash(tuple(tokens))
        block_id = bm._hash_to_block[h]
        assert bm._ref_counts[block_id] == 2

    def test_free_cached_blocks_reusable(self):
        bm = BlockManager(num_blocks=10, block_size=4, enable_prefix_cache=True)
        tokens = list(range(4))
        blocks = bm.allocate_with_prefix_cache(tokens)
        bm.free_with_prefix_cache(blocks, tokens)
        # Block still in cache, can be reused
        blocks2 = bm.allocate_with_prefix_cache(tokens)
        assert blocks2 == blocks  # reused

    def test_partial_prefix_match(self):
        bm = BlockManager(num_blocks=100, block_size=4, enable_prefix_cache=True)
        tokens_a = list(range(12))  # 3 blocks
        tokens_b = list(range(8)) + [100, 101, 102, 103]  # 2 shared + 1 different

        blocks_a = bm.allocate_with_prefix_cache(tokens_a)
        blocks_b = bm.allocate_with_prefix_cache(tokens_b)

        assert blocks_a[:2] == blocks_b[:2]  # shared prefix
        assert blocks_a[2] != blocks_b[2]  # different suffix

    def test_prefix_cache_hit_rate(self):
        """Shared-prefix workload should achieve >= 90% hit rate."""
        bm = BlockManager(num_blocks=1000, block_size=4, enable_prefix_cache=True)
        system_prefix = list(range(20))  # 5 blocks of shared prefix

        hits = 0
        total = 0
        for i in range(50):
            unique_suffix = list(range(1000 + i * 4, 1000 + i * 4 + 4))
            tokens = system_prefix + unique_suffix
            result = bm.allocate_with_prefix_cache(tokens)
            if i > 0:  # First request can't have hits
                # 5 prefix blocks should hit, 1 suffix block should miss
                hits += 5
                total += 6
            else:
                total += 6

        hit_rate = hits / total if total > 0 else 0
        assert hit_rate >= 0.8  # Conservative threshold


class TestEviction:
    """Tests for lazy eviction of zero-ref-count cached blocks."""

    def test_evict_cached_blocks_reclaims_zero_ref_blocks(self):
        """_evict_cached_blocks should return zero-ref blocks to the free pool."""
        bm = BlockManager(num_blocks=4, block_size=4, enable_prefix_cache=True)
        tokens = list(range(4))
        blocks = bm.allocate_with_prefix_cache(tokens)
        bm.free_with_prefix_cache(blocks, tokens)

        h = bm.compute_block_hash(tuple(tokens))
        block_id = bm._hash_to_block[h]
        assert bm._ref_counts[block_id] == 0

        free_before = bm.num_free_blocks
        evicted = bm._evict_cached_blocks(1)

        assert evicted == 1
        assert bm.num_free_blocks == free_before + 1
        assert h not in bm._hash_to_block
        assert block_id not in bm._ref_counts

    def test_evict_does_not_touch_active_refs(self):
        """Blocks with ref_count > 0 must never be evicted."""
        bm = BlockManager(num_blocks=8, block_size=4, enable_prefix_cache=True)
        tokens = list(range(4))
        bm.allocate_with_prefix_cache(tokens)
        bm.allocate_with_prefix_cache(tokens)

        h = bm.compute_block_hash(tuple(tokens))
        block_id = bm._hash_to_block[h]
        assert bm._ref_counts[block_id] == 2

        evicted = bm._evict_cached_blocks(4)
        assert evicted == 0
        assert bm._hash_to_block[h] == block_id

    def test_allocate_triggers_eviction_when_pool_exhausted(self):
        """allocate() should succeed after evicting zero-ref cached blocks."""
        bm = BlockManager(num_blocks=4, block_size=4, enable_prefix_cache=True)
        tokens = list(range(4))
        cached_blocks = bm.allocate_with_prefix_cache(tokens)
        bm.free_with_prefix_cache(cached_blocks, tokens)
        bm.allocate(bm.num_free_blocks)  # exhaust the free pool

        assert bm.num_free_blocks == 0
        new_blocks = bm.allocate(1)  # should trigger eviction
        assert len(new_blocks) == 1

    def test_double_free_does_not_make_ref_count_negative(self):
        """Calling free_with_prefix_cache twice must not drive ref_count below 0."""
        bm = BlockManager(num_blocks=10, block_size=4, enable_prefix_cache=True)
        tokens = list(range(4))
        blocks = bm.allocate_with_prefix_cache(tokens)
        bm.free_with_prefix_cache(blocks, tokens)
        bm.free_with_prefix_cache(blocks, tokens)  # double-free

        h = bm.compute_block_hash(tuple(tokens))
        block_id = bm._hash_to_block[h]
        assert bm._ref_counts[block_id] >= 0

    def test_allocate_raises_when_eviction_insufficient(self):
        """allocate() must still raise if eviction can't satisfy the request."""
        bm = BlockManager(num_blocks=2, block_size=4, enable_prefix_cache=True)
        # Both blocks have ref_count > 0 (not evictable)
        bm.allocate_with_prefix_cache(list(range(4)))
        bm.allocate_with_prefix_cache([10, 11, 12, 13])

        with pytest.raises(RuntimeError):
            bm.allocate(1)
