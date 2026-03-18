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
