"""Tests for ResourceManager lifecycle: prepare -> update -> free."""

import pytest
from tinytrtllm.engine.block_manager import BlockManager
from tinytrtllm.engine.resource_manager import KVCacheResourceManager
from tinytrtllm.engine.request import LlmRequest, RequestState


class TestKVCacheResourceManager:
    def _make_request(self, token_ids, max_tokens=10):
        return LlmRequest(request_id=0, token_ids=token_ids, max_tokens=max_tokens)

    def test_get_max_resource_count(self):
        bm = BlockManager(num_blocks=100, block_size=256)
        rm = KVCacheResourceManager(bm)
        assert rm.get_max_resource_count() == 100

    def test_get_needed_resource_to_completion(self):
        bm = BlockManager(num_blocks=100, block_size=256)
        rm = KVCacheResourceManager(bm)
        req = self._make_request([1] * 300, max_tokens=200)
        needed = rm.get_needed_resource_to_completion(req)
        # 300 context + 200 output = 500 tokens, 256 block_size -> 2 blocks
        assert needed == 2

    def test_prepare_resources(self):
        bm = BlockManager(num_blocks=100, block_size=256)
        rm = KVCacheResourceManager(bm)
        req = self._make_request([1] * 300)
        rm.prepare_resources([req])
        assert len(req.block_table) > 0
        assert bm.num_free_blocks < 100

    def test_update_resources_appends_slot(self):
        bm = BlockManager(num_blocks=100, block_size=4)
        rm = KVCacheResourceManager(bm)
        req = self._make_request([1, 2, 3, 4])  # exactly 1 block
        rm.prepare_resources([req])
        initial_blocks = len(req.block_table)
        # Simulate generation adding tokens beyond block boundary
        req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
        req.transition_to(RequestState.GENERATION_IN_PROGRESS)
        for i in range(4):
            req.append_output_token(100 + i)
        rm.update_resources([req])
        assert len(req.block_table) >= initial_blocks

    def test_free_resources(self):
        bm = BlockManager(num_blocks=100, block_size=256)
        rm = KVCacheResourceManager(bm)
        req = self._make_request([1] * 300)
        rm.prepare_resources([req])
        free_before = bm.num_free_blocks
        rm.free_resources(req)
        assert bm.num_free_blocks > free_before

    def test_full_lifecycle(self):
        bm = BlockManager(num_blocks=100, block_size=256)
        rm = KVCacheResourceManager(bm)
        req = self._make_request([1, 2, 3], max_tokens=5)

        # prepare
        rm.prepare_resources([req])
        assert len(req.block_table) > 0

        # update
        req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
        req.transition_to(RequestState.GENERATION_IN_PROGRESS)
        req.append_output_token(42)
        rm.update_resources([req])

        # free
        rm.free_resources(req)
        assert bm.num_free_blocks == 100

    def test_can_allocate_for_request(self):
        bm = BlockManager(num_blocks=100, block_size=256)
        rm = KVCacheResourceManager(bm)
        req = self._make_request([1] * 300, max_tokens=200)
        assert rm.can_allocate(req) is True
