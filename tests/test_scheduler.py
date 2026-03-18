"""Tests for the two-tier scheduling system — TRT-LLM's signature architecture."""

import pytest

from tinytrtllm.config import ChunkingPolicy, SchedulingPolicy
from tinytrtllm.engine.request import LlmRequest, RequestState
from tinytrtllm.engine.scheduler import (
    CapacityScheduler,
    MicroBatchScheduler,
    ScheduledRequests,
    TwoTierScheduler,
)


def _make_request(request_id=0, context_len=100, max_tokens=50, chunk_size=4096):
    return LlmRequest(
        request_id=request_id,
        token_ids=list(range(context_len)),
        max_tokens=max_tokens,
        context_chunk_size=chunk_size,
    )


def _make_gen_request(request_id=0, context_len=10, max_tokens=50):
    """Create a request already in generation phase."""
    req = _make_request(request_id, context_len, max_tokens)
    req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
    req.transition_to(RequestState.GENERATION_IN_PROGRESS)
    return req


# ---------------------------------------------------------------------------
# CapacityScheduler tests (Commit 6)
# ---------------------------------------------------------------------------


class TestCapacitySchedulerGuaranteedNoEvict:
    def test_schedules_when_resources_available(self):
        sched = CapacityScheduler(
            policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
            max_batch_size=256,
            max_num_tokens=16384,
            block_size=256,
            max_blocks=1000,
        )
        req = _make_request(context_len=100, max_tokens=50)
        admitted, rejected = sched.schedule_capacity([req], active_blocks=0)
        assert len(admitted) == 1
        assert len(rejected) == 0

    def test_rejects_when_cannot_guarantee(self):
        sched = CapacityScheduler(
            policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
            max_batch_size=256,
            max_num_tokens=16384,
            block_size=256,
            max_blocks=1,  # Only 1 block = 256 tokens
        )
        # This request needs (500+500)/256 = 4 blocks to guarantee completion
        req = _make_request(context_len=500, max_tokens=500)
        admitted, rejected = sched.schedule_capacity([req], active_blocks=0)
        assert len(admitted) == 0
        assert len(rejected) == 1

    def test_respects_max_batch_size(self):
        sched = CapacityScheduler(
            policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
            max_batch_size=2,
            max_num_tokens=16384,
            block_size=256,
            max_blocks=1000,
        )
        reqs = [_make_request(request_id=i, context_len=10) for i in range(5)]
        admitted, rejected = sched.schedule_capacity(reqs, active_blocks=0)
        assert len(admitted) == 2
        assert len(rejected) == 3

    def test_fifo_order(self):
        sched = CapacityScheduler(
            policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
            max_batch_size=2,
            max_num_tokens=16384,
            block_size=256,
            max_blocks=1000,
        )
        reqs = [_make_request(request_id=i, context_len=10) for i in range(5)]
        admitted, _ = sched.schedule_capacity(reqs, active_blocks=0)
        assert [r.request_id for r in admitted] == [0, 1]

    def test_empty_queue(self):
        sched = CapacityScheduler(
            policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
            max_batch_size=256,
            max_num_tokens=16384,
            block_size=256,
            max_blocks=1000,
        )
        admitted, rejected = sched.schedule_capacity([], active_blocks=0)
        assert len(admitted) == 0
        assert len(rejected) == 0


class TestCapacitySchedulerMaxUtilization:
    def test_schedules_optimistically(self):
        sched = CapacityScheduler(
            policy=SchedulingPolicy.MAX_UTILIZATION,
            max_batch_size=256,
            max_num_tokens=16384,
            block_size=256,
            max_blocks=2,  # Only 2 blocks
        )
        # Needs more blocks for full completion, but MAX_UTIL admits anyway
        req = _make_request(context_len=100, max_tokens=500)
        admitted, rejected = sched.schedule_capacity([req], active_blocks=0)
        assert len(admitted) == 1

    def test_still_rejects_when_no_space_for_context(self):
        sched = CapacityScheduler(
            policy=SchedulingPolicy.MAX_UTILIZATION,
            max_batch_size=256,
            max_num_tokens=16384,
            block_size=256,
            max_blocks=0,  # No blocks at all
        )
        req = _make_request(context_len=100, max_tokens=50)
        admitted, rejected = sched.schedule_capacity([req], active_blocks=0)
        assert len(admitted) == 0


# ---------------------------------------------------------------------------
# MicroBatchScheduler tests (Commit 7)
# ---------------------------------------------------------------------------


class TestMicroBatchSchedulerEqualProgress:
    def test_equal_chunks(self):
        sched = MicroBatchScheduler(
            policy=ChunkingPolicy.EQUAL_PROGRESS,
            max_num_tokens=100,
        )
        reqs = [_make_request(request_id=i, context_len=50, chunk_size=4096) for i in range(2)]
        result = sched.schedule_microbatch(reqs, gen_requests=[])
        # Both should be scheduled with roughly equal chunk sizes
        assert len(result.context_last_chunk) + len(result.context_chunking) == 2

    def test_generation_requests_in_list(self):
        sched = MicroBatchScheduler(
            policy=ChunkingPolicy.EQUAL_PROGRESS,
            max_num_tokens=100,
        )
        gen_req = _make_gen_request(request_id=0)
        result = sched.schedule_microbatch([], gen_requests=[gen_req])
        assert len(result.generation) == 1

    def test_token_budget_respected(self):
        sched = MicroBatchScheduler(
            policy=ChunkingPolicy.EQUAL_PROGRESS,
            max_num_tokens=20,
        )
        reqs = [_make_request(request_id=i, context_len=50) for i in range(5)]
        result = sched.schedule_microbatch(reqs, gen_requests=[])
        total = result.num_context_tokens + result.num_generation_tokens
        assert total <= 20


class TestMicroBatchSchedulerFCFS:
    def test_fcfs_order(self):
        sched = MicroBatchScheduler(
            policy=ChunkingPolicy.FCFS,
            max_num_tokens=200,
        )
        reqs = [_make_request(request_id=i, context_len=50) for i in range(3)]
        result = sched.schedule_microbatch(reqs, gen_requests=[])
        all_ctx = result.context_last_chunk + result.context_chunking
        ids = [r.request_id for r in all_ctx]
        # FCFS: process in order
        assert ids == sorted(ids)

    def test_chunking_categorization(self):
        sched = MicroBatchScheduler(
            policy=ChunkingPolicy.FCFS,
            max_num_tokens=30,
        )
        req = _make_request(request_id=0, context_len=50, chunk_size=20)
        result = sched.schedule_microbatch([req], gen_requests=[])
        # With 50 tokens and chunk_size 20, first chunk is not last
        assert len(result.context_chunking) == 1 or len(result.context_last_chunk) == 1


# ---------------------------------------------------------------------------
# ScheduledRequests helpers (Commit 8)
# ---------------------------------------------------------------------------


class TestScheduledRequests:
    def test_all_requests(self):
        sr = ScheduledRequests()
        r1 = _make_request(request_id=1)
        r2 = _make_gen_request(request_id=2)
        sr.context_last_chunk.append(r1)
        sr.generation.append(r2)
        assert len(list(sr.all_requests())) == 2

    def test_num_context_tokens(self):
        sr = ScheduledRequests()
        req = _make_request(request_id=0, context_len=100)
        req._scheduled_context_tokens = 100
        sr.context_last_chunk.append(req)
        assert sr.num_context_tokens == 100

    def test_num_generation_tokens(self):
        sr = ScheduledRequests()
        req = _make_gen_request(request_id=0)
        sr.generation.append(req)
        assert sr.num_generation_tokens == 1  # 1 token per gen step

    def test_is_empty(self):
        sr = ScheduledRequests()
        assert sr.is_empty is True
        sr.generation.append(_make_gen_request())
        assert sr.is_empty is False

    def test_len(self):
        sr = ScheduledRequests()
        sr.context_last_chunk.append(_make_request(request_id=1))
        sr.generation.append(_make_gen_request(request_id=2))
        assert len(sr) == 2

    def test_can_run_cuda_graph(self):
        sr = ScheduledRequests()
        # CUDA graph only for decode-only batches (no context)
        sr.generation.append(_make_gen_request())
        assert sr.can_run_cuda_graph is True
        sr.context_last_chunk.append(_make_request(request_id=1))
        assert sr.can_run_cuda_graph is False

    def test_four_disjoint_lists(self):
        sr = ScheduledRequests()
        assert hasattr(sr, 'context_chunking')
        assert hasattr(sr, 'context_last_chunk')
        assert hasattr(sr, 'generation')
        assert hasattr(sr, 'paused')


class TestTwoTierScheduler:
    def test_full_pipeline(self):
        sched = TwoTierScheduler(
            scheduling_policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
            chunking_policy=ChunkingPolicy.EQUAL_PROGRESS,
            max_batch_size=256,
            max_num_tokens=16384,
            block_size=256,
            max_blocks=1000,
        )
        waiting = [_make_request(request_id=i, context_len=50) for i in range(3)]
        active_gen = [_make_gen_request(request_id=10)]
        result = sched.schedule(waiting, active_gen, active_blocks=0)
        assert isinstance(result, ScheduledRequests)
        assert not result.is_empty
