"""Tests for PyExecutor iteration loop and OverlapExecutor."""

import threading
import time

import pytest
import torch
import torch.nn as nn

from tinytrtllm.config import SamplingParams, SchedulingPolicy, ChunkingPolicy
from tinytrtllm.engine.executor import (
    OverlapExecutor,
    PyExecutor,
    RequestOutput,
    ResponseManager,
)
from tinytrtllm.engine.model_engine import ModelEngine
from tinytrtllm.engine.request import LlmRequest, RequestState
from tinytrtllm.engine.sampler import Sampler
from tinytrtllm.engine.scheduler import TwoTierScheduler


class DummyModel(nn.Module):
    """Minimal model that returns random logits."""

    def __init__(self, vocab_size=100):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(1, vocab_size, bias=False)

    def forward(self, input_ids, positions=None):
        batch = input_ids.shape[0]
        # Return deterministic logits: token 42 is always max
        logits = torch.zeros(batch, self.vocab_size)
        logits[:, 42] = 100.0
        return logits


def _make_executor(max_batch_size=256, max_tokens=16384, vocab_size=100):
    model = DummyModel(vocab_size)
    engine = ModelEngine(model, vocab_size, torch.device("cpu"), enable_cuda_graph=False)
    scheduler = TwoTierScheduler(
        scheduling_policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
        chunking_policy=ChunkingPolicy.EQUAL_PROGRESS,
        max_batch_size=max_batch_size,
        max_num_tokens=max_tokens,
        block_size=256,
        max_blocks=1000,
    )
    sampler = Sampler()
    return PyExecutor(engine, scheduler, sampler)


def _make_request(request_id=0, context_len=5, max_tokens=3):
    return LlmRequest(
        request_id=request_id,
        token_ids=list(range(context_len)),
        max_tokens=max_tokens,
    )


class TestPyExecutor:
    def test_empty_iteration(self):
        executor = _make_executor()
        outputs = executor.iteration()
        assert outputs == []

    def test_single_request_lifecycle(self):
        executor = _make_executor()
        req = _make_request(request_id=1, context_len=5, max_tokens=2)
        sp = SamplingParams(temperature=0.0)  # greedy → always token 42
        executor.enqueue_request(req, sp)

        # First iteration: context phase → transition to generation
        outputs = executor.iteration()
        # May not complete in one iteration

        # Run enough iterations to complete
        all_outputs = list(outputs)
        for _ in range(10):
            if not executor.has_pending:
                break
            all_outputs.extend(executor.iteration())

        # Should have completed
        completed = [o for o in all_outputs if o.finished]
        assert len(completed) >= 1
        assert completed[0].request_id == 1
        assert len(completed[0].output_token_ids) == 2

    def test_multiple_requests_batched(self):
        executor = _make_executor()
        sp = SamplingParams(temperature=0.0)

        for i in range(3):
            req = _make_request(request_id=i, context_len=3, max_tokens=1)
            executor.enqueue_request(req, sp)

        # Run iterations
        all_outputs = []
        for _ in range(10):
            all_outputs.extend(executor.iteration())
            if not executor.has_pending:
                break

        completed = [o for o in all_outputs if o.finished]
        assert len(completed) == 3

    def test_callback_fires(self):
        executor = _make_executor()
        sp = SamplingParams(temperature=0.0)
        results = []

        def on_complete(output):
            results.append(output)

        req = _make_request(request_id=0, context_len=3, max_tokens=1)
        executor.enqueue_request(req, sp, callback=on_complete)

        for _ in range(10):
            executor.iteration()
            if not executor.has_pending:
                break

        assert len(results) == 1
        assert results[0].finished

    def test_cancellation(self):
        executor = _make_executor()
        sp = SamplingParams(temperature=0.0)

        req = _make_request(request_id=0, context_len=3, max_tokens=100)
        executor.enqueue_request(req, sp)
        executor.cancel_request(0)

        outputs = executor.iteration()
        assert len(outputs) == 0
        assert not executor.has_pending


class TestResponseManager:
    def test_register_and_notify(self):
        rm = ResponseManager()
        rm.register(0)
        output = RequestOutput(request_id=0, output_token_ids=[42], finished=True)
        rm.notify(0, output)
        result = rm.await_response(0, timeout=1.0)
        assert result is not None
        assert result.finished

    def test_timeout_returns_none(self):
        rm = ResponseManager()
        rm.register(0)
        result = rm.await_response(0, timeout=0.1)
        assert result is None

    def test_multiple_waiters(self):
        rm = ResponseManager()
        rm.register(0)
        rm.register(1)

        out0 = RequestOutput(request_id=0, output_token_ids=[1], finished=True)
        out1 = RequestOutput(request_id=1, output_token_ids=[2], finished=True)

        rm.notify(0, out0)
        rm.notify(1, out1)

        r0 = rm.await_response(0, timeout=1.0)
        r1 = rm.await_response(1, timeout=1.0)
        assert r0.request_id == 0
        assert r1.request_id == 1

    def test_streaming_successive_awaits(self):
        rm = ResponseManager()
        rm.register(0)

        # First token
        rm.notify(0, RequestOutput(request_id=0, output_token_ids=[1], finished=False))
        r1 = rm.await_response(0, timeout=1.0)
        assert r1 is not None
        assert not r1.finished

        # Second token
        rm.notify(0, RequestOutput(request_id=0, output_token_ids=[1, 2], finished=False))
        r2 = rm.await_response(0, timeout=1.0)
        assert r2 is not None

        # Completion
        rm.notify(0, RequestOutput(request_id=0, output_token_ids=[1, 2, 3], finished=True))
        r3 = rm.await_response(0, timeout=1.0)
        assert r3 is not None
        assert r3.finished

    def test_cleanup(self):
        rm = ResponseManager()
        rm.register(0)
        rm.cleanup(0)
        result = rm.await_response(0, timeout=0.1)
        assert result is None


class TestOverlapExecutor:
    def test_results_correct(self):
        model = DummyModel(100)
        engine = ModelEngine(model, 100, torch.device("cpu"), enable_cuda_graph=False)
        scheduler = TwoTierScheduler(
            scheduling_policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
            chunking_policy=ChunkingPolicy.EQUAL_PROGRESS,
            max_batch_size=256,
            max_num_tokens=16384,
            block_size=256,
            max_blocks=1000,
        )
        sampler = Sampler()
        executor = OverlapExecutor(engine, scheduler, sampler)

        sp = SamplingParams(temperature=0.0)
        req = _make_request(request_id=0, context_len=3, max_tokens=2)
        executor.enqueue_request(req, sp)

        all_outputs = []
        for _ in range(20):
            all_outputs.extend(executor.overlap_iteration())
            if not executor.has_pending:
                break

        completed = [o for o in all_outputs if o.finished]
        assert len(completed) >= 1

    def test_fallback_when_empty(self):
        model = DummyModel(100)
        engine = ModelEngine(model, 100, torch.device("cpu"), enable_cuda_graph=False)
        scheduler = TwoTierScheduler(
            scheduling_policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
            chunking_policy=ChunkingPolicy.EQUAL_PROGRESS,
            max_batch_size=256,
            max_num_tokens=16384,
            block_size=256,
            max_blocks=1000,
        )
        sampler = Sampler()
        executor = OverlapExecutor(engine, scheduler, sampler)
        outputs = executor.overlap_iteration()
        assert outputs == []


class TestBugFixes:
    """Regression tests for executor bug fixes."""

    def test_active_blocks_tracked_from_requests(self):
        """E2: active_blocks should reflect actual block_table sizes."""
        model = DummyModel(vocab_size=100)
        me = ModelEngine(model=model, vocab_size=100, device=torch.device("cpu"))
        sched = TwoTierScheduler(
            scheduling_policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
            chunking_policy=ChunkingPolicy.EQUAL_PROGRESS,
            max_batch_size=8, max_num_tokens=1024,
            block_size=16, max_blocks=10,
        )
        sampler = Sampler()
        executor = PyExecutor(me, sched, sampler)

        # Enqueue and run first iteration to move request to active
        req = LlmRequest(request_id=1, token_ids=[1, 2, 3], max_tokens=5)
        req.block_table = [0, 1, 2]  # Simulate 3 allocated blocks
        sp = SamplingParams(temperature=0.0)
        executor.enqueue_request(req, sp)
        executor.iteration()

        # The request should now be in _active with its block_table
        active_blocks = sum(len(r.block_table) for r in executor._active)
        assert active_blocks == 3, f"Expected 3 active blocks, got {active_blocks}"

    def test_chunked_prefill_completes(self):
        """E4: Multi-chunk context requests should not stall."""
        model = DummyModel(vocab_size=100)
        me = ModelEngine(model=model, vocab_size=100, device=torch.device("cpu"))
        # Small max_num_tokens forces chunking
        sched = TwoTierScheduler(
            scheduling_policy=SchedulingPolicy.MAX_UTILIZATION,
            chunking_policy=ChunkingPolicy.FCFS,
            max_batch_size=8, max_num_tokens=5,
            block_size=16, max_blocks=100000,
        )
        sampler = Sampler()
        executor = PyExecutor(me, sched, sampler)

        # 10-token context with max_num_tokens=5 requires 2 chunks
        req = LlmRequest(request_id=1, token_ids=list(range(10)), max_tokens=1)
        sp = SamplingParams(temperature=0.0)
        executor.enqueue_request(req, sp)

        # Run enough iterations to complete chunked prefill + generation
        outputs = []
        for _ in range(20):
            result = executor.iteration()
            outputs.extend(result)
            if not executor.has_pending:
                break

        assert len(outputs) >= 1, "Chunked prefill request should complete"
        assert outputs[0].finished

    def test_prepare_resources_not_called_on_in_progress(self):
        """E3: Only CONTEXT_INIT requests should get prepare_resources."""
        from unittest.mock import MagicMock
        from tinytrtllm.engine.resource_manager import KVCacheResourceManager
        from tinytrtllm.engine.block_manager import BlockManager

        bm = BlockManager(num_blocks=100, block_size=16)
        rm = KVCacheResourceManager(bm)
        rm.prepare_resources = MagicMock(wraps=rm.prepare_resources)

        model = DummyModel(vocab_size=100)
        me = ModelEngine(model=model, vocab_size=100, device=torch.device("cpu"))
        sched = TwoTierScheduler(
            scheduling_policy=SchedulingPolicy.MAX_UTILIZATION,
            chunking_policy=ChunkingPolicy.FCFS,
            max_batch_size=8, max_num_tokens=5,
            block_size=16, max_blocks=100,
        )
        sampler = Sampler()
        executor = PyExecutor(me, sched, sampler, resource_manager=rm)

        req = LlmRequest(request_id=1, token_ids=list(range(10)), max_tokens=1)
        sp = SamplingParams(temperature=0.0)
        executor.enqueue_request(req, sp)

        # First iteration: should prepare (CONTEXT_INIT)
        executor.iteration()
        first_call_args = rm.prepare_resources.call_args_list[0][0][0]
        assert len(first_call_args) == 1  # One request prepared

        # Reset mock
        rm.prepare_resources.reset_mock()

        # Second iteration: request is CONTEXT_IN_PROGRESS, should NOT prepare again
        executor.iteration()
        if rm.prepare_resources.called:
            second_call_args = rm.prepare_resources.call_args_list[0][0][0]
            # Should be empty list (no CONTEXT_INIT requests)
            assert len(second_call_args) == 0, "Should not prepare CONTEXT_IN_PROGRESS requests"

    def test_response_manager_no_missed_wakeup(self):
        """E6: notify before await should not cause timeout."""
        import time
        rm = ResponseManager()
        rm.register(1)

        output = RequestOutput(request_id=1, output_token_ids=[42], finished=True)
        rm.notify(1, output)

        start = time.monotonic()
        result = rm.await_response(1, timeout=5.0)
        elapsed = time.monotonic() - start

        assert result is not None, "Should get response without timeout"
        assert elapsed < 1.0, f"Should not wait long, waited {elapsed:.1f}s"
        assert result.output_token_ids == [42]
