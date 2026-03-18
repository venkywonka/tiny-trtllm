"""PyExecutor — the iteration-based executor loop, TRT-LLM's core runtime.

Each iteration: fetch → schedule → prepare → forward → sample_async → update → respond

OverlapExecutor extends this with dual-microbatch ping-pong:
CPU processes batch N-1 while GPU runs batch N.

ResponseManager uses threading.Condition for await-responses pattern.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from tinytrtllm.config import SamplingParams, TinyLlmArgs
from tinytrtllm.engine.model_engine import ModelEngine
from tinytrtllm.engine.request import LlmRequest, RequestState
from tinytrtllm.engine.sampler import Sampler
from tinytrtllm.engine.scheduler import ScheduledRequests, TwoTierScheduler


@dataclass
class RequestOutput:
    """Output for a completed or streaming request."""

    request_id: int
    output_token_ids: list[int]
    finished: bool
    text: str = ""


class ResponseManager:
    """Manages response notifications using threading.Condition.

    This is TRT-LLM's await-responses pattern — callers block on a
    Condition variable until their request has new tokens or completes.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._conditions: dict[int, threading.Condition] = {}
        self._responses: dict[int, list[RequestOutput]] = {}

    def register(self, request_id: int) -> None:
        with self._lock:
            self._conditions[request_id] = threading.Condition(self._lock)
            self._responses[request_id] = []

    def notify(self, request_id: int, output: RequestOutput) -> None:
        with self._lock:
            if request_id not in self._responses:
                return
            self._responses[request_id].append(output)
            cond = self._conditions.get(request_id)
            if cond:
                cond.notify_all()

    def await_response(
        self, request_id: int, timeout: float = 30.0
    ) -> Optional[RequestOutput]:
        """Block until a response is available for this request."""
        with self._lock:
            cond = self._conditions.get(request_id)
            if cond is None:
                return None
            # Check if response already available
            responses = self._responses.get(request_id, [])
            if responses:
                return responses.pop(0)
            # Wait atomically — releases _lock and waits on cond
            cond.wait(timeout=timeout)
            responses = self._responses.get(request_id, [])
            if responses:
                return responses.pop(0)
        return None

    def cleanup(self, request_id: int) -> None:
        with self._lock:
            self._conditions.pop(request_id, None)
            self._responses.pop(request_id, None)


class PyExecutor:
    """Non-overlapping executor loop — TRT-LLM's PyExecutor equivalent.

    Each iteration runs the full pipeline sequentially:
    fetch → schedule → prepare → forward → sample → update → respond
    """

    def __init__(
        self,
        model_engine: ModelEngine,
        scheduler: TwoTierScheduler,
        sampler: Sampler,
        resource_manager=None,
        kv_cache: Optional[torch.Tensor] = None,
    ):
        self.model_engine = model_engine
        self.scheduler = scheduler
        self.sampler = sampler
        self.resource_manager = resource_manager
        self.kv_cache = kv_cache
        self.response_manager = ResponseManager()

        # Request queues
        self._waiting: list[LlmRequest] = []
        self._active: list[LlmRequest] = []
        self._sampling_params: dict[int, SamplingParams] = {}
        self._callbacks: dict[int, Callable] = {}
        self._cancelled: set[int] = set()
        self._lock = threading.Lock()

    def enqueue_request(
        self,
        request: LlmRequest,
        sampling_params: Optional[SamplingParams] = None,
        callback: Optional[Callable] = None,
    ) -> None:
        """Add a request to the waiting queue."""
        with self._lock:
            self._waiting.append(request)
            self._sampling_params[request.request_id] = sampling_params or SamplingParams()
            if callback:
                self._callbacks[request.request_id] = callback
            self.response_manager.register(request.request_id)

    def cancel_request(self, request_id: int) -> None:
        """Cancel a pending or active request."""
        with self._lock:
            self._cancelled.add(request_id)
            # Remove from waiting
            self._waiting = [r for r in self._waiting if r.request_id != request_id]

    def iteration(self) -> list[RequestOutput]:
        """Run one executor iteration. Returns completed outputs."""
        # Remove cancelled requests
        with self._lock:
            self._active = [
                r for r in self._active
                if r.request_id not in self._cancelled
            ]
            waiting = list(self._waiting)

        # Schedule
        active_blocks = sum(len(r.block_table) for r in self._active)
        active_gen = [r for r in self._active if r.is_generation_phase]
        active_ctx = [r for r in self._active if r.state == RequestState.CONTEXT_IN_PROGRESS]
        scheduled = self.scheduler.schedule(waiting + active_ctx, active_gen, active_blocks)

        if scheduled.is_empty:
            return []

        # Move newly admitted context requests to active
        with self._lock:
            for req in list(scheduled.context_chunking) + list(scheduled.context_last_chunk):
                if req in self._waiting:
                    self._waiting.remove(req)
                if req not in self._active:
                    self._active.append(req)

        # Prepare resources
        if self.resource_manager:
            all_reqs = list(scheduled.all_requests())
            self.resource_manager.prepare_resources(
                [r for r in all_reqs if r.state == RequestState.CONTEXT_INIT]
            )

        # Forward pass
        logits = self.model_engine.forward(scheduled, self.kv_cache)

        # Sample (async phase 1)
        gen_requests = list(scheduled.generation)
        # For context-last-chunk requests, they transition to generation after this
        ctx_last = list(scheduled.context_last_chunk)

        # Only sample for gen requests (they produce tokens)
        # Context-last-chunk requests: advance chunk, transition state
        for req in list(scheduled.context_chunking):
            chunk_size = getattr(req, '_scheduled_context_tokens', 0)
            req.num_context_tokens_processed += chunk_size
            if req.state == RequestState.CONTEXT_INIT:
                req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
            # CONTEXT_IN_PROGRESS self-loop is implicit (no state change needed)

        for req in ctx_last:
            chunk_size = getattr(req, '_scheduled_context_tokens', req.context_len)
            req.num_context_tokens_processed += chunk_size
            if req.state == RequestState.CONTEXT_INIT:
                req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
            if req.state == RequestState.CONTEXT_IN_PROGRESS:
                req.transition_to(RequestState.GENERATION_IN_PROGRESS)

        # Sample logits for generation + context-last-chunk requests
        sampling_reqs = ctx_last + gen_requests
        if sampling_reqs and logits.shape[0] > 0:
            # Extract logits for sampling requests
            sample_logits = logits[-len(sampling_reqs):]
            sample_state = self.sampler.sample_async(
                sample_logits, sampling_reqs, self._sampling_params
            )
            completed_ids = self.sampler.update_requests(sample_state, sampling_reqs)
        else:
            completed_ids = []

        # Update resources
        if self.resource_manager:
            self.resource_manager.update_resources(list(scheduled.generation))

        # Handle completions
        outputs = []
        for req_id in completed_ids:
            req = next((r for r in self._active if r.request_id == req_id), None)
            if req:
                output = RequestOutput(
                    request_id=req_id,
                    output_token_ids=list(req.output_token_ids),
                    finished=True,
                )
                outputs.append(output)
                self.response_manager.notify(req_id, output)

                # Fire callback
                cb = self._callbacks.get(req_id)
                if cb:
                    cb(output)

                # Free resources
                if self.resource_manager:
                    self.resource_manager.free_resources(req)

                with self._lock:
                    self._active = [r for r in self._active if r.request_id != req_id]

        # Streaming: notify in-progress generation requests
        for req in gen_requests:
            if req.request_id not in completed_ids and req.output_token_ids:
                output = RequestOutput(
                    request_id=req.request_id,
                    output_token_ids=list(req.output_token_ids),
                    finished=False,
                )
                self.response_manager.notify(req.request_id, output)

        return outputs

    @property
    def has_pending(self) -> bool:
        with self._lock:
            return len(self._waiting) > 0 or len(self._active) > 0


class OverlapExecutor(PyExecutor):
    """Overlap executor — CPU processes batch N-1 while GPU runs batch N.

    Uses two CUDA streams for ping-pong overlap:
    - engine_stream: GPU model forward + sampling
    - cpu_stream: CPU scheduling, state updates, response handling
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_sample_state = None
        self._prev_requests = None

    def overlap_iteration(self) -> list[RequestOutput]:
        """Overlapped iteration: process previous batch while running current."""
        outputs = []

        # Phase 1: Process previous batch results on CPU
        if self._prev_sample_state is not None and self._prev_requests is not None:
            completed_ids = self.sampler.update_requests(
                self._prev_sample_state, self._prev_requests
            )
            for req_id in completed_ids:
                req = next(
                    (r for r in self._active if r.request_id == req_id), None
                )
                if req:
                    output = RequestOutput(
                        request_id=req_id,
                        output_token_ids=list(req.output_token_ids),
                        finished=True,
                    )
                    outputs.append(output)
                    self.response_manager.notify(req_id, output)
                    if self.resource_manager:
                        self.resource_manager.free_resources(req)
                    with self._lock:
                        self._active = [
                            r for r in self._active if r.request_id != req_id
                        ]

        # Phase 2: Schedule and launch next batch on GPU
        with self._lock:
            waiting = list(self._waiting)
        active_blocks = sum(len(r.block_table) for r in self._active)
        active_gen = [r for r in self._active if r.is_generation_phase]
        active_ctx = [r for r in self._active if r.state == RequestState.CONTEXT_IN_PROGRESS]
        scheduled = self.scheduler.schedule(waiting + active_ctx, active_gen, active_blocks)

        if scheduled.is_empty:
            self._prev_sample_state = None
            self._prev_requests = None
            return outputs

        with self._lock:
            for req in list(scheduled.context_chunking) + list(scheduled.context_last_chunk):
                if req in self._waiting:
                    self._waiting.remove(req)
                if req not in self._active:
                    self._active.append(req)

        # Forward + async sample (GPU)
        logits = self.model_engine.forward(scheduled, self.kv_cache)

        # Transition context requests
        for req in list(scheduled.context_chunking):
            chunk_size = getattr(req, '_scheduled_context_tokens', 0)
            req.num_context_tokens_processed += chunk_size
            if req.state == RequestState.CONTEXT_INIT:
                req.transition_to(RequestState.CONTEXT_IN_PROGRESS)

        ctx_last = list(scheduled.context_last_chunk)
        for req in ctx_last:
            chunk_size = getattr(req, '_scheduled_context_tokens', req.context_len)
            req.num_context_tokens_processed += chunk_size
            if req.state == RequestState.CONTEXT_INIT:
                req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
            if req.state == RequestState.CONTEXT_IN_PROGRESS:
                req.transition_to(RequestState.GENERATION_IN_PROGRESS)

        sampling_reqs = ctx_last + list(scheduled.generation)
        if sampling_reqs and logits.shape[0] > 0:
            sample_logits = logits[-len(sampling_reqs):]
            self._prev_sample_state = self.sampler.sample_async(
                sample_logits, sampling_reqs, self._sampling_params
            )
            self._prev_requests = sampling_reqs
        else:
            self._prev_sample_state = None
            self._prev_requests = None

        return outputs
