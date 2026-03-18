"""Two-tier scheduling — TRT-LLM's signature architectural pattern.

Tier 1 (CapacityScheduler): Decides which requests to admit based on memory.
  - GUARANTEED_NO_EVICT: Only admits if it can guarantee completion without eviction.
  - MAX_UTILIZATION: Admits optimistically; may pause/preempt under memory pressure.

Tier 2 (MicroBatchScheduler): Decides how to chunk admitted requests' context tokens.
  - EQUAL_PROGRESS: Split token budget equally across context requests (fairness).
  - FCFS: Process context requests in arrival order.

ScheduledRequests: 4 disjoint lists segmenting work by phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator

from tinytrtllm.config import ChunkingPolicy, SchedulingPolicy
from tinytrtllm.engine.request import LlmRequest


@dataclass
class ScheduledRequests:
    """Phase-segmented request lists — the 4 disjoint lists are TRT-LLM's signature."""

    context_chunking: list[LlmRequest] = field(default_factory=list)
    context_last_chunk: list[LlmRequest] = field(default_factory=list)
    generation: list[LlmRequest] = field(default_factory=list)
    paused: list[LlmRequest] = field(default_factory=list)

    def all_requests(self) -> Iterator[LlmRequest]:
        yield from self.context_chunking
        yield from self.context_last_chunk
        yield from self.generation

    @property
    def num_context_tokens(self) -> int:
        total = 0
        for req in self.context_chunking:
            total += getattr(req, "_scheduled_context_tokens", 0)
        for req in self.context_last_chunk:
            total += getattr(req, "_scheduled_context_tokens", 0)
        return total

    @property
    def num_generation_tokens(self) -> int:
        return len(self.generation)

    @property
    def is_empty(self) -> bool:
        return (
            len(self.context_chunking) == 0
            and len(self.context_last_chunk) == 0
            and len(self.generation) == 0
        )

    def __len__(self) -> int:
        return (
            len(self.context_chunking)
            + len(self.context_last_chunk)
            + len(self.generation)
            + len(self.paused)
        )

    @property
    def can_run_cuda_graph(self) -> bool:
        """CUDA graphs only work for decode-only batches (no variable-length context)."""
        return (
            len(self.context_chunking) == 0
            and len(self.context_last_chunk) == 0
            and len(self.generation) > 0
        )


class CapacityScheduler:
    """Tier 1: Admits requests based on memory capacity policy."""

    def __init__(
        self,
        policy: SchedulingPolicy,
        max_batch_size: int,
        max_num_tokens: int,
        block_size: int,
        max_blocks: int,
    ):
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.block_size = block_size
        self.max_blocks = max_blocks

    def _blocks_needed(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return math.ceil(num_tokens / self.block_size)

    def schedule_capacity(
        self,
        waiting: list[LlmRequest],
        active_blocks: int,
    ) -> tuple[list[LlmRequest], list[LlmRequest]]:
        """Return (admitted, rejected) lists."""
        admitted: list[LlmRequest] = []
        rejected: list[LlmRequest] = []
        blocks_used = active_blocks

        for req in waiting:
            if len(admitted) >= self.max_batch_size:
                rejected.append(req)
                continue

            if self.policy == SchedulingPolicy.GUARANTEED_NO_EVICT:
                # Must guarantee completion: need blocks for ALL tokens (context + output)
                total_tokens = req.context_len + req.max_tokens
                blocks_for_req = self._blocks_needed(total_tokens)
                if blocks_used + blocks_for_req <= self.max_blocks:
                    blocks_used += blocks_for_req
                    admitted.append(req)
                else:
                    rejected.append(req)

            elif self.policy == SchedulingPolicy.MAX_UTILIZATION:
                # Optimistic: only need blocks for context to start
                context_blocks = self._blocks_needed(req.context_len)
                if context_blocks == 0:
                    context_blocks = 1  # At least 1 block
                if blocks_used + context_blocks <= self.max_blocks:
                    blocks_used += context_blocks
                    admitted.append(req)
                else:
                    rejected.append(req)

        return admitted, rejected


class MicroBatchScheduler:
    """Tier 2: Chunks context tokens across admitted requests."""

    def __init__(self, policy: ChunkingPolicy, max_num_tokens: int):
        self.policy = policy
        self.max_num_tokens = max_num_tokens

    def schedule_microbatch(
        self,
        context_requests: list[LlmRequest],
        gen_requests: list[LlmRequest],
    ) -> ScheduledRequests:
        result = ScheduledRequests()

        # Generation requests always go through (1 token each)
        result.generation = list(gen_requests)
        gen_tokens = len(gen_requests)
        remaining_budget = self.max_num_tokens - gen_tokens

        if not context_requests or remaining_budget <= 0:
            return result

        if self.policy == ChunkingPolicy.EQUAL_PROGRESS:
            self._schedule_equal_progress(context_requests, remaining_budget, result)
        elif self.policy == ChunkingPolicy.FCFS:
            self._schedule_fcfs(context_requests, remaining_budget, result)

        return result

    def _schedule_equal_progress(
        self,
        requests: list[LlmRequest],
        budget: int,
        result: ScheduledRequests,
    ) -> None:
        """Split budget equally across context requests for fairness."""
        n = len(requests)
        per_request = max(1, budget // n)

        for req in requests:
            remaining_context = req.context_len - req.num_context_tokens_processed
            chunk_size = min(per_request, remaining_context)
            if chunk_size <= 0:
                continue
            req._scheduled_context_tokens = chunk_size
            is_last = (chunk_size >= remaining_context)
            if is_last:
                result.context_last_chunk.append(req)
            else:
                result.context_chunking.append(req)

    def _schedule_fcfs(
        self,
        requests: list[LlmRequest],
        budget: int,
        result: ScheduledRequests,
    ) -> None:
        """Process context requests in FIFO order."""
        tokens_left = budget
        for req in requests:
            if tokens_left <= 0:
                break
            remaining_context = req.context_len - req.num_context_tokens_processed
            chunk_size = min(tokens_left, remaining_context)
            if chunk_size <= 0:
                continue
            req._scheduled_context_tokens = chunk_size
            tokens_left -= chunk_size
            is_last = (chunk_size >= remaining_context)
            if is_last:
                result.context_last_chunk.append(req)
            else:
                result.context_chunking.append(req)


class TwoTierScheduler:
    """Combines CapacityScheduler (Tier 1) + MicroBatchScheduler (Tier 2)."""

    def __init__(
        self,
        scheduling_policy: SchedulingPolicy,
        chunking_policy: ChunkingPolicy,
        max_batch_size: int,
        max_num_tokens: int,
        block_size: int,
        max_blocks: int,
    ):
        self.capacity_scheduler = CapacityScheduler(
            policy=scheduling_policy,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            block_size=block_size,
            max_blocks=max_blocks,
        )
        self.microbatch_scheduler = MicroBatchScheduler(
            policy=chunking_policy,
            max_num_tokens=max_num_tokens,
        )

    def schedule(
        self,
        waiting: list[LlmRequest],
        active_gen: list[LlmRequest],
        active_blocks: int,
    ) -> ScheduledRequests:
        """Run both tiers: capacity → microbatch → ScheduledRequests."""
        admitted, _ = self.capacity_scheduler.schedule_capacity(waiting, active_blocks)
        return self.microbatch_scheduler.schedule_microbatch(admitted, active_gen)
