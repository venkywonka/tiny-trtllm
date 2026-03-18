# Two-Tier Scheduling

TRT-LLM's most distinctive pattern. No other inference engine separates admission control from token budgeting.

## Overview

```{mermaid}
graph TD
    WQ[Waiting Queue] --> CS["CapacityScheduler<br/><b>Tier 1: Admission</b>"]
    CS -->|admitted requests| MBS["MicroBatchScheduler<br/><b>Tier 2: Chunking</b>"]
    MBS --> SR[ScheduledRequests]
    SR --> CC["context_chunking<br/>still chunking prefill"]
    SR --> CLC["context_last_chunk<br/>final prefill chunk → gen"]
    SR --> GEN["generation<br/>decode: 1 token/step"]
    SR --> P["paused<br/>preempted (MAX_UTIL)"]
```

## Tier 1: CapacityScheduler (Admission Control)

Decides **whether** a request can be admitted into the active set.

Two policies:
- **`GUARANTEED_NO_EVICT`**: Only admit a request if we can guarantee it will finish without running out of KV cache blocks. Conservative but no preemptions.
- **`MAX_UTILIZATION`**: Admit optimistically. If memory runs short, preempt lower-priority requests. Higher throughput but requests may be paused.

## Tier 2: MicroBatchScheduler (Token Budgeting)

Decides **how many** context tokens each admitted prefill request gets this iteration.

Two policies:
- **`EQUAL_PROGRESS`**: Distribute the token budget fairly across all prefill requests. Every request makes progress each iteration.
- **`FCFS`**: Process requests in arrival order. The first request gets the full budget until complete.

## ScheduledRequests: 4 Disjoint Lists

After scheduling, requests are categorized into exactly one of four lists:

| List | Phase | Description |
|------|-------|-------------|
| `context_chunking` | Prefill | Still chunking — more prefill iterations to go |
| `context_last_chunk` | Prefill → Decode | Final prefill chunk, will transition to generation |
| `generation` | Decode | Generating tokens, 1 token per step |
| `paused` | Suspended | Preempted by `MAX_UTILIZATION` policy |

## Code Example

```python
from tinytrtllm.engine.scheduler import TwoTierScheduler
from tinytrtllm.config import SchedulingPolicy, ChunkingPolicy

sched = TwoTierScheduler(
    scheduling_policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
    chunking_policy=ChunkingPolicy.EQUAL_PROGRESS,
    max_batch_size=256,
    max_num_tokens=16384,
    block_size=256,
    max_blocks=1000,
)

result = sched.schedule(waiting_requests, active_gen_requests, active_blocks=0)
# result.context_chunking   → prefill requests still chunking
# result.context_last_chunk → prefill requests on their final chunk
# result.generation         → decode requests (1 token each)
# result.paused             → preempted requests
```
