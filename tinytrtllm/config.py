"""Pydantic configuration hierarchy mirroring TRT-LLM's validated config pattern."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class SchedulingPolicy(str, Enum):
    """TRT-LLM capacity scheduling policies.

    GUARANTEED_NO_EVICT: Only schedules if completion can be guaranteed without eviction.
    MAX_UTILIZATION: Schedules optimistically; may pause/preempt under memory pressure.
    """

    GUARANTEED_NO_EVICT = "guaranteed_no_evict"
    MAX_UTILIZATION = "max_utilization"


class ChunkingPolicy(str, Enum):
    """TRT-LLM chunked-prefill policies.

    EQUAL_PROGRESS: Split context tokens equally across requests for fairness.
    FCFS: Process context requests in arrival order (first-come first-served).
    """

    EQUAL_PROGRESS = "equal_progress"
    FCFS = "fcfs"


class TinyLlmArgs(BaseModel):
    """Top-level engine configuration, modeled after TRT-LLM's LlmArgs."""

    model_path: str
    dtype: str = "bfloat16"
    max_batch_size: int = Field(default=256, gt=0)
    max_seq_len: int = Field(default=4096, gt=0)
    max_num_tokens: int = Field(default=16384, gt=0)
    block_size: int = Field(default=256, gt=0)
    gpu_memory_utilization: float = Field(default=0.9, gt=0.0, le=1.0)
    tensor_parallel_size: int = Field(default=1, gt=0)
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.GUARANTEED_NO_EVICT
    chunking_policy: ChunkingPolicy = ChunkingPolicy.EQUAL_PROGRESS
    enable_overlap: bool = True
    enable_cuda_graph: bool = True
    enable_prefix_cache: bool = True
    attn_backend: str = "auto"


class SamplingParams(BaseModel):
    """Per-request sampling configuration."""

    temperature: float = Field(default=1.0, ge=0.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=64, gt=0)
    ignore_eos: bool = False
