"""Tests for Pydantic config hierarchy — TinyLlmArgs and SamplingParams."""

import pytest
from pydantic import ValidationError

from tinytrtllm.config import (
    ChunkingPolicy,
    SchedulingPolicy,
    SamplingParams,
    TinyLlmArgs,
)


class TestTinyLlmArgs:
    def test_minimal_creation(self):
        args = TinyLlmArgs(model_path="/tmp/model")
        assert args.model_path == "/tmp/model"

    def test_defaults(self):
        args = TinyLlmArgs(model_path="/tmp/model")
        assert args.dtype == "bfloat16"
        assert args.max_batch_size == 256
        assert args.max_seq_len == 4096
        assert args.max_num_tokens == 16384
        assert args.block_size == 256
        assert args.gpu_memory_utilization == pytest.approx(0.9)
        assert args.tensor_parallel_size == 1
        assert args.scheduling_policy == SchedulingPolicy.GUARANTEED_NO_EVICT
        assert args.chunking_policy == ChunkingPolicy.EQUAL_PROGRESS
        assert args.enable_overlap is True
        assert args.enable_cuda_graph is True
        assert args.enable_prefix_cache is True

    def test_missing_model_path_raises(self):
        with pytest.raises(ValidationError):
            TinyLlmArgs()

    def test_negative_batch_size_raises(self):
        with pytest.raises(ValidationError):
            TinyLlmArgs(model_path="/tmp/model", max_batch_size=-1)

    def test_zero_max_seq_len_raises(self):
        with pytest.raises(ValidationError):
            TinyLlmArgs(model_path="/tmp/model", max_seq_len=0)

    def test_negative_max_num_tokens_raises(self):
        with pytest.raises(ValidationError):
            TinyLlmArgs(model_path="/tmp/model", max_num_tokens=-10)

    def test_gpu_memory_utilization_out_of_range(self):
        with pytest.raises(ValidationError):
            TinyLlmArgs(model_path="/tmp/model", gpu_memory_utilization=1.5)
        with pytest.raises(ValidationError):
            TinyLlmArgs(model_path="/tmp/model", gpu_memory_utilization=0.0)

    def test_custom_values(self):
        args = TinyLlmArgs(
            model_path="/tmp/model",
            dtype="float16",
            max_batch_size=128,
            max_seq_len=2048,
            scheduling_policy=SchedulingPolicy.MAX_UTILIZATION,
            chunking_policy=ChunkingPolicy.FCFS,
            enable_overlap=False,
        )
        assert args.dtype == "float16"
        assert args.max_batch_size == 128
        assert args.scheduling_policy == SchedulingPolicy.MAX_UTILIZATION
        assert args.chunking_policy == ChunkingPolicy.FCFS

    def test_invalid_dtype_raises(self):
        """dtype validator should reject unknown dtype strings."""
        with pytest.raises(ValidationError):
            TinyLlmArgs(model_path="/tmp/model", dtype="fp16")
        with pytest.raises(ValidationError):
            TinyLlmArgs(model_path="/tmp/model", dtype="bf16")

    def test_valid_dtypes_accepted(self):
        for dt in ["bfloat16", "float16", "float32"]:
            args = TinyLlmArgs(model_path="/tmp/model", dtype=dt)
            assert args.dtype == dt

    def test_attn_backend_default(self):
        args = TinyLlmArgs(model_path="/tmp/model")
        assert args.attn_backend == "auto"

    def test_attn_backend_custom(self):
        args = TinyLlmArgs(model_path="/tmp/model", attn_backend="trtllm,fi")
        assert args.attn_backend == "trtllm,fi"


class TestSchedulingPolicy:
    def test_guaranteed_no_evict(self):
        assert SchedulingPolicy.GUARANTEED_NO_EVICT.value == "guaranteed_no_evict"

    def test_max_utilization(self):
        assert SchedulingPolicy.MAX_UTILIZATION.value == "max_utilization"


class TestChunkingPolicy:
    def test_equal_progress(self):
        assert ChunkingPolicy.EQUAL_PROGRESS.value == "equal_progress"

    def test_fcfs(self):
        assert ChunkingPolicy.FCFS.value == "fcfs"


class TestSamplingParams:
    def test_defaults(self):
        sp = SamplingParams()
        assert sp.temperature == pytest.approx(1.0)
        assert sp.top_k == 50
        assert sp.top_p == pytest.approx(1.0)
        assert sp.max_tokens == 64
        assert sp.ignore_eos is False

    def test_negative_temperature_raises(self):
        with pytest.raises(ValidationError):
            SamplingParams(temperature=-0.1)

    def test_zero_temperature_allowed(self):
        sp = SamplingParams(temperature=0.0)
        assert sp.temperature == 0.0

    def test_top_p_out_of_range(self):
        with pytest.raises(ValidationError):
            SamplingParams(top_p=1.5)
        with pytest.raises(ValidationError):
            SamplingParams(top_p=-0.1)

    def test_top_k_negative_raises(self):
        with pytest.raises(ValidationError):
            SamplingParams(top_k=-1)

    def test_max_tokens_zero_raises(self):
        with pytest.raises(ValidationError):
            SamplingParams(max_tokens=0)

    def test_custom_values(self):
        sp = SamplingParams(temperature=0.7, top_k=10, top_p=0.9, max_tokens=128)
        assert sp.temperature == pytest.approx(0.7)
        assert sp.top_k == 10
        assert sp.top_p == pytest.approx(0.9)
        assert sp.max_tokens == 128
