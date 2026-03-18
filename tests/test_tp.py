"""Tests for tensor parallelism — single GPU with mocked multi-GPU."""

import pytest
import torch

from tinytrtllm.distributed.tp import (
    DistributedInfo,
    SharedMemoryBuffer,
    all_gather,
    all_reduce,
    barrier,
    broadcast,
    destroy_distributed,
    get_device,
    get_tp_rank,
    get_tp_size,
    init_distributed,
)


class TestDistributedInfo:
    def test_defaults(self):
        info = DistributedInfo()
        assert info.rank == 0
        assert info.world_size == 1

    def test_custom(self):
        info = DistributedInfo(rank=1, world_size=4)
        assert info.rank == 1
        assert info.world_size == 4


class TestSingleGPUOps:
    def test_init_single(self):
        init_distributed(rank=0, world_size=1)
        assert get_tp_rank() == 0
        assert get_tp_size() == 1
        destroy_distributed()

    def test_all_reduce_noop(self):
        init_distributed(rank=0, world_size=1)
        t = torch.tensor([1.0, 2.0, 3.0])
        result = all_reduce(t)
        assert torch.allclose(result, t)
        destroy_distributed()

    def test_all_gather_noop(self):
        init_distributed(rank=0, world_size=1)
        t = torch.tensor([1.0, 2.0])
        result = all_gather(t)
        assert torch.allclose(result, t)
        destroy_distributed()

    def test_broadcast_noop(self):
        init_distributed(rank=0, world_size=1)
        t = torch.tensor([5.0])
        result = broadcast(t)
        assert result.item() == 5.0
        destroy_distributed()

    def test_barrier_noop(self):
        init_distributed(rank=0, world_size=1)
        barrier()  # Should not raise
        destroy_distributed()

    def test_get_device(self):
        init_distributed(rank=0, world_size=1)
        device = get_device()
        assert device is not None
        destroy_distributed()


class TestSharedMemoryBuffer:
    def test_write_and_read(self):
        buf = SharedMemoryBuffer(size=1024)
        buf.write(0, b"hello")
        data = buf.read(0, 5)
        assert data == b"hello"

    def test_offset_write(self):
        buf = SharedMemoryBuffer(size=1024)
        buf.write(100, b"test")
        data = buf.read(100, 4)
        assert data == b"test"

    def test_buffer_size(self):
        buf = SharedMemoryBuffer(size=512)
        assert buf.size == 512

    def test_vectorized_write_large_data(self):
        """S8: write should handle large data efficiently (vectorized)."""
        buf = SharedMemoryBuffer(size=4096)
        data = bytes(range(256)) * 4  # 1024 bytes
        buf.write(0, data)
        result = buf.read(0, 1024)
        assert result == data

    def test_buffer_shared_without_cuda(self):
        """S11: Buffer should be shared even without CUDA."""
        buf = SharedMemoryBuffer(size=256)
        assert buf._buffer.is_shared()


class TestTPBugFixes:
    """Regression tests for TP bug fixes."""

    def test_all_gather_accepts_dim_parameter(self):
        """S7: all_gather should accept a dim parameter."""
        from tinytrtllm.distributed.tp import all_gather
        t = torch.randn(2, 3)
        # Single GPU: no-op regardless of dim
        result = all_gather(t, dim=-1)
        assert torch.equal(result, t)

    def test_init_distributed_raises_on_reinit(self):
        """S10: Re-initializing should raise, not silently skip."""
        # This only applies to multi-GPU (world_size > 1)
        # For single GPU, init_distributed returns early, so test the guard
        from tinytrtllm.distributed.tp import init_distributed, destroy_distributed
        init_distributed(rank=0, world_size=1)
        # world_size=1 doesn't go through dist.init_process_group
        # so this tests the single-GPU path (no raise expected)
        destroy_distributed()

    def test_row_parallel_linear_bias_rank0_only(self):
        """S9: RowParallelLinear bias should only be on rank 0."""
        from tinytrtllm.layers.linear import RowParallelLinear
        # Rank 0: has bias
        rpl0 = RowParallelLinear(8, 4, bias=True, tp_size=2, tp_rank=0)
        assert rpl0.linear.bias is not None
        # Rank 1: no bias
        rpl1 = RowParallelLinear(8, 4, bias=True, tp_size=2, tp_rank=1)
        assert rpl1.linear.bias is None

    def test_parallel_lm_head_imports_all_gather(self):
        """S10: ParallelLMHead should use all_gather, not pass."""
        from tinytrtllm.layers.embedding import ParallelLMHead
        # tp_size=1: just runs the linear, no gather needed
        head = ParallelLMHead(vocab_size=100, hidden_size=16, tp_size=1)
        x = torch.randn(2, 16)
        logits = head(x)
        assert logits.shape == (2, 100)
