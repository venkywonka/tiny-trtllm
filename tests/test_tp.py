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
