"""Tensor parallelism with NCCL — multi-GPU ready, tested on single GPU.

Architecture:
- DistributedInfo: rank/size holder
- init/destroy distributed: NCCL process group management
- all_reduce/all_gather/broadcast/barrier: collective ops
- spawn_tp_workers: torch.multiprocessing.spawn wrapper
- SharedMemoryBuffer: IPC buffer for TP coordination
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist


@dataclass
class DistributedInfo:
    rank: int = 0
    world_size: int = 1
    device: Optional[torch.device] = None


# Global singleton
_dist_info = DistributedInfo()


def get_tp_rank() -> int:
    return _dist_info.rank


def get_tp_size() -> int:
    return _dist_info.world_size


def get_device() -> torch.device:
    if _dist_info.device is not None:
        return _dist_info.device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{_dist_info.rank}")
    return torch.device("cpu")


def init_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialize distributed process group."""
    global _dist_info
    _dist_info = DistributedInfo(rank=rank, world_size=world_size)

    if world_size <= 1:
        _dist_info.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        _dist_info.device = torch.device(f"cuda:{rank}")
    else:
        _dist_info.device = torch.device("cpu")


def destroy_distributed() -> None:
    """Destroy distributed process group."""
    global _dist_info
    if dist.is_initialized():
        dist.destroy_process_group()
    _dist_info = DistributedInfo()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """Sum tensor across all ranks. No-op for single GPU."""
    if _dist_info.world_size <= 1 or not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensors from all ranks into a single tensor."""
    if _dist_info.world_size <= 1 or not dist.is_initialized():
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(_dist_info.world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from src rank."""
    if _dist_info.world_size <= 1 or not dist.is_initialized():
        return tensor
    dist.broadcast(tensor, src=src)
    return tensor


def barrier() -> None:
    """Synchronize all ranks."""
    if _dist_info.world_size <= 1 or not dist.is_initialized():
        return
    dist.barrier()


def spawn_tp_workers(
    fn: Callable,
    world_size: int,
    args: tuple = (),
) -> None:
    """Spawn N worker processes for tensor parallelism."""
    if world_size <= 1:
        fn(0, world_size, *args)
        return

    import torch.multiprocessing as mp

    mp.spawn(
        fn,
        args=(world_size, *args),
        nprocs=world_size,
        join=True,
    )


class SharedMemoryBuffer:
    """Shared memory IPC buffer for TP coordination (following nanovllm pattern).

    Used for lightweight coordination between TP workers (schedule decisions,
    ready signals) without going through NCCL.
    """

    def __init__(self, size: int = 1024 * 1024):
        self.size = size
        self._buffer = torch.zeros(size, dtype=torch.uint8)
        if torch.cuda.is_available():
            self._buffer = self._buffer.share_memory_()

    def write(self, offset: int, data: bytes) -> None:
        for i, b in enumerate(data):
            if offset + i < self.size:
                self._buffer[offset + i] = b

    def read(self, offset: int, length: int) -> bytes:
        return bytes(self._buffer[offset : offset + length].tolist())
