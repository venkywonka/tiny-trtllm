"""Tensor parallelism stubs — single-GPU defaults."""

_tp_rank = 0
_tp_size = 1


def get_tp_rank() -> int:
    return _tp_rank


def get_tp_size() -> int:
    return _tp_size


def all_reduce(tensor):
    """No-op for single GPU."""
    return tensor
