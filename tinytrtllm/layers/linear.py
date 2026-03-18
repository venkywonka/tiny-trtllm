"""TP-aware linear layers with weight loaders."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinytrtllm.distributed.tp import all_reduce


class ReplicatedLinear(nn.Module):
    """Standard linear layer — replicated across all TP ranks."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ColumnParallelLinear(nn.Module):
    """Linear layer with output dimension sharded across TP ranks.

    Weight shape: (out_features // tp_size, in_features).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.out_features_per_rank = out_features // tp_size
        self.linear = nn.Linear(in_features, self.out_features_per_rank, bias=bias)

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Shard loaded weight along output dimension."""
        start = self.tp_rank * self.out_features_per_rank
        end = start + self.out_features_per_rank
        param.data.copy_(loaded_weight[start:end])


class RowParallelLinear(nn.Module):
    """Linear layer with input dimension sharded across TP ranks.

    Weight shape: (out_features, in_features // tp_size).
    Calls all_reduce in forward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.in_features_per_rank = in_features // tp_size
        self.linear = nn.Linear(self.in_features_per_rank, out_features, bias=bias)

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        return all_reduce(out)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Shard loaded weight along input dimension."""
        start = self.tp_rank * self.in_features_per_rank
        end = start + self.in_features_per_rank
        param.data.copy_(loaded_weight[:, start:end])


class MergedColumnParallelLinear(nn.Module):
    """Merged column-parallel linear for gate+up projections.

    Weight shape: (sum(out_features_list) // tp_size, in_features).
    """

    def __init__(
        self,
        in_features: int,
        out_features_list: list[int],
        bias: bool = True,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features_list = out_features_list
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.total_out_features = sum(out_features_list) // tp_size
        self.linear = nn.Linear(in_features, self.total_out_features, bias=bias)

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int):
        """Load a specific shard (e.g., gate or up projection)."""
        shard_size = self.out_features_list[shard_id] // self.tp_size
        start_idx = self.tp_rank * shard_size
        end_idx = start_idx + shard_size
        # Compute offset within merged weight
        offset = sum(s // self.tp_size for s in self.out_features_list[:shard_id])
        param.data[offset : offset + shard_size].copy_(loaded_weight[start_idx:end_idx])


class QKVParallelLinear(nn.Module):
    """Specialized QKV parallel linear layer.

    Computes q_size = num_heads * head_dim, kv_size = num_kv_heads * head_dim.
    Shards each by tp_size.
    Weight shape: ((q_size + 2 * kv_size) // tp_size, hidden_size).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = True,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.q_size_per_rank = self.q_size // tp_size
        self.kv_size_per_rank = self.kv_size // tp_size
        self.total_out = self.q_size_per_rank + 2 * self.kv_size_per_rank

        self.linear = nn.Linear(hidden_size, self.total_out, bias=bias)

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str):
        """Load Q, K, or V shard."""
        if shard_id == "q":
            shard_size = self.q_size_per_rank
            start = self.tp_rank * shard_size
            offset = 0
        elif shard_id == "k":
            shard_size = self.kv_size_per_rank
            start = self.tp_rank * shard_size
            offset = self.q_size_per_rank
        elif shard_id == "v":
            shard_size = self.kv_size_per_rank
            start = self.tp_rank * shard_size
            offset = self.q_size_per_rank + self.kv_size_per_rank
        else:
            raise ValueError(f"Unknown shard_id: {shard_id}")
        param.data[offset : offset + shard_size].copy_(loaded_weight[start : start + shard_size])
