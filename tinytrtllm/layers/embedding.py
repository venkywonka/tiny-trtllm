"""Vocab-parallel embedding and LM head layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinytrtllm.distributed.tp import all_gather, all_reduce


class VocabParallelEmbedding(nn.Module):
    """Embedding layer with vocabulary sharded across TP ranks.

    Each rank holds vocab_size // tp_size rows. During forward, token IDs
    outside this rank's shard are masked to zero, embedded, then all-reduced.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.vocab_per_rank = vocab_size // tp_size
        self.vocab_start = tp_rank * self.vocab_per_rank
        self.vocab_end = self.vocab_start + self.vocab_per_rank
        self.embedding = nn.Embedding(self.vocab_per_rank, hidden_size)

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return self.embedding(input_ids)

        # Mask out-of-range tokens for this shard
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        # Shift IDs to local range
        local_ids = input_ids - self.vocab_start
        local_ids = local_ids.clamp(0, self.vocab_per_rank - 1)
        output = self.embedding(local_ids)
        # Zero out embeddings for tokens not in this shard
        output = output * mask.unsqueeze(-1).to(output.dtype)
        return all_reduce(output)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Shard loaded weight along vocab dimension."""
        param.data.copy_(loaded_weight[self.vocab_start : self.vocab_end])


class ParallelLMHead(nn.Module):
    """Linear projection from hidden states to vocabulary logits.

    Supports optional last_token_indices for extracting specific positions
    before the projection, which is more efficient than projecting all positions.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        bias: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.vocab_per_rank = vocab_size // tp_size
        self.linear = nn.Linear(hidden_size, self.vocab_per_rank, bias=bias)

    @property
    def weight(self):
        return self.linear.weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if last_token_indices is not None:
            # Extract last token hidden states: (batch, hidden)
            batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            hidden_states = hidden_states[batch_indices, last_token_indices]

        logits = self.linear(hidden_states)

        if self.tp_size > 1:
            logits = all_gather(logits, dim=-1)

        return logits

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Shard loaded weight along vocab dimension."""
        start = self.tp_rank * self.vocab_per_rank
        end = start + self.vocab_per_rank
        param.data.copy_(loaded_weight[start:end])
