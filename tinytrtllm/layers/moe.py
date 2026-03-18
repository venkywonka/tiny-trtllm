"""Mixture of Experts — gate + topk + fused expert dispatch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEGate(nn.Module):
    """Router gate that produces logits over experts."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)  # (num_tokens, num_experts)


class FusedMoE(nn.Module):
    """Mixture-of-Experts with top-k routing and expert-parallel dispatch.

    Each expert contains gate_proj, up_proj (fused as gate_up) and down_proj,
    stored as batched parameter tensors for efficient indexing.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate = MoEGate(hidden_size, num_experts)

        # Expert weights: each expert has gate_proj + up_proj (fused), down_proj
        self.experts_gate_up = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.experts_down = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        nn.init.kaiming_uniform_(self.experts_gate_up)
        nn.init.kaiming_uniform_(self.experts_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])  # (num_tokens, hidden)
        num_tokens = x_2d.shape[0]

        # Gate
        router_logits = self.gate(x_2d)  # (num_tokens, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)

        # TopK selection
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Dispatch to experts
        output = torch.zeros(num_tokens, self.hidden_size, dtype=x.dtype, device=x.device)
        for k in range(self.top_k):
            expert_ids = topk_ids[:, k]  # (num_tokens,)
            weights = topk_weights[:, k]  # (num_tokens,)

            for expert_idx in range(self.num_experts):
                mask = expert_ids == expert_idx
                if not mask.any():
                    continue
                expert_input = x_2d[mask]

                # gate_up_proj
                gate_up = F.linear(expert_input, self.experts_gate_up[expert_idx])
                gate, up = gate_up.chunk(2, dim=-1)
                hidden = F.silu(gate) * up

                # down_proj
                expert_output = F.linear(hidden, self.experts_down[expert_idx])

                output[mask] += weights[mask].unsqueeze(-1) * expert_output

        return output.view(orig_shape)
