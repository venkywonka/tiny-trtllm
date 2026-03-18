"""Async two-phase sampler with grouped strategies.

Phase 1 (sample_async): Launch GPU sampling kernels, return SampleState
Phase 2 (update_requests): Synchronize, update CPU request state

Grouped strategy: batch requests by (top_k, top_p, temperature) → 1 kernel per group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from tinytrtllm.config import SamplingParams
from tinytrtllm.engine.request import LlmRequest


@dataclass
class SampleState:
    """Intermediate state between async sample and update."""

    token_ids: torch.Tensor  # (batch,) sampled token IDs
    request_ids: list[int]  # mapping back to requests
    logits: torch.Tensor  # (batch, vocab) for logging/debugging


class Sampler:
    """Async two-phase sampler with grouped sampling strategies."""

    def sample_async(
        self,
        logits: torch.Tensor,
        requests: list[LlmRequest],
        sampling_params_map: dict[int, SamplingParams],
    ) -> SampleState:
        """Phase 1: Launch sampling on GPU. Returns immediately with SampleState.

        Groups requests by (top_k, top_p, temperature) for efficient batch processing.
        """
        if logits.shape[0] == 0:
            return SampleState(
                token_ids=torch.empty(0, dtype=torch.long),
                request_ids=[],
                logits=logits,
            )

        batch_size = logits.shape[0]
        device = logits.device

        # Group by sampling strategy
        groups = self._group_by_strategy(requests, sampling_params_map)

        sampled_tokens = torch.empty(batch_size, dtype=torch.long, device=device)
        request_ids = [r.request_id for r in requests]

        for (top_k, top_p, temp), indices in groups.items():
            group_logits = logits[indices]
            tokens = self._sample_impl(group_logits, top_k, top_p, temp)
            sampled_tokens[indices] = tokens

        return SampleState(
            token_ids=sampled_tokens,
            request_ids=request_ids,
            logits=logits,
        )

    def update_requests(
        self,
        sample_state: SampleState,
        requests: list[LlmRequest],
    ) -> list[int]:
        """Phase 2: Synchronize and update CPU request state.

        Returns list of completed request IDs.
        """
        completed = []
        token_ids_cpu = sample_state.token_ids.cpu()

        for i, req in enumerate(requests):
            if i >= len(token_ids_cpu):
                break
            token_id = token_ids_cpu[i].item()
            req.append_output_token(token_id)

            if req.num_tokens_remaining <= 0:
                from tinytrtllm.engine.request import RequestState
                if req.state != RequestState.GENERATION_COMPLETE:
                    req.transition_to(RequestState.GENERATION_COMPLETE)
                completed.append(req.request_id)

        return completed

    def _group_by_strategy(
        self,
        requests: list[LlmRequest],
        params_map: dict[int, SamplingParams],
    ) -> dict[tuple, list[int]]:
        """Group request indices by (top_k, top_p, temperature) for batched sampling."""
        groups: dict[tuple, list[int]] = {}
        for i, req in enumerate(requests):
            sp = params_map.get(req.request_id, SamplingParams())
            key = (sp.top_k, sp.top_p, sp.temperature)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        return groups

    def _sample_impl(
        self,
        logits: torch.Tensor,
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> torch.Tensor:
        """Sample tokens from logits with top_k, top_p, temperature."""
        if temperature <= 1e-6:
            # Greedy
            return logits.argmax(dim=-1)

        # Temperature scaling
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0 and top_k < logits.shape[-1]:
            topk_vals, _ = torch.topk(logits, top_k, dim=-1)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float('-inf'))

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float('-inf')
            # Scatter back
            logits = torch.zeros_like(sorted_logits).scatter(-1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
