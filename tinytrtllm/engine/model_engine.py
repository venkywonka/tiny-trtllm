"""ModelEngine — forward pass orchestration with CUDA graph capture.

Responsibilities:
- Prepare model inputs from ScheduledRequests
- Run model forward pass (eager or CUDA graph replay)
- Extract logits for sampling
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from tinytrtllm.engine.scheduler import ScheduledRequests


class CUDAGraphRunner:
    """Captures and replays CUDA graphs for decode batches."""

    def __init__(self, max_batch_size: int = 256):
        self.max_batch_size = max_batch_size
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._input_buffers: dict[int, dict[str, torch.Tensor]] = {}
        self._output_buffers: dict[int, torch.Tensor] = {}

    def can_use(self, batch_size: int) -> bool:
        """Check if we have a captured graph for this batch size."""
        return batch_size in self._graphs

    def capture(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Capture a CUDA graph for the given batch size."""
        if not torch.cuda.is_available():
            return

        # Warmup
        with torch.cuda.stream(torch.cuda.Stream()):
            for _ in range(2):
                model(input_ids[:batch_size], positions[:batch_size])

        # Capture
        graph = torch.cuda.CUDAGraph()
        input_buf = {
            "input_ids": input_ids[:batch_size].clone(),
            "positions": positions[:batch_size].clone(),
        }
        with torch.cuda.graph(graph):
            output = model(input_buf["input_ids"], input_buf["positions"])

        self._graphs[batch_size] = graph
        self._input_buffers[batch_size] = input_buf
        self._output_buffers[batch_size] = output

    def replay(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Replay a captured graph with new inputs."""
        buf = self._input_buffers[batch_size]
        buf["input_ids"].copy_(input_ids[:batch_size])
        buf["positions"].copy_(positions[:batch_size])
        self._graphs[batch_size].replay()
        return self._output_buffers[batch_size]


class ModelEngine:
    """Orchestrates model forward pass with optional CUDA graph replay."""

    def __init__(
        self,
        model: nn.Module,
        vocab_size: int,
        device: torch.device,
        enable_cuda_graph: bool = True,
    ):
        self.model = model
        self.vocab_size = vocab_size
        self.device = device
        self.enable_cuda_graph = enable_cuda_graph
        self.graph_runner = CUDAGraphRunner() if enable_cuda_graph else None

    def forward(
        self,
        scheduled: ScheduledRequests,
        kv_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run model forward and return logits for sampling.

        Returns: (num_sampling_tokens, vocab_size) tensor of logits.
        """
        if scheduled.is_empty:
            return torch.empty(0, self.vocab_size, device=self.device)

        input_ids, positions, context_mask = self._prepare_inputs(scheduled)

        # Try CUDA graph replay for decode-only batches
        use_graph = (
            self.enable_cuda_graph
            and self.graph_runner is not None
            and scheduled.can_run_cuda_graph
            and self.graph_runner.can_use(input_ids.shape[0])
        )

        if use_graph:
            logits = self.graph_runner.replay(input_ids, positions, input_ids.shape[0])
        else:
            logits = self.model(input_ids, positions)

        # Extract logits for sampling:
        # - Context requests: take last token logits
        # - Generation requests: take the single token logits
        return self._extract_sampling_logits(logits, scheduled, context_mask)

    def _prepare_inputs(
        self, scheduled: ScheduledRequests
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare flat input_ids and positions from scheduled requests."""
        all_input_ids = []
        all_positions = []
        context_mask = []  # True for context tokens, False for gen tokens

        # Context requests (chunking + last chunk)
        for req in list(scheduled.context_chunking) + list(scheduled.context_last_chunk):
            chunk_size = getattr(req, '_scheduled_context_tokens', req.context_len)
            start = req.num_context_tokens_processed
            chunk_ids = req.token_ids[start : start + chunk_size]
            all_input_ids.extend(chunk_ids)
            all_positions.extend(range(start, start + chunk_size))
            context_mask.extend([True] * chunk_size)

        # Generation requests (1 token each)
        for req in scheduled.generation:
            # Input is the last generated token (or last context token)
            if req.output_token_ids:
                all_input_ids.append(req.output_token_ids[-1])
            else:
                all_input_ids.append(req.token_ids[-1])
            pos = req.context_len + len(req.output_token_ids) - 1
            all_positions.append(max(0, pos))
            context_mask.append(False)

        if not all_input_ids:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty, torch.empty(0, dtype=torch.bool, device=self.device)

        input_ids = torch.tensor(all_input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(all_positions, dtype=torch.long, device=self.device)
        ctx_mask = torch.tensor(context_mask, dtype=torch.bool, device=self.device)
        return input_ids, positions, ctx_mask

    def _extract_sampling_logits(
        self,
        logits: torch.Tensor,
        scheduled: ScheduledRequests,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract only the logits needed for sampling."""
        if logits.dim() == 2:
            # (total_tokens, vocab) — need to pick last-token for context seqs
            sampling_indices = []
            offset = 0

            for req in list(scheduled.context_chunking) + list(scheduled.context_last_chunk):
                chunk_size = getattr(req, '_scheduled_context_tokens', req.context_len)
                # Last token of each context chunk
                sampling_indices.append(offset + chunk_size - 1)
                offset += chunk_size

            for req in scheduled.generation:
                sampling_indices.append(offset)
                offset += 1

            if not sampling_indices:
                return torch.empty(0, self.vocab_size, device=self.device)

            indices = torch.tensor(sampling_indices, dtype=torch.long, device=self.device)
            return logits[indices]

        return logits
