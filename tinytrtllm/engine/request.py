"""LlmRequest state machine — the core request lifecycle."""

from __future__ import annotations

from enum import Enum, auto
from typing import List, Tuple


class RequestState(Enum):
    """States in the LLM request lifecycle."""

    CONTEXT_INIT = auto()
    CONTEXT_IN_PROGRESS = auto()
    GENERATION_IN_PROGRESS = auto()
    GENERATION_COMPLETE = auto()


class LlmRequest:
    """Tracks the full lifecycle of a single inference request.

    State machine:
        CONTEXT_INIT → CONTEXT_IN_PROGRESS → GENERATION_IN_PROGRESS → GENERATION_COMPLETE
                        ↻ (self-loop for chunked prefill)   ↻ (self-loop for decode steps)
    """

    VALID_TRANSITIONS: dict[RequestState, set[RequestState]] = {
        RequestState.CONTEXT_INIT: {RequestState.CONTEXT_IN_PROGRESS},
        RequestState.CONTEXT_IN_PROGRESS: {
            RequestState.CONTEXT_IN_PROGRESS,
            RequestState.GENERATION_IN_PROGRESS,
        },
        RequestState.GENERATION_IN_PROGRESS: {
            RequestState.GENERATION_IN_PROGRESS,
            RequestState.GENERATION_COMPLETE,
        },
        RequestState.GENERATION_COMPLETE: set(),
    }

    def __init__(
        self,
        request_id: int,
        token_ids: List[int],
        max_tokens: int,
        context_chunk_size: int = 4096,
    ) -> None:
        self.request_id = request_id
        self.token_ids = token_ids
        self.max_tokens = max_tokens
        self.context_chunk_size = context_chunk_size

        self.state = RequestState.CONTEXT_INIT
        self.output_token_ids: List[int] = []
        self.num_context_tokens_processed: int = 0
        self.block_table: List[int] = []

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def transition_to(self, new_state: RequestState) -> None:
        """Advance the request to *new_state*, raising on invalid moves."""
        if new_state not in self.VALID_TRANSITIONS[self.state]:
            raise ValueError(
                f"Invalid transition: {self.state.name} → {new_state.name}"
            )
        self.state = new_state

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    @property
    def is_context_phase(self) -> bool:
        return self.state in (
            RequestState.CONTEXT_INIT,
            RequestState.CONTEXT_IN_PROGRESS,
        )

    @property
    def is_generation_phase(self) -> bool:
        return self.state in (
            RequestState.GENERATION_IN_PROGRESS,
            RequestState.GENERATION_COMPLETE,
        )

    # ------------------------------------------------------------------
    # Token bookkeeping
    # ------------------------------------------------------------------

    @property
    def context_len(self) -> int:
        return len(self.token_ids)

    @property
    def num_tokens_remaining(self) -> int:
        return self.max_tokens - len(self.output_token_ids)

    @property
    def total_num_tokens(self) -> int:
        return self.context_len + len(self.output_token_ids)

    def append_output_token(self, token_id: int) -> None:
        """Record a newly generated token."""
        self.output_token_ids.append(token_id)

    # ------------------------------------------------------------------
    # Chunked prefill
    # ------------------------------------------------------------------

    def get_context_chunk(self) -> Tuple[List[int], bool]:
        """Return the next chunk of context tokens and whether it is the last.

        Returns:
            (chunk_token_ids, is_last_chunk)
        """
        start = self.num_context_tokens_processed
        end = min(start + self.context_chunk_size, self.context_len)
        chunk = self.token_ids[start:end]
        self.num_context_tokens_processed = end
        is_last = end >= self.context_len
        return chunk, is_last
