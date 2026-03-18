"""Tests for LlmRequest state machine — the core request lifecycle."""

import pytest
from tinytrtllm.engine.request import LlmRequest, RequestState


class TestRequestState:
    def test_states_exist(self):
        assert RequestState.CONTEXT_INIT is not None
        assert RequestState.CONTEXT_IN_PROGRESS is not None
        assert RequestState.GENERATION_IN_PROGRESS is not None
        assert RequestState.GENERATION_COMPLETE is not None


class TestLlmRequest:
    def test_initial_state(self):
        req = LlmRequest(request_id=0, token_ids=[1, 2, 3], max_tokens=10)
        assert req.state == RequestState.CONTEXT_INIT

    def test_valid_transition_context_init_to_in_progress(self):
        req = LlmRequest(request_id=0, token_ids=[1, 2, 3], max_tokens=10)
        req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
        assert req.state == RequestState.CONTEXT_IN_PROGRESS

    def test_context_in_progress_loops(self):
        """Chunked prefill: CONTEXT_IN_PROGRESS can loop back to itself."""
        req = LlmRequest(request_id=0, token_ids=list(range(100)), max_tokens=10)
        req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
        req.transition_to(RequestState.CONTEXT_IN_PROGRESS)  # Loop
        assert req.state == RequestState.CONTEXT_IN_PROGRESS

    def test_valid_transition_to_generation(self):
        req = LlmRequest(request_id=0, token_ids=[1, 2, 3], max_tokens=10)
        req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
        req.transition_to(RequestState.GENERATION_IN_PROGRESS)
        assert req.state == RequestState.GENERATION_IN_PROGRESS

    def test_valid_transition_to_complete(self):
        req = LlmRequest(request_id=0, token_ids=[1, 2, 3], max_tokens=10)
        req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
        req.transition_to(RequestState.GENERATION_IN_PROGRESS)
        req.transition_to(RequestState.GENERATION_COMPLETE)
        assert req.state == RequestState.GENERATION_COMPLETE

    def test_invalid_transition_raises(self):
        req = LlmRequest(request_id=0, token_ids=[1, 2, 3], max_tokens=10)
        with pytest.raises(ValueError):
            req.transition_to(RequestState.GENERATION_COMPLETE)

    def test_invalid_backward_transition(self):
        req = LlmRequest(request_id=0, token_ids=[1, 2, 3], max_tokens=10)
        req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
        req.transition_to(RequestState.GENERATION_IN_PROGRESS)
        with pytest.raises(ValueError):
            req.transition_to(RequestState.CONTEXT_IN_PROGRESS)

    def test_num_tokens_remaining(self):
        req = LlmRequest(request_id=0, token_ids=[1, 2, 3], max_tokens=10)
        assert req.num_tokens_remaining == 10

    def test_num_tokens_remaining_decreases(self):
        req = LlmRequest(request_id=0, token_ids=[1, 2, 3], max_tokens=10)
        req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
        req.transition_to(RequestState.GENERATION_IN_PROGRESS)
        req.append_output_token(100)
        assert req.num_tokens_remaining == 9

    def test_is_context_phase(self):
        req = LlmRequest(request_id=0, token_ids=[1, 2, 3], max_tokens=10)
        assert req.is_context_phase is True
        req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
        assert req.is_context_phase is True

    def test_is_generation_phase(self):
        req = LlmRequest(request_id=0, token_ids=[1, 2, 3], max_tokens=10)
        assert req.is_generation_phase is False
        req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
        req.transition_to(RequestState.GENERATION_IN_PROGRESS)
        assert req.is_generation_phase is True

    def test_append_output_token(self):
        req = LlmRequest(request_id=0, token_ids=[1, 2, 3], max_tokens=10)
        req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
        req.transition_to(RequestState.GENERATION_IN_PROGRESS)
        req.append_output_token(42)
        assert req.output_token_ids == [42]
        req.append_output_token(43)
        assert req.output_token_ids == [42, 43]

    def test_get_context_chunk_small(self):
        """When all tokens fit in one chunk, is_last_chunk is True."""
        req = LlmRequest(request_id=0, token_ids=[1, 2, 3], max_tokens=10, context_chunk_size=64)
        chunk_ids, is_last = req.get_context_chunk()
        assert chunk_ids == [1, 2, 3]
        assert is_last is True

    def test_get_context_chunk_large(self):
        """Chunked prefill: multiple chunks needed."""
        tokens = list(range(10))
        req = LlmRequest(request_id=0, token_ids=tokens, max_tokens=5, context_chunk_size=4)

        chunk1, is_last1 = req.get_context_chunk()
        assert len(chunk1) == 4
        assert is_last1 is False
        assert req.num_context_tokens_processed == 4

        chunk2, is_last2 = req.get_context_chunk()
        assert len(chunk2) == 4
        assert is_last2 is False
        assert req.num_context_tokens_processed == 8

        chunk3, is_last3 = req.get_context_chunk()
        assert len(chunk3) == 2
        assert is_last3 is True
        assert req.num_context_tokens_processed == 10

    def test_context_len(self):
        req = LlmRequest(request_id=0, token_ids=[1, 2, 3, 4, 5], max_tokens=10)
        assert req.context_len == 5

    def test_total_tokens(self):
        req = LlmRequest(request_id=0, token_ids=[1, 2, 3], max_tokens=10)
        req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
        req.transition_to(RequestState.GENERATION_IN_PROGRESS)
        req.append_output_token(42)
        req.append_output_token(43)
        assert req.total_num_tokens == 5  # 3 context + 2 output

    def test_block_table(self):
        req = LlmRequest(request_id=0, token_ids=[1, 2, 3], max_tokens=10)
        assert req.block_table == []
        req.block_table = [0, 1, 2]
        assert req.block_table == [0, 1, 2]
