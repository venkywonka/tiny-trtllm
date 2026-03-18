"""Tests for async two-phase sampler with grouped strategies."""

import pytest
import torch

from tinytrtllm.config import SamplingParams
from tinytrtllm.engine.request import LlmRequest, RequestState
from tinytrtllm.engine.sampler import SampleState, Sampler


def _make_gen_request(request_id=0, max_tokens=10):
    req = LlmRequest(request_id=request_id, token_ids=[1, 2, 3], max_tokens=max_tokens)
    req.transition_to(RequestState.CONTEXT_IN_PROGRESS)
    req.transition_to(RequestState.GENERATION_IN_PROGRESS)
    return req


class TestSampleState:
    def test_creation(self):
        ss = SampleState(
            token_ids=torch.tensor([1, 2, 3]),
            request_ids=[0, 1, 2],
            logits=torch.randn(3, 100),
        )
        assert ss.token_ids.shape == (3,)
        assert len(ss.request_ids) == 3


class TestSampler:
    def test_sample_async_returns_state(self):
        sampler = Sampler()
        logits = torch.randn(2, 100)
        reqs = [_make_gen_request(i) for i in range(2)]
        params = {i: SamplingParams() for i in range(2)}

        state = sampler.sample_async(logits, reqs, params)
        assert isinstance(state, SampleState)
        assert state.token_ids.shape == (2,)

    def test_update_requests_appends_tokens(self):
        sampler = Sampler()
        req = _make_gen_request(0, max_tokens=10)
        state = SampleState(
            token_ids=torch.tensor([42]),
            request_ids=[0],
            logits=torch.randn(1, 100),
        )
        sampler.update_requests(state, [req])
        assert req.output_token_ids == [42]

    def test_greedy_returns_argmax(self):
        sampler = Sampler()
        # Create logits where token 5 is clearly the max
        logits = torch.zeros(1, 100)
        logits[0, 5] = 100.0
        req = _make_gen_request(0)
        params = {0: SamplingParams(temperature=0.0)}

        state = sampler.sample_async(logits, [req], params)
        assert state.token_ids[0].item() == 5

    def test_temperature_scaling(self):
        sampler = Sampler()
        logits = torch.randn(1, 100)
        req = _make_gen_request(0)

        # High temp → more uniform distribution (just check it runs)
        params = {0: SamplingParams(temperature=2.0)}
        state = sampler.sample_async(logits, [req], params)
        assert state.token_ids.shape == (1,)

    def test_top_k_filtering(self):
        sampler = Sampler()
        logits = torch.randn(1, 100)
        req = _make_gen_request(0)
        params = {0: SamplingParams(top_k=5, temperature=0.5)}

        state = sampler.sample_async(logits, [req], params)
        assert state.token_ids.shape == (1,)

    def test_top_p_filtering(self):
        sampler = Sampler()
        logits = torch.randn(1, 100)
        req = _make_gen_request(0)
        params = {0: SamplingParams(top_p=0.9, temperature=0.5)}

        state = sampler.sample_async(logits, [req], params)
        assert state.token_ids.shape == (1,)

    def test_grouped_by_strategy(self):
        sampler = Sampler()
        logits = torch.randn(3, 100)
        reqs = [_make_gen_request(i) for i in range(3)]
        params = {
            0: SamplingParams(temperature=0.0),  # greedy group
            1: SamplingParams(temperature=0.0),  # same group
            2: SamplingParams(temperature=1.0, top_k=10),  # different group
        }

        groups = sampler._group_by_strategy(reqs, params)
        assert len(groups) == 2  # Two distinct groups

    def test_sample_async_empty(self):
        sampler = Sampler()
        logits = torch.empty(0, 100)
        state = sampler.sample_async(logits, [], {})
        assert state.token_ids.shape == (0,)

    def test_completion_detection(self):
        sampler = Sampler()
        req = _make_gen_request(0, max_tokens=1)  # Only 1 token left
        state = SampleState(
            token_ids=torch.tensor([42]),
            request_ids=[0],
            logits=torch.randn(1, 100),
        )
        completed = sampler.update_requests(state, [req])
        assert 0 in completed
        assert req.state == RequestState.GENERATION_COMPLETE
