"""Tests for LLM top-level API."""

import pytest
import torch
import torch.nn as nn

from unittest.mock import patch, MagicMock

from tinytrtllm.config import SamplingParams
from tinytrtllm.llm import LLM, GenerateOutput


class DummyModel(nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, positions=None):
        batch = input_ids.shape[0]
        logits = torch.zeros(batch, self.vocab_size)
        logits[:, 42] = 100.0
        return logits


class TestLLMAPI:
    def _make_llm(self):
        """Create an LLM with a dummy model (no real weights needed)."""
        llm = LLM(model_path="/tmp/fake")
        # Manually initialize with dummy model
        llm._model = DummyModel(100)
        llm._tokenizer = None

        from tinytrtllm.engine.model_engine import ModelEngine
        from tinytrtllm.engine.executor import PyExecutor
        from tinytrtllm.engine.sampler import Sampler
        from tinytrtllm.engine.scheduler import TwoTierScheduler
        from tinytrtllm.config import SchedulingPolicy, ChunkingPolicy

        engine = ModelEngine(llm._model, 100, torch.device("cpu"), enable_cuda_graph=False)
        scheduler = TwoTierScheduler(
            scheduling_policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
            chunking_policy=ChunkingPolicy.EQUAL_PROGRESS,
            max_batch_size=256,
            max_num_tokens=16384,
            block_size=256,
            max_blocks=1000,
        )
        llm._executor = PyExecutor(engine, scheduler, Sampler())
        llm._initialized = True
        return llm

    def test_generate_returns_outputs(self):
        llm = self._make_llm()
        outputs = llm.generate(
            [[1, 2, 3]],  # token IDs directly
            SamplingParams(temperature=0.0, max_tokens=3),
        )
        assert len(outputs) == 1
        assert isinstance(outputs[0], GenerateOutput)
        assert len(outputs[0].token_ids) == 3
        assert all(t == 42 for t in outputs[0].token_ids)  # greedy → token 42

    def test_batch_generate(self):
        llm = self._make_llm()
        outputs = llm.generate(
            [[1, 2], [3, 4], [5, 6]],
            SamplingParams(temperature=0.0, max_tokens=2),
        )
        assert len(outputs) == 3
        for out in outputs:
            assert len(out.token_ids) == 2

    def test_is_finished(self):
        llm = self._make_llm()
        assert llm.is_finished is True

    def test_generate_output_repr(self):
        out = GenerateOutput(request_id=0, prompt="hi", text="hello", token_ids=[1, 2])
        assert "hello" in repr(out)
