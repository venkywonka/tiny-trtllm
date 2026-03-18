"""Tests for OpenAI-compatible FastAPI server."""

import pytest
import torch
import torch.nn as nn

from httpx import ASGITransport, AsyncClient

from tinytrtllm.config import SamplingParams, SchedulingPolicy, ChunkingPolicy
from tinytrtllm.engine.executor import PyExecutor
from tinytrtllm.engine.model_engine import ModelEngine
from tinytrtllm.engine.sampler import Sampler
from tinytrtllm.engine.scheduler import TwoTierScheduler
from tinytrtllm.llm import LLM
from tinytrtllm.serve.server import create_app


class DummyModel(nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, positions=None):
        batch = input_ids.shape[0]
        logits = torch.zeros(batch, self.vocab_size)
        logits[:, 42] = 100.0
        return logits


def _make_llm():
    llm = LLM(model_path="/tmp/fake")
    llm._model = DummyModel(100)
    llm._tokenizer = None
    engine = ModelEngine(llm._model, 100, torch.device("cpu"), enable_cuda_graph=False)
    scheduler = TwoTierScheduler(
        scheduling_policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
        chunking_policy=ChunkingPolicy.EQUAL_PROGRESS,
        max_batch_size=256, max_num_tokens=16384, block_size=256, max_blocks=1000,
    )
    llm._executor = PyExecutor(engine, scheduler, Sampler())
    llm._initialized = True
    return llm


@pytest.fixture
def app():
    return create_app(_make_llm(), model_name="test-model")


@pytest.mark.asyncio
async def test_health(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_list_models(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "test-model"


@pytest.mark.asyncio
async def test_chat_completions_non_streaming(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 3,
            "temperature": 0.0,
            "stream": False,
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_chat_completions_streaming(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 3,
            "temperature": 0.0,
            "stream": True,
        })
    assert resp.status_code == 200
    # SSE format
    text = resp.text
    assert "data:" in text
    assert "[DONE]" in text


@pytest.mark.asyncio
async def test_no_model_returns_503():
    app = create_app(llm=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
        })
    assert resp.status_code == 503


class TestServerBugFixes:
    """Regression tests for server bug fixes."""

    @pytest.mark.asyncio
    async def test_max_tokens_above_limit_rejected(self, app):
        """S4: max_tokens > 4096 should be rejected."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5000,
            })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_unknown_role_returns_error(self, app):
        """S5: Unknown roles should cause an error, not be silently dropped."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "tool", "content": "I am a tool"},
                ],
                "max_tokens": 3,
                "temperature": 0.0,
            })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_sse_chunks_exclude_null_fields(self, app):
        """S6: SSE chunks should not contain null fields."""
        import json
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 3,
                "temperature": 0.0,
                "stream": True,
            })
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"]
                for v in delta.values():
                    assert v is not None, f"Null field in SSE delta: {delta}"

    def test_format_messages_rejects_unknown_role(self):
        """S5: _format_messages should raise on unknown roles."""
        from tinytrtllm.serve.server import _format_messages, ChatMessage
        msgs = [ChatMessage(role="tool", content="data")]
        with pytest.raises(ValueError, match="Unsupported message role"):
            _format_messages(msgs)
