"""End-to-end server tests: startup, streaming, concurrent requests."""

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


def _make_test_app():
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
    return create_app(llm, model_name="test-model")


@pytest.mark.asyncio
async def test_server_startup():
    app = _make_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_streaming_sse_format():
    app = _make_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 3,
            "temperature": 0.0,
            "stream": True,
        })
    assert resp.status_code == 200
    lines = resp.text.strip().split("\n")
    data_lines = [l for l in lines if l.startswith("data:")]
    assert len(data_lines) >= 2  # At least role chunk + done
    assert any("[DONE]" in l for l in data_lines)


@pytest.mark.asyncio
async def test_non_streaming_json():
    app = _make_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 3,
            "temperature": 0.0,
            "stream": False,
        })
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["usage"]["completion_tokens"] > 0


@pytest.mark.asyncio
async def test_concurrent_requests():
    """Multiple concurrent requests should all complete."""
    import asyncio
    app = _make_test_app()

    async def make_request(ac, idx):
        resp = await ac.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": f"Request {idx}"}],
            "max_tokens": 2,
            "temperature": 0.0,
            "stream": False,
        })
        return resp.status_code

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        tasks = [make_request(ac, i) for i in range(5)]
        results = await asyncio.gather(*tasks)

    assert all(status == 200 for status in results)
