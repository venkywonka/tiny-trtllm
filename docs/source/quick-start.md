# Quick Start Guide

## Install

```bash
# Core (vanilla attention backend — works everywhere)
pip install torch>=2.4.0 pydantic>=2.0 transformers>=4.51.0 \
    xxhash>=3.0.0 fastapi>=0.110.0 uvicorn>=0.29.0 safetensors>=0.4.0

# Optional: TRT-LLM attention kernel (recommended for performance)
pip install tensorrt-llm

# Optional: FlashAttention backend
pip install flash-attn>=2.5.0

# Dev dependencies
pip install pytest pytest-cov pytest-asyncio httpx ruff
```

## Generate Text (Python API)

```python
from tinytrtllm.llm import LLM
from tinytrtllm.config import SamplingParams

llm = LLM("Qwen/Qwen3-0.6B")
outputs = llm.generate(
    ["Explain transformers in one sentence."],
    SamplingParams(temperature=0.0, max_tokens=64),
)
print(outputs[0].text)
```

## Serve with OpenAI-Compatible API

```python
from tinytrtllm.serve.server import run_server

run_server("Qwen/Qwen3-0.6B", host="0.0.0.0", port=8000)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64,
    "stream": true
  }'
```

## Server Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check → `{"status": "ok"}` |
| GET | `/v1/models` | List loaded models |
| POST | `/v1/chat/completions` | Chat completion (streaming SSE or JSON) |

## Run Tests

```bash
# All unit tests (CPU, no GPU needed)
pytest tests/ -x -v

# With coverage
pytest tests/ --cov=tinytrtllm --cov-report=term-missing

# GPU integration tests only
pytest tests/ -m gpu -v
```

## Attention Backend Selection

```python
# Auto-detect (default): trtllm → fa → vanilla
args = TinyLlmArgs(model_path="...", attn_backend="auto")

# Force specific backend
args = TinyLlmArgs(model_path="...", attn_backend="vanilla")

# Hybrid: TRT-LLM for prefill, FlashInfer for decode
args = TinyLlmArgs(model_path="...", attn_backend="trtllm,fi")
```

Availability at import time:
- `tensorrt-llm` installed → `"trtllm"` registered
- `flash-attn` installed → `"fa"` registered
- `flashinfer` installed → `"fi"` registered
- Always available → `"vanilla"` (PyTorch SDPA)
