# tiny-trtllm

A ~3,700 LOC minimal reimplementation of [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)'s PyTorch backend, built to teach the architectural patterns that distinguish TRT-LLM from vllm and sglang.

**mini-sglang** exists for sglang (~5K LOC). **nano-vllm** exists for vllm (~1.3K LOC). **tiny-trtllm** fills this gap for TRT-LLM.

## Why TRT-LLM is Different

Most inference engines look similar at a distance, but TRT-LLM has a distinct architecture. This project preserves 16 core differentiators:

| # | Feature | Why It's Unique |
|---|---------|-----------------|
| 1 | **Two-tier scheduling** | CapacityScheduler (admission) + MicroBatchScheduler (chunking) — vllm/sglang have single-tier |
| 2 | **Capacity policies** | `GUARANTEED_NO_EVICT` vs `MAX_UTILIZATION` — no other engine offers this guarantee model |
| 3 | **Chunking policies** | `EQUAL_PROGRESS` (fairness) vs `FCFS` (throughput) |
| 4 | **Iteration-based executor** | Fixed loop: fetch → schedule → forward → sample → respond |
| 5 | **Phase-segmented requests** | 4 disjoint lists: `context_chunking`, `context_last_chunk`, `generation`, `paused` |
| 6 | **Async two-phase sampling** | `sample_async()` → GPU kernels; `update_requests()` → CPU state |
| 7 | **Grouped strategy sampling** | Batch by `(top_k, top_p, temp)` → 1 kernel per group |
| 8 | **Plan/execute attention** | `plan() → PreparedAttention → run()` enables CUDA graph capture |
| 9 | **ResourceManager lifecycle** | `prepare → update → free` pluggable pattern |
| 10 | **Overlap executor** | CPU processes batch N-1 while GPU runs batch N (ping-pong) |
| 11 | **Await-responses** | `threading.Condition` based (vs sglang's ZMQ / vllm's async generators) |
| 12 | **Pydantic config** | Validated config hierarchy with enums |
| 13 | **MoE** | Gate → TopK → fused expert dispatch (single-GPU) |
| 14 | **TRT-LLM attention kernel** | `thop.attention()` via `pip install tensorrt-llm` (zero C++ in our repo) |
| 15 | **Hybrid attention** | Config-driven prefill/decode backend split (e.g., `"trtllm,fi"`) |
| 16 | **Multi-GPU TP** | NCCL wrappers + `spawn_tp_workers` (tested single-GPU, ready for N) |

## What's Removed

TensorRT compilation, speculative decoding, guided decoding, disaggregated serving, pipeline parallelism, LoRA/PEFT, quantization (FP8/INT4/AWQ), beam search, multimodal, expert parallelism, C++ nanobind bindings, 25+ model architectures, weight streaming, health checks/OTLP, sparse attention, MLA.

## Quick Start

### Install

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

### Generate Text (Python API)

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

### Serve with OpenAI-Compatible API

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

### Run Tests

```bash
# All unit tests (CPU, no GPU needed)
pytest tests/ -x -v

# With coverage
pytest tests/ --cov=tinytrtllm --cov-report=term-missing

# GPU integration tests only
pytest tests/ -m gpu -v
```

## Architecture

```
tinytrtllm/
├── config.py                 # Pydantic: TinyLlmArgs, SamplingParams, enums
├── llm.py                    # LLM top-level API (generate, tokenize)
├── engine/
│   ├── request.py            # LlmRequest state machine (4 states)
│   ├── scheduler.py          # Two-tier: CapacityScheduler + MicroBatchScheduler
│   ├── block_manager.py      # Paged KV cache + prefix caching (xxhash)
│   ├── resource_manager.py   # prepare → update → free lifecycle
│   ├── model_engine.py       # Forward pass + CUDA graph runner
│   ├── sampler.py            # Async two-phase + grouped strategies
│   └── executor.py           # PyExecutor + OverlapExecutor + ResponseManager
├── layers/
│   ├── attention.py          # Plan/execute interface + backend registry
│   ├── attention_trtllm.py   # TRT-LLM thop.attention() backend
│   ├── attention_fa.py       # FlashAttention backend
│   ├── attention_fi.py       # FlashInfer backend
│   ├── attention_vanilla.py  # PyTorch SDPA fallback
│   ├── attention_hybrid.py   # Prefill/decode backend dispatcher
│   ├── linear.py             # TP-aware: Column/Row/Merged/QKV parallel
│   ├── embedding.py          # VocabParallelEmbedding + ParallelLMHead
│   ├── norm.py               # RMSNorm with fused residual
│   ├── rotary.py             # Rotary Position Embedding (RoPE)
│   ├── activation.py         # SiLU*Mul gated activation
│   └── moe.py                # MoE gate + TopK + fused expert dispatch
├── models/
│   ├── base.py               # BaseModel + HF safetensors weight loading
│   ├── llama.py              # LlamaForCausalLM
│   ├── qwen3.py              # Qwen3ForCausalLM (QK norms, tied embeddings)
│   └── qwen3_moe.py          # Qwen3MoEForCausalLM (routed + shared experts)
├── distributed/
│   └── tp.py                 # NCCL TP: all_reduce, spawn_tp_workers, IPC buffer
└── serve/
    └── server.py             # FastAPI: /v1/chat/completions, /v1/models, /health
```

## How the Pieces Fit Together

### Request Lifecycle

```
User prompt
  → LLM.generate() tokenizes
    → PyExecutor.enqueue_request()
      → CapacityScheduler: can we admit this request?
        → MicroBatchScheduler: how many context tokens per chunk?
          → ModelEngine.forward(): run model (eager or CUDA graph)
            → Sampler.sample_async(): GPU sampling kernels
              → Sampler.update_requests(): CPU state update
                → ResponseManager.notify(): wake up waiters
                  → LLM.generate() returns GenerateOutput
```

### Two-Tier Scheduling (the signature pattern)

```
               Waiting Queue
                    │
        ┌───────────▼───────────┐
        │   CapacityScheduler   │  Tier 1: Admission control
        │  (memory guarantees)  │  GUARANTEED_NO_EVICT │ MAX_UTILIZATION
        └───────────┬───────────┘
                    │ admitted requests
        ┌───────────▼───────────┐
        │  MicroBatchScheduler  │  Tier 2: Token budgeting
        │  (chunk allocation)   │  EQUAL_PROGRESS │ FCFS
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   ScheduledRequests   │  4 disjoint lists:
        │  context_chunking     │  ├─ still chunking prefill
        │  context_last_chunk   │  ├─ final prefill chunk
        │  generation           │  ├─ decode phase
        │  paused               │  └─ preempted (MAX_UTIL only)
        └───────────────────────┘
```

### Plan/Execute Attention

```
plan(seq_lens, block_offsets, ...) → PreparedAttention
  │
  │  PreparedAttention holds all metadata:
  │  seq_lens, context_lens, slot_mapping, is_context flags
  │
run(q, k, v, PreparedAttention) → output
  │
  │  Why separate? CUDA graphs need fixed metadata.
  │  plan() runs on CPU once, run() replays on GPU.
  │
  └→ Backend dispatch: trtllm │ fa │ fi │ vanilla │ hybrid
```

### Attention Backend Selection

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

## Supported Models

| Model | Architecture Class | Notes |
|-------|-------------------|-------|
| Llama 2/3 | `LlamaForCausalLM` | GQA, SiLU-gated MLP |
| Qwen3 | `Qwen3ForCausalLM` | Optional QK norms, tied embeddings |
| Qwen3-MoE | `Qwen3MoEForCausalLM` | Routed experts + optional shared expert |

Adding a new model: implement the `nn.Module`, call `register_model("ArchName", YourClass)` in the module, and add an import in `models/__init__.py`.

## Stats

| Metric | Value |
|--------|-------|
| Production code | 3,654 LOC |
| Test code | 2,578 LOC |
| Test coverage | 86% |
| Tests | 206 passing |
| C++ files | 0 (Python + Triton only) |
| Python files | 51 |

## License

MIT
