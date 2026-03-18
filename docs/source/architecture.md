# Architecture Overview

tiny-trtllm mirrors TRT-LLM's real module layout. Each component maps 1:1 to its TRT-LLM counterpart.

## Module Layout

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

## Request Lifecycle

```{mermaid}
graph TD
    A[User prompt] --> B["LLM.generate() tokenizes"]
    B --> C["PyExecutor.enqueue_request()"]
    C --> D["CapacityScheduler: can we admit?"]
    D --> E["MicroBatchScheduler: how many tokens per chunk?"]
    E --> F["ModelEngine.forward(): run model"]
    F --> G["Sampler.sample_async(): GPU kernels"]
    G --> H["Sampler.update_requests(): CPU state"]
    H --> I["ResponseManager.notify(): wake waiters"]
    I --> J["LLM.generate() returns GenerateOutput"]
```

## How the Pieces Fit Together

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

## Module Categories

### Engine (Scheduling & Execution)

The engine layer handles request management, scheduling, memory management, and the main execution loop. This is where TRT-LLM's most distinctive patterns live — the two-tier scheduler, the async two-phase sampler, and the overlap executor.

### Layers (Neural Network Primitives)

Custom attention, linear, normalization, and activation layers that support tensor parallelism and the plan/execute pattern. The attention backend registry enables swapping between TRT-LLM kernels, FlashAttention, FlashInfer, and vanilla PyTorch SDPA.

### Models (Architecture Implementations)

Llama, Qwen3, and Qwen3-MoE implementations using the layer primitives. Each model registers itself via `register_model()` and loads HuggingFace safetensors weights.

### Infrastructure

Pydantic configuration hierarchy, NCCL-based tensor parallelism, and a FastAPI OpenAI-compatible server.
