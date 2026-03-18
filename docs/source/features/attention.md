# Plan/Execute Attention

The separation of `plan()` and `run()` is what enables CUDA graph capture. The plan computes all metadata on CPU once; the run replays on GPU.

## Interface

```{mermaid}
graph LR
    P["plan(seq_lens, block_offsets, slot_mapping)"] --> PA[PreparedAttention]
    PA --> R["run(q, k, v, PreparedAttention)"]
    R --> O[output tensor]
```

### PreparedAttention

The `plan()` call produces a `PreparedAttention` dataclass containing:

| Field | Shape | Description |
|-------|-------|-------------|
| `seq_lens` | `(batch,)` | Sequence lengths |
| `context_lens` | `(batch,)` | Context (cached) lengths |
| `block_offsets` | `(batch, max_blocks_per_seq)` | Paged KV cache block table |
| `slot_mapping` | `(total_tokens,)` | Token → KV cache slot mapping |
| `is_context` | `(batch,)` | Whether each request is in prefill |

### Why Separate?

CUDA graphs require fixed tensor shapes and addresses. By computing all dynamic metadata in `plan()` on CPU, the `run()` call can be captured as a CUDA graph and replayed without re-planning.

## Attention Backends

Five backends, one interface:

| Backend | Module | Requires | Best For |
|---------|--------|----------|----------|
| **trtllm** | `attention_trtllm.py` | `pip install tensorrt-llm` | Best overall performance |
| **fa** | `attention_fa.py` | `pip install flash-attn` | Good prefill performance |
| **fi** | `attention_fi.py` | `pip install flashinfer` | Good decode performance |
| **vanilla** | `attention_vanilla.py` | (always available) | Debugging, CPU fallback |
| **hybrid** | `attention_hybrid.py` | Two backends installed | Mix best prefill + decode |

## Backend Selection

```python
from tinytrtllm.config import TinyLlmArgs

# Auto-detect best available: trtllm → fa → vanilla
args = TinyLlmArgs(model_path="...", attn_backend="auto")

# Force vanilla (always works, slow)
args = TinyLlmArgs(model_path="...", attn_backend="vanilla")

# Hybrid: TRT-LLM prefill + FlashInfer decode
args = TinyLlmArgs(model_path="...", attn_backend="trtllm,fi")
```

## Hybrid Attention

The hybrid backend dispatches prefill tokens to one backend and decode tokens to another. This is configured via a comma-separated string:

```python
# Format: "prefill_backend,decode_backend"
attn_backend = "trtllm,fi"  # TRT-LLM for prefill, FlashInfer for decode
attn_backend = "fa,fi"      # FlashAttention for prefill, FlashInfer for decode
```

This is unique to TRT-LLM — other engines use a single attention kernel for both phases.
