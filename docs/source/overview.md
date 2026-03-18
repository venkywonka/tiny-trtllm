# Overview

**tiny-trtllm** is a ~3,700 LOC minimal reimplementation of [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)'s PyTorch backend, built to teach the architectural patterns that distinguish TRT-LLM from vllm and sglang.

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

## Project Stats

| Metric | Value |
|--------|-------|
| Production code | 3,654 LOC |
| Test code | 2,578 LOC |
| Test coverage | 86% |
| Tests | 206 passing |
| C++ files | 0 (Python + Triton only) |
