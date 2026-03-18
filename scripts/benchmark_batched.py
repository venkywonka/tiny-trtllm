#!/usr/bin/env python3
"""Benchmark: batched tiny-trtllm vs serial vs HF baseline.

Measures the actual speedup from KV cache + batching.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 32


def hf_greedy_batch(model, tokenizer, prompts, max_tokens):
    """HF baseline: serial greedy decode."""
    total_tokens = 0
    start = time.perf_counter()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                temperature=None, top_p=None,
            )
        total_tokens += out.shape[1] - input_len
    elapsed = time.perf_counter() - start
    return total_tokens, elapsed


def tinytrtllm_serial(model, tokenizer, prompts, max_tokens):
    """tiny-trtllm serial: one request at a time with KV cache."""
    from tinytrtllm.engine.hf_adapter import HFKVCacheAdapter

    total_tokens = 0
    start = time.perf_counter()
    for prompt in prompts:
        adapter = HFKVCacheAdapter(model)
        token_ids = tokenizer.encode(prompt)
        logits = adapter.prefill(token_ids)
        next_token = logits[-1].argmax().item()
        generated = [next_token]
        for _ in range(max_tokens - 1):
            logits = adapter.decode([next_token])
            next_token = logits[0].argmax().item()
            generated.append(next_token)
        total_tokens += len(generated)
    elapsed = time.perf_counter() - start
    return total_tokens, elapsed


def tinytrtllm_batched(model, tokenizer, prompts, max_tokens):
    """tiny-trtllm batched: all requests in one batch with KV cache."""
    from tinytrtllm.engine.hf_adapter import BatchedHFModel

    token_id_lists = [tokenizer.encode(p) for p in prompts]
    start = time.perf_counter()

    bm = BatchedHFModel(model)
    logits_list = bm.prefill_batch(token_id_lists)

    # Greedy decode loop
    batch_tokens = [[l.argmax().item()] for l in logits_list]
    for _ in range(max_tokens - 1):
        next_toks = [bt[-1] for bt in batch_tokens]
        dl = bm.decode_batch(next_toks)
        for i, l in enumerate(dl):
            batch_tokens[i].append(l.argmax().item())

    total_tokens = sum(len(bt) for bt in batch_tokens)
    elapsed = time.perf_counter() - start
    return total_tokens, elapsed


def tinytrtllm_executor(model, tokenizer, prompts, max_tokens):
    """tiny-trtllm BatchedExecutor: full engine pipeline."""
    from tinytrtllm.engine.batched_executor import create_batched_executor

    start = time.perf_counter()
    executor = create_batched_executor(model, tokenizer, max_tokens=max_tokens)
    results = executor.generate_batch(prompts)
    total_tokens = sum(len(v) for v in results.values())
    elapsed = time.perf_counter() - start
    return total_tokens, elapsed


if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"Batched Inference Benchmark")
    print(f"{'='*70}")
    print(f"Model: {MODEL_ID}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    # Warmup
    inp = tokenizer("Hi", return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**inp, max_new_tokens=5, do_sample=False, temperature=None, top_p=None)

    for num_requests in [1, 4, 8, 16]:
        prompts = [f"Question {i}: explain the concept of machine learning in detail." for i in range(num_requests)]

        print(f"\n--- {num_requests} requests, {MAX_NEW_TOKENS} tokens each ---")

        # HF serial baseline
        tok, t = hf_greedy_batch(model, tokenizer, prompts, MAX_NEW_TOKENS)
        hf_tps = tok / t
        print(f"  HF serial:           {tok} tok in {t:.2f}s = {hf_tps:.0f} tok/s")

        # tiny-trtllm serial (KV cache, no batching)
        tok, t = tinytrtllm_serial(model, tokenizer, prompts, MAX_NEW_TOKENS)
        serial_tps = tok / t
        print(f"  tiny serial+KV:      {tok} tok in {t:.2f}s = {serial_tps:.0f} tok/s ({serial_tps/hf_tps:.2f}x vs HF)")

        # tiny-trtllm batched (KV cache + batching)
        tok, t = tinytrtllm_batched(model, tokenizer, prompts, MAX_NEW_TOKENS)
        batch_tps = tok / t
        print(f"  tiny batched+KV:     {tok} tok in {t:.2f}s = {batch_tps:.0f} tok/s ({batch_tps/hf_tps:.2f}x vs HF)")

        # tiny-trtllm full executor
        tok, t = tinytrtllm_executor(model, tokenizer, prompts, MAX_NEW_TOKENS)
        exec_tps = tok / t
        print(f"  tiny executor:       {tok} tok in {t:.2f}s = {exec_tps:.0f} tok/s ({exec_tps/hf_tps:.2f}x vs HF)")

    # Correctness spot-check
    print(f"\n--- Correctness Check ---")
    from tinytrtllm.engine.hf_adapter import BatchedHFModel
    prompts = ["The capital of France is", "def fibonacci(n):"]
    token_id_lists = [tokenizer.encode(p) for p in prompts]

    hf_results = []
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt").to("cuda")
        il = inp.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=16, do_sample=False, temperature=None, top_p=None)
        hf_results.append(out[0][il:].tolist())

    bm = BatchedHFModel(model)
    logits_list = bm.prefill_batch(token_id_lists)
    batch_tokens = [[l.argmax().item()] for l in logits_list]
    for _ in range(15):
        dl = bm.decode_batch([bt[-1] for bt in batch_tokens])
        for i, l in enumerate(dl):
            batch_tokens[i].append(l.argmax().item())

    for i, prompt in enumerate(prompts):
        match = batch_tokens[i] == hf_results[i]
        print(f"  '{prompt[:35]}': {'MATCH' if match else 'MISMATCH'}")
        if not match:
            print(f"    HF:    {hf_results[i][:8]}...")
            print(f"    Batch: {batch_tokens[i][:8]}...")
