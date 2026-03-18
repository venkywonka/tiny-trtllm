"""TDD tests for batched inference with KV cache.

These tests define the expected behavior of the optimized engine:
1. KV cache: decode with 1 token per step (not full sequence replay)
2. Batching: multiple requests processed in one forward pass
3. Correctness: batched output == serial output == HF output
"""

import pytest
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load Qwen3-0.6B once for the whole module."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model_id = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def hf_greedy(model, tokenizer, prompt, max_tokens=16):
    """HF ground truth: greedy decode."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            temperature=None, top_p=None,
        )
    return out[0][input_len:].tolist()


# ---------------------------------------------------------------------------
# RED: KV-cache backed HF adapter
# ---------------------------------------------------------------------------

class TestHFKVCacheAdapter:
    """The adapter must use HF's use_cache=True for efficient decode."""

    def test_single_request_matches_hf(self, model_and_tokenizer):
        """Single request through KV-cache adapter must match HF greedy."""
        model, tokenizer = model_and_tokenizer
        from tinytrtllm.engine.hf_adapter import HFKVCacheAdapter

        adapter = HFKVCacheAdapter(model)
        prompt = "The capital of France is"
        token_ids = tokenizer.encode(prompt)

        # Prefill
        logits = adapter.prefill(token_ids)
        assert logits.shape == (len(token_ids), model.config.vocab_size)

        # Greedy decode 16 tokens
        generated = []
        next_token = logits[-1].argmax().item()
        generated.append(next_token)
        for _ in range(15):
            logits = adapter.decode([next_token])
            next_token = logits[0].argmax().item()
            generated.append(next_token)

        expected = hf_greedy(model, tokenizer, prompt, 16)
        assert generated == expected, f"\nGot:      {generated}\nExpected: {expected}"

    def test_multiple_requests_independent(self, model_and_tokenizer):
        """Two requests processed independently produce correct results."""
        model, tokenizer = model_and_tokenizer
        from tinytrtllm.engine.hf_adapter import HFKVCacheAdapter

        prompts = ["The capital of France is", "def fibonacci(n):"]
        expected = [hf_greedy(model, tokenizer, p, 8) for p in prompts]

        for i, prompt in enumerate(prompts):
            adapter = HFKVCacheAdapter(model)
            token_ids = tokenizer.encode(prompt)
            logits = adapter.prefill(token_ids)
            generated = []
            next_token = logits[-1].argmax().item()
            generated.append(next_token)
            for _ in range(7):
                logits = adapter.decode([next_token])
                next_token = logits[0].argmax().item()
                generated.append(next_token)
            assert generated == expected[i], f"Prompt {i} mismatch"

    def test_reset_clears_kv_cache(self, model_and_tokenizer):
        """After reset, adapter should behave as fresh."""
        model, tokenizer = model_and_tokenizer
        from tinytrtllm.engine.hf_adapter import HFKVCacheAdapter

        adapter = HFKVCacheAdapter(model)
        token_ids = tokenizer.encode("Hello")
        adapter.prefill(token_ids)
        adapter.reset()
        # Should be able to prefill a new sequence
        token_ids2 = tokenizer.encode("World")
        logits = adapter.prefill(token_ids2)
        assert logits.shape[0] == len(token_ids2)


# ---------------------------------------------------------------------------
# RED: Batched model wrapper that handles multiple sequences
# ---------------------------------------------------------------------------

class TestBatchedModelWrapper:
    """Wrapper that processes multiple requests in one forward pass."""

    def test_batched_prefill(self, model_and_tokenizer):
        """Prefill multiple sequences in one call."""
        model, tokenizer = model_and_tokenizer
        from tinytrtllm.engine.hf_adapter import BatchedHFModel

        bm = BatchedHFModel(model)
        prompts = ["The capital of France is", "def fibonacci(n):"]
        token_id_lists = [tokenizer.encode(p) for p in prompts]

        # Prefill returns per-request last-token logits
        logits_list = bm.prefill_batch(token_id_lists)
        assert len(logits_list) == 2
        for logits in logits_list:
            assert logits.shape == (model.config.vocab_size,)

    def test_batched_decode(self, model_and_tokenizer):
        """Decode multiple sequences in one call."""
        model, tokenizer = model_and_tokenizer
        from tinytrtllm.engine.hf_adapter import BatchedHFModel

        bm = BatchedHFModel(model)
        prompts = ["The capital of France is", "def fibonacci(n):"]
        token_id_lists = [tokenizer.encode(p) for p in prompts]

        # Prefill
        logits_list = bm.prefill_batch(token_id_lists)
        next_tokens = [l.argmax().item() for l in logits_list]

        # Decode 1 step
        decode_logits = bm.decode_batch(next_tokens)
        assert len(decode_logits) == 2
        for logits in decode_logits:
            assert logits.shape == (model.config.vocab_size,)

    def test_batched_matches_serial(self, model_and_tokenizer):
        """Batched output must exactly match serial (per-request) output."""
        model, tokenizer = model_and_tokenizer
        from tinytrtllm.engine.hf_adapter import BatchedHFModel

        prompts = ["The capital of France is", "def fibonacci(n):"]
        token_id_lists = [tokenizer.encode(p) for p in prompts]
        max_tokens = 8

        # Serial: one request at a time
        serial_results = []
        for token_ids in token_id_lists:
            bm = BatchedHFModel(model)
            logits_list = bm.prefill_batch([token_ids])
            generated = [logits_list[0].argmax().item()]
            for _ in range(max_tokens - 1):
                dl = bm.decode_batch([generated[-1]])
                generated.append(dl[0].argmax().item())
            serial_results.append(generated)

        # Batched: both requests together
        bm = BatchedHFModel(model)
        logits_list = bm.prefill_batch(token_id_lists)
        batch_tokens = [[l.argmax().item()] for l in logits_list]
        for _ in range(max_tokens - 1):
            next_toks = [bt[-1] for bt in batch_tokens]
            dl = bm.decode_batch(next_toks)
            for i, l in enumerate(dl):
                batch_tokens[i].append(l.argmax().item())

        for i in range(len(prompts)):
            assert batch_tokens[i] == serial_results[i], \
                f"Prompt {i}: batched {batch_tokens[i]} != serial {serial_results[i]}"

    def test_batched_matches_hf(self, model_and_tokenizer):
        """Batched output must match HF model.generate() greedy."""
        model, tokenizer = model_and_tokenizer
        from tinytrtllm.engine.hf_adapter import BatchedHFModel

        prompts = [
            "The capital of France is",
            "def fibonacci(n):",
            "Explain gravity in one sentence:",
        ]
        max_tokens = 16

        # HF ground truth
        hf_results = [hf_greedy(model, tokenizer, p, max_tokens) for p in prompts]

        # Batched inference
        bm = BatchedHFModel(model)
        token_id_lists = [tokenizer.encode(p) for p in prompts]
        logits_list = bm.prefill_batch(token_id_lists)
        batch_tokens = [[l.argmax().item()] for l in logits_list]
        for _ in range(max_tokens - 1):
            next_toks = [bt[-1] for bt in batch_tokens]
            dl = bm.decode_batch(next_toks)
            for i, l in enumerate(dl):
                batch_tokens[i].append(l.argmax().item())

        for i, prompt in enumerate(prompts):
            assert batch_tokens[i] == hf_results[i], \
                f"Prompt '{prompt[:30]}': got {batch_tokens[i][:5]}... expected {hf_results[i][:5]}..."


# ---------------------------------------------------------------------------
# RED: Full engine integration — batched executor with KV cache
# ---------------------------------------------------------------------------

class TestBatchedExecutor:
    """Full pipeline: scheduler → batched model → sampler → executor."""

    def test_concurrent_requests_correct(self, model_and_tokenizer):
        """Multiple requests enqueued simultaneously all produce correct output."""
        model, tokenizer = model_and_tokenizer
        from tinytrtllm.engine.batched_executor import create_batched_executor

        prompts = [
            "The capital of France is",
            "def fibonacci(n):",
            "Explain gravity in one sentence:",
        ]
        max_tokens = 16

        # HF ground truth
        hf_results = {p: hf_greedy(model, tokenizer, p, max_tokens) for p in prompts}

        # All requests enqueued at once → continuous batching
        executor = create_batched_executor(model, tokenizer, max_tokens=max_tokens)
        results = executor.generate_batch(prompts)

        for prompt in prompts:
            assert results[prompt] == hf_results[prompt], \
                f"'{prompt[:30]}': got {results[prompt][:5]}... expected {hf_results[prompt][:5]}..."

    def test_throughput_higher_than_serial(self, model_and_tokenizer):
        """Batched execution must be faster than serial."""
        import time
        model, tokenizer = model_and_tokenizer
        from tinytrtllm.engine.batched_executor import create_batched_executor

        prompts = [f"Question {i}: explain" for i in range(8)]
        max_tokens = 16

        # Serial
        t0 = time.perf_counter()
        for p in prompts:
            executor = create_batched_executor(model, tokenizer, max_tokens=max_tokens)
            executor.generate_batch([p])
        serial_time = time.perf_counter() - t0

        # Batched
        executor = create_batched_executor(model, tokenizer, max_tokens=max_tokens)
        t0 = time.perf_counter()
        executor.generate_batch(prompts)
        batched_time = time.perf_counter() - t0

        speedup = serial_time / batched_time
        print(f"\nSerial: {serial_time:.2f}s, Batched: {batched_time:.2f}s, Speedup: {speedup:.1f}x")
        assert speedup > 1.5, f"Batched should be >1.5x faster, got {speedup:.1f}x"
