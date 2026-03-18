#!/usr/bin/env python3
"""Benchmark: tiny-trtllm engine throughput vs HF transformers baseline.

Compares our full engine pipeline (two-tier scheduler → model_engine →
async sampler → executor) against raw HF model.generate().

TRT-LLM comparison note: TRT-LLM v1.2 requires CUDA 13 (not available on
this L40S with CUDA 12.6). When a compatible TRT-LLM is available, add a
third comparison using `from tensorrt_llm import LLM as TrtLLM`.
"""

import json
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
NUM_REQUESTS = 64
MAX_NEW_TOKENS = 64


def generate_prompts(tokenizer, n, min_len=50, max_len=200):
    """Generate varied-length prompts."""
    import random
    random.seed(42)
    base_prompts = [
        "Explain the concept of",
        "Write a Python function that",
        "The history of",
        "Compare and contrast",
        "What are the benefits of",
        "Describe in detail how",
        "List the top 5 reasons why",
        "In a formal essay, explain",
    ]
    topics = [
        "machine learning", "quantum computing", "neural networks",
        "database indexing", "compiler optimization", "operating systems",
        "distributed systems", "cryptography", "graph algorithms",
        "natural language processing", "computer vision", "robotics",
    ]
    prompts = []
    for i in range(n):
        base = base_prompts[i % len(base_prompts)]
        topic = topics[i % len(topics)]
        prompt = f"{base} {topic}"
        # Pad to varied lengths
        padding = " ".join(["The"] * random.randint(5, 40))
        prompts.append(f"{prompt}. {padding}\n\nAnswer:")
    return prompts


def benchmark_hf(model, tokenizer, prompts, max_new_tokens):
    """Benchmark HF transformers greedy decode."""
    total_input_tokens = 0
    total_output_tokens = 0
    start = time.perf_counter()

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]
        total_input_tokens += input_len

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        total_output_tokens += outputs.shape[1] - input_len

    elapsed = time.perf_counter() - start
    return {
        "system": "HuggingFace Transformers",
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "elapsed_s": round(elapsed, 2),
        "output_tok_per_s": round(total_output_tokens / elapsed, 1),
        "total_tok_per_s": round((total_input_tokens + total_output_tokens) / elapsed, 1),
    }


def benchmark_tinytrtllm(model, tokenizer, prompts, max_new_tokens):
    """Benchmark tiny-trtllm engine with full pipeline."""
    from tinytrtllm.config import SamplingParams, SchedulingPolicy, ChunkingPolicy
    from tinytrtllm.engine.executor import PyExecutor
    from tinytrtllm.engine.model_engine import ModelEngine
    from tinytrtllm.engine.request import LlmRequest
    from tinytrtllm.engine.sampler import Sampler
    from tinytrtllm.engine.scheduler import TwoTierScheduler

    class HFWrapper(torch.nn.Module):
        def __init__(self, hf_model):
            super().__init__()
            self.hf_model = hf_model
        def forward(self, input_ids, positions=None):
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            with torch.no_grad():
                out = self.hf_model(input_ids=input_ids)
            return out.logits.squeeze(0)

    device = next(model.parameters()).device
    wrapped = HFWrapper(model)
    model_engine = ModelEngine(
        model=wrapped, vocab_size=model.config.vocab_size,
        device=device, enable_cuda_graph=False,
        use_full_sequence_decode=True,
    )
    scheduler = TwoTierScheduler(
        scheduling_policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
        chunking_policy=ChunkingPolicy.EQUAL_PROGRESS,
        max_batch_size=256, max_num_tokens=16384,
        block_size=256, max_blocks=100000,
    )
    executor = PyExecutor(model_engine, scheduler, Sampler())
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    total_input_tokens = 0
    total_output_tokens = 0
    start = time.perf_counter()

    for i, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt)
        total_input_tokens += len(token_ids)
        req = LlmRequest(request_id=i, token_ids=token_ids, max_tokens=max_new_tokens)
        executor.enqueue_request(req, sp)

        for _ in range(max_new_tokens * 3 + 100):
            outputs = executor.iteration()
            done = False
            for out in outputs:
                if out.finished and out.request_id == i:
                    total_output_tokens += len(out.output_token_ids)
                    done = True
                    break
            if done:
                break

    elapsed = time.perf_counter() - start
    return {
        "system": "tiny-trtllm Engine",
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "elapsed_s": round(elapsed, 2),
        "output_tok_per_s": round(total_output_tokens / elapsed, 1),
        "total_tok_per_s": round((total_input_tokens + total_output_tokens) / elapsed, 1),
    }


if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"Benchmark: tiny-trtllm vs HuggingFace Transformers")
    print(f"{'='*60}")
    print(f"Model: {MODEL_ID}")
    print(f"Requests: {NUM_REQUESTS}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Decode mode: greedy (temperature=0)")

    # Load model once
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    prompts = generate_prompts(tokenizer, NUM_REQUESTS)
    print(f"Generated {len(prompts)} prompts")

    # Warmup
    print(f"\nWarming up...")
    warmup_input = tokenizer("Hello", return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**warmup_input, max_new_tokens=5, do_sample=False, temperature=None, top_p=None)

    # Benchmark HF
    print(f"\nRunning HF benchmark...")
    hf_result = benchmark_hf(model, tokenizer, prompts, MAX_NEW_TOKENS)

    # Benchmark tiny-trtllm
    print(f"Running tiny-trtllm benchmark...")
    tiny_result = benchmark_tinytrtllm(model, tokenizer, prompts, MAX_NEW_TOKENS)

    # Results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    for r in [hf_result, tiny_result]:
        print(f"\n  {r['system']}:")
        print(f"    Input tokens:  {r['total_input_tokens']}")
        print(f"    Output tokens: {r['total_output_tokens']}")
        print(f"    Time:          {r['elapsed_s']}s")
        print(f"    Output tok/s:  {r['output_tok_per_s']}")
        print(f"    Total tok/s:   {r['total_tok_per_s']}")

    ratio = tiny_result["output_tok_per_s"] / hf_result["output_tok_per_s"] if hf_result["output_tok_per_s"] > 0 else 0
    print(f"\n  tiny-trtllm / HF ratio: {ratio:.2f}x")

    # Save results
    os.makedirs("benchmarks", exist_ok=True)
    results = {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(),
        "num_requests": NUM_REQUESTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "hf": hf_result,
        "tinytrtllm": tiny_result,
        "ratio": round(ratio, 2),
    }
    with open("benchmarks/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to benchmarks/results.json")
