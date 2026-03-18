#!/usr/bin/env python3
"""End-to-end verification: tiny-trtllm engine vs HuggingFace transformers.

Tests the FULL engine pipeline (scheduler → model_engine → sampler → executor)
by plugging in an HF model. For greedy decode (temperature=0), outputs must
be IDENTICAL to HF's model.generate().

This validates that our two-tier scheduling, async sampling, and iteration
loop produce correct results.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain gravity in one sentence:",
]
MAX_NEW_TOKENS = 32


def run_hf_baseline(model, tokenizer, prompts, max_new_tokens):
    """HF transformers greedy decode — ground truth."""
    print(f"\n{'='*60}")
    print(f"HuggingFace Transformers Baseline (greedy)")
    print(f"{'='*60}")

    results = []
    total_tokens = 0
    start = time.perf_counter()

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        new_tokens = outputs[0][input_len:].tolist()
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        total_tokens += len(new_tokens)
        results.append({
            "prompt": prompt,
            "token_ids": new_tokens,
            "text": text,
        })
        print(f"\n  Prompt: {prompt!r}")
        print(f"  Output ({len(new_tokens)} tokens): {text!r}")
        print(f"  Token IDs: {new_tokens[:10]}{'...' if len(new_tokens) > 10 else ''}")

    elapsed = time.perf_counter() - start
    print(f"\n  Total: {total_tokens} tokens in {elapsed:.2f}s = {total_tokens/elapsed:.0f} tok/s")
    return results


def run_tinytrtllm_engine(model, tokenizer, prompts, max_new_tokens):
    """Run prompts through tiny-trtllm's FULL engine pipeline.

    Uses the real two-tier scheduler, model engine, async sampler, and
    executor — but with the HF model as the forward pass.
    """
    print(f"\n{'='*60}")
    print(f"tiny-trtllm Engine (greedy, full pipeline)")
    print(f"{'='*60}")

    from tinytrtllm.config import SamplingParams, SchedulingPolicy, ChunkingPolicy
    from tinytrtllm.engine.executor import PyExecutor
    from tinytrtllm.engine.model_engine import ModelEngine
    from tinytrtllm.engine.request import LlmRequest, RequestState
    from tinytrtllm.engine.sampler import Sampler
    from tinytrtllm.engine.scheduler import TwoTierScheduler

    # Wrap HF model so engine can call it
    class HFWrapper(torch.nn.Module):
        def __init__(self, hf_model):
            super().__init__()
            self.hf_model = hf_model

        def forward(self, input_ids, positions=None):
            """Engine sends flat (total_tokens,) → we forward through HF."""
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            with torch.no_grad():
                out = self.hf_model(input_ids=input_ids)
            return out.logits.squeeze(0)

    vocab_size = model.config.vocab_size
    device = next(model.parameters()).device
    wrapped = HFWrapper(model)

    # Build engine components — the actual tiny-trtllm architecture
    model_engine = ModelEngine(
        model=wrapped,
        vocab_size=vocab_size,
        device=device,
        enable_cuda_graph=False,
        use_full_sequence_decode=True,  # No KV cache — resend full sequence
    )
    scheduler = TwoTierScheduler(
        scheduling_policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
        chunking_policy=ChunkingPolicy.EQUAL_PROGRESS,
        max_batch_size=256,
        max_num_tokens=16384,
        block_size=256,
        max_blocks=100000,
    )
    sampler = Sampler()
    executor = PyExecutor(model_engine, scheduler, sampler)

    # Process ONE request at a time (for correctness validation —
    # batched requests would need proper causal masking in the adapter)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    results = []
    total_tokens = 0
    start = time.perf_counter()

    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)
        req = LlmRequest(request_id=len(results), token_ids=token_ids, max_tokens=max_new_tokens)
        executor.enqueue_request(req, sp)

        # Run executor iterations until this request completes
        completed = False
        for _ in range(max_new_tokens * 3 + 100):
            outputs = executor.iteration()
            for out in outputs:
                if out.finished and out.request_id == req.request_id:
                    completed = True
                    new_tokens = out.output_token_ids
                    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    total_tokens += len(new_tokens)
                    results.append({
                        "prompt": prompt,
                        "token_ids": new_tokens,
                        "text": text,
                    })
                    print(f"\n  Prompt: {prompt!r}")
                    print(f"  Output ({len(new_tokens)} tokens): {text!r}")
                    print(f"  Token IDs: {new_tokens[:10]}{'...' if len(new_tokens) > 10 else ''}")
                    break
            if completed:
                break

        if not completed:
            print(f"\n  Prompt: {prompt!r}")
            print(f"  FAILED: did not complete in time")
            # Gather what we have
            results.append({
                "prompt": prompt,
                "token_ids": list(req.output_token_ids),
                "text": tokenizer.decode(req.output_token_ids, skip_special_tokens=True),
            })

    elapsed = time.perf_counter() - start
    if total_tokens > 0:
        print(f"\n  Total: {total_tokens} tokens in {elapsed:.2f}s = {total_tokens/elapsed:.0f} tok/s")
    return results


def compare_results(hf_results, tiny_results):
    """Compare token-by-token. For greedy decode, must be IDENTICAL."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: Token-by-Token")
    print(f"{'='*60}")

    all_match = True
    for i, (hf, tiny) in enumerate(zip(hf_results, tiny_results)):
        match = hf["token_ids"] == tiny["token_ids"]
        status = "MATCH" if match else "MISMATCH"
        print(f"\n  Prompt {i} [{hf['prompt'][:40]}]: {status}")
        if not match:
            all_match = False
            min_len = min(len(hf["token_ids"]), len(tiny["token_ids"]))
            for j in range(min_len):
                if hf["token_ids"][j] != tiny["token_ids"][j]:
                    print(f"    First divergence at token {j}:")
                    print(f"      HF:   {hf['token_ids'][j]}")
                    print(f"      Tiny: {tiny['token_ids'][j]}")
                    break
            print(f"    HF   ({len(hf['token_ids'])} tok): {hf['token_ids']}")
            print(f"    Tiny ({len(tiny['token_ids'])} tok): {tiny['token_ids']}")

    print(f"\n{'='*60}")
    if all_match:
        print(f"  RESULT: ALL {len(hf_results)} PROMPTS MATCH")
        print(f"  Greedy decode is IDENTICAL between HF and tiny-trtllm engine.")
    else:
        print(f"  RESULT: MISMATCH DETECTED — debugging needed")
    print(f"{'='*60}")

    return all_match


if __name__ == "__main__":
    print(f"Model: {MODEL_ID}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"GPU: {torch.cuda.get_device_name()}")

    # Load model ONCE, share between both runs
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded: {model.config.architectures}")

    # Step 1: HF baseline
    hf_results = run_hf_baseline(model, tokenizer, PROMPTS, MAX_NEW_TOKENS)

    # Step 2: tiny-trtllm engine (same model, our scheduler/sampler/executor)
    tiny_results = run_tinytrtllm_engine(model, tokenizer, PROMPTS, MAX_NEW_TOKENS)

    # Step 3: Compare
    success = compare_results(hf_results, tiny_results)
    sys.exit(0 if success else 1)
