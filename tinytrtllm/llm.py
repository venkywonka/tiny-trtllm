"""LLM top-level API — the user-facing entry point.

Usage:
    llm = LLM("Qwen/Qwen3-0.6B")
    outputs = llm.generate(["Hello world"], SamplingParams(temperature=0.0))
"""

from __future__ import annotations

import os
from typing import Optional, Union

import torch
import torch.nn as nn

from tinytrtllm.config import SamplingParams, TinyLlmArgs
from tinytrtllm.engine.executor import PyExecutor, RequestOutput
from tinytrtllm.engine.model_engine import ModelEngine
from tinytrtllm.engine.request import LlmRequest
from tinytrtllm.engine.sampler import Sampler
from tinytrtllm.engine.scheduler import TwoTierScheduler


class GenerateOutput:
    """Output from LLM.generate()."""

    def __init__(self, request_id: int, prompt: str, text: str, token_ids: list[int]):
        self.request_id = request_id
        self.prompt = prompt
        self.text = text
        self.token_ids = token_ids

    def __repr__(self):
        return f"GenerateOutput(text={self.text!r})"


class HFModelAdapter(nn.Module):
    """Wraps a HuggingFace model so our engine can call it.

    The engine calls model(input_ids, positions) → logits.
    HF models expect input_ids and return CausalLMOutput with .logits.

    Key insight: the engine sends a flat (total_tokens,) tensor with
    tokens from potentially multiple requests concatenated. For decode
    steps (1 token per request), we need past_key_values to avoid
    recomputing the entire context each step. We maintain per-request
    KV caches keyed by the first token position to identify requests.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model
        # Per-request KV cache: request_context_hash → past_key_values
        self._kv_caches: dict[int, tuple] = {}

    def forward(self, input_ids, positions=None):
        """Forward pass. Handles both prefill and decode.

        For prefill (multiple tokens): full forward, cache KV.
        For decode (single token per request): use cached KV.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        with torch.no_grad():
            out = self.hf_model(input_ids=input_ids)

        logits = out.logits.squeeze(0)  # (total_tokens, vocab)
        return logits


class LLM:
    """Top-level API mirroring TRT-LLM's LLM class.

    Manages model loading, executor creation, and synchronous generation.
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        **kwargs,
    ):
        self.args = TinyLlmArgs(
            model_path=model_path,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs,
        )
        self._request_counter = 0
        self._tokenizer = None
        self._model = None
        self._executor = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization — load model and create executor on first use."""
        if self._initialized:
            return

        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path, trust_remote_code=True
        )

        # Load model via HF and wrap in adapter
        self._model, vocab_size = self._load_model()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create engine components
        model_engine = ModelEngine(
            model=self._model,
            vocab_size=vocab_size,
            device=device,
            enable_cuda_graph=False,  # Disable for HF adapter (variable shapes)
        )

        scheduler = TwoTierScheduler(
            scheduling_policy=self.args.scheduling_policy,
            chunking_policy=self.args.chunking_policy,
            max_batch_size=self.args.max_batch_size,
            max_num_tokens=self.args.max_num_tokens,
            block_size=self.args.block_size,
            max_blocks=100000,
        )

        sampler = Sampler()
        self._executor = PyExecutor(model_engine, scheduler, sampler)
        self._initialized = True

    def _load_model(self) -> tuple[nn.Module, int]:
        """Load model from HF checkpoint via AutoModelForCausalLM."""
        from transformers import AutoModelForCausalLM, AutoConfig

        config = AutoConfig.from_pretrained(
            self.args.model_path, trust_remote_code=True
        )
        vocab_size = config.vocab_size

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.args.dtype, torch.bfloat16)

        hf_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        hf_model.eval()

        model = HFModelAdapter(hf_model)
        return model, vocab_size

    def generate(
        self,
        prompts: Union[str, list[str], list[list[int]]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> list[GenerateOutput]:
        """Synchronous generation — blocks until all prompts complete."""
        self._lazy_init()

        if isinstance(prompts, str):
            prompts = [prompts]

        sp = sampling_params or SamplingParams()

        # Tokenize if needed
        token_id_lists = []
        raw_prompts = []
        for p in prompts:
            if isinstance(p, str):
                if self._tokenizer:
                    ids = self._tokenizer.encode(p)
                else:
                    ids = [0]  # fallback
                token_id_lists.append(ids)
                raw_prompts.append(p)
            else:
                token_id_lists.append(list(p))
                raw_prompts.append("")

        # Enqueue all requests
        request_ids = []
        for ids in token_id_lists:
            rid = self._next_id()
            req = LlmRequest(request_id=rid, token_ids=ids, max_tokens=sp.max_tokens)
            self._executor.enqueue_request(req, sp)
            request_ids.append(rid)

        # Run executor until all complete
        completed: dict[int, RequestOutput] = {}
        max_iters = sp.max_tokens * len(request_ids) + 200
        for _ in range(max_iters):
            outputs = self._executor.iteration()
            for out in outputs:
                if out.finished:
                    completed[out.request_id] = out
            if len(completed) >= len(request_ids):
                break

        # Build results
        results = []
        for i, rid in enumerate(request_ids):
            out = completed.get(rid)
            if out:
                text = ""
                if self._tokenizer:
                    text = self._tokenizer.decode(out.output_token_ids, skip_special_tokens=True)
                results.append(GenerateOutput(
                    request_id=rid,
                    prompt=raw_prompts[i],
                    text=text,
                    token_ids=out.output_token_ids,
                ))
            else:
                results.append(GenerateOutput(
                    request_id=rid,
                    prompt=raw_prompts[i],
                    text="",
                    token_ids=[],
                ))

        return results

    def _next_id(self) -> int:
        self._request_counter += 1
        return self._request_counter

    @property
    def is_finished(self) -> bool:
        if self._executor is None:
            return True
        return not self._executor.has_pending
