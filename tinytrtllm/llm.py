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

        # Load tokenizer
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_path, trust_remote_code=True
            )
        except Exception:
            self._tokenizer = None

        # Load model
        self._model, vocab_size = self._load_model()

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._model is not None:
            self._model = self._model.to(device)
            if self.args.dtype == "bfloat16":
                self._model = self._model.to(torch.bfloat16)
            elif self.args.dtype == "float16":
                self._model = self._model.to(torch.float16)

        # Create engine components
        model_engine = ModelEngine(
            model=self._model,
            vocab_size=vocab_size,
            device=device,
            enable_cuda_graph=self.args.enable_cuda_graph,
        )

        scheduler = TwoTierScheduler(
            scheduling_policy=self.args.scheduling_policy,
            chunking_policy=self.args.chunking_policy,
            max_batch_size=self.args.max_batch_size,
            max_num_tokens=self.args.max_num_tokens,
            block_size=self.args.block_size,
            max_blocks=1000,  # TODO: compute from GPU memory
        )

        sampler = Sampler()

        self._executor = PyExecutor(model_engine, scheduler, sampler)
        self._initialized = True

    def _load_model(self) -> tuple[Optional[nn.Module], int]:
        """Load model from HF checkpoint."""
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                self.args.model_path, trust_remote_code=True
            )
            vocab_size = config.vocab_size

            # Try to load from our model registry
            from tinytrtllm.models import get_model_class
            arch = config.architectures[0] if config.architectures else None
            if arch:
                model_cls = get_model_class(arch)
                model = model_cls(config)
                # Load weights
                from tinytrtllm.models.base import TinyModel
                TinyModel.load_weights(
                    model,
                    self.args.model_path,
                    dtype=getattr(torch, self.args.dtype.replace("float", "float")),
                )
                return model, vocab_size

        except Exception:
            pass

        return None, 32000  # fallback vocab size

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
        max_iters = sp.max_tokens * 2 + 100
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
