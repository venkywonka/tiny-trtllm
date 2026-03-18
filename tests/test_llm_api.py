"""Tests for LLM top-level API."""

import threading
import pytest
import torch
import torch.nn as nn

from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from tinytrtllm.config import SamplingParams, TinyLlmArgs
from tinytrtllm.llm import LLM, GenerateOutput, HFModelAdapter


class DummyModel(nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, positions=None):
        batch = input_ids.shape[0]
        logits = torch.zeros(batch, self.vocab_size)
        logits[:, 42] = 100.0
        return logits


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_llm():
    """Create an LLM with a dummy model (no real weights needed)."""
    llm = LLM(model_path="/tmp/fake")
    llm._model = DummyModel(100)
    llm._tokenizer = None

    from tinytrtllm.engine.model_engine import ModelEngine
    from tinytrtllm.engine.executor import PyExecutor
    from tinytrtllm.engine.sampler import Sampler
    from tinytrtllm.engine.scheduler import TwoTierScheduler
    from tinytrtllm.config import SchedulingPolicy, ChunkingPolicy

    engine = ModelEngine(llm._model, 100, torch.device("cpu"), enable_cuda_graph=False)
    scheduler = TwoTierScheduler(
        scheduling_policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
        chunking_policy=ChunkingPolicy.EQUAL_PROGRESS,
        max_batch_size=256,
        max_num_tokens=16384,
        block_size=256,
        max_blocks=1000,
    )
    llm._executor = PyExecutor(engine, scheduler, Sampler())
    llm._initialized = True
    return llm


# ---------------------------------------------------------------------------
# TestHFModelAdapter -- Bug N1: position_ids forwarding
# ---------------------------------------------------------------------------

class TestHFModelAdapter:
    """Tests that HFModelAdapter correctly threads position_ids to the HF model."""

    def _make_hf_model_mock(self, seq_len=4, vocab_size=100):
        """Return a mock HF model whose __call__ records its kwargs."""
        mock_hf = MagicMock()
        # Simulate CausalLMOutput with .logits shape (1, seq_len, vocab_size)
        mock_hf.return_value.logits = torch.zeros(1, seq_len, vocab_size)
        return mock_hf

    def test_forward_without_positions_omits_position_ids(self):
        """When positions=None the HF model must NOT receive position_ids."""
        mock_hf = self._make_hf_model_mock(seq_len=3)
        adapter = HFModelAdapter(mock_hf)

        input_ids = torch.tensor([1, 2, 3])
        adapter.forward(input_ids, positions=None)

        call_kwargs = mock_hf.call_args.kwargs
        assert "position_ids" not in call_kwargs, (
            "position_ids must be absent when positions=None"
        )

    def test_forward_with_positions_passes_position_ids(self):
        """When positions is provided it must arrive as position_ids in the HF call."""
        mock_hf = self._make_hf_model_mock(seq_len=3)
        adapter = HFModelAdapter(mock_hf)

        input_ids = torch.tensor([10, 20, 30])
        positions = torch.tensor([5, 6, 7])  # non-zero offset -- real-world decode step
        adapter.forward(input_ids, positions=positions)

        call_kwargs = mock_hf.call_args.kwargs
        assert "position_ids" in call_kwargs, (
            "position_ids must be forwarded to the HF model when positions is given"
        )
        expected = positions.unsqueeze(0)
        assert torch.equal(call_kwargs["position_ids"], expected), (
            f"position_ids mismatch: got {call_kwargs['position_ids']}, "
            f"expected {expected}"
        )

    def test_forward_batched_positions_forwarded_unchanged(self):
        """2-D position tensors (already batched) must be passed through as-is."""
        mock_hf = self._make_hf_model_mock(seq_len=2)
        mock_hf.return_value.logits = torch.zeros(2, 2, 100)
        adapter = HFModelAdapter(mock_hf)

        input_ids = torch.tensor([[1, 2], [3, 4]])   # (batch=2, seq=2)
        positions = torch.tensor([[0, 1], [0, 1]])   # already 2-D
        adapter.forward(input_ids, positions=positions)

        call_kwargs = mock_hf.call_args.kwargs
        assert torch.equal(call_kwargs["position_ids"], positions)

    def test_forward_1d_input_ids_unsqueezed(self):
        """A flat (seq,) input_ids tensor must be unsqueezed to (1, seq) for the HF model."""
        mock_hf = self._make_hf_model_mock(seq_len=4)
        adapter = HFModelAdapter(mock_hf)

        input_ids = torch.tensor([1, 2, 3, 4])  # 1-D
        adapter.forward(input_ids)

        call_kwargs = mock_hf.call_args.kwargs
        assert call_kwargs["input_ids"].dim() == 2, (
            "HF model must receive a 2-D input_ids tensor"
        )
        assert call_kwargs["input_ids"].shape == (1, 4)

    def test_forward_logits_squeezed(self):
        """The batch-dim must be squeezed out before returning logits."""
        vocab_size = 50
        mock_hf = self._make_hf_model_mock(seq_len=3, vocab_size=vocab_size)
        adapter = HFModelAdapter(mock_hf)

        logits = adapter.forward(torch.tensor([1, 2, 3]))
        assert logits.shape == (3, vocab_size), (
            f"Expected squeezed logits (3, {vocab_size}), got {logits.shape}"
        )

    def test_no_grad_during_forward(self):
        """Forward must run inside torch.no_grad() to avoid building a compute graph."""
        mock_hf = self._make_hf_model_mock(seq_len=2)
        adapter = HFModelAdapter(mock_hf)

        input_ids = torch.tensor([1, 2])
        logits = adapter.forward(input_ids)
        assert not logits.requires_grad, "Logits should not require grad under no_grad()"

    def test_kv_caches_attribute_removed(self):
        """The obsolete _kv_caches attribute must not exist on the adapter."""
        mock_hf = MagicMock()
        mock_hf.return_value.logits = torch.zeros(1, 1, 10)
        adapter = HFModelAdapter(mock_hf)
        assert not hasattr(adapter, "_kv_caches"), (
            "_kv_caches is dead code and must be removed from HFModelAdapter"
        )


# ---------------------------------------------------------------------------
# TestDtypeValidation -- Bug: missing dtype validator in config.py
# ---------------------------------------------------------------------------

class TestDtypeValidation:
    """Tests that TinyLlmArgs rejects invalid dtype strings."""

    @pytest.mark.parametrize("valid_dtype", ["bfloat16", "float16", "float32"])
    def test_valid_dtypes_accepted(self, valid_dtype):
        args = TinyLlmArgs(model_path="/tmp/model", dtype=valid_dtype)
        assert args.dtype == valid_dtype

    @pytest.mark.parametrize("bad_dtype", [
        "float8",
        "int8",
        "fp16",       # common alias but not in allowed set
        "bf16",       # another common alias
        "",
        "FLOAT32",    # case-sensitive
        "float 32",  # whitespace
        "auto",
    ])
    def test_invalid_dtype_raises_validation_error(self, bad_dtype):
        with pytest.raises(ValidationError) as exc_info:
            TinyLlmArgs(model_path="/tmp/model", dtype=bad_dtype)
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("dtype",) for err in errors), (
            f"Expected a validation error on field 'dtype', got: {errors}"
        )

    def test_default_dtype_is_bfloat16(self):
        args = TinyLlmArgs(model_path="/tmp/model")
        assert args.dtype == "bfloat16"


# ---------------------------------------------------------------------------
# TestLazyInitThreadSafety -- Bug N2: double-checked locking
# ---------------------------------------------------------------------------

class TestLazyInitThreadSafety:
    """Tests that _lazy_init is idempotent and safe under concurrent calls."""

    def _make_uninit_llm(self):
        return LLM(model_path="/tmp/fake")

    def test_llm_has_init_lock(self):
        """LLM.__init__ must create a threading.Lock for double-checked locking."""
        llm = self._make_uninit_llm()
        assert hasattr(llm, "_init_lock"), (
            "LLM must have an _init_lock attribute after __init__"
        )
        # threading.Lock() returns an instance of _thread.lock; compare via isinstance
        # to the lock type rather than the function.
        assert isinstance(llm._init_lock, type(threading.Lock())), (
            "_init_lock must be a threading.Lock"
        )

    def test_lazy_init_already_initialized_skips_lock(self):
        """When _initialized is True the method must return before acquiring the lock."""
        llm = self._make_uninit_llm()
        llm._initialized = True

        # Replace the lock with one that raises if entered
        broken_lock = MagicMock()
        broken_lock.__enter__ = MagicMock(
            side_effect=RuntimeError("lock must not be acquired when already initialized")
        )
        broken_lock.__exit__ = MagicMock(return_value=False)
        llm._init_lock = broken_lock

        # Should not raise -- the outer guard short-circuits before the lock
        llm._lazy_init()

    def test_lazy_init_safe_under_concurrent_access(self):
        """Multiple threads racing on _lazy_init must initialize exactly once."""
        from tinytrtllm.engine.model_engine import ModelEngine
        from tinytrtllm.engine.executor import PyExecutor
        from tinytrtllm.engine.sampler import Sampler
        from tinytrtllm.engine.scheduler import TwoTierScheduler
        from tinytrtllm.config import SchedulingPolicy, ChunkingPolicy

        init_count = [0]
        errors: list[Exception] = []

        class CountingLLM(LLM):
            """Subclass that replaces _lazy_init with a lightweight but thread-safe stub."""

            def _lazy_init(self):  # type: ignore[override]
                if self._initialized:
                    return
                with self._init_lock:
                    if self._initialized:
                        return
                    init_count[0] += 1
                    self._tokenizer = None
                    dummy = DummyModel(100)
                    self._model = dummy
                    engine = ModelEngine(
                        dummy, 100, torch.device("cpu"), enable_cuda_graph=False
                    )
                    scheduler = TwoTierScheduler(
                        scheduling_policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
                        chunking_policy=ChunkingPolicy.EQUAL_PROGRESS,
                        max_batch_size=256,
                        max_num_tokens=16384,
                        block_size=256,
                        max_blocks=1000,
                    )
                    self._executor = PyExecutor(engine, scheduler, Sampler())
                    self._initialized = True

        llm = CountingLLM(model_path="/tmp/fake")

        NUM_THREADS = 16
        barrier = threading.Barrier(NUM_THREADS)

        def worker():
            try:
                barrier.wait()  # synchronize so all threads start at once
                llm._lazy_init()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions in worker threads: {errors}"
        assert llm._initialized is True
        assert init_count[0] == 1, (
            f"_lazy_init body must run exactly once, ran {init_count[0]} times"
        )


# ---------------------------------------------------------------------------
# TestIterationCap -- Bug: max_iters too low for chunked prefill
# ---------------------------------------------------------------------------

class TestIterationCap:
    """Tests that generate() does not time-out on long prompts with chunked prefill."""

    def test_long_prompt_completes_with_chunked_prefill(self):
        """A prompt longer than max_num_tokens requires multiple prefill iterations.
        The iteration cap must account for this or the request is never completed."""
        import math
        from tinytrtllm.config import SchedulingPolicy, ChunkingPolicy
        from tinytrtllm.engine.model_engine import ModelEngine
        from tinytrtllm.engine.executor import PyExecutor
        from tinytrtllm.engine.sampler import Sampler
        from tinytrtllm.engine.scheduler import TwoTierScheduler

        # Use a very small max_num_tokens to force chunking
        MAX_TOKENS_PER_ITER = 4
        PROMPT_LEN = 10  # will require ceil(10/4) = 3 prefill iterations
        MAX_NEW_TOKENS = 2

        llm = LLM(model_path="/tmp/fake")
        llm._tokenizer = None
        dummy = DummyModel(100)
        llm._model = dummy

        engine = ModelEngine(dummy, 100, torch.device("cpu"), enable_cuda_graph=False)
        scheduler = TwoTierScheduler(
            scheduling_policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
            chunking_policy=ChunkingPolicy.EQUAL_PROGRESS,
            max_batch_size=256,
            max_num_tokens=MAX_TOKENS_PER_ITER,
            block_size=256,
            max_blocks=1000,
        )
        llm._executor = PyExecutor(engine, scheduler, Sampler())
        llm.args.max_num_tokens = MAX_TOKENS_PER_ITER
        llm._initialized = True

        prompt_ids = list(range(PROMPT_LEN))
        outputs = llm.generate(
            [prompt_ids],
            SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS),
        )
        assert len(outputs) == 1
        assert len(outputs[0].token_ids) == MAX_NEW_TOKENS, (
            "Request must complete even when chunked prefill needs many iterations"
        )


# ---------------------------------------------------------------------------
# TestLLMAPI -- original tests (preserved)
# ---------------------------------------------------------------------------

class TestLLMAPI:
    def _make_llm(self):
        return _make_llm()

    def test_generate_returns_outputs(self):
        llm = self._make_llm()
        outputs = llm.generate(
            [[1, 2, 3]],  # token IDs directly
            SamplingParams(temperature=0.0, max_tokens=3),
        )
        assert len(outputs) == 1
        assert isinstance(outputs[0], GenerateOutput)
        assert len(outputs[0].token_ids) == 3
        assert all(t == 42 for t in outputs[0].token_ids)  # greedy -> token 42

    def test_batch_generate(self):
        llm = self._make_llm()
        outputs = llm.generate(
            [[1, 2], [3, 4], [5, 6]],
            SamplingParams(temperature=0.0, max_tokens=2),
        )
        assert len(outputs) == 3
        for out in outputs:
            assert len(out.token_ids) == 2

    def test_is_finished(self):
        llm = self._make_llm()
        assert llm.is_finished is True

    def test_generate_output_repr(self):
        out = GenerateOutput(request_id=0, prompt="hi", text="hello", token_ids=[1, 2])
        assert "hello" in repr(out)
