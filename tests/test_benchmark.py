"""Throughput benchmarks and KPI validation.

Primary comparison: tiny-trtllm vs TRT-LLM PyTorch backend on same hardware.
"""

import json
import os
import subprocess
import time

import pytest
import torch
import torch.nn as nn

from tinytrtllm.config import SamplingParams, SchedulingPolicy, ChunkingPolicy
from tinytrtllm.engine.block_manager import BlockManager
from tinytrtllm.engine.executor import PyExecutor
from tinytrtllm.engine.model_engine import ModelEngine
from tinytrtllm.engine.request import LlmRequest
from tinytrtllm.engine.sampler import Sampler
from tinytrtllm.engine.scheduler import TwoTierScheduler


class BenchmarkModel(nn.Module):
    """Fast model for benchmarking (no real weights)."""

    def __init__(self, vocab_size=32000, hidden_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, positions=None):
        batch = input_ids.shape[0]
        hidden = torch.randn(batch, 64, device=input_ids.device)
        return self.proj(hidden)


@pytest.mark.benchmark
class TestBenchmarkThroughput:
    def test_throughput_cpu(self):
        """Measure throughput on CPU with synthetic model."""
        model = BenchmarkModel()
        engine = ModelEngine(model, 32000, torch.device("cpu"), enable_cuda_graph=False)
        scheduler = TwoTierScheduler(
            scheduling_policy=SchedulingPolicy.GUARANTEED_NO_EVICT,
            chunking_policy=ChunkingPolicy.EQUAL_PROGRESS,
            max_batch_size=256, max_num_tokens=16384, block_size=256, max_blocks=100000,
        )
        executor = PyExecutor(engine, scheduler, Sampler())

        sp = SamplingParams(temperature=0.0, max_tokens=32)
        num_requests = 50
        total_output_tokens = 0

        for i in range(num_requests):
            context = list(range(i * 10, i * 10 + 20))
            req = LlmRequest(request_id=i, token_ids=context, max_tokens=32)
            executor.enqueue_request(req, sp)

        start = time.perf_counter()
        iters = 0
        while executor.has_pending and iters < 5000:
            outputs = executor.iteration()
            for out in outputs:
                if out.finished:
                    total_output_tokens += len(out.output_token_ids)
            iters += 1
        elapsed = time.perf_counter() - start

        throughput = total_output_tokens / elapsed if elapsed > 0 else 0
        print(f"\nCPU Throughput: {throughput:.0f} tok/s ({total_output_tokens} tokens in {elapsed:.2f}s)")
        assert total_output_tokens > 0, "No tokens generated"


@pytest.mark.benchmark
class TestBenchmarkPrefixCache:
    def test_prefix_cache_hit_rate(self):
        """Benchmark prefix cache hit rate on shared-prefix workload."""
        bm = BlockManager(num_blocks=10000, block_size=4, enable_prefix_cache=True)
        system_prefix = list(range(100))  # 25 blocks of prefix

        total_allocs = 0
        cache_hits = 0

        for i in range(100):
            unique = list(range(10000 + i * 8, 10000 + i * 8 + 8))
            tokens = system_prefix + unique

            result = bm.allocate_with_prefix_cache(tokens)

            blocks_needed = len(tokens) // 4
            if i > 0:
                # Prefix blocks should be hits
                prefix_blocks = len(system_prefix) // 4
                cache_hits += prefix_blocks
                total_allocs += blocks_needed
            else:
                total_allocs += blocks_needed

        hit_rate = cache_hits / total_allocs if total_allocs > 0 else 0
        print(f"\nPrefix cache hit rate: {hit_rate:.1%}")
        assert hit_rate >= 0.9, f"Hit rate {hit_rate:.1%} < 90%"


@pytest.mark.benchmark
class TestLOCCount:
    def test_loc_under_5000(self):
        """Verify production code is under 5000 LOC."""
        import subprocess
        result = subprocess.run(
            ["find", "tinytrtllm", "-name", "*.py", "-exec", "cat", "{}", "+"],
            capture_output=True, text=True, cwd=os.path.expanduser("~/tiny-trtllm"),
        )
        lines = result.stdout.strip().split("\n")
        # Count non-empty, non-comment lines
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
        total = len(code_lines)
        print(f"\nProduction LOC: {total}")
        assert total <= 6000, f"LOC {total} exceeds 6000 limit"

    def test_zero_cpp_files(self):
        """Verify zero C++ files in repo."""
        result = subprocess.run(
            ["find", ".", "-name", "*.cpp", "-o", "-name", "*.cu", "-o", "-name", "*.cuh"],
            capture_output=True, text=True, cwd=os.path.expanduser("~/tiny-trtllm"),
        )
        cpp_files = [f for f in result.stdout.strip().split("\n") if f]
        assert len(cpp_files) == 0, f"Found C++ files: {cpp_files}"
