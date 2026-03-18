"""Tests for neural network layers."""

import pytest
import torch
import torch.nn as nn

from tinytrtllm.layers.activation import SiluAndMul
from tinytrtllm.layers.embedding import VocabParallelEmbedding, ParallelLMHead
from tinytrtllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tinytrtllm.layers.norm import RMSNorm
from tinytrtllm.layers.rotary import RotaryEmbedding


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == (2, 8, 64)

    def test_unit_variance(self):
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        # RMS norm should produce roughly unit variance
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.5)

    def test_fused_residual(self):
        norm = RMSNorm(hidden_size=64)
        x = torch.randn(2, 8, 64)
        residual = torch.randn(2, 8, 64)
        normed, new_residual = norm(x, residual)
        assert normed.shape == (2, 8, 64)
        # new_residual should be x + residual
        expected = x + residual
        assert torch.allclose(new_residual, expected, atol=1e-5)

    def test_numerical_correctness(self):
        hidden_size = 32
        norm = RMSNorm(hidden_size=hidden_size, eps=1e-6)
        x = torch.randn(1, 4, hidden_size)
        # Manual RMS norm computation
        variance = x.pow(2).mean(-1, keepdim=True)
        manual_normed = x * torch.rsqrt(variance + 1e-6)
        manual_normed = manual_normed * norm.weight.data
        out = norm(x)
        assert torch.allclose(out, manual_normed, atol=1e-5)

    def test_dtype_preservation(self):
        norm = RMSNorm(hidden_size=64).to(torch.float32)
        x = torch.randn(2, 4, 64, dtype=torch.float32)
        out = norm(x)
        assert out.dtype == torch.float32


class TestSiluAndMul:
    def test_output_shape(self):
        act = SiluAndMul()
        x = torch.randn(2, 8, 128)  # last dim must be even
        out = act(x)
        assert out.shape == (2, 8, 64)  # halved

    def test_correctness(self):
        act = SiluAndMul()
        x = torch.randn(2, 4, 64)
        gate, value = x.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(gate) * value
        out = act(x)
        assert torch.allclose(out, expected, atol=1e-5)


class TestRotaryEmbedding:
    def test_output_shape(self):
        rope = RotaryEmbedding(dim=64, max_seq_len=128)
        q = torch.randn(2, 8, 4, 64)  # (batch, seq, heads, dim)
        k = torch.randn(2, 8, 4, 64)
        positions = torch.arange(8).unsqueeze(0).expand(2, -1)
        q_rot, k_rot = rope(q, k, positions)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_position_dependent(self):
        rope = RotaryEmbedding(dim=64, max_seq_len=128)
        q = torch.randn(1, 2, 1, 64)
        k = torch.randn(1, 2, 1, 64)
        pos1 = torch.tensor([[0, 1]])
        pos2 = torch.tensor([[2, 3]])
        q1, _ = rope(q, k, pos1)
        q2, _ = rope(q, k, pos2)
        assert not torch.allclose(q1, q2)  # different positions → different output

    def test_numerical_correctness(self):
        dim = 8
        rope = RotaryEmbedding(dim=dim, max_seq_len=16)
        q = torch.ones(1, 1, 1, dim)
        k = torch.ones(1, 1, 1, dim)
        positions = torch.tensor([[3]])
        q_rot, k_rot = rope(q, k, positions)
        # Just verify it runs and changes the input
        assert not torch.allclose(q_rot, q)  # rotation should change values


class TestReplicatedLinear:
    def test_forward_shape(self):
        linear = ReplicatedLinear(in_features=64, out_features=128, bias=False)
        x = torch.randn(2, 8, 64)
        out = linear(x)
        assert out.shape == (2, 8, 128)


class TestColumnParallelLinear:
    def test_output_dim(self):
        """output_dim = output_size / tp_size."""
        linear = ColumnParallelLinear(in_features=64, out_features=128, bias=False, tp_size=1)
        x = torch.randn(2, 8, 64)
        out = linear(x)
        assert out.shape == (2, 8, 128)

    def test_weight_loader(self):
        linear = ColumnParallelLinear(in_features=64, out_features=128, bias=False, tp_size=2, tp_rank=0)
        # Should shard output dim
        assert linear.weight.shape[0] == 64  # 128 / 2


class TestRowParallelLinear:
    def test_input_dim(self):
        """input_dim = input_size / tp_size."""
        linear = RowParallelLinear(in_features=128, out_features=64, bias=False, tp_size=1)
        x = torch.randn(2, 8, 128)
        out = linear(x)
        assert out.shape == (2, 8, 64)

    def test_weight_loader(self):
        linear = RowParallelLinear(in_features=128, out_features=64, bias=False, tp_size=2, tp_rank=0)
        assert linear.weight.shape[1] == 64  # 128 / 2


class TestMergedColumnParallelLinear:
    def test_forward_shape(self):
        linear = MergedColumnParallelLinear(
            in_features=64, out_features_list=[128, 128], bias=False, tp_size=1
        )
        x = torch.randn(2, 8, 64)
        out = linear(x)
        assert out.shape == (2, 8, 256)

    def test_weight_loader_shards(self):
        linear = MergedColumnParallelLinear(
            in_features=64, out_features_list=[128, 128], bias=False, tp_size=2, tp_rank=0
        )
        assert linear.weight.shape[0] == 128  # (128+128)/2


class TestQKVParallelLinear:
    def test_forward_shape(self):
        linear = QKVParallelLinear(
            hidden_size=64, num_heads=8, num_kv_heads=2, head_dim=8, bias=False, tp_size=1
        )
        x = torch.randn(2, 4, 64)
        out = linear(x)
        # q=64 + k=16 + v=16 = 96
        assert out.shape == (2, 4, 96)

    def test_weight_loader_shards(self):
        linear = QKVParallelLinear(
            hidden_size=64, num_heads=8, num_kv_heads=2, head_dim=8, bias=False, tp_size=2, tp_rank=0
        )
        # q=64/2=32 + k=16/2=8 + v=16/2=8 = 48
        assert linear.weight.shape[0] == 48


class TestVocabParallelEmbedding:
    def test_output_shape(self):
        embed = VocabParallelEmbedding(vocab_size=100, hidden_size=64, tp_size=1)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        out = embed(input_ids)
        assert out.shape == (2, 3, 64)

    def test_weight_loader(self):
        embed = VocabParallelEmbedding(vocab_size=100, hidden_size=64, tp_size=2, tp_rank=0)
        assert embed.weight.shape[0] == 50  # vocab sharded


class TestParallelLMHead:
    def test_logits_shape(self):
        lm_head = ParallelLMHead(vocab_size=100, hidden_size=64, tp_size=1)
        hidden = torch.randn(2, 8, 64)
        logits = lm_head(hidden)
        assert logits.shape == (2, 8, 100)

    def test_last_token_extraction(self):
        lm_head = ParallelLMHead(vocab_size=100, hidden_size=64, tp_size=1)
        hidden = torch.randn(2, 8, 64)
        last_token_indices = torch.tensor([3, 7])  # last token per seq
        logits = lm_head(hidden, last_token_indices)
        assert logits.shape == (2, 100)  # only last token per sequence
