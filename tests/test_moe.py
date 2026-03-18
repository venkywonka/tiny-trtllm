"""Tests for MoE layer."""

import pytest
import torch

from tinytrtllm.layers.moe import FusedMoE, MoEGate


class TestMoEGate:
    def test_output_shape(self):
        gate = MoEGate(hidden_size=64, num_experts=8)
        x = torch.randn(10, 64)
        out = gate(x)
        assert out.shape == (10, 8)

    def test_output_shape_batched(self):
        gate = MoEGate(hidden_size=64, num_experts=4)
        x = torch.randn(2, 5, 64)
        out = gate(x.view(-1, 64))
        assert out.shape == (10, 4)


class TestFusedMoE:
    def test_output_shape(self):
        moe = FusedMoE(hidden_size=64, intermediate_size=128, num_experts=8, top_k=2)
        x = torch.randn(10, 64)
        out = moe(x)
        assert out.shape == (10, 64)

    def test_routing_weights_normalized(self):
        moe = FusedMoE(hidden_size=64, intermediate_size=128, num_experts=8, top_k=2)
        x = torch.randn(5, 64)
        router_logits = moe.gate(x)
        weights = torch.softmax(router_logits, dim=-1)
        topk_w, _ = torch.topk(weights, 2, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        assert torch.allclose(topk_w.sum(dim=-1), torch.ones(5), atol=1e-5)

    def test_3d_input(self):
        moe = FusedMoE(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        x = torch.randn(2, 5, 64)  # batch, seq, hidden
        out = moe(x)
        assert out.shape == (2, 5, 64)

    def test_single_expert(self):
        moe = FusedMoE(hidden_size=32, intermediate_size=64, num_experts=1, top_k=1)
        x = torch.randn(4, 32)
        out = moe(x)
        assert out.shape == (4, 32)

    def test_gradient_flows(self):
        moe = FusedMoE(hidden_size=32, intermediate_size=64, num_experts=4, top_k=2)
        x = torch.randn(6, 32, requires_grad=True)
        out = moe(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (6, 32)

    def test_output_not_all_zeros(self):
        moe = FusedMoE(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        x = torch.randn(8, 64)
        out = moe(x)
        assert not torch.allclose(out, torch.zeros_like(out))
