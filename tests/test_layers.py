"""Tests for neural network layers."""

import pytest
import torch
import torch.nn as nn

from tinytrtllm.layers.norm import RMSNorm


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
