"""Tests for the plan/execute attention interface and backend implementations."""

import pytest
import torch

from tinytrtllm.layers.attention import (
    BACKEND_REGISTRY,
    Attention,
    AttentionBackend,
    PreparedAttention,
    get_attention_backend,
    store_kvcache,
)


class TestPreparedAttention:
    def test_creation(self):
        pa = PreparedAttention(
            seq_lens=torch.tensor([3, 5]),
            context_lens=torch.tensor([3, 0]),
            max_context_len=3,
            max_seq_len=5,
            block_offsets=torch.tensor([[0, 1], [2, 3]]),
            slot_mapping=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            is_context=torch.tensor([True, False]),
            num_tokens=8,
        )
        assert pa.num_tokens == 8
        assert pa.max_seq_len == 5


class TestAttentionBackend:
    def test_backend_registry_has_vanilla(self):
        assert "vanilla" in BACKEND_REGISTRY

    def test_get_backend_vanilla(self):
        backend = get_attention_backend("vanilla")
        assert isinstance(backend, AttentionBackend)

    def test_get_backend_unknown_raises(self):
        with pytest.raises(KeyError):
            get_attention_backend("nonexistent_backend")


class TestStoreKVCache:
    def test_store_and_retrieve(self):
        num_layers = 1
        num_heads = 2
        head_dim = 4
        block_size = 4
        num_blocks = 4
        # KV cache: (num_layers, 2, num_blocks, block_size, num_heads, head_dim)
        kv_cache = torch.zeros(num_layers, 2, num_blocks, block_size, num_heads, head_dim)

        k = torch.randn(3, num_heads, head_dim)  # 3 tokens
        v = torch.randn(3, num_heads, head_dim)
        slot_mapping = torch.tensor([0, 1, 2])  # tokens go to slots 0,1,2
        layer_idx = 0

        store_kvcache(kv_cache, k, v, slot_mapping, layer_idx, block_size)

        # Check that values were stored
        # slot 0 → block 0, offset 0
        assert torch.allclose(kv_cache[0, 0, 0, 0], k[0])
        assert torch.allclose(kv_cache[0, 1, 0, 0], v[0])

    def test_skip_negative_slot(self):
        num_layers = 1
        num_heads = 2
        head_dim = 4
        block_size = 4
        num_blocks = 4
        kv_cache = torch.zeros(num_layers, 2, num_blocks, block_size, num_heads, head_dim)

        k = torch.randn(2, num_heads, head_dim)
        v = torch.randn(2, num_heads, head_dim)
        slot_mapping = torch.tensor([0, -1])  # second token skipped
        layer_idx = 0

        store_kvcache(kv_cache, k, v, slot_mapping, layer_idx, block_size)

        # Slot 0 should have data
        assert not torch.allclose(kv_cache[0, 0, 0, 0], torch.zeros(num_heads, head_dim))


class TestVanillaBackend:
    def test_plan_returns_prepared(self):
        backend = get_attention_backend("vanilla")
        pa = backend.plan(
            seq_lens=torch.tensor([4, 3]),
            context_lens=torch.tensor([4, 0]),
            block_offsets=torch.tensor([[0], [1]]),
            slot_mapping=torch.tensor([0, 1, 2, 3, 4, 5, 6]),
            max_seq_len=4,
        )
        assert isinstance(pa, PreparedAttention)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_run_output_shape(self):
        backend = get_attention_backend("vanilla")
        batch_size = 2
        seq_len = 4
        num_heads = 4
        head_dim = 8
        num_kv_heads = 2

        q = torch.randn(batch_size * seq_len, num_heads, head_dim, device="cuda")
        k = torch.randn(batch_size * seq_len, num_kv_heads, head_dim, device="cuda")
        v = torch.randn(batch_size * seq_len, num_kv_heads, head_dim, device="cuda")

        pa = PreparedAttention(
            seq_lens=torch.tensor([seq_len, seq_len], device="cuda"),
            context_lens=torch.tensor([seq_len, seq_len], device="cuda"),
            max_context_len=seq_len,
            max_seq_len=seq_len,
            block_offsets=torch.zeros(batch_size, 1, dtype=torch.int32, device="cuda"),
            slot_mapping=torch.arange(batch_size * seq_len, device="cuda"),
            is_context=torch.tensor([True, True], device="cuda"),
            num_tokens=batch_size * seq_len,
        )

        output = backend.run(
            q, k, v, pa,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_cache=None,  # For prefill, no KV cache needed
            layer_idx=0,
            scale=head_dim**-0.5,
        )
        assert output.shape == (batch_size * seq_len, num_heads, head_dim)


class TestAttentionModule:
    def test_creation(self):
        attn = Attention(
            num_heads=8,
            num_kv_heads=2,
            head_dim=64,
            backend_name="vanilla",
        )
        assert attn.num_heads == 8
        assert attn.num_kv_heads == 2

    def test_backend_auto_detection(self):
        """Auto should resolve to an available backend."""
        backend = get_attention_backend("auto")
        assert isinstance(backend, AttentionBackend)


class TestBackendSelection:
    def test_vanilla_always_available(self):
        backend = get_attention_backend("vanilla")
        assert backend is not None

    def test_auto_fallback(self):
        backend = get_attention_backend("auto")
        assert backend is not None
