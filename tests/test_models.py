"""Tests for model registry, Llama, Qwen3, and Qwen3-MoE architectures."""

import pytest
import torch

from tinytrtllm.models import get_model_class, list_models, register_model


# ---------------------------------------------------------------------------
# Mock configs
# ---------------------------------------------------------------------------


class MockLlamaConfig:
    vocab_size = 1000
    hidden_size = 64
    intermediate_size = 128
    num_hidden_layers = 2
    num_attention_heads = 4
    num_key_value_heads = 2
    rms_norm_eps = 1e-6
    max_position_embeddings = 512


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_register_and_retrieve(self):
        class DummyModel:
            pass

        register_model("DummyArch", DummyModel)
        assert get_model_class("DummyArch") is DummyModel

    def test_unknown_architecture_raises(self):
        with pytest.raises(KeyError, match="Unknown architecture"):
            get_model_class("NonExistentArch_12345")

    def test_list_models_contains_registered(self):
        # LlamaForCausalLM is registered on import
        from tinytrtllm.models import llama  # noqa: F401

        assert "LlamaForCausalLM" in list_models()


# ---------------------------------------------------------------------------
# Llama tests
# ---------------------------------------------------------------------------


class TestLlamaForCausalLM:
    def test_instantiation(self):
        from tinytrtllm.models.llama import LlamaForCausalLM

        model = LlamaForCausalLM(MockLlamaConfig)
        assert model.config is MockLlamaConfig

    def test_forward_shape(self):
        from tinytrtllm.models.llama import LlamaForCausalLM

        model = LlamaForCausalLM(MockLlamaConfig)
        model.eval()
        input_ids = torch.randint(0, MockLlamaConfig.vocab_size, (5,))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (5, MockLlamaConfig.vocab_size)

    def test_forward_batch(self):
        from tinytrtllm.models.llama import LlamaForCausalLM

        model = LlamaForCausalLM(MockLlamaConfig)
        model.eval()
        input_ids = torch.randint(0, MockLlamaConfig.vocab_size, (2, 8))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (2, 8, MockLlamaConfig.vocab_size)

    def test_packed_modules_mapping(self):
        from tinytrtllm.models.llama import LlamaForCausalLM

        mapping = LlamaForCausalLM.packed_modules_mapping
        assert "qkv_proj" in mapping
        assert mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
        assert "gate_up_proj" in mapping
        assert mapping["gate_up_proj"] == ["gate_proj", "up_proj"]

    def test_registry_lookup(self):
        from tinytrtllm.models import llama  # noqa: F401

        cls = get_model_class("LlamaForCausalLM")
        from tinytrtllm.models.llama import LlamaForCausalLM

        assert cls is LlamaForCausalLM

    def test_parameter_count_positive(self):
        from tinytrtllm.models.llama import LlamaForCausalLM

        model = LlamaForCausalLM(MockLlamaConfig)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_num_decoder_layers(self):
        from tinytrtllm.models.llama import LlamaForCausalLM

        model = LlamaForCausalLM(MockLlamaConfig)
        assert len(model.model.layers) == MockLlamaConfig.num_hidden_layers
