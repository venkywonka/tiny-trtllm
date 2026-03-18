import os

import pytest
import torch


@pytest.fixture
def device():
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def cuda_device():
    """Return CUDA device, skip test if unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def small_model_path():
    """Path to Qwen3-0.6B (download if not cached)."""
    model_id = "Qwen/Qwen3-0.6B"
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    # Return model ID — HF will handle caching
    return model_id


@pytest.fixture
def tokenizer(small_model_path):
    """AutoTokenizer for the small model."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    return AutoTokenizer.from_pretrained(small_model_path, trust_remote_code=True)
