"""Base model class with HF safetensors weight loading."""

from __future__ import annotations

import os

import torch
import torch.nn as nn


class TinyModel(nn.Module):
    """Base class for all tiny-trtllm models."""

    # Subclasses override this to map merged weight names
    packed_modules_mapping: dict[str, list[str]] = {}

    def __init__(self, config):
        super().__init__()
        self.config = config

    @staticmethod
    def load_weights(
        model: nn.Module, model_path: str, dtype: torch.dtype = torch.bfloat16
    ):
        """Load weights from HF safetensors checkpoint."""
        from safetensors import safe_open

        weight_files = sorted(
            f for f in os.listdir(model_path) if f.endswith(".safetensors")
        )

        param_map = dict(model.named_parameters())
        loaded: set[str] = set()

        for wf in weight_files:
            path = os.path.join(model_path, wf)
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    # Try direct mapping first
                    if key in param_map:
                        param_map[key].data.copy_(f.get_tensor(key).to(dtype))
                        loaded.add(key)
                    else:
                        # Try with weight_loader for TP-sharded params
                        for _pname, param in param_map.items():
                            if hasattr(param, "weight_loader"):
                                param.weight_loader(param, f.get_tensor(key), key)
                                loaded.add(key)

        return loaded
