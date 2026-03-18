"""tiny-trtllm: Minimal reimplementation of TensorRT-LLM's PyTorch backend."""

__version__ = "0.1.0"

from tinytrtllm.config import SamplingParams, TinyLlmArgs

__all__ = ["TinyLlmArgs", "SamplingParams", "__version__"]
