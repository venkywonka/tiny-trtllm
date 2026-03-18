"""Neural network layers for tiny-trtllm."""

# Import backends to trigger registration
import tinytrtllm.layers.attention_vanilla  # noqa: F401

try:
    import tinytrtllm.layers.attention_trtllm  # noqa: F401
except Exception:
    pass

try:
    import tinytrtllm.layers.attention_fa  # noqa: F401
except Exception:
    pass

try:
    import tinytrtllm.layers.attention_fi  # noqa: F401
except Exception:
    pass
