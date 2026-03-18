# Supported Models

## Model Matrix

| Model | Architecture Class | Key Features |
|-------|-------------------|--------------|
| Llama 2/3 | `LlamaForCausalLM` | GQA, SiLU-gated MLP, RoPE |
| Qwen3 | `Qwen3ForCausalLM` | Optional QK norms, tied word embeddings |
| Qwen3-MoE | `Qwen3MoEForCausalLM` | Routed experts + optional shared expert, top-k routing |

## Model Registry

Models self-register using the `register_model()` decorator. At import time, each model module registers its architecture name:

```python
from tinytrtllm.models import register_model

@register_model("LlamaForCausalLM")
class LlamaForCausalLM(TinyModel):
    ...
```

The `get_model_class()` function resolves the architecture name from the HuggingFace config's `architectures` field.

## Weight Loading

All models load weights from HuggingFace safetensors format. The base class handles:

- Automatic safetensors file discovery
- Packed module mapping (e.g., QKV projection packed into a single tensor)
- Tensor parallel weight slicing when `tensor_parallel_size > 1`
