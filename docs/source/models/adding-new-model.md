# Adding a New Model

## Steps

1. Create a new file in `tinytrtllm/models/` (e.g., `mistral.py`)
2. Implement the model class inheriting from `TinyModel`
3. Register it with `register_model()`
4. Add an import in `models/__init__.py`

## Template

```python
from tinytrtllm.models import register_model
from tinytrtllm.models.base import TinyModel


@register_model("MyModelForCausalLM")
class MyModelForCausalLM(TinyModel):
    # Map packed HF weight names to constituent projections
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(self, config):
        super().__init__(config)
        # Build layers: embeddings, transformer blocks, LM head
        ...

    def forward(self, input_ids, positions=None, prepared_attention=None,
                kv_caches=None, attn_metadata=None):
        # Forward pass → logits
        ...
```

## Key Points

- **`packed_modules_mapping`**: Tells the weight loader how to split packed HF tensors (e.g., a single QKV weight into separate Q, K, V)
- **`forward()` signature**: Must accept `prepared_attention` (from the plan/execute pattern) and `kv_caches`
- **Weight names**: The base class automatically maps HuggingFace weight names to your module's parameter names
