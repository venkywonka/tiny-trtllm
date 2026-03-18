# Overlap Executor

The `OverlapExecutor` is TRT-LLM's technique for hiding CPU overhead behind GPU compute via ping-pong execution.

## How It Works

```{mermaid}
sequenceDiagram
    participant CPU
    participant GPU
    Note over CPU,GPU: Standard PyExecutor
    CPU->>GPU: Forward batch N
    GPU-->>CPU: Results batch N
    CPU->>CPU: Process results N
    CPU->>GPU: Forward batch N+1
    Note over CPU,GPU: OverlapExecutor (ping-pong)
    CPU->>GPU: Forward batch N
    par CPU processes batch N-1
        CPU->>CPU: Sample, schedule, prepare N+1
    and GPU runs batch N
        GPU-->>GPU: Forward pass
    end
    CPU->>GPU: Forward batch N+1
```

## Key Idea

In the standard `PyExecutor`, CPU work (sampling results, scheduling next batch, preparing inputs) happens **sequentially** with GPU work. The `OverlapExecutor` overlaps them:

- **GPU** runs the forward pass for batch N
- **CPU** simultaneously processes results from batch N-1 and prepares batch N+1

This hides CPU overhead, improving throughput especially when CPU scheduling/sampling is non-trivial.

## Enabling

```python
from tinytrtllm.config import TinyLlmArgs

# Overlap is enabled by default
args = TinyLlmArgs(model_path="...", enable_overlap=True)

# Disable for debugging (simpler sequential execution)
args = TinyLlmArgs(model_path="...", enable_overlap=False)
```
