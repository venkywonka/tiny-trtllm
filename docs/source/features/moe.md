# Mixture of Experts (MoE)

tiny-trtllm supports Mixture of Experts models like Qwen3-MoE with a single-GPU implementation.

## Architecture

```{mermaid}
graph TD
    H[Hidden states] --> G[Gate linear]
    G --> S[Softmax]
    S --> TK["TopK routing<br/>(select top_k experts)"]
    TK --> D["Dispatch to experts"]
    D --> E1["Expert 1<br/>SiLU-gated MLP"]
    D --> E2["Expert 2<br/>SiLU-gated MLP"]
    D --> EN["Expert N<br/>SiLU-gated MLP"]
    E1 --> C[Weighted combine]
    E2 --> C
    EN --> C
    C --> O[Output]
```

## Pipeline

1. **Gate**: Linear projection from hidden dim to num_experts
2. **Softmax**: Convert gate logits to routing probabilities
3. **TopK**: Select `top_k` experts per token (typically 2-4)
4. **Dispatch**: Route tokens to their assigned experts
5. **Expert MLP**: Each expert is a SiLU-gated MLP (gate_proj, up_proj, down_proj)
6. **Combine**: Weighted sum of expert outputs using routing weights

## Supported Models

| Model | Experts | Top-K | Shared Expert |
|-------|---------|-------|---------------|
| Qwen3-MoE | Varies | Configurable | Optional |
