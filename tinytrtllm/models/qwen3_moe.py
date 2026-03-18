"""Qwen3-MoE model architecture — Qwen3 with mixture-of-experts MLP layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from tinytrtllm.layers.moe import FusedMoE
from tinytrtllm.models import register_model
from tinytrtllm.models.base import TinyModel


class Qwen3MoEAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = config.hidden_size // self.num_heads
        self.qk_norms = getattr(config, "qk_norms", False)

        self.qkv_proj = nn.Linear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )

        if self.qk_norms:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6))
            self.k_norm = nn.LayerNorm(self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6))

    def forward(
        self, hidden_states, positions=None, prepared_attn=None, kv_cache=None,
        layer_idx=0,
    ):
        orig_shape = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        total_tokens = q.shape[0] if q.dim() == 2 else q.shape[0] * q.shape[1]
        q = q.reshape(total_tokens, self.num_heads, self.head_dim)
        k = k.reshape(total_tokens, self.num_kv_heads, self.head_dim)
        v = v.reshape(total_tokens, self.num_kv_heads, self.head_dim)

        if self.qk_norms:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        scale = self.head_dim**-0.5
        attn = torch.bmm(
            q.transpose(0, 1), k.transpose(0, 1).transpose(-2, -1)
        ) * scale
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v.transpose(0, 1))
        out = out.transpose(0, 1).contiguous().view(total_tokens, -1)

        out = self.o_proj(out)
        return out.view(*orig_shape)


class Qwen3MoESparseMLP(nn.Module):
    """MoE MLP with optional shared expert running in parallel with routed experts."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = getattr(config, "num_experts_per_tok", 2)

        self.moe = FusedMoE(
            hidden_size=config.hidden_size,
            intermediate_size=getattr(config, "moe_intermediate_size", config.intermediate_size),
            num_experts=self.num_experts,
            top_k=self.top_k,
        )

        # Optional shared expert
        self.has_shared_expert = getattr(config, "shared_expert_intermediate_size", 0) > 0
        if self.has_shared_expert:
            shared_size = config.shared_expert_intermediate_size
            self.shared_expert_gate_up = nn.Linear(
                config.hidden_size, shared_size * 2, bias=False,
            )
            self.shared_expert_down = nn.Linear(
                shared_size, config.hidden_size, bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routed_output = self.moe(x)

        if self.has_shared_expert:
            gate_up = self.shared_expert_gate_up(x)
            gate, up = gate_up.chunk(2, dim=-1)
            shared_output = self.shared_expert_down(
                torch.nn.functional.silu(gate) * up
            )
            return routed_output + shared_output

        return routed_output


class Qwen3MoEDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.self_attn = Qwen3MoEAttention(config)
        self.mlp = Qwen3MoESparseMLP(config)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
            elementwise_affine=True,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
            elementwise_affine=True,
        )

    def forward(
        self, hidden_states, positions=None, prepared_attn=None, kv_cache=None,
        layer_idx=0,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, positions, prepared_attn, kv_cache, layer_idx,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3MoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3MoEDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )

    def forward(self, input_ids, positions=None, prepared_attn=None, kv_cache=None):
        hidden_states = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, positions, prepared_attn, kv_cache, i,
            )
        return self.norm(hidden_states)


class Qwen3MoEForCausalLM(TinyModel):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3MoEModel(config)
        self.tied_word_embeddings = getattr(config, "tied_word_embeddings", False)
        if self.tied_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False,
            )

    def forward(self, input_ids, positions=None, prepared_attn=None, kv_cache=None):
        hidden_states = self.model(input_ids, positions, prepared_attn, kv_cache)
        if self.tied_word_embeddings:
            return torch.nn.functional.linear(
                hidden_states, self.model.embed_tokens.weight,
            )
        return self.lm_head(hidden_states)


register_model("Qwen3MoEForCausalLM", Qwen3MoEForCausalLM)
