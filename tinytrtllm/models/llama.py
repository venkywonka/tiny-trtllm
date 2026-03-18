"""Llama model architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from tinytrtllm.models import register_model
from tinytrtllm.models.base import TinyModel


class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = config.hidden_size // self.num_heads

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

    def forward(
        self, hidden_states, positions=None, prepared_attn=None, kv_cache=None,
        layer_idx=0,
    ):
        orig_shape = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Flatten to 2D: (total_tokens, hidden)
        total_tokens = q.shape[0] if q.dim() == 2 else q.shape[0] * q.shape[1]
        q = q.reshape(total_tokens, self.num_heads, self.head_dim)
        k = k.reshape(total_tokens, self.num_kv_heads, self.head_dim)
        v = v.reshape(total_tokens, self.num_kv_heads, self.head_dim)

        # GQA repeat for KV heads
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        # (heads, tokens, dim) layout for bmm
        attn = torch.bmm(
            q.transpose(0, 1), k.transpose(0, 1).transpose(-2, -1)
        ) * scale
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v.transpose(0, 1))  # (heads, tokens, dim)
        out = out.transpose(0, 1).contiguous().view(total_tokens, -1)

        out = self.o_proj(out)
        # Restore original shape (batch, seq, hidden) if needed
        return out.view(*orig_shape)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size * 2,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(torch.nn.functional.silu(gate) * up)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
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


class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
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


class LlamaForCausalLM(TinyModel):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, positions=None, prepared_attn=None, kv_cache=None):
        hidden_states = self.model(input_ids, positions, prepared_attn, kv_cache)
        return self.lm_head(hidden_states)


register_model("LlamaForCausalLM", LlamaForCausalLM)
