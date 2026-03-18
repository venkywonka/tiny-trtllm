"""HuggingFace model adapters with KV cache for efficient batched inference.

HFKVCacheAdapter: Single-request prefill + KV-cached decode.
BatchedHFModel: Batched prefill and decode for multiple requests.
"""

import torch
from transformers.cache_utils import DynamicCache


class HFKVCacheAdapter:
    """Wraps an HF model with KV cache for efficient single-request decode.

    Usage:
        adapter = HFKVCacheAdapter(hf_model)
        logits = adapter.prefill(token_ids)       # (seq_len, vocab)
        logits = adapter.decode([next_token_id])   # (1, vocab)
        adapter.reset()                            # clear KV cache
    """

    def __init__(self, hf_model):
        self.model = hf_model
        self.past_key_values = None

    def prefill(self, token_ids: list[int]) -> torch.Tensor:
        """Run prefill (full context), cache KV, return (seq_len, vocab) logits."""
        input_ids = torch.tensor([token_ids], device=self._device())
        with torch.no_grad():
            out = self.model(input_ids=input_ids, use_cache=True)
        self.past_key_values = out.past_key_values
        return out.logits[0]  # (seq_len, vocab)

    def decode(self, token_ids: list[int]) -> torch.Tensor:
        """Decode one step using KV cache, return (1, vocab) logits."""
        input_ids = torch.tensor([token_ids], device=self._device())
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=self.past_key_values,
                use_cache=True,
            )
        self.past_key_values = out.past_key_values
        return out.logits[0]  # (1, vocab)

    def reset(self):
        """Clear KV cache so the adapter behaves as fresh."""
        self.past_key_values = None

    def _device(self):
        return next(self.model.parameters()).device


class BatchedHFModel:
    """Batched HF model wrapper for multi-request inference.

    Prefills each request individually to avoid numerical contamination from
    padding, then left-pads KV caches to enable batched decode. Decode steps
    use attention_mask to correctly mask padding positions.

    Usage:
        bm = BatchedHFModel(hf_model)
        logits_list = bm.prefill_batch([token_ids_1, token_ids_2])
        logits_list = bm.decode_batch([next_tok_1, next_tok_2])
    """

    def __init__(self, hf_model):
        self.model = hf_model
        self.past_key_values = None
        self._pad_lens = []  # per-request left-padding lengths
        self._total_len = 0  # total KV cache sequence length

    def prefill_batch(self, token_id_lists: list[list[int]]) -> list[torch.Tensor]:
        """Prefill multiple sequences.

        Each sequence is processed individually to preserve numerical accuracy,
        then KV caches are left-padded and combined for batched decode.

        Returns list of last-token logits (vocab_size,) per request.
        """
        device = next(self.model.parameters()).device
        batch_size = len(token_id_lists)
        seq_lens = [len(ids) for ids in token_id_lists]
        max_len = max(seq_lens)

        # Individual prefills to avoid padding contamination
        individual_caches = []
        results = []
        for ids in token_id_lists:
            input_ids = torch.tensor([ids], device=device)
            with torch.no_grad():
                out = self.model(input_ids=input_ids, use_cache=True)
            individual_caches.append(out.past_key_values)
            results.append(out.logits[0, -1])  # last-token logits (vocab_size,)

        # Left-pad KV caches and combine into a batched cache
        pad_lens = [max_len - sl for sl in seq_lens]
        num_layers = len(individual_caches[0])

        batch_cache = DynamicCache()
        for layer_idx in range(num_layers):
            layer_ref = individual_caches[0].layers[layer_idx]
            _, num_heads, _, head_dim = layer_ref.keys.shape

            padded_keys = []
            padded_values = []
            for i in range(batch_size):
                k = individual_caches[i].layers[layer_idx].keys  # [1, H, S, D]
                v = individual_caches[i].layers[layer_idx].values
                if pad_lens[i] > 0:
                    pad_k = torch.zeros(
                        1, num_heads, pad_lens[i], head_dim,
                        device=device, dtype=k.dtype,
                    )
                    pad_v = torch.zeros(
                        1, num_heads, pad_lens[i], head_dim,
                        device=device, dtype=v.dtype,
                    )
                    k = torch.cat([pad_k, k], dim=2)
                    v = torch.cat([pad_v, v], dim=2)
                padded_keys.append(k)
                padded_values.append(v)

            batch_k = torch.cat(padded_keys, dim=0)  # [B, H, max_len, D]
            batch_v = torch.cat(padded_values, dim=0)
            batch_cache.update(batch_k, batch_v, layer_idx)

        self.past_key_values = batch_cache
        self._pad_lens = pad_lens
        self._total_len = max_len

        return results

    def decode_batch(self, next_tokens: list[int]) -> list[torch.Tensor]:
        """Decode one token per request using cached KV.

        next_tokens: list of token IDs, one per request.
        Returns list of (vocab_size,) logits.
        """
        device = next(self.model.parameters()).device
        batch_size = len(next_tokens)

        input_ids = torch.tensor([[t] for t in next_tokens], device=device)

        # Extend total length by 1 for the new decode token
        self._total_len += 1

        # Build attention mask: (batch, total_kv_len + 1_for_current_token)
        # Padding positions from prefill are 0, all real tokens are 1
        attn_mask = torch.zeros(
            batch_size, self._total_len, device=device, dtype=torch.long,
        )
        for i in range(batch_size):
            attn_mask[i, self._pad_lens[i]:] = 1

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                past_key_values=self.past_key_values,
                use_cache=True,
            )

        self.past_key_values = out.past_key_values
        return [out.logits[i, 0] for i in range(batch_size)]
