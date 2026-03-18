"""Batched executor for concurrent request processing with KV cache.

Provides a full pipeline: tokenize -> batched prefill -> batched decode -> detokenize.
"""

import torch

from tinytrtllm.engine.hf_adapter import BatchedHFModel


class BatchedExecutor:
    """Full engine: batched prefill + decode with greedy sampling.

    Processes all prompts concurrently in a single batch, using
    BatchedHFModel for efficient KV-cached inference.
    """

    def __init__(self, batched_model: BatchedHFModel, tokenizer, max_tokens: int = 16):
        self.batched_model = batched_model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def generate_batch(self, prompts: list[str]) -> dict[str, list[int]]:
        """Generate for all prompts concurrently.

        Args:
            prompts: List of prompt strings.

        Returns:
            Dictionary mapping each prompt to its list of generated token IDs.
        """
        if not prompts:
            return {}

        # Tokenize all prompts
        token_id_lists = [self.tokenizer.encode(p) for p in prompts]

        # Prefill all at once
        logits_list = self.batched_model.prefill_batch(token_id_lists)

        # Initialize generated tokens with first greedy token from prefill
        batch_tokens = [[l.argmax().item()] for l in logits_list]

        # Track which requests are still active
        eos_token_id = self.tokenizer.eos_token_id
        active = [True] * len(prompts)

        # Check if any first token is EOS
        for i in range(len(prompts)):
            if batch_tokens[i][-1] == eos_token_id:
                active[i] = False

        # Greedy decode loop
        for step in range(self.max_tokens - 1):
            if not any(active):
                break

            # Get next tokens for all requests (use last generated token)
            next_toks = [bt[-1] for bt in batch_tokens]

            # Decode one step for all requests
            dl = self.batched_model.decode_batch(next_toks)

            # Greedy sample and append
            for i in range(len(prompts)):
                if active[i]:
                    next_token = dl[i].argmax().item()
                    batch_tokens[i].append(next_token)
                    if next_token == eos_token_id:
                        active[i] = False
                else:
                    # Still need to append something for inactive requests
                    # to keep them in sync, but we already stopped them
                    batch_tokens[i].append(batch_tokens[i][-1])

        # Trim inactive request tokens (remove trailing padding after EOS)
        results = {}
        for i, prompt in enumerate(prompts):
            tokens = batch_tokens[i]
            # Truncate to max_tokens
            tokens = tokens[: self.max_tokens]
            # Truncate at EOS if present
            if eos_token_id is not None and eos_token_id in tokens:
                eos_idx = tokens.index(eos_token_id)
                tokens = tokens[: eos_idx + 1]
            results[prompt] = tokens

        return results


def create_batched_executor(
    model, tokenizer, max_tokens: int = 16
) -> BatchedExecutor:
    """Factory function to create a BatchedExecutor from an HF model.

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: HuggingFace tokenizer.
        max_tokens: Maximum number of tokens to generate per request.

    Returns:
        A BatchedExecutor instance.
    """
    batched_model = BatchedHFModel(model)
    return BatchedExecutor(batched_model, tokenizer, max_tokens=max_tokens)
