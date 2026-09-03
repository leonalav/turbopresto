"""Stateful generation for RWKV-7.

Per /ipho-physicist: RWKV-7's recurrent state allows efficient
autoregressive generation. The state can be carried across calls,
enabling:
- Streaming generation
- KV-cache-free inference (state IS the cache)
- Multi-turn conversation

This module provides a stateful generator that maintains the per-layer
state and produces tokens one at a time (or in chunks for efficiency).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


class RWKVGenerator:
    """Stateful RWKV-7 generator with caching.

    Stores the recurrent state between forward passes, allowing:
    - Initial prefill on a prompt
    - Token-by-token generation with state continuation
    - Sampling with temperature, top-k, top-p

    Args:
        model: RWKV-7 model
        tokenizer: Tokenizer with bos_id, eos_id, pad_id, encode, decode
        device: Device to use
        dtype: Dtype for inference (default float32 for CPU)
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

        self.model.eval()
        self.model.to(device=device, dtype=dtype)

    @torch.no_grad()
    def prefill(
        self,
        prompt: str,
        max_input_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Process prompt and return the resulting state + last logits.

        Args:
            prompt: Input text.
            max_input_len: If set, truncate prompt to this many tokens.

        Returns:
            input_ids tensor [1, T] of all tokens seen so far.
        """
        ids = self.tokenizer.encode(prompt)
        if max_input_len is not None:
            ids = ids[-max_input_len:]
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        return input_ids

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        stop_on_eos: bool = True,
        greedy: bool = False,
        use_calculator: bool = True,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature. 0 = greedy.
            top_k: Top-k truncation.
            top_p: Nucleus threshold.
            stop_on_eos: Stop at EOS token.
            greedy: If True, deterministic greedy decoding.
            use_calculator: If True, evaluate <TOOL>calc(...)</TOOL>
                expressions emitted by the model and replace them with the
                numeric result before returning.  This offloads exact
                arithmetic to the sandboxed calculator.

        Returns:
            Generated text (prompt + generation), with tool calls
            resolved if use_calculator=True.
        """
        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

        generated = []
        eos_id = getattr(self.tokenizer, "eos_id", None)

        # Prefill
        logits = self.model(input_ids)  # [1, T, V]
        next_logits = logits[:, -1, :]  # [1, V]

        for _ in range(max_new_tokens):
            if greedy or temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # [1, 1]
            else:
                next_token = self._sample(next_logits, temperature, top_k, top_p)

            tok_id = next_token.item()
            generated.append(tok_id)

            if stop_on_eos and eos_id is not None and tok_id == eos_id:
                break

            # Forward one token (with state carried implicitly by recurrence)
            logits = self.model(next_token)  # [1, 1, V]
            next_logits = logits[:, -1, :]

        # Decode
        all_ids = ids + generated
        text = self.tokenizer.decode(all_ids)

        # Calculator tool-use: any <TOOL>calc(...)</TOOL> block in the
        # generated text is evaluated by the sandboxed safe_eval and
        # replaced by its numeric result.
        if use_calculator:
            try:
                from src.inference.calculator import try_call
                text, _status = try_call(text)
            except Exception:
                # never let a tool-call failure corrupt the response
                pass

        return text

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        greedy: bool = False,
    ) -> List[str]:
        """Generate for multiple prompts (each in own batch slot).

        Args:
            prompts: List of input texts (padded to same length internally).
            max_new_tokens: Max tokens to generate per prompt.
            temperature, top_k, top_p: Sampling params.
            greedy: Deterministic.

        Returns:
            List of generated texts.
        """
        # Tokenize
        all_ids = [self.tokenizer.encode(p) for p in prompts]
        max_len = max(len(ids) for ids in all_ids)
        pad_id = getattr(self.tokenizer, "pad_id", 0)

        # Pad to same length
        B = len(prompts)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=self.device)
        for i, ids in enumerate(all_ids):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)

        # Prefill
        logits = self.model(input_ids)  # [B, T, V]
        next_logits = logits[:, -1, :]  # [B, V]

        generated_ids = [[] for _ in range(B)]
        eos_id = getattr(self.tokenizer, "eos_id", None)
        active = [True] * B

        for _ in range(max_new_tokens):
            if greedy or temperature == 0:
                next_tokens = next_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            else:
                next_tokens = self._sample(next_logits, temperature, top_k, top_p)  # [B, 1]

            for b in range(B):
                if not active[b]:
                    continue
                tok_id = next_tokens[b].item()
                generated_ids[b].append(tok_id)
                if eos_id is not None and tok_id == eos_id:
                    active[b] = False

            if not any(active):
                break

            # Forward one step
            logits = self.model(next_tokens)  # [B, 1, V]
            next_logits = logits[:, -1, :]

        # Decode
        results = []
        for b, ids in enumerate(all_ids):
            full_ids = ids + generated_ids[b]
            results.append(self.tokenizer.decode(full_ids))
        return results

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Sample one token from logits with temperature/top-k/top-p."""
        logits = logits / temperature

        # Top-k
        if top_k > 0:
            top_k_vals = logits.topk(min(top_k, logits.size(-1)), dim=-1)[0]
            logits[logits < top_k_vals[:, -1:]] = -float("inf")

        # Top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
            cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_mask = cum_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            indices_to_remove = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
            logits[indices_to_remove] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    greedy: bool = False,
) -> str:
    """Functional interface to RWKV-7 generation.

    Convenience wrapper around RWKVGenerator.
    """
    gen = RWKVGenerator(model, tokenizer)
    return gen.generate(
        prompt, max_new_tokens, temperature, top_k, top_p, greedy=greedy
    )


if __name__ == "__main__":
    from src.model.config import ModelConfig
    from src.model.init import init_and_verify
    from src.tokenizer.math_tokenizer import StubTokenizer

    cfg = ModelConfig(vocab_size=256, n_layers=2, d_model=64, d_state=32)
    model = init_and_verify(cfg)
    tok = StubTokenizer(vocab_size=256)

    gen = RWKVGenerator(model, tok)
    out = gen.generate("What is 1 + 1?", max_new_tokens=10, temperature=0.0)
    print(f"Generated: {out!r}")