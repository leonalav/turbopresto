"""Data collator for RWKV-7 training.

Per /ipho-physicist: RWKV is recurrent (not attention-based), so we don't
need to pad sequences to a common length. We pack multiple short sequences
into a single long sequence using <BOS> and <EOS> tokens.

This is more efficient than padding and is the standard approach for
RWKV training (BlinkDL/RWKV-LM uses the same).
"""

from __future__ import annotations

import random
from typing import Dict, List

import torch


class RWKVCollator:
    """Packs variable-length sequences into fixed-size chunks.

    Args:
        tokenizer: MathTokenizer (or StubTokenizer) with bos_id/eos_id/pad_id.
        seq_len: Target sequence length (default ctx_len).
        pad_to_multiple: Pad batches to multiple of N (for efficient kernels).
        append_eos: Whether to append <EOS> after each example.
        random_shuffle: If True, shuffle the packing order (default True).
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 1024,
        pad_to_multiple: int = 1,
        append_eos: bool = True,
        random_shuffle: bool = True,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_to_multiple = pad_to_multiple
        self.append_eos = append_eos
        self.random_shuffle = random_shuffle

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text and optionally append EOS."""
        ids = self.tokenizer.encode(text)
        if self.append_eos:
            ids = ids + [self.tokenizer.eos_id]
        return ids

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Pack texts into fixed-size sequences.

        Concatenates tokenized texts with BOS at boundaries, then chunks
        into seq_len blocks.

        Returns:
            Tensor of shape [N, seq_len] where N is the number of chunks.
        """
        all_ids: List[int] = []
        for text in texts:
            all_ids.extend(self._tokenize(text))

        # Truncate or pad to multiple of seq_len
        target_len = ((len(all_ids) + self.seq_len - 1) // self.seq_len) * self.seq_len
        if len(all_ids) < target_len:
            # Pad with pad_id (last tokens are mostly ignored by loss masking)
            all_ids = all_ids + [self.tokenizer.pad_id] * (target_len - len(all_ids))

        # Reshape to [N, seq_len]
        n_chunks = target_len // self.seq_len
        tensor = torch.tensor(all_ids[:target_len], dtype=torch.long).reshape(n_chunks, self.seq_len)
        return tensor

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader.

        Args:
            batch: List of {"text": str} or {"prompt": str, "target": str} dicts.

        Returns:
            Dict with "input_ids" [N, seq_len] and "labels" [N, seq_len].
        """
        # Extract text from batch
        texts = []
        for ex in batch:
            if "text" in ex:
                texts.append(ex["text"])
            elif "prompt" in ex and "target" in ex:
                texts.append(ex["prompt"] + ex["target"])
            else:
                raise ValueError(f"Unknown example format: {ex.keys()}")

        if self.random_shuffle:
            texts = list(texts)
            random.shuffle(texts)

        input_ids = self.encode_batch(texts)
        # Labels = input_ids (next-token prediction)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def collate_for_pretrain(batch: List[Dict], tokenizer, seq_len: int = 1024
                        ) -> Dict[str, torch.Tensor]:
    """Convenience function for pretrain collator."""
    collator = RWKVCollator(tokenizer, seq_len=seq_len)
    return collator.collate_fn(batch)


def collate_for_sft(batch: List[Dict], tokenizer, seq_len: int = 1024,
                   mask_prompt: bool = True) -> Dict[str, torch.Tensor]:
    """Collate for SFT with prompt-mask support.

    Args:
        batch: List of {"prompt": str, "target": str} dicts.
        tokenizer: Tokenizer.
        seq_len: Target sequence length.
        mask_prompt: If True, set labels for prompt tokens to -100 (ignore).

    Returns:
        Dict with "input_ids" [N, seq_len] and "labels" [N, seq_len].
        Labels have -100 for prompt positions (if mask_prompt=True).
    """
    all_input_ids: List[int] = []
    all_labels: List[int] = []

    for ex in batch:
        prompt_ids = tokenizer.encode(ex["prompt"])
        target_ids = tokenizer.encode(ex["target"])
        if hasattr(tokenizer, "eos_id"):
            target_ids = target_ids + [tokenizer.eos_id]

        input_ids = prompt_ids + target_ids
        if mask_prompt:
            labels = [-100] * len(prompt_ids) + target_ids
        else:
            labels = list(input_ids)

        all_input_ids.extend(input_ids)
        all_labels.extend(labels)

    # Pad/truncate to seq_len multiple
    target_len = ((len(all_input_ids) + seq_len - 1) // seq_len) * seq_len
    if len(all_input_ids) < target_len:
        pad_id = getattr(tokenizer, "pad_id", 0)
        all_input_ids = all_input_ids + [pad_id] * (target_len - len(all_input_ids))
        all_labels = all_labels + [-100] * (target_len - len(all_labels))

    n_chunks = target_len // seq_len
    input_ids = torch.tensor(all_input_ids[:target_len], dtype=torch.long).reshape(n_chunks, seq_len)
    labels = torch.tensor(all_labels[:target_len], dtype=torch.long).reshape(n_chunks, seq_len)
    return {"input_ids": input_ids, "labels": labels}


if __name__ == "__main__":
    from src.tokenizer.math_tokenizer import StubTokenizer
    tok = StubTokenizer()
    collator = RWKVCollator(tok, seq_len=32)

    batch = [
        {"text": "What is 1 + 1? 2"},
        {"text": "What is 2 + 2? 4"},
    ]
    out = collator.collate_fn(batch)
    print(f"Input ids shape: {out['input_ids'].shape}")
    print(f"Labels shape: {out['labels'].shape}")
    print(f"First sequence: {out['input_ids'][0][:20].tolist()}...")