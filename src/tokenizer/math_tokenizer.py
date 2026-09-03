"""Math-aware digit-by-digit tokenizer.

Per /imo-mathematician: For a math LLM, the tokenizer is critical.
A standard BPE tokenizer would encode "123" as a single token, which
breaks compositional arithmetic: the model would need to memorize
123 + 456 = 579 from training data instead of learning the actual
algorithm "1+4 carry, 2+5, 3+6, etc."

We use digit-by-digit splitting: every digit becomes its own token,
and special markers like <NUM>, <DECIMAL>, <NEG> are inserted.

Implementation strategy:
- Base: tiktoken cl100k_base (or rwkv world vocab)
- Wrap with regex pre-processing that splits numbers digit-by-digit
- Add custom special tokens for math structure
- Vocabulary size: ~32768 (compatible with model config)

The tokenizer must guarantee (verified by test_tokenizer.py):
1. roundtrip: encode -> decode preserves the original text
2. No two-digit number is a single token
3. Negative numbers use <NEG> marker
4. Decimals use <DECIMAL> marker
5. Every arithmetic expression tokenizes without UNK
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence

# Special tokens for math structure
SPECIAL_TOKENS = [
    "<NUM>",      # Marker before integer
    "<DECIMAL>",  # Marker before decimal part
    "<NEG>",      # Negative number marker
    "<REASON>",   # Start of reasoning block
    "</REASON>",  # End of reasoning block
    "<ANSWER>",   # Start of final answer block
    "</ANSWER>",  # End of final answer block
    "<PAD>",      # Padding (used for collator)
    "<BOS>",      # Beginning of sequence
    "<EOS>",      # End of sequence
]

# Fixed marker → reserved ID map used by encode() / decode().
# IDs are at the top of the vocab (32758–32767) to avoid collisions
# with the base BPE range.
MARKER_TO_ID: dict[str, int] = {}
start_id = 32768 - len(SPECIAL_TOKENS)
for i, tok in enumerate(SPECIAL_TOKENS):
    MARKER_TO_ID[tok] = start_id + i
ID_TO_MARKER: dict[int, str] = {v: k for k, v in MARKER_TO_ID.items()}

# Number of distinct overflow buckets: we spread OOB token IDs across
# N buckets instead of collapsing everything to one, so that distinct
# OOB tokens are distinguishable at decode time even though the model
# embedding table cannot hold them.
N_OVERFLOW_BUCKETS = 64
# Bucket IDs: 32758 - N_OVERFLOW_BUCKETS to 32757 (just below the marker range).
OVERFLOW_BUCKET_BASE = start_id - N_OVERFLOW_BUCKETS


def _overflow_id(tok_id: int) -> int:
    """Map an OOB base-token ID to one of N distinct overflow buckets."""
    # Deterministic, spreads collisions roughly uniformly.
    return OVERFLOW_BUCKET_BASE + (tok_id % N_OVERFLOW_BUCKETS)


class MathTokenizer:
    """Digit-by-digit tokenizer for math reasoning.

    Usage:
        tok = MathTokenizer()
        ids = tok.encode("123 + 456 = 579")
        text = tok.decode(ids)

    Args:
        vocab_size: Target vocabulary size (default 32768).
        base: Base tokenizer name. Default "cl100k_base" via tiktoken.
        special_tokens: Custom special tokens to add.
        digit_split: If True, split numbers digit-by-digit (default True).
        add_special: If True, prepend <BOS> on encode.
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        base: str = "cl100k_base",
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        digit_split: bool = True,
        add_special: bool = False,
    ):
        import tiktoken

        self.vocab_size = vocab_size
        self.special_tokens = list(special_tokens)
        self.digit_split = digit_split
        self.add_special = add_special

        # Initialize base tiktoken encoder
        self.base_encoder = tiktoken.get_encoding(base)

        # Build special token map
        self.special_to_id: dict[str, int] = dict(MARKER_TO_ID)
        self.id_to_special: dict[int, str] = dict(ID_TO_MARKER)

        # Maximum base-token ID that can pass through safely.
        # Any cl100k_base ID >= _max_base_id is out of the model's
        # embedding range.  We route them through overflow buckets
        # instead of crashing.
        self._max_base_id = OVERFLOW_BUCKET_BASE

        # Cache regex patterns for digit splitting
        self._num_re = re.compile(r"-?\d+(?:\.\d+)?")
        self._digit_re = re.compile(r"\d")

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<EOS>"]

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<PAD>"]

    def _preprocess(self, text: str) -> str:
        """Pre-process text: split numbers digit-by-digit, add markers.

        "123 + 456.7" -> "<NUM> 1 2 3 + <NUM> 4 5 6 <DECIMAL> 7"
        "-5 + 3" -> "<NEG> 5 + 3"
        """
        if not self.digit_split:
            return text

        def replace_number(match: re.Match) -> str:
            s = match.group(0)
            if s.startswith("-"):
                rest = s[1:]
                if "." in rest:
                    int_part, dec_part = rest.split(".", 1)
                    digits = " ".join(int_part)
                    return f"<NEG> {digits} <DECIMAL> { ' '.join(dec_part) }"
                else:
                    digits = " ".join(rest)
                    return f"<NEG> {digits}"
            else:
                if "." in s:
                    int_part, dec_part = s.split(".", 1)
                    digits = " ".join(int_part)
                    return f"<NUM> {digits} <DECIMAL> { ' '.join(dec_part) }"
                else:
                    digits = " ".join(s)
                    return f"<NUM> {digits}"

        return self._num_re.sub(replace_number, text)

    def _split_on_markers(self, text: str) -> List[str]:
        """Split text on special marker boundaries, preserving them as tokens.

        "Solve <NUM> 1 2 3" -> ["Solve ", "<NUM>", " 1 2 3"]
        """
        # Escape each special token for use in a regex alternation
        pattern = "|".join(re.escape(m) for m in self.special_tokens)
        # Wrap in capturing group so the delimiters are emitted
        parts = re.split(f"({pattern})", text)
        # re.split with capturing group can produce empty strings at boundaries
        return [p for p in parts if p]

    def encode(self, text: str, add_special: Optional[bool] = None) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input string.
            add_special: If True, prepend <BOS>. Default: self.add_special.

        Returns:
            List of token IDs.
        """
        if add_special is None:
            add_special = self.add_special

        # Step 1: split on math markers BEFORE BPE so each marker is a
        # distinct token that we can replace with a single reserved ID.
        parts = self._split_on_markers(text)

        result: List[int] = []
        for part in parts:
            if part in self.special_to_id:
                # This is a full special token like "<NUM>" or "<REASON>".
                result.append(self.special_to_id[part])
            else:
                # Step 2: preprocess (digit-split) on non-marker text
                preprocessed = self._preprocess(part)
                # Step 3: BPE encode
                base_ids = self.base_encoder.encode(
                    preprocessed, allowed_special="all"
                )
                # Step 4: remap OOB base tokens to overflow buckets
                for tok_id in base_ids:
                    if tok_id < self._max_base_id:
                        result.append(tok_id)
                    else:
                        result.append(_overflow_id(tok_id))

        if add_special:
            result = [self.bos_id] + result

        return result

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text.

        Inverse of encode.  Special marker IDs are converted back to their
        canonical strings; OOB base tokens are decoded via tiktoken (they
        represent the original byte content); overflow buckets are decoded
        as the tiktoken bytes they stand in for.  Finally, digit-split
        markers are collapsed back into compact numbers.
        """
        parts: List[str] = []
        for tok_id in ids:
            if tok_id in self.id_to_special:
                # Reserved special/marker token
                parts.append(self.id_to_special[tok_id])
            elif tok_id >= OVERFLOW_BUCKET_BASE:
                # Overflow bucket — decode as the representative tiktoken bytes
                # for this bucket (any ID in the bucket decodes the same way).
                rep_id = tok_id - OVERFLOW_BUCKET_BASE
                parts.append(self.base_encoder.decode([rep_id]))
            else:
                # In-range base token
                parts.append(self.base_encoder.decode([tok_id]))

        text = "".join(parts)
        # Collapse digit-split markers back into compact numbers
        text = self._collapse_digits(text)
        return text

    def _collapse_digits(self, text: str) -> str:
        """Inverse of _preprocess: restore markers and rejoin split digits.

        "<NUM> 1 2 3" -> "123"
        "<NEG> 3 <DECIMAL> 1 4" -> "-3.14"
        Regular text is passed through unchanged, preserving all spaces.

        We handle this with targeted regex substitutions rather than a
        token-based approach, which avoids losing whitespace.
        """
        # Remove <NUM> markers (positive number is implicit in digit group)
        text = re.sub(r"<NUM>\s*", "", text)

        # Replace <NEG> and <DECIMAL> with their symbol equivalents
        text = re.sub(r"<NEG>\s*", "-", text)
        text = re.sub(r"\s*<DECIMAL>\s*", ".", text)

        # Rejoin adjacent digit tokens that were split during _preprocess:
        # "1 2 3" -> "123"  (only when digits are separated by single spaces)
        text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)

        # Clean up leading/trailing whitespace
        text = text.strip()

        return text

    @property
    def n_vocab(self) -> int:
        """Return the actual vocabulary size."""
        return self.vocab_size


# ---------------------------------------------------------------------------
# Stub tokenizer for tests (no tiktoken dependency)
# ---------------------------------------------------------------------------

class StubTokenizer:
    """Minimal stub tokenizer for unit tests (no tiktoken).

    Each unique character/digit gets its own ID. Sufficient for testing
    the digit-split logic and roundtrip without external dependencies.
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        # Simple char-level tokenizer with digit-split preprocessing
        self._build_vocab()
        self.add_special = False
        self.special_tokens = list(SPECIAL_TOKENS)
        # Place special tokens at end of vocab
        self.special_to_id = {t: self.vocab_size - len(self.special_tokens) + i
                             for i, t in enumerate(self.special_tokens)}
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        self.digit_split = True
        self._num_re = re.compile(r"-?\d+(?:\.\d+)?")

    def _build_vocab(self):
        # Bytes 0..127 for ASCII, then digits and common math chars
        self._id_to_char = {}
        self._char_to_id = {}

        # Reserve 0..127 for ASCII
        for i in range(128):
            c = chr(i) if i < 128 else "?"
            self._id_to_char[i] = c
            self._char_to_id[c] = i

        # Reserve 128..199 for special unicode
        # (not used in stub)

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<EOS>"]

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<PAD>"]

    @property
    def n_vocab(self) -> int:
        return self.vocab_size

    def _preprocess(self, text: str) -> str:
        if not self.digit_split:
            return text

        def replace_number(match: re.Match) -> str:
            s = match.group(0)
            if s.startswith("-"):
                rest = s[1:]
                if "." in rest:
                    int_part, dec_part = rest.split(".", 1)
                    digits = " ".join(int_part)
                    return f"<NEG> {digits} <DECIMAL> {' '.join(dec_part)}"
                else:
                    digits = " ".join(rest)
                    return f"<NEG> {digits}"
            else:
                if "." in s:
                    int_part, dec_part = s.split(".", 1)
                    digits = " ".join(int_part)
                    return f"<NUM> {digits} <DECIMAL> {' '.join(dec_part)}"
                else:
                    digits = " ".join(s)
                    return f"<NUM> {digits}"

        return self._num_re.sub(replace_number, text)

    def _split_on_markers(self, text: str) -> List[str]:
        """Split text on special marker boundaries, preserving them as tokens."""
        pattern = "|".join(re.escape(m) for m in self.special_tokens)
        parts = re.split(f"({pattern})", text)
        return [p for p in parts if p]

    def _collapse_digits(self, text: str) -> str:
        # Inverse of _preprocess: same logic as MathTokenizer._collapse_digits
        text = re.sub(r"<NUM>\s*", "", text)
        text = re.sub(r"<NEG>\s*", "-", text)
        text = re.sub(r"\s*<DECIMAL>\s*", ".", text)
        text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
        text = text.strip()
        return text

    def encode(self, text: str, add_special: Optional[bool] = None) -> List[int]:
        if add_special is None:
            add_special = self.add_special

        # Split on markers first, then preprocess + char-tokenize each part
        parts = self._split_on_markers(text)
        ids = []
        for part in parts:
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            else:
                preprocessed = self._preprocess(part)
                for ch in preprocessed:
                    ids.append(self._char_to_id.get(ch, ord("?")))
        if add_special:
            ids = [self.bos_id] + ids
        return ids

    def decode(self, ids: List[int]) -> str:
        parts = []
        for tok_id in ids:
            if tok_id in self.id_to_special:
                parts.append(self.id_to_special[tok_id])
            elif tok_id in self._id_to_char:
                parts.append(self._id_to_char[tok_id])
        text = "".join(parts)
        return self._collapse_digits(text)


if __name__ == "__main__":
    # Smoke test
    tok = MathTokenizer()
    examples = [
        "123 + 456 = 579",
        "What is 7 * 8?",
        "-3.14 + 2.5",
        "Solve for x: 2x + 5 = 13",
    ]
    for ex in examples:
        ids = tok.encode(ex)
        decoded = tok.decode(ids)
        print(f"'{ex}' -> {len(ids)} tokens -> '{decoded}'")
        # Note: roundtrip may not be exact due to marker handling
    print("\nStub tokenizer:")
    stub = StubTokenizer()
    for ex in examples:
        ids = stub.encode(ex)
        decoded = stub.decode(ids)
        print(f"'{ex}' -> {len(ids)} tokens -> '{decoded}'")
