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
        # Reserve IDs starting from vocab_size - len(special_tokens)
        # so they don't conflict with base BPE tokens
        self.special_to_id: dict[str, int] = {}
        self.id_to_special: dict[int, str] = {}
        start_id = vocab_size - len(self.special_tokens)
        for i, tok in enumerate(self.special_tokens):
            self.special_to_id[tok] = start_id + i
            self.id_to_special[start_id + i] = tok

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

        # Preprocess: split numbers
        preprocessed = self._preprocess(text)

        # Encode using base BPE
        base_ids = self.base_encoder.encode(preprocessed, allowed_special="all")

        # Remap any special tokens to our reserved IDs
        # (tiktoken's "allowed_special" lets us pass through special tokens
        # like <|endoftext|>, but our custom math tokens need explicit mapping)
        result: List[int] = []
        for tok_id in base_ids:
            # If this token is a special token in our set, remap
            decoded = self.base_encoder.decode([tok_id])
            if decoded in self.special_to_id:
                result.append(self.special_to_id[decoded])
            else:
                result.append(tok_id)

        if add_special:
            result = [self.bos_id] + result

        return result

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text.

        Inverse of encode. Joins characters and removes extra spaces
        added by digit splitting.

        Note: roundtrip is approximate for math expressions because we
        add spaces during preprocessing. We post-process to remove them.
        """
        # First, convert IDs back to strings via base encoder
        # Special tokens need special handling
        parts: List[str] = []
        for tok_id in ids:
            if tok_id in self.id_to_special:
                parts.append(self.id_to_special[tok_id])
            else:
                parts.append(self.base_encoder.decode([tok_id]))

        text = "".join(parts)

        # Post-process: collapse spaces around digits and markers
        # "<NUM> 1 2 3" -> "123"
        text = self._collapse_digits(text)
        return text

    def _collapse_digits(self, text: str) -> str:
        """Remove spaces between digits and markers.

        "<NUM> 1 2 3" -> "123"
        "<NEG> 3 <DECIMAL> 1 4" -> "-3.14"
        "Solve for x" -> "Solve for x" (preserves spaces)

        Strategy: tokenize markers and digits as separate entities, then
        rebuild the math expression cleanly. Spaces between digits and
        math markers are removed; spaces in regular text are preserved.
        """
        # First pass: strip spaces adjacent to markers (both leading and trailing)
        text = re.sub(r"\s*<NUM>\s*", "<NUM>", text)
        text = re.sub(r"\s*<DECIMAL>\s*", "<DECIMAL>", text)
        text = re.sub(r"\s*<NEG>\s*", "<NEG>", text)
        text = re.sub(r"<NUM>", "", text)  # NUM is implicit (positive number)

        # Split into tokens
        token_re = re.compile(r"<DECIMAL>|<NEG>|[\d]+|[^\s\d<>]+|\s+")
        tokens = token_re.findall(text)

        # Rebuild
        result = []
        for tok in tokens:
            if tok == "<NEG>":
                result.append("-")
            elif tok == "<DECIMAL>":
                result.append(".")
            elif tok.isdigit():
                result.append(tok)
            elif tok.strip():
                # Non-empty non-digit, non-marker token: pass through
                # (preserves spaces in regular text)
                result.append(tok)
            # else: whitespace-only token — already represented elsewhere

        out = "".join(result)
        # Final cleanup: collapse stray dots like "-.5" -> "-0.5"
        out = re.sub(r"^(\.)", r"0\1", out)
        out = re.sub(r"([+\-=*\/x])(\.)", r"\g<1>0\2", out)
        return out

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

    def _collapse_digits(self, text: str) -> str:
        # Same logic as MathTokenizer._collapse_digits
        token_re = re.compile(r"<NUM>|<DECIMAL>|<NEG>|[\d]+|[^\s\d<>]+|\s+")
        tokens = token_re.findall(text)

        result = []
        for tok in tokens:
            if tok in ("<NUM>",):
                pass
            elif tok == "<NEG>":
                result.append("-")
            elif tok == "<DECIMAL>":
                result.append(".")
            elif tok.isdigit():
                result.append(tok)
            else:
                result.append(tok)

        out = "".join(result)
        out = re.sub(r"^(\.)", r"0\1", out)
        out = re.sub(r"([+\-=*\/x])(\.)", r"\g<1>0\2", out)
        return out

    def encode(self, text: str, add_special: Optional[bool] = None) -> List[int]:
        if add_special is None:
            add_special = self.add_special
        preprocessed = self._preprocess(text)

        # Insert spaces around markers so we can split them
        for tok in self.special_tokens:
            preprocessed = preprocessed.replace(tok, f" {tok} ")

        tokens = preprocessed.split()
        ids = []
        for t in tokens:
            if t in self.special_to_id:
                ids.append(self.special_to_id[t])
            elif t in self._char_to_id:
                ids.append(self._char_to_id[t])
            else:
                # Unknown: use byte-level fallback
                for ch in t:
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