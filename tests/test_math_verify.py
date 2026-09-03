r"""M1 regression tests: is_equiv last-token fallback must not false-positive.

The old implementation returned True for "x^2 + y^2 = z^2" vs "z^2" because
both have the same last token "z^2".  This was a false positive.

After the M1 fix, the last-token fallback only matches when:
- Both strings have <= 4 tokens
- The last token is a plain number (regex -?\d+(?:\.\d+)?)
- Both strings have the same number of tokens

A separate "answer = N" vs "N" rule preserves the legitimate use case.
"""

from __future__ import annotations

import pytest

from src.utils.math_verify import is_equiv


class TestIsEquivLastTokenFallback:
    """M1 fix: last-token fallback must not false-positive on symbolic math."""

    def test_symbolic_vs_number_false_positive(self):
        """M1 fix: 'x^2 + y^2 = z^2' must NOT match 'z^2'.

        These are mathematically different: one is an equation, the other is
        a single variable. The old last-token match returned True incorrectly.
        """
        assert is_equiv("x^2 + y^2 = z^2", "z^2") is False, (
            "M1 regression: symbolic expression matched on last-token 'z^2'. "
            "This false positive would corrupt GRPO reward signals."
        )

    def test_symbolic_vs_symbolic_same_last_token(self):
        """Both symbolic but different equations must not match."""
        assert is_equiv("x^2 + y^2 = z^2", "a^2 + b^2 = z^2") is False
        assert is_equiv("a = b", "c = b") is False

    def test_plain_number_exact(self):
        """Plain numbers should still match exactly."""
        assert is_equiv("42", "42") is True
        assert is_equiv("3.14", "3.14") is True
        assert is_equiv("99", "99") is True

    def test_numeric_equivalence_still_works(self):
        """Numeric equivalence (step 2) must remain unaffected by M1."""
        assert is_equiv("1/2", "0.5") is True
        assert is_equiv("0.5", "1/2") is True
        assert is_equiv("50%", "0.5") is True
        assert is_equiv("-3.14", "-3.14") is True

    def test_numeric_tolerance(self):
        """Numeric comparison within tolerance."""
        # Clearly within 1e-6
        assert is_equiv("3.14159", "3.14159") is True  # identical
        assert is_equiv("3.14", "3.14") is True
        # Clearly outside (diff = 1e-5 >> tol)
        assert is_equiv("3.14159", "3.14160") is False
        # Inside 1e-6 tolerance (diff = 1e-8)
        assert is_equiv("3.14159265", "3.14159266") is True

    def test_answer_equals_n_vs_n(self):
        """M1 fix: 'answer = N' vs 'N' must still match (the legitimate case)."""
        assert is_equiv("answer = 42", "42") is True, (
            "M1 regression: 'answer = 42' vs '42' should still match. "
            "This is the legitimate last-token use case."
        )
        assert is_equiv("ans = 7", "7") is True
        assert is_equiv("result = 3.14", "3.14") is True

    def test_answer_equals_n_vs_different_n(self):
        """'answer = 42' vs '7' must not match."""
        assert is_equiv("answer = 42", "7") is False
        assert is_equiv("answer = 42", "answer = 7") is False

    def test_plain_number_same_token_count(self):
        """Short number strings with same token count can still match."""
        assert is_equiv("42", "42") is True
        assert is_equiv("42", "43") is False  # different value

    def test_none_input(self):
        """None inputs must return False."""
        assert is_equiv(None, "42") is False
        assert is_equiv("42", None) is False
        assert is_equiv(None, None) is False

    def test_empty_input(self):
        """Empty inputs must return False."""
        assert is_equiv("", "42") is False
        assert is_equiv("42", "") is False
        assert is_equiv(" ", "42") is False
