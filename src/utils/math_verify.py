"""Math answer verification utilities.

Per /imo-mathematician: the correctness of math reasoning depends on
exact match between generated and reference answers. Naive string
comparison fails on equivalent forms like "1/2" vs "0.5" or "x=2" vs "2".

This module provides:
- `extract_boxed`: extract \\boxed{...} content
- `normalize_number`: parse various number formats
- `is_equiv`: check if two answers are equivalent (numerically or symbolically)

All functions are pure and deterministic for testability.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional, Union


# Module-level compiled regex (M1 fix).
# Matches plain integer or decimal strings only — no operators, no variables.
_PLAIN_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def extract_boxed(text: str) -> Optional[str]:
    """Extract the LAST \\boxed{...} content from text.

    Handles nested braces up to 2 levels. Returns the content without
    the \\boxed wrapper, or None if not found.
    """
    # Find \boxed{ ... } with matching braces (up to 3 levels deep)
    # We'll use a simple state-machine approach
    idx = text.rfind("\\boxed{")
    if idx < 0:
        # Also try \boxed without backslash (LaTeX variations)
        idx = text.rfind("boxed{")
    if idx < 0:
        return None

    # Find the opening brace
    start = text.find("{", idx)
    if start < 0:
        return None

    # Walk forward counting braces
    depth = 1
    i = start + 1
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
        i += 1
    return None


def normalize_number(s: str) -> Optional[Union[float, Fraction, str]]:
    """Parse a number string in various formats.

    Handles:
    - Plain integers: "42"
    - Decimals: "3.14"
    - Fractions: "1/2"
    - Percentages: "50%"
    - Negative: "-5"

    Returns:
    - float for decimals
    - Fraction for exact rationals
    - str for unparseable content (variables, expressions)
    - None if empty/garbage
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None

    # Strip surrounding delimiters
    s = s.strip("$").strip()
    s = s.strip("()").strip()
    s = s.strip()

    # Try percentage: "50%"
    pct_match = re.match(r"^(-?\d+(?:\.\d+)?)\s*%$", s)
    if pct_match:
        try:
            return float(pct_match.group(1)) / 100.0
        except ValueError:
            pass

    # Try fraction: "1/2"
    frac_match = re.match(r"^(-?\d+)\s*/\s*(\d+)$", s)
    if frac_match:
        try:
            return Fraction(int(frac_match.group(1)), int(frac_match.group(2)))
        except (ValueError, ZeroDivisionError):
            pass

    # Try plain integer or decimal
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass

    # Try as fraction with implicit denominator: "1\\frac{1}{2}"
    if "\\frac" in s:
        frac_re = re.search(r"\\frac\{(-?\d+)\}\{(\d+)\}", s)
        if frac_re:
            try:
                return Fraction(int(frac_re.group(1)), int(frac_re.group(2)))
            except (ValueError, ZeroDivisionError):
                pass

    # Return as string (might be expression like "x = 2")
    return s


def is_equiv(pred: Optional[str], gold: Optional[str], tol: float = 1e-6) -> bool:
    """Check if two answers are equivalent.

    Tries multiple comparison strategies in order:
    1. Exact string match
    2. Numeric equivalence (with tolerance)
    3. Normalized form match (strip $, whitespace, etc.)

    Args:
        pred: Predicted answer string
        gold: Gold/reference answer string
        tol: Tolerance for numeric comparison

    Returns:
        True if equivalent
    """
    if pred is None or gold is None:
        return False

    pred_n = pred.strip().strip("$").strip()
    gold_n = gold.strip().strip("$").strip()

    if not pred_n or not gold_n:
        return False

    # 1. Exact match
    if pred_n == gold_n:
        return True

    # 2. Numeric equivalence
    p_val = normalize_number(pred_n)
    g_val = normalize_number(gold_n)
    if p_val is not None and g_val is not None:
        if isinstance(p_val, (int, float)) and isinstance(g_val, (int, float)):
            return abs(float(p_val) - float(g_val)) <= tol
        if isinstance(p_val, Fraction) and isinstance(g_val, Fraction):
            return p_val == g_val
        if isinstance(p_val, (int, float)) and isinstance(g_val, Fraction):
            return abs(float(p_val) - float(g_val)) <= tol
        if isinstance(p_val, Fraction) and isinstance(g_val, (int, float)):
            return abs(float(p_val) - float(g_val)) <= tol

    # 3. Tight last-token numeric match (M1 fix: reject symbolic expressions).
    # Matches only when both sides are short and the last token is a plain number.
    # This eliminates false positives like "x^2 + y^2 = z^2" vs "z^2".
    pred_tokens = pred_n.split()
    gold_tokens = gold_n.split()
    if (pred_tokens and gold_tokens
            and _PLAIN_NUMBER_RE.match(pred_tokens[-1])
            and _PLAIN_NUMBER_RE.match(gold_tokens[-1])
            and len(pred_tokens) == len(gold_tokens)
            and len(pred_tokens) <= 4):
        if pred_tokens[-1] == gold_tokens[-1]:
            return True

    # 3b. Explicit "answer = N" vs "N" fallback.
    # Catches the GSM8K convention where the predicted side has extra framing.
    if (pred_tokens and gold_tokens
            and len(pred_tokens) == len(gold_tokens) + 2
            and pred_tokens[-2] == "="
            and pred_tokens[-3].lower() in {"answer", "ans", "result"}
            and gold_tokens[-1] == pred_tokens[-1]
            and _PLAIN_NUMBER_RE.match(pred_tokens[-1])):
        return True

    # 4. Numeric substring match — only when numbers are in the same non-numeric context.
    # (M1 fix: step 4 was too loose and matched "2" inside "x^2" with "2" inside "z^2".)
    p_nums = re.findall(r"-?\d+(?:\.\d+)?", pred_n)
    g_nums = re.findall(r"-?\d+(?:\.\d+)?", gold_n)
    if p_nums and g_nums and p_nums[-1] == g_nums[-1]:
        # Same last number; now verify the non-numeric context matches.
        # Strip all digits from each string — if the remaining symbolic parts are
        # identical (or one is a non-empty prefix of the other), the numbers are
        # in the same expression context.
        p_sym = re.sub(r"\d+(?:\.\d+)?", "", pred_n).strip()
        g_sym = re.sub(r"\d+(?:\.\d+)?", "", gold_n).strip()
        # Non-empty but different symbolic parts mean numbers are embedded in
        # different expressions (e.g. "x^2" vs "z^2" in "x^2 + y^2 = z^2" vs "z^2").
        if p_sym and g_sym and p_sym != g_sym:
            pass  # numbers in different contexts — not a match
        else:
            # Same context (both empty) or one is a superset of the other:
            # "x = 5" vs "5" and "the answer is 42" vs "42" are legitimate.
            try:
                pf = float(p_nums[-1])
                gf = float(g_nums[-1])
                return abs(pf - gf) <= tol
            except ValueError:
                pass

    return False


def extract_number(s: str) -> Optional[str]:
    """Extract the LAST number from a string.

    Returns:
    - The matched string (e.g., "42", "-3.14", "1/2")
    - None if no number found
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None

    # Try fractions first (greedy, including negatives)
    frac_matches = re.findall(r"-?\d+\s*/\s*\d+", s)
    if frac_matches:
        return frac_matches[-1].replace(" ", "")

    # Try plain numbers
    num_matches = re.findall(r"-?\d+(?:\.\d+)?", s)
    if num_matches:
        return num_matches[-1]

    return None


def extract_first_number(s: str) -> Optional[str]:
    """Extract the FIRST number from a string."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    frac_matches = re.findall(r"-?\d+\s*/\s*\d+", s)
    if frac_matches:
        return frac_matches[0].replace(" ", "")
    num_matches = re.findall(r"-?\d+(?:\.\d+)?", s)
    if num_matches:
        return num_matches[0]
    return None


# ---------------------------------------------------------------------------
# GSM8K-style answer extraction
# ---------------------------------------------------------------------------

GSM8K_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract answer from GSM8K format (#### answer)."""
    m = GSM8K_ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    # Fall back to last number
    return extract_number(text)


# ---------------------------------------------------------------------------
# MATH dataset style
# ---------------------------------------------------------------------------

def extract_math_answer(text: str) -> Optional[str]:
    """Extract answer from MATH dataset format (\\boxed{...})."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    # Fall back to last number
    return extract_number(text)


if __name__ == "__main__":
    # Smoke tests
    tests = [
        ("123", "123"),
        ("42", "42"),
        ("1/2", "0.5"),
        ("0.5", "1/2"),
        ("-3.14", "-3.14"),
        ("50%", "0.5"),
        ("x = 5", "5"),
        ("the answer is 42", "42"),
        ("x^2 + y^2 = z^2", "z^2"),  # symbolic, will fall to str compare
    ]
    for p, g in tests:
        eq = is_equiv(p, g)
        print(f"is_equiv({p!r}, {g!r}) = {eq}")

    print("\nBoxed extraction:")
    samples = [
        r"The answer is $\boxed{42}$.",
        r"Therefore $\boxed{\frac{1}{2}}$",
        r"Result: $\boxed{3.14}$",
        r"No boxed answer here",
    ]
    for s in samples:
        print(f"  {s!r} -> {extract_boxed(s)!r}")