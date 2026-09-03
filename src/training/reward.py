"""Verifiable reward function for math reasoning.

Per /ipho-physicist: a verifiable reward is one whose correctness can be
checked deterministically (not by an LLM judge). For math, this means:
- Answer matches reference (numeric or symbolic equivalence)
- Format follows expected structure (<REASON>...</REASON><ANSWER>...</ANSWER>)
- Length is reasonable (shorter correct answers preferred)

The reward is a weighted sum:
    R = w_correct * R_correct + w_format * R_format + w_length * R_length

Weights default to (1.0, 0.3, 0.2).
"""

from __future__ import annotations

import re
from typing import Optional

from src.utils.math_verify import extract_boxed, extract_number, is_equiv, normalize_number


def reward_correctness(pred_text: str, gold: str) -> float:
    """Reward for answer correctness (0 or 1).

    Extracts the answer from \\boxed{...} or last number in pred_text
    and compares to gold.
    """
    if not pred_text or not gold:
        return 0.0

    # Strategy: try is_equiv on the whole text first (handles "50%", "1/2", "\\boxed{42}", etc.)
    if is_equiv(pred_text, gold):
        return 1.0

    # Try \\boxed extraction
    pred_boxed = extract_boxed(pred_text)
    if pred_boxed is not None and is_equiv(pred_boxed, gold):
        return 1.0

    # Try last number (e.g. "the answer is 50%" -> 50, but if pred is "50%",
    # extract_number returns "50", losing the % info)
    pred_num = extract_number(pred_text)
    if pred_num is not None:
        # Compare as numbers (handles fraction/decimal conversions)
        pred_val = normalize_number(pred_num)
        gold_val = normalize_number(gold)
        if pred_val is not None and gold_val is not None and not isinstance(pred_val, str) and not isinstance(gold_val, str):
            try:
                return 1.0 if abs(float(pred_val) - float(gold_val)) <= 1e-6 else 0.0
            except (ValueError, TypeError):
                pass

    # Try the whole pred_text as a numeric value (handles "50%" -> 0.5)
    pred_val = normalize_number(pred_text)
    gold_val = normalize_number(gold)
    if pred_val is not None and gold_val is not None and not isinstance(pred_val, str) and not isinstance(gold_val, str):
        try:
            return 1.0 if abs(float(pred_val) - float(gold_val)) <= 1e-6 else 0.0
        except (ValueError, TypeError):
            pass

    return 0.0


def reward_format(text: str) -> float:
    """Reward for following the expected format.

    Expected structure:
    <REASON>...</REASON>
    <ANSWER>...</ANSWER>
    """
    if not text:
        return 0.0

    has_reason_open = "<REASON>" in text
    has_reason_close = "</REASON>" in text
    has_answer_open = "<ANSWER>" in text
    has_answer_close = "</ANSWER>" in text

    score = 0.0

    # Check reasoning block
    if has_reason_open and has_reason_close:
        idx_r_open = text.find("<REASON>")
        idx_r_close = text.find("</REASON>")
        if idx_r_open < idx_r_close:
            score += 0.5
        else:
            score += 0.25
    elif has_reason_open or has_reason_close:
        # Half a tag pair
        score += 0.0

    # Check answer block
    if has_answer_open and has_answer_close:
        idx_a_open = text.find("<ANSWER>")
        idx_a_close = text.find("</ANSWER>")
        if idx_a_open < idx_a_close:
            # Need REASON block to come BEFORE answer for full credit
            if has_reason_open and has_reason_close:
                idx_r_close = text.find("</REASON>")
                if idx_r_close < idx_a_open:
                    score += 0.5  # proper ordering
                else:
                    score += 0.25  # wrong order: partial credit
            else:
                # No reasoning block at all: half credit for answer
                score += 0.25
        else:
            score += 0.25
    elif has_answer_open or has_answer_close:
        score += 0.0

    return score


def reward_length(text: str, gold: str, optimal_length: int = 200) -> float:
    """Reward for concise correct answers.

    - If answer is wrong: no length bonus
    - If answer is correct and length is close to optimal: full bonus
    - Penalty for very long responses
    """
    if not is_equiv(extract_boxed(text) or extract_number(text) or "", gold):
        return 0.0

    length = len(text)
    if length == 0:
        return 0.0
    if length <= optimal_length:
        return 1.0
    # Decay linearly until 2x optimal, then 0
    if length >= 2 * optimal_length:
        return 0.0
    return 1.0 - (length - optimal_length) / optimal_length


def compute_reward(
    pred_text: str,
    gold: str,
    w_correct: float = 1.0,
    w_format: float = 0.3,
    w_length: float = 0.2,
) -> float:
    """Compute total reward as weighted sum.

    Args:
        pred_text: Model-generated response.
        gold: Reference answer (string).
        w_correct, w_format, w_length: Component weights.

    Returns:
        Scalar reward in [0, w_correct + w_format + w_length].
    """
    r_c = reward_correctness(pred_text, gold)
    r_f = reward_format(pred_text)
    r_l = reward_length(pred_text, gold)
    return w_correct * r_c + w_format * r_f + w_length * r_l


def compute_batch_rewards(
    pred_texts: list,
    golds: list,
    w_correct: float = 1.0,
    w_format: float = 0.3,
    w_length: float = 0.2,
) -> list:
    """Compute rewards for a batch of (pred, gold) pairs."""
    return [
        compute_reward(p, g, w_correct, w_format, w_length)
        for p, g in zip(pred_texts, golds)
    ]


def reward_breakdown(pred_text: str, gold: str) -> dict:
    """Return each component for debugging/analysis."""
    return {
        "correctness": reward_correctness(pred_text, gold),
        "format": reward_format(pred_text),
        "length": reward_length(pred_text, gold),
        "total": compute_reward(pred_text, gold),
    }


if __name__ == "__main__":
    # Smoke tests
    cases = [
        ("The answer is $\\boxed{42}$.", "42"),
        ("x = 5. The answer is 5.", "5"),
        ("This is wrong.", "42"),
        ("<REASON>simple</REASON><ANSWER>42</ANSWER>", "42"),
        ("<REASON>long reasoning ...</REASON>\n<ANSWER>42</ANSWER>", "42"),
    ]
    for pred, gold in cases:
        bd = reward_breakdown(pred, gold)
        print(f"{bd}")