"""Reward function tests.

Verified by /imo-mathematician:
- Correct answer -> reward = 1.0
- Wrong answer -> reward = 0.0
- Format bonus for <REASON>/<ANSWER> structure
- Length bonus for short correct answers
- Boxed answer extraction
"""

from __future__ import annotations

import pytest

from src.training.reward import (
    compute_reward,
    compute_batch_rewards,
    reward_correctness,
    reward_format,
    reward_length,
    reward_breakdown,
)


class TestCorrectnessReward:
    """Correctness reward is binary (0 or 1)."""

    @pytest.mark.parametrize("pred,gold", [
        ("42", "42"),
        ("$\\boxed{42}$", "42"),
        ("The answer is 42.", "42"),
        ("x = 42", "42"),
        ("42.0", "42"),
        ("0.5", "1/2"),
        ("1/2", "0.5"),
        ("50%", "0.5"),
        ("-3.14", "-3.14"),
        ("0", "0"),
    ])
    def test_correct_exact_match(self, pred, gold):
        assert reward_correctness(pred, gold) == 1.0

    @pytest.mark.parametrize("pred,gold", [
        ("wrong", "42"),
        ("41", "42"),
        ("", "42"),
        (None, "42"),
    ])
    def test_wrong_no_match(self, pred, gold):
        assert reward_correctness(pred, gold) == 0.0

    def test_none_pred(self):
        assert reward_correctness(None, "42") == 0.0

    def test_none_gold(self):
        assert reward_correctness("42", None) == 0.0

    def test_boxed_extraction(self):
        """\\boxed{...} is extracted and matched."""
        text = r"Therefore the answer is $\boxed{579}$."
        gold = "579"
        assert reward_correctness(text, gold) == 1.0


class TestFormatReward:
    """Format reward for <REASON>...</REASON><ANSWER>...</ANSWER>."""

    def test_full_format(self):
        """Full format gets max score."""
        text = "<REASON>Step 1... step 2...</REASON><ANSWER>42</ANSWER>"
        assert reward_format(text) == 1.0

    def test_reason_only(self):
        """Only reason open/close."""
        text = "<REASON>thinking...</REASON>"
        assert 0 < reward_format(text) <= 0.5

    def test_answer_only(self):
        """Only answer open/close."""
        text = "<ANSWER>42</ANSWER>"
        assert 0 < reward_format(text) <= 0.5

    def test_no_format(self):
        """No markers gets 0."""
        assert reward_format("42") == 0.0

    def test_empty(self):
        assert reward_format("") == 0.0
        assert reward_format(None) == 0.0

    def test_reason_before_answer(self):
        """Reason must come before answer for full format credit."""
        text = "<ANSWER>42</ANSWER><REASON>mistake</REASON>"
        # Wrong order: partial credit (format score should be < 1.0)
        score = reward_format(text)
        assert 0 < score < 1.0, f"Wrong order should give partial credit, got {score}"


class TestLengthReward:
    """Short correct answers get length bonus."""

    def test_short_correct(self):
        """Short correct answer: full bonus."""
        gold = "42"
        text = "<ANSWER>42</ANSWER>"
        assert reward_length(text, gold) == 1.0

    def test_correct_long(self):
        """Correct but very long: zero bonus."""
        gold = "42"
        text = "Here is my reasoning step by step... " * 20 + "<ANSWER>42</ANSWER>"
        assert reward_length(text, gold) == 0.0

    def test_wrong_no_bonus(self):
        """Wrong answer: no length bonus regardless of length."""
        gold = "42"
        wrong = "99"
        # Even if it's short, wrong = 0 length bonus
        assert reward_length(wrong, gold) == 0.0


class TestTotalReward:
    """Total reward = weighted sum of components."""

    def test_correct_full_reward(self):
        """Perfect: correct + full format + optimal length."""
        pred = "<REASON>42</REASON><ANSWER>42</ANSWER>"
        gold = "42"
        r = compute_reward(pred, gold)
        # correctness=1, format=1, length=1
        # default: 1.0*1 + 0.3*1 + 0.2*1 = 1.5
        assert r == pytest.approx(1.5)

    def test_wrong_partial_reward(self):
        """Wrong answer: no correctness reward but format may still apply."""
        pred = "<REASON>wrong</REASON><ANSWER>wrong</ANSWER>"
        gold = "42"
        r = compute_reward(pred, gold)
        # correctness=0, format=1.0 (full), length=0
        # total = 0 + 0.3*1.0 + 0.2*0 = 0.3
        assert r == pytest.approx(0.3)

    def test_wrong_no_format_no_reward(self):
        """Wrong answer, no format: no reward."""
        pred = "I think it's 99"
        gold = "42"
        r = compute_reward(pred, gold)
        # correctness=0, format=0, length=0
        assert r == 0.0

    def test_batch_rewards(self):
        """compute_batch_rewards works for lists."""
        preds = ["<ANSWER>42</ANSWER>", "wrong"]
        golds = ["42", "42"]
        rewards = compute_batch_rewards(preds, golds)
        assert len(rewards) == 2
        assert rewards[0] > 0
        assert rewards[1] == 0.0

    def test_breakdown(self):
        """reward_breakdown returns all components."""
        pred = "<REASON>42</REASON><ANSWER>42</ANSWER>"
        gold = "42"
        bd = reward_breakdown(pred, gold)
        assert "correctness" in bd
        assert "format" in bd
        assert "length" in bd
        assert "total" in bd


class TestBoxedExtraction:
    """Test the answer extraction from various formats."""

    @pytest.mark.parametrize("text,expected", [
        (r"$\boxed{42}$", "42"),
        (r"$\boxed{\frac{1}{2}}$", "\\frac{1}{2}"),
        (r"The answer is $\boxed{-3.14}$.", "-3.14"),
        ("No boxed here", None),
        ("", None),
        (None, None),
    ])
    def test_boxed_patterns(self, text, expected):
        from src.utils.math_verify import extract_boxed
        result = extract_boxed(text) if text else None
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])