"""Tests for the calculator tool / column-by-column CoT helpers."""

import pytest

from src.inference.calculator import (
    CalculatorError,
    call_calculator,
    column_cot_addition,
    column_cot_multiplication,
    extract_tool_call,
    safe_eval,
    try_call,
)


# ---------------------------------------------------------------------------
# safe_eval
# ---------------------------------------------------------------------------

class TestSafeEval:
    @pytest.mark.parametrize("expr,expected", [
        ("2 + 3", 5),
        ("(2 + 3) * 4", 20),
        ("100 - 25", 75),
        ("2 ** 10", 1024),
        ("7 // 2", 3),
        ("7 % 3", 1),
        ("3.14 * 2", 6.28),
        ("-5 + 10", 5),
        ("abs(-7)", 7),
        ("round(3.5)", 4),
        ("sqrt(16)", 4.0),
        ("max(1, 2, 3)", 3),
        ("min(1, 2, 3)", 1),
        ("pow(2, 8)", 256),
        ("3 == 3", True),
        ("3 != 4", True),
        ("5 < 10", True),
    ])
    def test_valid(self, expr, expected):
        assert safe_eval(expr) == pytest.approx(expected) \
               if isinstance(expected, float) else safe_eval(expr) == expected

    @pytest.mark.parametrize("bad", [
        "__import__('os')",
        "open('foo')",
        "x.y",
        "[1, 2, 3]",
        "{'a': 1}",
        "lambda x: x",
        "class X: pass",
        "1; 2",                # semicolon
        "x_underscore",        # underscore
        "$dollar",
        "\\backslash",
        "@decorator",
        "for i in []: pass",
        "",                    # empty
        " " * 600,             # too long
    ])
    def test_invalid(self, bad):
        with pytest.raises(CalculatorError):
            safe_eval(bad)


# ---------------------------------------------------------------------------
# try_call / extract_tool_call
# ---------------------------------------------------------------------------

class TestToolExtraction:
    def test_extract_basic(self):
        found = extract_tool_call("x = <TOOL>calc(2+3)</TOOL>")
        assert found is not None
        tool, expr, _, _ = found
        assert tool == "calc"
        assert expr.strip() == "2+3"

    def test_extract_case_insensitive(self):
        found = extract_tool_call("try <tool>CALC(7*6)</tool>")
        assert found is not None
        assert found[0] == "calc"
        assert found[1].strip() == "7*6"

    def test_no_call(self):
        assert extract_tool_call("plain text, no tool") is None

    def test_try_call_replaces(self):
        text = "answer is <TOOL>calc(567 + 489)</TOOL>"
        new, status = try_call(text)
        assert status == "ok"
        assert "1056" in new
        assert "<TOOL>" not in new

    def test_try_call_no_call(self):
        text = "no tool here"
        new, status = try_call(text)
        assert status == "no_call"
        assert new == text

    def test_try_call_error_status(self):
        # Use an expression that's syntactically valid in the regex but
        # rejected by safe_eval (e.g. underscore identifier).
        text = "bad <TOOL>calc(foo_bar)</TOOL> call"
        new, status = try_call(text)
        assert status == "error"
        # original is returned unchanged on error
        assert new == text


# ---------------------------------------------------------------------------
# Column-by-column CoT
# ---------------------------------------------------------------------------

class TestColumnCoT:
    def test_addition_567_489(self):
        out = column_cot_addition(567, 489)
        # First line should be ones column
        assert "ones" in out.split("\n")[0]
        # Should contain 1056
        assert "1056" in out
        # Three intermediate carries (ones, tens, hundreds); final thousands
        # row writes the carry-out without producing another carry.
        assert out.count("carry 1") == 3

    def test_addition_no_carry(self):
        out = column_cot_addition(123, 456)
        assert "579" in out
        # No "carry N" should appear (only "Result:" line)
        assert "carry " not in out

    def test_addition_unequal_lengths(self):
        # 12345 + 678
        out = column_cot_addition(12345, 678)
        assert "13023" in out
        # Should start with ones column
        first = out.split("\n")[0]
        assert "ones" in first

    def test_addition_zero(self):
        out = column_cot_addition(0, 0)
        assert "Result: 0" in out

    def test_addition_single_digit(self):
        out = column_cot_addition(7, 8)
        assert "Result: 15" in out

    def test_multiplication_small(self):
        out = column_cot_multiplication(23, 47)
        assert "Result: 1081" in out
        # Should reference both 7 and 4
        assert "× 7" in out
        assert "× 4" in out

    def test_multiplication_single_digit(self):
        out = column_cot_multiplication(6, 7)
        assert "Result: 42" in out


# ---------------------------------------------------------------------------
# try_call: multi-call iteration (added after the fix for the
# "single search, only first call resolved" path)
# ---------------------------------------------------------------------------

class TestTryCallMultiIteration:
    """After the fix, ``try_call`` resolves every ``<TOOL>calc(...)</TOOL>``
    block in left-to-right order rather than only the first.  These tests
    pin the new behaviour while keeping the legacy single-call contract
    intact (``no_call`` / ``ok`` for the happy paths)."""

    def test_two_calls_left_to_right(self):
        text = (
            "step1: <TOOL>calc(2+3)</TOOL>, "
            "step2: <TOOL>calc(10*4)</TOOL>"
        )
        new, status = try_call(text)
        assert status == "ok"
        assert new == "step1: 5, step2: 40"
        assert "<TOOL>" not in new

    def test_three_calls(self):
        text = (
            "<TOOL>calc(100-25)</TOOL> "
            "<TOOL>calc(75/3)</TOOL> "
            "<TOOL>calc(25+1)</TOOL>"
        )
        new, status = try_call(text)
        assert status == "ok"
        assert new == "75 25 26"

    def test_no_call_still_status_no_call(self):
        # The "no_call" status is the legacy contract for input that has
        # no tool calls -- verify we preserved it after the loop refactor.
        text = "just plain text"
        new, status = try_call(text)
        assert status == "no_call"
        assert new == text

    def test_error_status_on_bad_expression(self):
        # Underscore triggers the calculator's safety rejection.
        text = "<TOOL>calc(foo_bar)</TOOL>"
        new, status = try_call(text)
        assert status == "error"
        assert new == text  # original returned unchanged

    def test_error_does_not_drop_earlier_resolved_call(self):
        # First call resolves to a benign integer; second fails.
        # We should keep the first resolution and report "error".
        text = "<TOOL>calc(1+1)</TOOL> then <TOOL>calc(bad_thing)</TOOL>"
        new, status = try_call(text)
        assert status == "error"
        # First call's "2" should remain spliced in even though the
        # second errored.
        assert "2" in new
        # The failing call should still be present (unresolved).
        assert "<TOOL>calc(bad_thing)</TOOL>" in new
