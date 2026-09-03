"""Tests for the GSM8K ``<<expr=result>>`` -> ``<TOOL>calc(expr)</TOOL>``
conversion pass.

The native HuggingFace ``gsm8k/main`` answer field embeds inline
calculator annotations in the form ``<<expr=result>>`` next to every
arithmetic step.  Without rewriting them at format time, the supervised
target contains the raw digit that follows the closing ``>>``, which
recapitulates the multi-digit-arithmetic failure mode the calculator
machinery was built to avoid.

These tests pin the conversion in isolation from the dataset and from
the calculator evaluator.  Integration with the calculator is covered
by ``tests/test_calculator.py::TestTryCallMultiIteration``.
"""

from __future__ import annotations

import pytest

from src.data.gsm8k import (
    GSM8KDataset,
    SYNTHETIC_GSM8K,
    convert_gsm8k_calc_annotations,
    make_synthetic_gsm8k,
)


# ---------------------------------------------------------------------------
# convert_gsm8k_calc_annotations in isolation
# ---------------------------------------------------------------------------

class TestConvertGsm8kCalcAnnotations:
    """Direct tests of the regex conversion."""

    def test_single_annotation_becomes_tool_call(self):
        text = "Janet has <<3+5=8>>8 apples"
        out = convert_gsm8k_calc_annotations(text)
        assert out == "Janet has <TOOL>calc(3+5)</TOOL> apples"

    def test_multiple_annotations(self):
        text = "Area = <<8*5=40>>40 sq units."
        out = convert_gsm8k_calc_annotations(text)
        assert out == "Area = <TOOL>calc(8*5)</TOOL> sq units."

    def test_no_annotations_unchanged(self):
        text = "Janet starts with 3 apples.\n3 + 5 = 8\n#### 8"
        assert convert_gsm8k_calc_annotations(text) == text

    def test_preserves_surrounding_text(self):
        # The conversion should leave all non-annotation text exactly as-is.
        text = "Question? Answer: <<16-3-4=9>>9 (verbatim digits stripped)"
        out = convert_gsm8k_calc_annotations(text)
        assert out == "Question? Answer: <TOOL>calc(16-3-4)</TOOL> (verbatim digits stripped)"

    def test_strips_gsm8k_native_result(self):
        # We must NOT leak the dataset's known result into the training
        # target -- this is the rule that keeps the model from
        # memorising "9" while still teaching it to emit the tool call.
        text = "Step <<2+2=4>>4"
        out = convert_gsm8k_calc_annotations(text)
        assert "4" not in out.split("<TOOL>")[1].split("</TOOL>")[0]

    def test_handles_whitespace_inside_annotation(self):
        text = "<< 10 - 3 = 7 >>"
        out = convert_gsm8k_calc_annotations(text)
        # The whitespace inside the expression is preserved -- this is
        # fine because ``safe_eval`` accepts whitespace, and a more
        # readable trace is friendlier in spot-checks during eval.
        assert "<TOOL>calc(10 - 3)</TOOL>" in out

    def test_realistic_gsm8k_sentence(self):
        # A representative sentence from a real GSM8K answer (with the
        # trailing "result" stripped on purpose).
        text = "She has <<16-3-4=9>>9 apples now."
        out = convert_gsm8k_calc_annotations(text)
        assert out == "She has <TOOL>calc(16-3-4)</TOOL> apples now."


# ---------------------------------------------------------------------------
# format_for_sft integrates the conversion (default-on, off-by-flag)
# ---------------------------------------------------------------------------

class TestGsm8kSftIntegration:
    """Verify the conversion is applied where the training pipeline sees it."""

    def test_format_for_sft_default_applies_conversion(self):
        ds = GSM8KDataset.__new__(GSM8KDataset)
        ds.examples = [
            {
                "question": "How many?",
                "answer": "5 + 3 = <<5+3=8>>8 apples\n#### 8",
            }
        ]
        ds.split = "train"
        _prompt, target = ds.format_for_sft(0)
        assert "<TOOL>calc(5+3)</TOOL>" in target
        assert ">>8" not in target  # native annotation must be gone

    def test_format_for_sft_can_be_disabled(self):
        ds = GSM8KDataset.__new__(GSM8KDataset)
        ds.examples = [
            {
                "question": "How many?",
                "answer": "5 + 3 = <<5+3=8>>8 apples\n#### 8",
            }
        ]
        ds.split = "train"
        _prompt, target = ds.format_for_sft(0, convert_calc_annotations=False)
        # Annotation survives unchanged.
        assert "<<5+3=8>>8" in target
        assert "<TOOL>" not in target

    def test_format_for_pretrain_applies_conversion(self):
        ds = GSM8KDataset.__new__(GSM8KDataset)
        ds.examples = [
            {
                "question": "How many?",
                "answer": "5 + 3 = <<5+3=8>>8 apples\n#### 8",
            }
        ]
        ds.split = "train"
        text = ds.format_for_pretrain(0)
        assert "<TOOL>calc(5+3)</TOOL>" in text
        assert ">>8" not in text


# ---------------------------------------------------------------------------
# End-to-end on the synthetic GSM8K examples shipped with the repo
# ---------------------------------------------------------------------------

class TestSyntheticGsm8kNoAnnotations:
    """The synthetic GSM8K builtins in this repo don't carry
    ``<<...>>`` annotations -- they're plain-text.  Confirm that the
    conversion pass is a no-op on them so we don't accidentally corrupt
    the test fixtures.
    """

    @pytest.mark.parametrize("ex", SYNTHETIC_GSM8K)
    def test_synthetic_unchanged(self, ex):
        out = convert_gsm8k_calc_annotations(ex["answer"])
        assert out == ex["answer"], "synthetic fixture must not be mutated"
