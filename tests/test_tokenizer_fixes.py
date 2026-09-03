"""Smoke test for C3 + C4 fixes in math_tokenizer.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer.math_tokenizer import StubTokenizer


def test_default_digit_split_disabled():
    """C3 fix: digit_split default is False."""
    tok = StubTokenizer()
    assert tok.digit_split is False, "digit_split should default to False"
    # "123" should encode as 3 separate chars (or a few), not bloated with markers
    ids = tok.encode("123")
    # Should NOT contain <NUM> marker
    assert tok.special_to_id["<NUM>"] not in ids, "Should not insert <NUM> when digit_split=False"
    print(f"  PASS: digit_split=False default; '123' -> {ids} (no marker)")
    return True


def test_digit_split_roundtrip():
    """C4 fix: digit-split markers decode back to original digits."""
    tok = StubTokenizer()
    tok.digit_split = True

    cases = [
        "123",
        "123 + 456",
        "-3.14",
        "5",
        "0",
        "12.34",
    ]
    for ex in cases:
        ids = tok.encode(ex)
        dec = tok.decode(ids)
        # Normalize: strip spaces from both (digit splitting adds them)
        norm_in = ex.replace(" ", "")
        norm_out = dec.replace(" ", "")
        ok = (norm_in == norm_out)
        marker_present = (
            tok.special_to_id["<NUM>"] in ids
            or tok.special_to_id["<NEG>"] in ids
        )
        print(f"  '{ex}' -> {ids} -> '{dec}' | {'PASS' if ok else 'FAIL'}")
        assert ok, f"Roundtrip failed for {ex!r}: got {dec!r}"
        if any(c.isdigit() for c in ex):
            assert marker_present, f"Markers should be present for {ex!r}"
    return True


def test_preexisting_markers():
    """C1 still works: <REASON>, <ANSWER>, etc. are recognized."""
    tok = StubTokenizer()
    tok.digit_split = False
    text = "Answer: <REASON> reasoning </REASON> <ANSWER> 42 </ANSWER>"
    ids = tok.encode(text)
    # All four markers should map to single reserved IDs
    assert tok.special_to_id["<REASON>"] in ids
    assert tok.special_to_id["</REASON>"] in ids
    assert tok.special_to_id["<ANSWER>"] in ids
    assert tok.special_to_id["</ANSWER>"] in ids
    print(f"  PASS: all 4 markers in IDs ({ids})")
    return True


def test_plain_text_unchanged():
    """Plain text roundtrips perfectly (no digit splitting)."""
    tok = StubTokenizer()
    tok.digit_split = False
    cases = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "What is 7 * 8?",
        "Solve for x: 2x + 5 = 13",
    ]
    for ex in cases:
        ids = tok.encode(ex)
        dec = tok.decode(ids)
        # With digit_split=False, roundtrip should be exact
        assert ex == dec, f"Roundtrip failed: {ex!r} -> {dec!r}"
        print(f"  PASS: '{ex}' -> '{dec}'")
    return True


if __name__ == "__main__":
    print("=== C3+C4 verification: MathTokenizer roundtrip + markers ===\n")
    print("Test 1: digit_split default is False (C3 fix)")
    test_default_digit_split_disabled()
    print("\nTest 2: digit_split=True markers decode back (C4 fix)")
    test_digit_split_roundtrip()
    print("\nTest 3: Pre-existing markers <REASON>, <ANSWER> still work (C1)")
    test_preexisting_markers()
    print("\nTest 4: Plain text roundtrip exact (no digit splitting)")
    test_plain_text_unchanged()
    print("\n=== ALL TESTS PASSED ===")
