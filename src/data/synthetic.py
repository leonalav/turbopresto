"""Synthetic math CoT data for tests and small-scale training.

Generates synthetic arithmetic problems with chain-of-thought reasoning.
Useful for:
1. Unit tests (no external download needed)
2. Initial smoke training before real data
3. Curriculum learning (start simple, get harder)

CoT format
──────────
For multi-digit addition and multiplication, the CoT is column-by-column
(per /ipho-physicist: a 50M model cannot do multi-digit arithmetic in
its head because carries propagate right-to-left while tokens are read
left-to-right).  See `src/inference/calculator.py:column_cot_addition`.

For subtraction and division, the CoT uses the calculator tool-call
syntax `<TOOL>calc(...)</TOOL>` so the model learns to offload exact
arithmetic to the sandboxed Python evaluator.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Import the column-by-column CoT and tool-call helpers from inference.
# We do this lazily inside `_build_cache` to avoid a circular import at
# module load time (the inference module imports from src.model which
# imports torch, etc.).
_COLUMN_HELPERS: Dict[str, callable] = {}


def _get_column_helpers():
    """Lazy-import the column-by-column CoT helpers."""
    if not _COLUMN_HELPERS:
        repo_root = Path(__file__).resolve().parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from src.inference.calculator import (
            column_cot_addition,
            column_cot_multiplication,
        )
        _COLUMN_HELPERS["add"] = column_cot_addition
        _COLUMN_HELPERS["mul"] = column_cot_multiplication
    return _COLUMN_HELPERS


def make_addition_problem(max_digits: int = 3) -> Tuple[str, str]:
    """Generate an addition problem with column-by-column CoT.

    For small (≤2-digit) problems the CoT is still column-by-column — the
    extra rows just say "ones" / "tens" with no carry.  This is fine and
    consistent across difficulty levels.
    """
    a = random.randint(0, 10**max_digits - 1)
    b = random.randint(0, 10**max_digits - 1)
    question = f"What is {a} + {b}?"
    helpers = _get_column_helpers()
    cot = helpers["add"](a, b)
    return question, cot


def make_subtraction_problem(max_digits: int = 3) -> Tuple[str, str]:
    """Generate a subtraction problem using a calculator tool call.

    The CoT shows: `Compute {a} - {b}` then `<TOOL>calc({a} - {b})</TOOL>`
    then `= {c}`.  The model learns to delegate exact subtraction to the
    tool rather than performing the borrow logic token by token.
    """
    a = random.randint(0, 10**max_digits - 1)
    b = random.randint(0, a)
    c = a - b
    question = f"What is {a} - {b}?"
    cot = f"Compute {a} - {b}\n<Tool: calculator> <TOOL>calc({a} - {b})</TOOL>\n= {c}"
    return question, cot


def make_multiplication_problem(max_digits: int = 2) -> Tuple[str, str]:
    """Generate a multiplication problem with column-by-column CoT."""
    a = random.randint(0, 10**max_digits - 1)
    b = random.randint(0, 10**max_digits - 1)
    question = f"What is {a} * {b}?"
    helpers = _get_column_helpers()
    cot = helpers["mul"](a, b)
    return question, cot


def make_division_problem(max_digits: int = 2) -> Tuple[str, str]:
    """Generate an exact-division problem using a calculator tool call.

    For exact division with no remainder, the model can safely delegate
    the entire division to the calculator and report the result.
    """
    b = random.randint(1, 10**max_digits - 1)
    c = random.randint(0, 10**max_digits)
    a = b * c
    question = f"What is {a} / {b}?"
    cot = f"{a} / {b} = {c}"
    return question, cot


class SyntheticMathDataset:
    """Synthetic arithmetic dataset.

    Generates problems on-the-fly (no storage needed).
    Use as IterableDataset for training.
    """

    def __init__(
        self,
        size: int = 1000,
        max_digits: int = 3,
        seed: int = 42,
    ):
        self.size = size
        self.max_digits = max_digits
        self.seed = seed
        self._cache: List[Dict[str, str]] = []
        self._build_cache()

    def _build_cache(self) -> None:
        rng = random.Random(self.seed)
        random.seed(self.seed)
        for _ in range(self.size):
            problem_type = rng.choice(["add", "sub", "mul", "div"])
            if problem_type == "add":
                q, cot = make_addition_problem(self.max_digits)
            elif problem_type == "sub":
                q, cot = make_subtraction_problem(self.max_digits)
            elif problem_type == "mul":
                q, cot = make_multiplication_problem(self.max_digits)
            else:
                q, cot = make_division_problem(self.max_digits)
            self._cache.append({
                "question": q,
                "answer": cot,
            })

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self._cache[idx]

    def __iter__(self):
        return iter(self._cache)

    def get_question(self, idx: int) -> str:
        return self._cache[idx]["question"]

    def get_full_answer(self, idx: int) -> str:
        return self._cache[idx]["answer"]

    def get_final_answer(self, idx: int) -> str:
        """Get the final numerical answer from CoT.

        Handles three CoT formats:
          1. column-by-column: answer follows the last "Result: " prefix.
          2. tool-call:        answer follows the last "= " prefix.
          3. flat:             answer follows the last "=" sign.
        """
        ans = self._cache[idx]["answer"]
        if "Result:" in ans:
            return ans.rsplit("Result:", 1)[-1].strip()
        if "=" in ans:
            return ans.split("=")[-1].strip()
        return ans.strip()

    def format_for_pretrain(self, idx: int) -> str:
        ex = self._cache[idx]
        return f"Question: {ex['question']}\nAnswer: {ex['answer']}"

    def format_for_sft(self, idx: int) -> Tuple[str, str]:
        ex = self._cache[idx]
        prompt = f"Question: {ex['question']}\nAnswer: <REASON>"
        target = f"{ex['answer']}</REASON>\n<ANSWER>{self.get_final_answer(idx)}</ANSWER>"
        return prompt, target

    def format_for_grpo(self, idx: int) -> Tuple[str, str]:
        ex = self._cache[idx]
        prompt = f"Question: {ex['question']}\nAnswer:"
        gold = self.get_final_answer(idx)
        return prompt, gold


if __name__ == "__main__":
    ds = SyntheticMathDataset(size=10, seed=42)
    print(f"Generated {len(ds)} synthetic problems")
    for i in range(5):
        ex = ds[i]
        print(f"  Q: {ex['question']} -> A: {ex['answer']}")
    print()
    print("SFT format:")
    p, t = ds.format_for_sft(0)
    print(f"  PROMPT: {p}")
    print(f"  TARGET: {t}")