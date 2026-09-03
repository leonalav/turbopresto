"""GSM8K dataset loader.

GSM8K (Grade School Math 8K) is a dataset of 8K linguistically diverse
grade-school math word problems. Format:
- question: natural language problem
- answer: chain-of-thought reasoning ending with #### <number>

The native HuggingFace ``gsm8k/main`` answer field also embeds inline
calculator annotations of the form ``<<expr=result>>`` next to every
arithmetic step (e.g. ``She has <<16-3-4=9>>9 apples``).  These are the
dataset's own way of marking a step that should be offloaded to a
calculator.

Without rewriting them, the model is supervised to predict the literal
substring ``9`` straight out of the CoT, which re-introduces the exact
multi-digit-arithmetic failure mode that the
``<TOOL>calc(...)</TOOL>`` + ``column_cot_*`` machinery was built to
prevent.  See ``format_for_sft`` / ``format_for_pretrain`` below for the
conversion pass.

For training:
- Pretraining: use the raw text (questions + answers concatenated)
- SFT: format as <REASON>...cot...</REASON><ANSWER>number</ANSWER>
- GRPO: use the question as prompt, generate, evaluate answer
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from src.utils.math_verify import extract_gsm8k_answer


# Matches a GSM8K calculator annotation: ``<<expr=result>>`` (optionally
# followed by a short bare numeric token, which is what GSM8K actually
# emits -- the closed-form ``<<expr=result>>RES`` where ``RES`` is the
# same digit(s) repeated literally in the surrounding prose so that a
# non-tool-reading model can still see it).
#
# Capture groups:
#   expr:     everything between ``<<`` and ``=`` (the expression)
#   after:    optional bare numeric token immediately after ``>>``;
#             we drop this to avoid teaching the model to both emit the
#             tool call *and* memorise the result verbatim.
_GSM8K_CALC_RE = re.compile(
    r"<<\s*(?P<expr>[^=\n]+?)\s*=\s*[^\n]*?>>(?P<after>\s*-?\d+(?:\.\d+)?)?"
)


def _convert_one(match: re.Match) -> str:
    expr = match.group("expr").strip()
    # When there is no trailing numeric token we keep the conversion
    # loss-less: the annotation is a one-for-one swap.  When there *is*
    # a trailing numeric token we additionally drop it; otherwise the
    # supervised target would still contain the dataset's known result,
    # which is exactly the "model predicts raw multi-digit sums" failure
    # mode this conversion is here to prevent.
    return f"<TOOL>calc({expr})</TOOL>"


def convert_gsm8k_calc_annotations(text: str) -> str:
    """Convert GSM8K ``<<expr=result>>`` annotations to ``<TOOL>calc(expr)</TOOL>``.

    The native GSM8K answer field marks every arithmetic step with an
    inline ``<<expr=result>>`` annotation, with the model supposed to
    emit a short numeric token (typically the same value as ``result``)
    that follows the closing ``>>``.  At training time we want those
    steps to flow through the project's ``<TOOL>calc(...)</TOOL>``
    machinery so the calculator/column-CoT fixes actually reach the
    data we're training on, *and* we want the supervised target to teach
    the model to emit the tool call rather than predict the digit
    string verbatim.

    The dataset repeats the result as ``result`` inside the annotation
    *and* as the literal token that follows -- we drop both.  When the
    annotation has no following token (rare; happens in some synthetic
    variants), the conversion is a one-for-one swap and no other text
    is affected.

    Args:
        text: A GSM8K answer (or CoT) string.

    Returns:
        The same string with each ``<<expr=result>>[RES]`` replaced by
        ``<TOOL>calc(expr)</TOOL>``.
    """
    return _GSM8K_CALC_RE.sub(_convert_one, text)


class GSM8KDataset:
    """GSM8K dataset wrapper.

    Loads from either:
    - HuggingFace datasets (gsm8k config, "main" subset)
    - Local JSONL files with {"question": ..., "answer": ...}

    Args:
        split: "train" or "test"
        local_path: Optional path to local JSONL file
    """

    def __init__(
        self,
        split: str = "train",
        local_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.split = split
        self.examples: List[Dict[str, str]] = []
        self._loaded_from = None

        if local_path is not None:
            self._load_local(local_path)
        else:
            try:
                self._load_hf(cache_dir)
            except Exception as e:
                # Fallback: leave empty, user must provide local data
                self.examples = []
                self._load_error = str(e)

    def _load_local(self, path: str) -> None:
        """Load from local JSONL file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"GSM8K file not found: {path}")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                self.examples.append({
                    "question": ex.get("question", ""),
                    "answer": ex.get("answer", ""),
                })
        self._loaded_from = f"local:{path}"

    def _load_hf(self, cache_dir: Optional[str]) -> None:
        """Load from HuggingFace datasets."""
        from datasets import load_dataset
        ds = load_dataset(
            "gsm8k",
            "main",
            split=self.split,
            cache_dir=cache_dir,
        )
        for ex in ds:
            self.examples.append({
                "question": ex["question"],
                "answer": ex["answer"],
            })
        self._loaded_from = f"huggingface:gsm8k:main:{self.split}"

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.examples[idx]

    def __iter__(self) -> Iterator[Dict[str, str]]:
        return iter(self.examples)

    def get_question(self, idx: int) -> str:
        return self.examples[idx]["question"]

    def get_full_answer(self, idx: int) -> str:
        """Get the full chain-of-thought answer."""
        return self.examples[idx]["answer"]

    def get_final_answer(self, idx: int) -> Optional[str]:
        """Extract the final numerical answer (after ####)."""
        return extract_gsm8k_answer(self.examples[idx]["answer"])

    def format_for_pretrain(self, idx: int, convert_calc_annotations: bool = True
                            ) -> str:
        """Format for pre-training: question + answer concatenated.

        If ``convert_calc_annotations`` is True (default), each
        ``<<expr=result>>`` annotation in the answer is rewritten to
        ``<TOOL>calc(expr)</TOOL>`` so the calculator machinery is
        supervised on real GSM8K text.
        """
        ex = self.examples[idx]
        answer = ex["answer"]
        if convert_calc_annotations:
            answer = convert_gsm8k_calc_annotations(answer)
        return f"Question: {ex['question']}\nAnswer: {answer}"

    def format_for_sft(self, idx: int, convert_calc_annotations: bool = True
                       ) -> Tuple[str, str]:
        """Format for SFT: prompt and target with CoT structure.

        If ``convert_calc_annotations`` is True (default), each
        ``<<expr=result>>`` annotation in the CoT is rewritten to
        ``<TOOL>calc(expr)</TOOL>`` so the calculator/column-CoT
        machinery the rest of the repo is built around actually reaches
        the benchmark we're training on.  Without this conversion the
        raw multi-digit-arithmetic substring that follows the ``>>``
        gets used as the supervised target verbatim, which re-creates
        the carry-direction failure mode the calculator was built to
        fix.

        Returns:
            (prompt, target) where target contains reasoning and final answer.
        """
        ex = self.examples[idx]
        prompt = (
            f"Question: {ex['question']}\n"
            f"Answer: <REASON>"
        )
        # Extract CoT (everything before ####) and final answer
        full_answer = ex["answer"]
        if convert_calc_annotations:
            full_answer = convert_gsm8k_calc_annotations(full_answer)
        if "####" in full_answer:
            cot_part, final_part = full_answer.split("####", 1)
            cot = cot_part.strip()
            final = final_part.strip()
        else:
            cot = full_answer.strip()
            final = ""

        target = (
            f"{cot}</REASON>\n"
            f"<ANSWER>{final}</ANSWER>"
        )
        return prompt, target

    def format_for_grpo(self, idx: int, convert_calc_annotations: bool = True
                      ) -> Tuple[str, str]:
        """Format for GRPO: prompt and gold answer (no CoT).

        ``convert_calc_annotations`` is unused here (the gold answer is
        just the final number), but the flag is accepted for symmetry
        with ``format_for_sft`` / ``format_for_pretrain``.

        Returns:
            (prompt, gold_answer) — model generates full response.
        """
        ex = self.examples[idx]
        prompt = (
            f"Question: {ex['question']}\n"
            f"Answer:"
        )
        gold = self.get_final_answer(idx) or ""
        return prompt, gold


# ---------------------------------------------------------------------------
# Synthetic data for tests (no external download needed)
# ---------------------------------------------------------------------------

SYNTHETIC_GSM8K = [
    {
        "question": "Janet has 3 apples. She buys 5 more. How many apples does she have now?",
        "answer": "Janet starts with 3 apples. She buys 5 more.\n3 + 5 = 8\n#### 8",
    },
    {
        "question": "A train travels 60 mph for 3 hours. How far does it go?",
        "answer": "Distance = speed * time = 60 * 3 = 180.\n#### 180",
    },
    {
        "question": "If 3x = 12, what is x?",
        "answer": "Divide both sides by 3: x = 12/3 = 4.\n#### 4",
    },
    {
        "question": "Sarah has 24 cookies. She gives 1/3 to her friend. How many does she keep?",
        "answer": "Sarah gives 24 * 1/3 = 8 cookies.\nShe keeps 24 - 8 = 16.\n#### 16",
    },
    {
        "question": "A rectangle has length 8 and width 5. What is its area?",
        "answer": "Area = length * width = 8 * 5 = 40.\n#### 40",
    },
    {
        "question": "Tom is 5 years older than Jerry. Tom is 17. How old is Jerry?",
        "answer": "Jerry = Tom - 5 = 17 - 5 = 12.\n#### 12",
    },
    {
        "question": "What is 15% of 200?",
        "answer": "15% of 200 = 0.15 * 200 = 30.\n#### 30",
    },
    {
        "question": "A book costs $12. With 8% tax, what's the total?",
        "answer": "Tax = 12 * 0.08 = 0.96.\nTotal = 12 + 0.96 = 12.96.\n#### 12.96",
    },
]


def make_synthetic_gsm8k(n: int = 100) -> List[Dict[str, str]]:
    """Generate n synthetic GSM8K-style examples by cycling."""
    return [SYNTHETIC_GSM8K[i % len(SYNTHETIC_GSM8K)] for i in range(n)]


if __name__ == "__main__":
    # Smoke test with synthetic data
    import json
    tmp = Path("/tmp/synthetic_gsm8k.jsonl")
    with open(tmp, "w") as f:
        for ex in SYNTHETIC_GSM8K:
            f.write(json.dumps(ex) + "\n")

    ds = GSM8KDataset(split="train", local_path=str(tmp))
    print(f"Loaded {len(ds)} examples")
    for i in range(min(3, len(ds))):
        prompt, target = ds.format_for_sft(i)
        print(f"\n--- Example {i} ---")
        print(f"PROMPT: {prompt!r}")
        print(f"TARGET: {target!r}")
        print(f"GOLD: {ds.get_final_answer(i)}")