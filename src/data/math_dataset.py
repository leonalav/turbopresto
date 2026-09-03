"""MATH dataset loader.

The MATH dataset (Hendrycks et al.) contains 12.5K competition math
problems with LaTeX-formatted answers (\\boxed{...}).

For our 50M model, MATH is challenging; we'll use a subset for SFT and
GRPO. For pre-training, we use a smaller mix or skip MATH entirely.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from src.utils.math_verify import extract_math_answer


class MATHDataset:
    """MATH dataset wrapper.

    Loads from either:
    - HuggingFace datasets (hendrycks/competition_math)
    - Local JSONL files with {"problem": ..., "solution": ..., "level": ..., "type": ...}
    """

    def __init__(
        self,
        split: str = "train",
        local_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.split = split
        self.examples: List[Dict[str, str]] = []

        if local_path is not None:
            self._load_local(local_path)
        else:
            try:
                self._load_hf(cache_dir)
            except Exception:
                self.examples = []

    def _load_local(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"MATH file not found: {path}")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                self.examples.append({
                    "problem": ex.get("problem", ""),
                    "solution": ex.get("solution", ""),
                    "level": ex.get("level", ""),
                    "type": ex.get("type", ""),
                })

    def _load_hf(self, cache_dir: Optional[str]) -> None:
        from datasets import load_dataset
        ds = load_dataset(
            "hendrycks/competition_math",
            split=self.split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        for ex in ds:
            self.examples.append({
                "problem": ex["problem"],
                "solution": ex["solution"],
                "level": ex.get("level", ""),
                "type": ex.get("type", ""),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.examples[idx]

    def __iter__(self) -> Iterator[Dict[str, str]]:
        return iter(self.examples)

    def get_problem(self, idx: int) -> str:
        return self.examples[idx]["problem"]

    def get_solution(self, idx: int) -> str:
        return self.examples[idx]["solution"]

    def get_gold_answer(self, idx: int) -> Optional[str]:
        """Extract \\boxed{...} answer from the solution."""
        return extract_math_answer(self.examples[idx]["solution"])

    def format_for_pretrain(self, idx: int) -> str:
        """Format for pre-training."""
        ex = self.examples[idx]
        return f"Problem: {ex['problem']}\nSolution: {ex['solution']}"

    def format_for_sft(self, idx: int) -> Tuple[str, str]:
        """Format for SFT with CoT structure."""
        ex = self.examples[idx]
        prompt = f"Problem: {ex['problem']}\nSolution: <REASON>"
        solution = ex["solution"]
        gold = self.get_gold_answer(idx) or ""
        target = f"{solution}</REASON>\n<ANSWER>{gold}</ANSWER>"
        return prompt, target

    def format_for_grpo(self, idx: int) -> Tuple[str, str]:
        """Format for GRPO."""
        ex = self.examples[idx]
        prompt = f"Problem: {ex['problem']}\nSolution:"
        gold = self.get_gold_answer(idx) or ""
        return prompt, gold


# ---------------------------------------------------------------------------
# Synthetic MATH-style examples for tests
# ---------------------------------------------------------------------------

SYNTHETIC_MATH = [
    {
        "problem": "Find the value of $x$ in $2x + 5 = 13$.",
        "solution": "Subtract 5 from both sides: $2x = 8$. Divide by 2: $x = 4$. \\boxed{4}",
        "level": "Level 1",
        "type": "Algebra",
    },
    {
        "problem": "What is $\\frac{1}{2} + \\frac{1}{3}$?",
        "solution": "Find common denominator 6: $\\frac{3}{6} + \\frac{2}{6} = \\frac{5}{6}$. \\boxed{\\frac{5}{6}}",
        "level": "Level 1",
        "type": "Prealgebra",
    },
    {
        "problem": "Compute $7^2 - 3 \\cdot 4$.",
        "solution": "$7^2 = 49$. $3 \\cdot 4 = 12$. $49 - 12 = 37$. \\boxed{37}",
        "level": "Level 1",
        "type": "Prealgebra",
    },
    {
        "problem": "If $f(x) = 3x + 2$, find $f(5)$.",
        "solution": "$f(5) = 3(5) + 2 = 17$. \\boxed{17}",
        "level": "Level 1",
        "type": "Algebra",
    },
    {
        "problem": "Solve for $x$: $x^2 - 9 = 0$.",
        "solution": "$x^2 = 9$. $x = \\pm 3$. \\boxed{\\pm 3}",
        "level": "Level 2",
        "type": "Algebra",
    },
]


def make_synthetic_math(n: int = 50) -> List[Dict[str, str]]:
    return [SYNTHETIC_MATH[i % len(SYNTHETIC_MATH)] for i in range(n)]


if __name__ == "__main__":
    import json
    tmp = Path("/tmp/synthetic_math.jsonl")
    with open(tmp, "w") as f:
        for ex in SYNTHETIC_MATH:
            f.write(json.dumps(ex) + "\n")

    ds = MATHDataset(split="train", local_path=str(tmp))
    print(f"Loaded {len(ds)} examples")
    for i in range(min(3, len(ds))):
        prompt, target = ds.format_for_sft(i)
        print(f"\n--- Example {i} ---")
        print(f"PROMPT: {prompt[:100]}...")
        print(f"TARGET: {target[:100]}...")
        print(f"GOLD: {ds.get_gold_answer(i)}")