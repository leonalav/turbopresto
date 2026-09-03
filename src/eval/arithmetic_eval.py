"""Pure arithmetic evaluation benchmark.

Tests the model's ability to do step-by-step arithmetic on
synthetic problems of varying difficulty.

This is a quick sanity check — much faster than GSM8K/MATH.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.data.synthetic import SyntheticMathDataset
from src.inference.generation import RWKVGenerator
from src.inference.voting import sample_and_vote
from src.model.config import ModelConfig
from src.model.init import init_and_verify
from src.tokenizer.math_tokenizer import MathTokenizer, StubTokenizer
from src.utils.math_verify import extract_boxed, extract_number, is_equiv


def evaluate_arithmetic(
    model,
    tokenizer,
    dataset: SyntheticMathDataset,
    max_new_tokens: int = 256,
    n_samples: int = 1,
    method: str = "greedy",
) -> Dict:
    """Evaluate on synthetic arithmetic.

    Args:
        model: RWKV-7 model
        tokenizer: Tokenizer
        dataset: SyntheticMathDataset
        max_new_tokens: Max generation
        n_samples: N for majority vote
        method: "greedy" or "majority"

    Returns:
        Dict with accuracy and per-operation breakdown.
    """
    gen = RWKVGenerator(model, tokenizer)

    n = len(dataset)
    correct = 0
    per_op_correct = {"add": 0, "sub": 0, "mul": 0, "div": 0}
    per_op_total = {"add": 0, "sub": 0, "mul": 0, "div": 0}

    for i in range(n):
        prompt, gold = dataset.format_for_grpo(i)

        # Determine op type from question
        if "+" in dataset.get_question(i):
            op = "add"
        elif "-" in dataset.get_question(i):
            op = "sub"
        elif "*" in dataset.get_question(i):
            op = "mul"
        elif "/" in dataset.get_question(i):
            op = "div"
        else:
            op = "add"

        per_op_total[op] += 1

        if n_samples <= 1:
            text = gen.generate(prompt, max_new_tokens=max_new_tokens, greedy=(method == "greedy"))
            pred_ans = extract_boxed(text) or extract_number(text)
        else:
            pred_ans, _ = sample_and_vote(
                model, tokenizer, prompt,
                n_samples=n_samples,
                max_new_tokens=max_new_tokens,
                method="majority",
                gold=gold,
            )

        is_corr = is_equiv(pred_ans or "", gold or "")
        correct += int(is_corr)
        per_op_correct[op] += int(is_corr)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n}] acc={correct/(i+1)*100:.1f}%")

    accuracy = correct / max(n, 1)
    per_op_acc = {
        op: per_op_correct[op] / max(per_op_total[op], 1)
        for op in per_op_correct
    }

    return {
        "total": n,
        "correct": correct,
        "accuracy": accuracy,
        "per_op_accuracy": per_op_acc,
        "per_op_correct": per_op_correct,
        "per_op_total": per_op_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Arithmetic evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--max-digits", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--method", type=str, default="greedy", choices=["greedy", "majority"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = init_and_verify(ModelConfig())
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(args.device)

    tokenizer = MathTokenizer()
    dataset = SyntheticMathDataset(
        size=args.n_examples,
        max_digits=args.max_digits,
        seed=42,
    )

    print(f"Arithmetic eval: {args.n_examples} examples, max_digits={args.max_digits}")
    results = evaluate_arithmetic(
        model, tokenizer, dataset,
        max_new_tokens=args.max_new_tokens,
        n_samples=args.n_samples,
        method=args.method,
    )

    print(f"\n=== Results ===")
    print(f"Overall accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    print(f"Per-op accuracy:")
    for op, acc in results['per_op_accuracy'].items():
        print(f"  {op}: {acc*100:.1f}% ({results['per_op_correct'][op]}/{results['per_op_total'][op]})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()