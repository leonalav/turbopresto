"""MATH dataset evaluation.

Usage:
    python -m src.eval.math_eval --checkpoint checkpoints/grpo/final.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from src.data.math_dataset import MATHDataset
from src.inference.generation import RWKVGenerator
from src.inference.voting import sample_and_vote
from src.model.config import ModelConfig
from src.model.init import init_and_verify
from src.tokenizer.math_tokenizer import MathTokenizer
from src.utils.math_verify import extract_boxed, extract_number, is_equiv


def evaluate_math(
    model,
    tokenizer,
    dataset: MATHDataset,
    max_new_tokens: int = 1024,
    n_samples: int = 1,
    method: str = "majority",
    max_examples: int = None,
) -> Dict:
    """Evaluate on MATH dataset.

    Args:
        model: RWKV-7 model
        tokenizer: Tokenizer
        dataset: MATHDataset
        max_new_tokens: Max generation length
        n_samples: N for majority vote
        method: "greedy" or "majority"
        max_examples: Limit examples

    Returns:
        Dict with accuracy and per-example results.
    """
    gen = RWKVGenerator(model, tokenizer)

    n = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    correct = 0
    results = []

    for i in range(n):
        prompt, gold_ans = dataset.format_for_grpo(i)

        if n_samples <= 1:
            text = gen.generate(prompt, max_new_tokens=max_new_tokens, greedy=(method == "greedy"))
            pred_ans = extract_boxed(text) or extract_number(text)
        else:
            pred_ans, _ = sample_and_vote(
                model, tokenizer, prompt,
                n_samples=n_samples,
                max_new_tokens=max_new_tokens,
                method="majority",
                gold=gold_ans,
            )

        is_corr = is_equiv(pred_ans or "", gold_ans or "")
        correct += int(is_corr)

        results.append({
            "idx": i,
            "gold": gold_ans,
            "pred": pred_ans,
            "correct": is_corr,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] acc={correct/(i+1)*100:.1f}%")

    accuracy = correct / max(n, 1)
    return {
        "total": n,
        "correct": correct,
        "accuracy": accuracy,
        "method": method,
        "n_samples": n_samples,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on MATH")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--method", type=str, default="majority",
                       choices=["greedy", "majority"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = init_and_verify(ModelConfig())
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(args.device)

    tokenizer = MathTokenizer()
    dataset = MATHDataset(split=args.split)

    print(f"Evaluating on MATH {args.split} ({len(dataset)} examples)")
    results = evaluate_math(
        model, tokenizer, dataset,
        max_new_tokens=args.max_new_tokens,
        n_samples=args.n_samples,
        method=args.method,
        max_examples=args.max_examples,
    )

    print(f"\n=== Results ===")
    print(f"Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()