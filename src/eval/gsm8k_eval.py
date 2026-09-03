"""GSM8K evaluation script.

Runs the model on GSM8K test set and reports accuracy.

Usage:
    python -m src.eval.gsm8k_eval --checkpoint checkpoints/grpo/final.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from src.data.gsm8k import GSM8KDataset
from src.inference.generation import RWKVGenerator
from src.inference.voting import majority_vote, sample_and_vote
from src.model.config import ModelConfig
from src.model.init import init_and_verify
from src.tokenizer.math_tokenizer import MathTokenizer, StubTokenizer
from src.utils.math_verify import extract_boxed, extract_number, is_equiv


def evaluate_gsm8k(
    model,
    tokenizer,
    dataset: GSM8KDataset,
    max_new_tokens: int = 512,
    n_samples: int = 1,
    method: str = "greedy",
    max_examples: int = None,
) -> Dict:
    """Evaluate model on GSM8K.

    Args:
        model: RWKV-7 model
        tokenizer: Tokenizer
        dataset: GSM8KDataset (typically test split)
        max_new_tokens: Max generation length per sample
        n_samples: N for majority vote (1 = greedy/single sample)
        method: "greedy", "majority", or "best_of_n"
        max_examples: Limit to first N examples (None = all)

    Returns:
        Dict with accuracy and per-example results.
    """
    gen = RWKVGenerator(model, tokenizer)

    n = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    correct = 0
    results = []

    for i in range(n):
        prompt, gold = dataset.format_for_grpo(i)
        gold_ans = gold

        if n_samples <= 1:
            # Single sample (greedy or sampled)
            text = gen.generate(prompt, max_new_tokens=max_new_tokens, greedy=(method == "greedy"))
            pred_ans = extract_boxed(text) or extract_number(text)
        else:
            # Majority vote
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
            "question": prompt[:200],
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
    parser = argparse.ArgumentParser(description="Evaluate on GSM8K")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-samples", type=int, default=1, help="N for majority vote")
    parser.add_argument("--method", type=str, default="greedy",
                       choices=["greedy", "majority"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", type=str, default=None,
                       help="Save detailed results to JSON file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load model
    model = init_and_verify(ModelConfig())
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(args.device)

    # Load tokenizer
    tokenizer = MathTokenizer()

    # Load dataset
    dataset = GSM8KDataset(split=args.split)

    # Evaluate
    print(f"Evaluating on GSM8K {args.split} ({len(dataset)} examples)")
    print(f"Method: {args.method}, n_samples: {args.n_samples}")
    results = evaluate_gsm8k(
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