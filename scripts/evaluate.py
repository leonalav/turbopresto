#!/usr/bin/env python
"""Evaluate RWKV-7 model on math benchmarks.

Usage:
    python scripts/evaluate.py gsm8k --checkpoint checkpoints/grpo/final.pt
    python scripts/evaluate.py math --checkpoint checkpoints/grpo/final.pt
    python scripts/evaluate.py arithmetic --checkpoint checkpoints/grpo/final.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.model.config import ModelConfig
from src.model.init import init_and_verify
from src.training.optimizer import load_checkpoint
from src.tokenizer.math_tokenizer import MathTokenizer
from src.eval.gsm8k_eval import evaluate_gsm8k, main as gsm8k_main
from src.eval.math_eval import evaluate_math, main as math_main
from src.eval.arithmetic_eval import evaluate_arithmetic, main as arith_main
from src.data.gsm8k import GSM8KDataset
from src.data.math_dataset import MATHDataset
from src.data.synthetic import SyntheticMathDataset
from src.inference.generation import RWKVGenerator


def load_model_and_tokenizer(checkpoint: str, device: str):
    """Load model and tokenizer from checkpoint."""
    cfg = ModelConfig()
    model = init_and_verify(cfg)
    state = torch.load(checkpoint, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    tokenizer = MathTokenizer()
    return model, tokenizer


def eval_gsm8k(args):
    """Evaluate on GSM8K."""
    import json
    print(f"Loading from {args.checkpoint}...")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.device)
    dataset = GSM8KDataset(split=args.split)
    print(f"Dataset: {len(dataset)} examples")

    from src.eval.gsm8k_eval import evaluate_gsm8k
    results = evaluate_gsm8k(
        model, tokenizer, dataset,
        max_new_tokens=args.max_new_tokens,
        n_samples=args.n_samples,
        method="majority" if args.n_samples > 1 else "greedy",
    )

    print(f"\n=== GSM8K {args.split} ===")
    print(f"Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved to {args.output}")
    return results


def eval_math(args):
    """Evaluate on MATH."""
    import json
    print(f"Loading from {args.checkpoint}...")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.device)
    dataset = MATHDataset(split=args.split)
    print(f"Dataset: {len(dataset)} examples")

    from src.eval.math_eval import evaluate_math
    results = evaluate_math(
        model, tokenizer, dataset,
        max_new_tokens=args.max_new_tokens,
        n_samples=args.n_samples,
        method="majority" if args.n_samples > 1 else "greedy",
    )

    print(f"\n=== MATH {args.split} ===")
    print(f"Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved to {args.output}")
    return results


def eval_arithmetic(args):
    """Evaluate on synthetic arithmetic."""
    import json
    print(f"Loading from {args.checkpoint}...")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.device)
    dataset = SyntheticMathDataset(size=args.n_examples, max_digits=args.max_digits, seed=42)
    print(f"Dataset: {len(dataset)} examples")

    from src.eval.arithmetic_eval import evaluate_arithmetic
    results = evaluate_arithmetic(
        model, tokenizer, dataset,
        max_new_tokens=args.max_new_tokens,
        n_samples=args.n_samples,
        method="majority" if args.n_samples > 1 else "greedy",
    )

    print(f"\n=== Arithmetic (max_digits={args.max_digits}) ===")
    print(f"Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    print("Per-op accuracy:")
    for op, acc in results['per_op_accuracy'].items():
        print(f"  {op}: {acc*100:.1f}% ({results['per_op_correct'][op]}/{results['per_op_total'][op]})")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved to {args.output}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RWKV-7 model")
    sub = parser.add_subparsers(dest="benchmark", required=True,
                               choices=["gsm8k", "math", "arithmetic"])

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", type=str, required=True)
    common.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")
    common.add_argument("--max-new-tokens", type=int, default=512)
    common.add_argument("--n-samples", type=int, default=1,
                       help="N for majority vote (1=greedy)")
    common.add_argument("--output", type=str, default=None)

    # GSM8K
    g = sub.add_parser("gsm8k", parents=[common])
    g.add_argument("--split", type=str, default="test", choices=["train", "test"])

    # MATH
    m = sub.add_parser("math", parents=[common])
    m.add_argument("--split", type=str, default="test", choices=["train", "test"])

    # Arithmetic
    a = sub.add_parser("arithmetic", parents=[common])
    a.add_argument("--n-examples", type=int, default=100)
    a.add_argument("--max-digits", type=int, default=3)

    args = parser.parse_args()

    if args.benchmark == "gsm8k":
        eval_gsm8k(args)
    elif args.benchmark == "math":
        eval_math(args)
    elif args.benchmark == "arithmetic":
        eval_arithmetic(args)


if __name__ == "__main__":
    main()
