#!/usr/bin/env python
"""Prepare data for RWKV-7 training.

Downloads and preprocesses:
- GSM8K dataset
- MATH dataset
- Optional: OpenWebText subset

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --datasets gsm8k math
    python scripts/prepare_data.py --output data/processed/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.gsm8k import GSM8KDataset, SYNTHETIC_GSM8K
from src.data.math_dataset import MATHDataset, SYNTHETIC_MATH
from src.data.synthetic import SyntheticMathDataset


def prepare_gsm8k(output_dir: Path, split: str = "train"):
    """Prepare GSM8K dataset."""
    print(f"Preparing GSM8K {split}...")
    ds = GSM8KDataset(split=split)
    print(f"  Loaded {len(ds)} examples")
    out = output_dir / f"gsm8k_{split}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i in range(len(ds)):
            ex = ds[i]
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Saved to {out}")
    return out


def prepare_math(output_dir: Path, split: str = "train"):
    """Prepare MATH dataset."""
    print(f"Preparing MATH {split}...")
    ds = MATHDataset(split=split)
    print(f"  Loaded {len(ds)} examples")
    out = output_dir / f"math_{split}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i in range(len(ds)):
            ex = ds[i]
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Saved to {out}")
    return out


def prepare_synthetic(output_dir: Path, size: int = 5000):
    """Generate synthetic arithmetic data."""
    print(f"Generating synthetic data ({size} examples)...")
    ds = SyntheticMathDataset(size=size)
    out = output_dir / "synthetic.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i in range(len(ds)):
            ex = ds[i]
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Saved {size} examples to {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "math", "synthetic"],
        choices=["gsm8k", "math", "synthetic"],
    )
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--size", type=int, default=5000, help="Synthetic data size")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                       choices=["train", "test"])
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    for ds_name in args.datasets:
        if ds_name == "gsm8k":
            for split in args.splits:
                try:
                    prepare_gsm8k(output_dir, split)
                except Exception as e:
                    print(f"  ERROR: {e}")
        elif ds_name == "math":
            for split in args.splits:
                try:
                    prepare_math(output_dir, split)
                except Exception as e:
                    print(f"  ERROR: {e}")
        elif ds_name == "synthetic":
            prepare_synthetic(output_dir, args.size)

    print("\nData preparation complete!")
    print(f"Files in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
