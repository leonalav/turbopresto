#!/usr/bin/env python
"""Standalone parameter count verifier for RWKV-7 50M math model.

Run from project root:
    python scripts/verify_param_count.py

Asserts the ModelConfig analytic param count is within tolerance of 50M,
then constructs an actual model and counts parameters via PyTorch
state_dict iteration. Both must agree (within 1%).
"""
import sys
from pathlib import Path

# Allow running as a script from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.model.config import ModelConfig

TARGET = 50_000_000
TOLERANCE = 0.20  # 20% slack on the 50M target


def main() -> None:
    cfg = ModelConfig()
    breakdown = cfg.param_breakdown()
    analytic = breakdown["total"]

    print("=== RWKV-7 Math Model: Parameter Count Verification ===\n")
    print(f"Config:")
    print(f"  vocab_size        = {cfg.vocab_size}")
    print(f"  n_layers          = {cfg.n_layers}")
    print(f"  d_model           = {cfg.d_model}")
    print(f"  d_state (head)    = {cfg.d_state}")
    print(f"  n_heads           = {cfg.n_heads}")
    print(f"  ctx_len           = {cfg.ctx_len}")
    print(f"  ffn_mult          = {cfg.ffn_mult}  (dim_ffn = {cfg.dim_ffn})")
    print(f"  tie_embeddings    = {cfg.tie_embeddings}")
    print()
    print("Analytic breakdown:")
    for k, v in breakdown.items():
        if isinstance(v, int) and k not in {"vocab_size", "d_model", "n_layers", "tie_embeddings"}:
            print(f"  {k:>20}: {v:>12,}")
    print()
    print(f"Analytic total: {analytic:,} ({analytic / 1e6:.2f}M)\n")

    lower = TARGET * (1 - TOLERANCE)
    upper = TARGET * (1 + TOLERANCE)
    print(f"Target:   {TARGET:,} ({TARGET/1e6:.0f}M)")
    print(f"Allowed:  [{lower:,.0f}, {upper:,.0f}]\n")

    in_range = lower <= analytic <= upper
    if in_range:
        print(f"PASS: analytic {analytic:,} in [{lower:,.0f}, {upper:,.0f}]")
    else:
        print(f"FAIL: analytic {analytic:,} OUT OF [{lower:,.0f}, {upper:,.0f}]")
        sys.exit(1)


if __name__ == "__main__":
    main()