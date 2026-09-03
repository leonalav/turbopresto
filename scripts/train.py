#!/usr/bin/env python
"""Train RWKV-7 model.

Usage:
    # Pretrain
    python scripts/train.py pretrain --max-steps 50000 --batch-size 16 --lr 6e-4

    # SFT
    python scripts/train.py sft --max-steps 5000 --batch-size 8 --lr 1e-5

    # GRPO
    python scripts/train.py grpo --max-steps 3000 --group-size 8 --lr 5e-6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.config import ModelConfig
from src.model.init import init_and_verify
from src.tokenizer.math_tokenizer import MathTokenizer
from src.training.optimizer import load_checkpoint
from src.training.pretrain import PretrainConfig, pretrain, build_pretrain_dataset
from src.training.sft import SFTConfig, sft, build_sft_examples
from src.training.grpo import GRPOConfig, grpo
from src.data.synthetic import SyntheticMathDataset


def train_pretrain(args):
    """Run pretraining."""
    print(f"Pretrain: {args.max_steps} steps, bs={args.batch_size}, lr={args.lr}")

    model_cfg = ModelConfig()
    model = init_and_verify(model_cfg)

    # Load from checkpoint if specified
    if args.load:
        step = load_checkpoint(args.load, model, map_location="cpu")
        print(f"Loaded checkpoint from step {step}")

    tokenizer = MathTokenizer()

    cfg = PretrainConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        seq_len=args.seq_len or model_cfg.ctx_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        save_every=args.save_every,
        log_every=args.log_every,
        grad_clip=args.grad_clip,
        save_dir=args.output or "checkpoints/pretrain",
        log_file=args.log_file,
        device=args.device,
        dtype=args.dtype,
    )

    texts = build_pretrain_dataset(cfg)
    print(f"Training data: {len(texts)} texts")

    logs = pretrain(model, cfg, tokenizer, texts)
    print(f"\nPretrain complete. Final loss: {logs[-1]['loss']:.4f}")


def train_sft(args):
    """Run SFT."""
    print(f"SFT: {args.max_steps} steps, bs={args.batch_size}, lr={args.lr}")

    model_cfg = ModelConfig()
    model = init_and_verify(model_cfg)

    if args.load:
        step = load_checkpoint(args.load, model, map_location="cpu")
        print(f"Loaded checkpoint from step {step}")

    tokenizer = MathTokenizer()

    cfg = SFTConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        seq_len=args.seq_len or 2048,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        save_every=args.save_every,
        log_every=args.log_every,
        grad_clip=args.grad_clip,
        save_dir=args.output or "checkpoints/sft",
        log_file=args.log_file,
        device=args.device,
        dtype=args.dtype,
    )

    logs = sft(model, cfg, tokenizer)
    print(f"\nSFT complete. Final loss: {logs[-1]['loss']:.4f}")


def train_grpo(args):
    """Run GRPO."""
    print(f"GRPO: {args.max_steps} steps, G={args.group_size}, lr={args.lr}")

    model_cfg = ModelConfig()
    model = init_and_verify(model_cfg)

    if args.load:
        step = load_checkpoint(args.load, model, map_location="cpu")
        print(f"Loaded checkpoint from step {step}")

    ref_model = None
    if args.ref_load:
        ref_model = init_and_verify(model_cfg)
        step = load_checkpoint(args.ref_load, ref_model, map_location="cpu")
        print(f"Loaded ref model from step {step}")

    tokenizer = MathTokenizer()

    # Build GRPO examples from synthetic data
    syn = SyntheticMathDataset(size=args.n_examples, seed=args.seed)
    examples = [syn.format_for_grpo(i) for i in range(len(syn))]

    cfg = GRPOConfig(
        group_size=args.group_size,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        seq_len=args.seq_len or 1024,
        lr=args.lr,
        kl_coef=args.kl_coef,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        save_every=args.save_every,
        log_every=args.log_every,
        grad_clip=args.grad_clip,
        save_dir=args.output or "checkpoints/grpo",
        device=args.device,
        dtype=args.dtype,
    )

    logs = grpo(model, cfg, tokenizer, examples, ref_model=ref_model)
    print(f"\nGRPO complete. Final stats: {logs[-1]}")


def main():
    parser = argparse.ArgumentParser(description="Train RWKV-7 model")
    sub = parser.add_subparsers(dest="stage", help="Training stage")

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--load", type=str, default=None, help="Checkpoint to load")
    common.add_argument("--output", type=str, default=None, help="Output dir")
    common.add_argument("--log-file", type=str, default=None)
    common.add_argument("--device", type=str, default="cuda" if _cuda_available() else "cpu")
    common.add_argument("--dtype", type=str, default="bfloat16" if _cuda_available() else "float32")

    # Pretrain
    p = sub.add_parser("pretrain", parents=[common], help="Pretraining")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # SFT
    s = sub.add_parser("sft", parents=[common], help="SFT")
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--max-steps", type=int, default=5000)
    s.add_argument("--seq-len", type=int, default=2048)
    s.add_argument("--lr", type=float, default=1e-5)
    s.add_argument("--weight-decay", type=float, default=0.01)
    s.add_argument("--warmup-steps", type=int, default=100)
    s.add_argument("--save-every", type=int, default=1000)
    s.add_argument("--log-every", type=int, default=20)
    s.add_argument("--grad-clip", type=float, default=1.0)

    # GRPO
    g = sub.add_parser("grpo", parents=[common], help="GRPO")
    g.add_argument("--load", type=str, default=None, help="SFT checkpoint to load")
    g.add_argument("--ref-load", type=str, default=None, help="Reference model checkpoint")
    g.add_argument("--batch-size", type=int, default=4)
    g.add_argument("--group-size", type=int, default=8)
    g.add_argument("--max-steps", type=int, default=3000)
    g.add_argument("--seq-len", type=int, default=1024)
    g.add_argument("--lr", type=float, default=5e-6)
    g.add_argument("--kl-coef", type=float, default=0.04)
    g.add_argument("--clip-eps", type=float, default=0.2)
    g.add_argument("--entropy-coef", type=float, default=0.01)
    g.add_argument("--save-every", type=int, default=500)
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--n-examples", type=int, default=500, help="Synthetic examples for GRPO")
    g.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.stage is None:
        parser.print_help()
        return

    # Convert dtype string to torch dtype
    import torch
    args.dtype = getattr(torch, args.dtype)

    if args.stage == "pretrain":
        train_pretrain(args)
    elif args.stage == "sft":
        train_sft(args)
    elif args.stage == "grpo":
        train_grpo(args)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
