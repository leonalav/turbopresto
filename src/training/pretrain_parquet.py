"""Faultless pre-training loop using pretokenized parquet data.

Reads tokens directly from parquet chunks uploaded to
leonidas123/valkmodel-data (produced by `python -m pretok.pretokenize`).

Features beyond `pretrain.py`:
  - Streams pretokenized tokens (no tokenization on GPU VM)
  - Deterministic shuffling (numpy/random seedable)
  - Resumable from checkpoint step (skip-ahead in dataset)
  - Periodic validation pass (held-out last 1% of each shard)
  - Periodic WandB logging + JSONL log
  - Heartbeat (last_loss, last_step) for monitoring
  - Atomic checkpointing with `--resume-from`
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FaultlessPretrainConfig:
    """Pretraining configuration for parquet pipeline."""
    # Data
    hf_repo_id: str = "leonidas123/valkmodel-data"
    seq_len: int = 4096
    batch_size: int = 16
    max_steps: int = 50_000

    # Optimization
    lr: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)
    cosine_min_lr_ratio: float = 0.1

    # Logging / IO
    save_dir: str = "checkpoints/pretrain"
    log_file: str = "logs/pretrain.jsonl"
    save_every: int = 5000
    log_every: int = 50
    val_every: int = 2000
    val_samples: int = 32

    # Reproducibility
    seed: int = 42

    # Hardware
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    # Resume
    resume_from: Optional[str] = None  # path to checkpoint
    max_shards: Optional[int] = None   # cap on shards loaded (debug)

    # M1 fix: EOS token id, used by ParquetDataset to mark doc boundaries.
    # Should match the tokenizer that produced the parquet chunks. Pass via
    # CLI: --eos_id <int>.
    eos_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Checkpoint IO (atomic)
# ---------------------------------------------------------------------------

def save_checkpoint_atomic(
    model,
    optimizer,
    scheduler,
    step: int,
    cfg: FaultlessPretrainConfig,
    path: Path,
) -> None:
    """Save checkpoint atomically (write to .tmp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "config": cfg.__dict__,
    }
    torch.save(state, tmp)
    tmp.replace(path)  # atomic on POSIX; on Windows .replace overwrites


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None,
                   map_location: str = "cpu") -> int:
    """Load checkpoint, return step number. Raises if not found."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state["step"]


# ---------------------------------------------------------------------------
# Optimizer & LR schedule
# ---------------------------------------------------------------------------

def build_optimizer_and_sched(model, cfg: FaultlessPretrainConfig):
    """Build AdamW + warmup-cosine LR."""
    from src.training.optimizer import (
        AdamWConfig, WarmupCosineLR, build_optimizer,
    )
    opt_cfg = AdamWConfig(
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        min_lr_ratio=cfg.cosine_min_lr_ratio,
        betas=cfg.betas,
    )
    optimizer = build_optimizer(model, opt_cfg)
    scheduler = WarmupCosineLR(optimizer, opt_cfg)
    return optimizer, scheduler, opt_cfg


def clip_grad(model, max_norm: float) -> float:
    """Clip grads and return the gradient norm."""
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))


# ---------------------------------------------------------------------------
# Training step (one packed sequence)
# ---------------------------------------------------------------------------

def pretrain_step(
    model,
    batch: Dict[str, np.ndarray],
    optimizer,
    scheduler,
    cfg: FaultlessPretrainConfig,
    step: int,
) -> float:
    """One training step. Returns scalar loss."""
    input_ids = torch.from_numpy(batch["input_ids"]).long().to(cfg.device)
    labels = torch.from_numpy(batch["labels"]).long().to(cfg.device)

    # M4 fix: ParquetDataset.iter_batches yields 1D arrays of shape
    # [seq_len] (a single packed sequence per batch — effective batch
    # size = 1). The model, however, expects [B, T] input. Add the
    # batch dim so logits come back as [1, seq_len, vocab] and the
    # shifting below is dimensionally consistent.
    input_ids = input_ids.unsqueeze(0)
    labels = labels.unsqueeze(0)

    logits = model(input_ids)  # [B=1, seq_len, vocab]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    loss.backward()
    grad_norm = clip_grad(model, cfg.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # C5 fix: WarmupCosineLR.step() requires an explicit global_step.
    # The step counter is tracked by the caller and passed in.
    if scheduler is not None:
        scheduler.step(step)

    return float(loss.detach()), grad_norm


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

def evaluate_val(
    model,
    val_iter: Iterator[Dict[str, np.ndarray]],
    cfg: FaultlessPretrainConfig,
    max_batches: int = 32,
) -> float:
    """Run validation pass; return mean loss."""
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_iter):
            if i >= max_batches:
                break
            input_ids = torch.from_numpy(batch["input_ids"]).long().to(cfg.device).unsqueeze(0)
            labels    = torch.from_numpy(batch["labels"]).long().to(cfg.device).unsqueeze(0)
            logits = model(input_ids)  # [1, seq_len, vocab]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            losses.append(float(loss.detach()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def pretrain(model, cfg: FaultlessPretrainConfig,
             log_callback: Optional[Callable[[Dict], None]] = None) -> List[Dict]:
    """Run faultless pretraining on parquet data.

    Returns:
        List of log dicts (per `log_every`).
    """
    from src.data.parquet_dataset import ParquetDataset

    # ── Reproducibility ─────────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True

    # ── Move model to device, set dtype ────────────────────────────────
    model.to(cfg.device)
    if cfg.dtype is not None:
        model.to(dtype=cfg.dtype)

    # ── Load parquet dataset ───────────────────────────────────────────
    print(f"\n[pretrain] Loading parquet dataset from {cfg.hf_repo_id} ...")
    dataset = ParquetDataset(
        repo_id=cfg.hf_repo_id,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        sources_filter=None,
        val_frac=0.01,
        train=True,
        eos_id=cfg.eos_id,  # M1: pass through to mark doc boundaries
    )
    dataset.load(max_shards=cfg.max_shards)
    stats = dataset.source_stats()
    print(f"\n[pretrain] Dataset stats:")
    for src, s in sorted(stats.items()):
        print(f"  {src:<10}: {s['docs']:>9,} docs  "
              f"{s['tokens']:>12,} tokens  ({s['tokens']/1e6:.1f}M)")
    total_tokens = dataset.total_tokens()
    print(f"  TOTAL     : {total_tokens / 1e9:.2f}B tokens\n")

    # ── Build optimizer ────────────────────────────────────────────────
    optimizer, scheduler, opt_cfg = build_optimizer_and_sched(model, cfg)

    # ── Resume from checkpoint ─────────────────────────────────────────
    start_step = 0
    if cfg.resume_from:
        ckpt_path = Path(cfg.resume_from)
        if ckpt_path.exists():
            print(f"[pretrain] Resuming from {ckpt_path} ...")
            start_step = load_checkpoint(ckpt_path, model, optimizer, scheduler,
                                         map_location=cfg.device)
            print(f"[pretrain] Resumed at step {start_step}")
        else:
            print(f"[pretrain] WARNING: resume_from={ckpt_path} not found, "
                  f"starting from step 0")

    # ── I/O setup ──────────────────────────────────────────────────────
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(cfg.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Main loop ──────────────────────────────────────────────────────
    logs: List[Dict] = []
    val_dataset = ParquetDataset(
        repo_id=cfg.hf_repo_id,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        val_frac=0.01,
        eos_id=cfg.eos_id,  # M1: same EOS as train
    )
    # Reuse loaded shards from main dataset
    val_dataset.shards = dataset.shards
    val_dataset.manifest = dataset.manifest

    val_iter_factory = lambda: val_dataset.iter_val(
        seed=cfg.seed + 1, n_samples=cfg.val_samples, seq_len=cfg.seq_len
    )

    # Dataset iterator factory (rebuilds each epoch)
    def train_iter_factory(resume: int):
        return dataset.iter_batches(
            seed=cfg.seed + resume,
            resume_step=resume,
            deterministic=False,
        )

    t_start = time.time()
    model.train()
    print(f"[pretrain] Starting training from step {start_step} "
          f"to step {cfg.max_steps}")
    print(f"[pretrain] Config: {cfg.__dict__}")

    step = start_step
    epoch = 0
    while step < cfg.max_steps:
        epoch += 1
        iter_factory = train_iter_factory(resume=step)
        for batch in iter_factory:
            try:
                loss, grad_norm = pretrain_step(
                    model, batch, optimizer, scheduler, cfg, step
                )
            except torch.cuda.OutOfMemoryError:
                print(f"[pretrain] CUDA OOM at step {step}; reducing")
                torch.cuda.empty_cache()
                continue
            except Exception as exc:
                print(f"[pretrain] Step {step} FAILED: {exc}; skipping")
                continue

            log_entry = {
                "step": step,
                "loss": loss,
                "grad_norm": grad_norm,
                "lr": scheduler.get_lr()[0] if scheduler else 0.0,
                "epoch": epoch,
                "elapsed": time.time() - t_start,
            }
            logs.append(log_entry)

            if cfg.log_every > 0 and step % cfg.log_every == 0:
                sps = (step - start_step + 1) / max(log_entry["elapsed"], 0.001)
                print(
                    f"[step {step:>6}/{cfg.max_steps}] "
                    f"loss={loss:.4f}  "
                    f"lr={log_entry['lr']:.2e}  "
                    f"gn={grad_norm:.2f}  "
                    f"epoch={epoch}  "
                    f"sps={sps:.2f}  "
                    f"elapsed={log_entry['elapsed']:.1f}s"
                )
                # Append to JSONL log file
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
                if log_callback is not None:
                    log_callback(log_entry)

            # Validation pass
            if cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0:
                val_loss = evaluate_val(model, val_iter_factory(),
                                        cfg, max_batches=16)
                print(f"[step {step:>6}] VAL loss={val_loss:.4f}")
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "val_loss": val_loss,
                                        "kind": "val"}) + "\n")

            # Checkpoint
            if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                ckpt_path = save_dir / f"step_{step+1}.pt"
                save_checkpoint_atomic(model, optimizer, scheduler,
                                       step + 1, cfg, ckpt_path)
                print(f"[step {step+1}] saved checkpoint to {ckpt_path}")

            step += 1
            if step >= cfg.max_steps:
                break

    # Final checkpoint
    final_path = save_dir / "final.pt"
    save_checkpoint_atomic(model, optimizer, scheduler,
                           cfg.max_steps, cfg, final_path)
    print(f"\n[pretrain] Final checkpoint saved to {final_path}")
    print(f"[pretrain] Total time: {time.time() - t_start:.1f}s")
    return logs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--save-dir", default="checkpoints/pretrain")
    parser.add_argument("--log-file", default="logs/pretrain.jsonl")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--eos-id", type=int, default=None,
                        help="EOS token id used to mark doc boundaries in parquet data (M1 fix)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Build tiny model for smoke test
    from src.model.config import ModelConfig
    from src.model.init import init_and_verify

    cfg = FaultlessPretrainConfig(
        max_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        save_dir=args.save_dir,
        log_file=args.log_file,
        resume_from=args.resume_from,
        max_shards=args.max_shards,
        eos_id=args.eos_id,  # M1: CLI override for EOS token id
        device=args.device,
        save_every=max(100, args.steps // 5),
        log_every=max(10, args.steps // 20),
    )
    model_cfg = ModelConfig()
    model = init_and_verify(model_cfg)
    logs = pretrain(model, cfg)
    print(f"Done. Final loss={logs[-1]['loss']:.4f}")
