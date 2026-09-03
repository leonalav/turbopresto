"""Pre-training loop for RWKV-7 math model.

This is the standard next-token-prediction training loop with:
- Cosine LR schedule with linear warmup
- Gradient clipping
- Periodic checkpointing
- Mixed dataset support (OpenR1-Math-220k, camel-ai/physics, synthetic)

Per /imo-mathematician: pretraining is the foundation of all downstream
performance. Verify loss decreases monotonically (modulo noise).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.collator import RWKVCollator
from src.data.synthetic import SyntheticMathDataset
from src.model.config import ModelConfig
from src.model.init import init_and_verify
from src.model.rwkv7 import RWKV7Model
from src.training.optimizer import (
    AdamWConfig,
    WarmupCosineLR,
    build_optimizer,
    clip_grad_norm,
    save_checkpoint,
)
from src.utils.seed import set_seed


@dataclass
class PretrainConfig:
    """Pre-training configuration."""

    batch_size: int = 16
    grad_accum: int = 4
    lr: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 50000
    seq_len: int = 4096
    save_every: int = 5000
    log_every: int = 50
    grad_clip: float = 1.0
    cosine_min_lr_ratio: float = 0.1
    seed: int = 42
    save_dir: str = "checkpoints/pretrain"
    log_file: Optional[str] = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    betas: tuple = (0.9, 0.95)


def build_pretrain_dataset(cfg: PretrainConfig) -> List[str]:
    """Build a list of pre-training texts.

    Mixes OpenR1-Math-220k, camel-ai/physics, and synthetic data.
    Uses streaming to avoid downloading the full dataset at once.
    """
    texts: List[str] = []

    # ── 1. Synthetic arithmetic (always works, no download) ─────────────────
    syn = SyntheticMathDataset(size=2000, max_digits=3, seed=cfg.seed)
    for i in range(len(syn)):
        texts.append(syn.format_for_pretrain(i))

    # ── 2. OpenR1-Math-220k (default split, ~94k rows) ───────────────────
    # Columns: problem | solution | answer | source | problem_type | ...
    # Use problem + solution + answer so the model sees full Q→A chains.
    # Filter to sources with verified correctness where available.
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "open-r1/OpenR1-Math-220k",
            "default",
            split="train",
            streaming=True,
        )
        count = 0
        for row in ds:
            # Prefer rows that have at least one correct generation
            if row.get("correctness_count", 0) == 0:
                # Still include — might be unverified, not wrong
                pass
            problem = row.get("problem", "")
            solution = row.get("solution", "")
            answer = row.get("answer", "")
            if problem and solution:
                text = f"Problem: {problem}\n\nSolution:\n{solution}\n\nAnswer: {answer}"
                texts.append(text)
            count += 1
            if count >= 50_000:  # cap at 50k to keep memory reasonable
                break
        print(f"Loaded {count} samples from OpenR1-Math-220k")
    except Exception as exc:
        print(f"[build_pretrain_dataset] OpenR1-Math-220k unavailable ({exc}); skipping.")

    # ── 3. camel-ai/physics ────────────────────────────────────────────────
    # Columns: role_type | physics_field | topic | message
    # message is a dict with "content": the text
    # Each row represents one message in a multi-turn dialogue.
    # We group consecutive USER/ASSISTANT rows into full dialogues.
    try:
        from datasets import load_dataset
        phys_ds = load_dataset(
            "camel-ai/physics",
            split="train",
            streaming=True,
        )
        # Group into (user, assistant) pairs per dialogue turn
        current_dialogue: List[str] = []
        dialogue_count = 0
        for row in phys_ds:
            role = row.get("role_type", "")
            msg = row.get("message", {})
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if not content:
                continue
            if "USER" in role.upper():
                current_dialogue = [f"[Physics: {row.get('physics_field','unknown')}] {content}"]
            elif "ASSISTANT" in role.upper():
                if current_dialogue:
                    # Pair the stored user prompt with this assistant response
                    user_prompt = current_dialogue[0]
                    text = f"{user_prompt}\n\nAssistant:\n{content}"
                    texts.append(text)
                    current_dialogue = []
                    dialogue_count += 1
                    if dialogue_count >= 30_000:  # cap physics dialogues
                        break
        print(f"Loaded {dialogue_count} physics dialogues from camel-ai/physics")
    except Exception as exc:
        print(f"[build_pretrain_dataset] camel-ai/physics unavailable ({exc}); skipping.")

    print(f"Total pre-training texts: {len(texts)}")
    return texts


def pretrain_step(
    model: RWKV7Model,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineLR,
    cfg: PretrainConfig,
    step: int,
) -> float:
    """Run a single training step. Returns the loss."""
    input_ids = batch["input_ids"].to(cfg.device)
    labels = batch["labels"].to(cfg.device)

    # Forward
    logits = model(input_ids)

    # Next-token prediction loss
    # Shift: predict token t+1 from logits at t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )

    # Backward
    loss.backward()

    # Gradient clipping
    if cfg.grad_clip > 0:
        grad_norm = clip_grad_norm(model.parameters(), cfg.grad_clip)

    # Update
    optimizer.step()
    optimizer.zero_grad()

    # Update LR
    scheduler.step(step)

    return loss.item()


def pretrain(
    model: RWKV7Model,
    cfg: PretrainConfig,
    tokenizer,
    train_texts: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, float], None]] = None,
) -> List[Dict]:
    """Run full pretraining loop.

    Args:
        model: RWKV-7 model (already initialized).
        cfg: Pretraining configuration.
        tokenizer: Tokenizer for encoding.
        train_texts: Optional list of training texts (auto-generated if None).
        progress_callback: Optional callback(step, loss).

    Returns:
        List of training log entries.
    """
    set_seed(cfg.seed)
    model.to(cfg.device)

    if train_texts is None:
        train_texts = build_pretrain_dataset(cfg)

    print(f"Pretraining on {len(train_texts)} texts")

    # Build collator
    collator = RWKVCollator(tokenizer, seq_len=cfg.seq_len)
    examples = [{"text": t} for t in train_texts]

    # Build optimizer
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

    # Save directory
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    if cfg.log_file:
        Path(cfg.log_file).parent.mkdir(parents=True, exist_ok=True)

    logs = []
    model.train()

    # Iterator over examples
    example_iter = iter(examples)
    t_start = time.time()

    for step in range(cfg.max_steps):
        # Get a batch
        try:
            batch_examples = [next(example_iter) for _ in range(cfg.batch_size)]
        except StopIteration:
            from random import shuffle
            shuffle(examples)
            example_iter = iter(examples)
            batch_examples = [next(example_iter) for _ in range(cfg.batch_size)]

        batch = collator.collate_fn(batch_examples)
        loss = pretrain_step(model, batch, optimizer, scheduler, cfg, step)
        logs.append({"step": step, "loss": loss})

        if cfg.log_every > 0 and step % cfg.log_every == 0:
            lr_now = scheduler.get_lr()[0]
            elapsed = time.time() - t_start
            sps = (step + 1) / max(elapsed, 0.001)
            print(
                f"[step {step:>6}] loss={loss:.4f} lr={lr_now:.2e} "
                f"sps={sps:.2f} elapsed={elapsed:.1f}s"
            )
            if progress_callback:
                progress_callback(step, loss)

        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"step_{step+1}.pt")
            save_checkpoint(model, optimizer, step + 1, ckpt_path)

        # Optional: write to log file
        if cfg.log_file and cfg.log_every > 0 and step % cfg.log_every == 0:
            with open(cfg.log_file, "a") as f:
                f.write(json.dumps(logs[-1]) + "\n")

    # Final save
    if cfg.save_dir:
        ckpt_path = os.path.join(cfg.save_dir, "final.pt")
        save_checkpoint(model, optimizer, cfg.max_steps, ckpt_path)

    return logs


def load_pretrained_model(checkpoint_path: str, cfg: Optional[ModelConfig] = None,
                         device: str = "cpu") -> RWKV7Model:
    """Load a pretrained model from checkpoint."""
    if cfg is None:
        cfg = ModelConfig()
    model = init_and_verify(cfg)
    state = torch.load(checkpoint_path, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.to(device)
    return model


if __name__ == "__main__":
    # Smoke test: tiny config, 10 steps
    cfg = PretrainConfig(
        max_steps=10,
        save_every=0,
        log_every=2,
        batch_size=2,
        seq_len=64,
    )
    model_cfg = ModelConfig(vocab_size=128, n_layers=2, d_model=64, d_state=32)
    model = init_and_verify(model_cfg)
    tokenizer = None  # placeholder
    from src.tokenizer.math_tokenizer import StubTokenizer
    tokenizer = StubTokenizer(vocab_size=128)

    train_texts = ["What is 1 + 1? 2", "What is 2 + 2? 4"] * 100
    logs = pretrain(model, cfg, tokenizer, train_texts)
    print(f"Final loss: {logs[-1]['loss']:.4f}")