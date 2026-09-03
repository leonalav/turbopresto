"""SFT (Supervised Fine-Tuning) loop for math reasoning.

SFT trains the model to follow the expected output format:
    Question: ...
    Answer: <REASON>chain of thought</REASON><ANSWER>number</ANSWER>

Key differences from pretraining:
- Lower LR (1e-5 vs 6e-4)
- Mask prompt tokens from loss (only train on target)
- Format-specific data
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from src.data.collator import collate_for_sft
from src.data.gsm8k import GSM8KDataset
from src.data.math_dataset import MATHDataset
from src.data.synthetic import SyntheticMathDataset
from src.model.config import ModelConfig
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
class SFTConfig:
    """SFT configuration."""

    batch_size: int = 8
    grad_accum: int = 8
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    seq_len: int = 2048
    save_every: int = 1000
    log_every: int = 20
    grad_clip: float = 1.0
    cosine_min_lr_ratio: float = 0.1
    seed: int = 42
    save_dir: str = "checkpoints/sft"
    log_file: Optional[str] = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    betas: tuple = (0.9, 0.95)


def build_sft_examples(cfg: SFTConfig) -> List[Dict[str, str]]:
    """Build SFT (prompt, target) pairs from multiple sources."""
    examples: List[Dict[str, str]] = []

    # Synthetic
    syn = SyntheticMathDataset(size=2000, max_digits=3, seed=cfg.seed)
    for i in range(len(syn)):
        examples.append({"source": "synthetic", **dict(zip(["prompt", "target"], syn.format_for_sft(i)))})

    # GSM8K
    try:
        gsm = GSM8KDataset(split="train")
        if len(gsm) > 0:
            for i in range(min(len(gsm), 5000)):
                p, t = gsm.format_for_sft(i)
                examples.append({"source": "gsm8k", "prompt": p, "target": t})
    except Exception:
        pass

    # MATH
    try:
        math_ds = MATHDataset(split="train")
        if len(math_ds) > 0:
            for i in range(min(len(math_ds), 5000)):
                p, t = math_ds.format_for_sft(i)
                examples.append({"source": "math", "prompt": p, "target": t})
    except Exception:
        pass

    return examples


def sft_step(
    model: RWKV7Model,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineLR,
    cfg: SFTConfig,
    step: int,
) -> float:
    """Single SFT step. Returns loss."""
    input_ids = batch["input_ids"].to(cfg.device)
    labels = batch["labels"].to(cfg.device)

    logits = model(input_ids)

    # Loss: only on non-masked positions (labels != -100)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )

    loss.backward()

    if cfg.grad_clip > 0:
        clip_grad_norm(model.parameters(), cfg.grad_clip)

    optimizer.step()
    optimizer.zero_grad()

    scheduler.step(step)
    return loss.item()


def sft(
    model: RWKV7Model,
    cfg: SFTConfig,
    tokenizer,
    examples: Optional[List[Dict]] = None,
    progress_callback: Optional[Callable[[int, float], None]] = None,
) -> List[Dict]:
    """Run full SFT loop."""
    set_seed(cfg.seed)
    model.to(cfg.device)

    if examples is None:
        examples = build_sft_examples(cfg)
    print(f"SFT on {len(examples)} examples")

    # Strip source field for collator
    collate_examples = [
        {"prompt": ex["prompt"], "target": ex["target"]} for ex in examples
    ]

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

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    logs = []
    model.train()

    example_iter = iter(collate_examples)
    t_start = time.time()

    for step in range(cfg.max_steps):
        try:
            batch_examples = [next(example_iter) for _ in range(cfg.batch_size)]
        except StopIteration:
            from random import shuffle
            shuffle(collate_examples)
            example_iter = iter(collate_examples)
            batch_examples = [next(example_iter) for _ in range(cfg.batch_size)]

        batch = collate_for_sft(batch_examples, tokenizer, seq_len=cfg.seq_len)
        loss = sft_step(model, batch, optimizer, scheduler, cfg, step)
        logs.append({"step": step, "loss": loss})

        if cfg.log_every > 0 and step % cfg.log_every == 0:
            lr_now = scheduler.get_lr()[0]
            elapsed = time.time() - t_start
            print(
                f"[step {step:>5}] loss={loss:.4f} lr={lr_now:.2e} "
                f"elapsed={elapsed:.1f}s"
            )
            if progress_callback:
                progress_callback(step, loss)

        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"step_{step+1}.pt")
            save_checkpoint(model, optimizer, step + 1, ckpt_path)

        if cfg.log_file and cfg.log_every > 0 and step % cfg.log_every == 0:
            with open(cfg.log_file, "a") as f:
                f.write(json.dumps(logs[-1]) + "\n")

    # Final save
    ckpt_path = os.path.join(cfg.save_dir, "final.pt")
    save_checkpoint(model, optimizer, cfg.max_steps, ckpt_path)
    return logs


if __name__ == "__main__":
    cfg = SFTConfig(max_steps=5, save_every=0, log_every=1, batch_size=2, seq_len=128)
    from src.model.init import init_and_verify
    model = init_and_verify(ModelConfig(vocab_size=256, n_layers=2, d_model=64, d_state=32))
    from src.tokenizer.math_tokenizer import StubTokenizer
    tok = StubTokenizer(vocab_size=256)
    logs = sft(model, cfg, tok)
    print(f"Final loss: {logs[-1]['loss']:.4f}")