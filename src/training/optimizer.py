"""Optimizer and learning rate schedule.

Per /imo-mathematician: AdamW with cosine schedule and warmup is the
canonical choice for transformer/RWKV pretraining. Key properties:

1. AdamW (decoupled weight decay) is standard for LLM training
   - LoRA-style: weight_decay applies to "main" weights, not biases/norms
   - betas=(0.9, 0.95) or (0.9, 0.999) depending on stage

2. Cosine schedule with linear warmup
   - LR goes from 0 -> peak (warmup) -> min_lr (cosine decay)
   - Warmup prevents early gradient explosion

3. Gradient clipping
   - clip_grad_norm_ at 1.0 prevents O(T) RWKV gradient explosion
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional

import torch
import torch.nn as nn


@dataclass
class AdamWConfig:
    """AdamW configuration."""

    lr: float = 6e-4
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1

    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 50000
    min_lr_ratio: float = 0.1  # final_lr = lr * min_lr_ratio


def cosine_schedule(
    step: int,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Cosine schedule with linear warmup.

    Per /ipho-physicist: cosine annealing is smoother than step decay.
    The schedule is:
        LR(s) = peak_lr * f(s / max_steps)

    where:
        f(t) = t/warmup_steps                    if t < warmup_steps (linear warmup)
        f(t) = min_lr + (1 - min_lr) * cos_decay if t >= warmup_steps

    Args:
        step: Current step (0-indexed).
        warmup_steps: Number of warmup steps.
        max_steps: Total number of steps.
        min_lr_ratio: Final LR as fraction of peak.

    Returns:
        Multiplier in [min_lr_ratio, 1.0].
    """
    if step < warmup_steps:
        # Linear warmup: 0 -> 1
        return step / max(1, warmup_steps)

    if step >= max_steps:
        return min_lr_ratio

    # Cosine decay: 1 -> min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_optimizer(
    model: nn.Module,
    cfg: AdamWConfig,
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with proper parameter groups.

    Separates parameters into:
    - decay: weights with dim >= 2 (Linear, Embedding weights)
    - no_decay: biases, LayerNorm, scalars (anything dim < 2)
    """
    decay_params = []
    no_decay_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Don't decay biases, norms, or scalar mixers (1,1,C) shape
        if (
            p.dim() < 2
            or "bias" in name
            or "norm" in name.lower()
            or "ln" in name.lower()
        ):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
    )


class WarmupCosineLR:
    """Custom LR scheduler: linear warmup then cosine decay.

    Does NOT inherit from torch.optim.lr_scheduler.LRScheduler so that
    callers control exactly when step() is called (with explicit
    global_step rather than internal counters).
    """

    def __init__(self, optimizer: torch.optim.Optimizer, cfg: AdamWConfig):
        self.optimizer = optimizer
        self.cfg = cfg
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._step_count = 0

    def step(self, global_step: Optional[int] = None) -> None:
        """Update learning rate.

        Args:
            global_step: Current training step.  If omitted the internal
                         counter is incremented by 1 (for callers that
                         prefer auto-increment).
        """
        if global_step is not None:
            self._step_count = global_step
        else:
            self._step_count += 1
        mult = cosine_schedule(
            self._step_count,
            self.cfg.warmup_steps,
            self.cfg.max_steps,
            self.cfg.min_lr_ratio,
        )
        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base_lr * mult

    def get_lr(self) -> List[float]:
        return [g["lr"] for g in self.optimizer.param_groups]

    # ------------------------------------------------------------------
    # Checkpoint support (required by save_checkpoint_atomic / load_checkpoint)
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Return serializable state for checkpointing."""
        return {"step_count": self._step_count}

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler from checkpoint state."""
        self._step_count = int(state["step_count"])
        # Re-apply the LR for the restored step so param_groups are current
        self.step(self._step_count)


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

def clip_grad_norm(
    parameters: Iterator[nn.Parameter],
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """Clip gradient norm in-place. Returns the total norm."""
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
    extra: Optional[dict] = None,
) -> None:
    """Save model + optimizer + step."""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    if extra is not None:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> int:
    """Load checkpoint. Returns the saved step."""
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state.get("step", 0)


if __name__ == "__main__":
    # Smoke test
    cfg = AdamWConfig(lr=6e-4, warmup_steps=100, max_steps=1000)
    print(f"LR schedule at step 0: {cosine_schedule(0, 100, 1000):.4f}")
    print(f"LR schedule at step 50: {cosine_schedule(50, 100, 1000):.4f}")
    print(f"LR schedule at step 100: {cosine_schedule(100, 100, 1000):.4f}")
    print(f"LR schedule at step 500: {cosine_schedule(500, 100, 1000):.4f}")
    print(f"LR schedule at step 1000: {cosine_schedule(1000, 100, 1000):.4f}")
    print(f"LR schedule at step 2000: {cosine_schedule(2000, 100, 1000):.4f}")

    # Test optimizer construction
    from src.model.config import ModelConfig
    from src.model.init import init_and_verify
    model = init_and_verify(ModelConfig())
    opt = build_optimizer(model, cfg)
    print(f"\nOptimizer param groups: {len(opt.param_groups)}")
    for i, g in enumerate(opt.param_groups):
        n_params = sum(p.numel() for p in g["params"])
        print(f"  Group {i}: {n_params:,} params, wd={g['weight_decay']}, lr={g['lr']}")