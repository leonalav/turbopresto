"""SFT (Supervised Fine-Tuning) loop for math reasoning.

SFT trains the model to follow the expected output format:
    Question: ...
    Answer: <REASON>chain of thought</REASON><ANSWER>number</ANSWER>

Key differences from pretraining:
- Lower LR (1e-5 vs 6e-4)
- Mask prompt tokens from loss (only train on target)
- Format-specific data

────────────────────────────────────────────────────────────────────────────
SFT data sources:
  - tatsu-lab/alpaca              — General instruction-following (52k rows)
  - open-r1/OpenR1-Math-220k      — R1 reasoning traces (94k math problems)
  - camel-ai/physics              — Physics dialogues (camel-ai/physics)
  - SyntheticMathDataset          — Simple arithmetic (2000 examples)

Format separation rationale:
  - Math/physics use `<REASON>...</REASON><ANSWER>...</ANSWER>` delimiters.
  - Alpaca uses bare `Instruction: ... Response:` to keep it from mixing
    with math delimiters — this prevents the model from accidentally emitting
    `<ANSWER>` after non-math prompts.

NOTE: GSM8K and MATH (hendrycks/competition_math) are BENCHMARK datasets.
They must NEVER appear in training data — only in evaluation.
────────────────────────────────────────────────────────────────────────────
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
    """Build SFT (prompt, target) pairs from multiple sources.

    IMPORTANT: GSM8K and MATH are evaluation benchmarks only.  Do NOT add them
    to this function under any circumstances.
    """
    examples: List[Dict[str, str]] = []

    # ── Synthetic ────────────────────────────────────────────────────────────
    syn = SyntheticMathDataset(size=2000, max_digits=3, seed=cfg.seed)
    for i in range(len(syn)):
        examples.append({"source": "synthetic", **dict(zip(["prompt", "target"], syn.format_for_sft(i)))})

    # ── tatsu-lab/alpaca ────────────────────────────────────────────────────
    # General instruction-following data (Self-Instruct, 52k rows).
    # Schema: {instruction: str, input: str (often empty), output: str, text: str}.
    # ~41k rows have empty `input`; ~11k have non-empty `input`.
    # Total ≈ 52,002 rows on the train split.
    #
    # We use bare "Instruction: ... Response:" framing (NOT the math
    # <REASON>/<ANSWER> delimiters) so the model learns a clean separation
    # between non-math instructions and math reasoning — no risk of the
    # model emitting <ANSWER> tags after "Hello!".
    #
    # Dataset size rationale: 50M base model, standard Alpaca recipe.  52k
    # is the original Stanford Alpaca count that taught LLaMA-7B to be a
    # helpful assistant.  Going lower (LIMA 1k) is too sparse for a small
    # base to generalize, going higher dilutes math signal.
    try:
        from datasets import load_dataset
        alpaca_ds = load_dataset(
            "tatsu-lab/alpaca",
            split="train",
            trust_remote_code=True,
        )
        count = 0
        skipped = 0
        for row in alpaca_ds:
            instruction = row.get("instruction", "").strip()
            inp         = row.get("input", "").strip()
            output      = row.get("output", "").strip()
            if not instruction or not output:
                skipped += 1
                continue
            # Drop the few impossible/safety-flagged rows already marked <nooutput>
            # by Alpaca (e.g. "Render a 3D model of a house" → "<nooutput>").
            # These teach the model to refuse work, which we don't want for
            # a math assistant.
            if output.lower() == "<nooutput>" or output.lower().startswith("<nooutput>"):
                skipped += 1
                continue
            prompt = (
                f"Instruction: {instruction}\n"
                + (f"Input: {inp}\n" if inp else "")
                + "Response:"
            )
            target = f" {output}"
            examples.append({"source": "alpaca", "prompt": prompt, "target": target})
            count += 1
        print(f"[sft] Loaded {count:,} alpaca examples ({skipped:,} skipped)")
    except Exception as exc:
        print(f"[sft] tatsu-lab/alpaca unavailable ({exc}); skipping.")

    # ── open-r1/OpenR1-Math-220k ──────────────────────────────────────────
    # R1 traces: long chain-of-thought reasoning on math problems.
    # Covers GSM8K-level, MATH-level, and AoPS-level difficulty.
    try:
        from datasets import load_dataset
        openr1_ds = load_dataset(
            "open-r1/OpenR1-Math-220k",
            "default",
            split="train",
            streaming=True,
        )
        count = 0
        for row in openr1_ds:
            problem   = row.get("problem", "")
            solution  = row.get("solution", "")
            answer    = row.get("answer", "")
            if not problem or not solution:
                continue
            prompt = f"Problem: {problem}\nSolution: <REASON>"
            target = f"{solution}</REASON>\n<ANSWER>{answer}</ANSWER>"
            examples.append({"source": "openr1", "prompt": prompt, "target": target})
            count += 1
            if count >= 50_000:
                break
        print(f"[sft] Loaded {count:,} openr1 examples")
    except Exception as exc:
        print(f"[sft] open-r1/OpenR1-Math-220k unavailable ({exc}); skipping.")

    # ── camel-ai/physics ───────────────────────────────────────────────────
    # Physics Q+A dialogues: diverse physics sub-fields (mechanics, E&M, QM,
    # thermo, etc.). Uses <REASON>/<ANSWER> format for consistency.
    try:
        from datasets import load_dataset
        physics_ds = load_dataset("camel-ai/physics", split="train", streaming=True)
        count = 0
        pending_user: Optional[str] = None
        pending_field: Optional[str] = None
        for row in physics_ds:
            role    = str(row.get("role_type", "")).upper()
            msg     = row.get("message", {})
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if not content:
                continue
            field = str(row.get("physics_field", "unknown"))
            if "USER" in role:
                pending_user  = f"[Physics: {field}] {content}"
                pending_field = field
            elif "ASSISTANT" in role and pending_user is not None:
                prompt = f"{pending_user}\nAnswer: <REASON>"
                target = f"{content}</REASON>\n<ANSWER>Done</ANSWER>"
                examples.append({"source": "physics", "prompt": prompt, "target": target})
                pending_user = None
                count += 1
                if count >= 30_000:
                    break
        print(f"[sft] Loaded {count:,} physics examples")
    except Exception as exc:
        print(f"[sft] camel-ai/physics unavailable ({exc}); skipping.")

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

    logits = model(input_ids)  # [B, T, V]

    # Next-token prediction loss (C1 fix: shift to align logits[t] with labels[t+1]).
    # Without shifting, logits[t] would be trained against labels[t] — i.e. the model
    # would learn to predict the current token given context that includes itself,
    # which is not autoregressive next-token prediction.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Loss: only on non-masked positions (labels != -100)
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
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