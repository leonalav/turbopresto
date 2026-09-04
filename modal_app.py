"""Modal A100 deployment for RWKV-7 50M math LLM.

Usage:
    # Pretrain
    modal run modal_app.py --stage pretrain

    # SFT
    modal run modal_app.py --stage sft

    # GRPO
    modal run modal_app.py --stage grpo

    # Evaluate
    modal run modal_app.py --stage eval
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# App definition
# ---------------------------------------------------------------------------

app = modal.App("rwkv7-math-llm")

# Image: CUDA 12.1 + PyTorch 2.1 + all dependencies
IMAGE = (
    modal.Image.debian_slim(python_version=(3, 11))
    .pip install(
        "torch>=2.1.0",
        "einops>=0.7.0",
        "numpy>=1.24",
        "tiktoken>=0.5.0",
        "datasets>=2.14.0",
        "huggingface-hub>=0.20.0",
        "safetensors>=0.4.0",
        "pyyaml>=6.0",
        "wandb>=0.15.0",
        "pyarrow>=14.0.0",
    )
    # RWKV CUDA kernel — compile from source
    .run_commands(
        "pip install torch.utils.cpp_extension || true",
    )
)

# GPU + compute: L40S @ 2 CPU, 4 GiB RAM (per user-specified profile, ~$2.33/hr)
GPU_CONFIG = modal.gpu.L40S(count=1)
CPU = 2
MEM = 8192  # MiB


# ---------------------------------------------------------------------------
# Volumes for persistence
# ---------------------------------------------------------------------------

DATA_VOL = modal.Volume.from_name("rwkv7-data", create_if_missing=True)
CKPT_VOL = modal.Volume.from_name("rwkv7-checkpoints", create_if_missing=True)
LOGS_VOL = modal.Volume.from_name("rwkv7-logs", create_if_missing=True)


# ---------------------------------------------------------------------------
# Helper to build model (imported at runtime)
# ---------------------------------------------------------------------------

def get_model_config():
    from src.model.config import ModelConfig
    return ModelConfig()


# ---------------------------------------------------------------------------
# Stage: pretrain
# ---------------------------------------------------------------------------

@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    cpu=CPU,
    memory=MEM,
    volumes={
        "/data": DATA_VOL,
        "/checkpoints": CKPT_VOL,
        "/logs": LOGS_VOL,
    },
    timeout=3600 * 24,  # 24 hours max
    retries=modal.Retries(max_retries=2),
)
def run_pretrain(
    batch_size: int = 16,
    max_steps: int = 50000,
    seq_len: int = 4096,
    lr: float = 6e-4,
    save_every: int = 5000,
    resume_from: str = "",
    max_shards: int = 0,
    data_source: str = "parquet",  # "parquet" (HF Hub) or "stream" (original)
    eos_id: Optional[int] = None,  # tokenizer.eos_id; marks doc boundaries
):
    """Run pretraining on A100.

    Args:
        data_source: "parquet" loads pretokenized data from
            leonidas123/valkmodel-data (recommended).  "stream" uses the
            original stream-from-HF tokenization pipeline.
        eos_id: EOS token id used by ParquetDataset to mark document
            boundaries so the model learns end-of-document. Must match the
            tokenizer that produced the parquet chunks. If None,
            ParquetDataset falls back to token 0 with a one-shot warning.
    """
    import os
    os.makedirs("/checkpoints/pretrain", exist_ok=True)
    os.makedirs("/logs/pretrain", exist_ok=True)

    import torch
    torch.backends.cudnn.benchmark = True

    from src.model.config import ModelConfig
    from src.model.init import init_and_verify

    cfg = ModelConfig()
    model = init_and_verify(cfg)
    model = model.to(dtype=torch.bfloat16)
    model = model.cuda()

    # Resolve eos_id: prefer the explicit CLI value, otherwise read from
    # MathTokenizer so the user doesn't need to remember the id.
    if eos_id is None:
        from src.tokenizer.math_tokenizer import MathTokenizer
        eos_id = MathTokenizer().eos_id
        print(f"[run_pretrain] eos_id not provided; using "
              f"MathTokenizer.eos_id = {eos_id}")

    import wandb
    wandb.init(
        project="rwkv7-math-llm",
        name="pretrain",
        config={
            "batch_size": batch_size,
            "max_steps": max_steps,
            "seq_len": seq_len,
            "lr": lr,
            "total_params": cfg.total_params(),
            "data_source": data_source,
            "eos_id": eos_id,
        },
    )

    if data_source == "parquet":
        # Use the new faultless parquet pipeline
        from src.training.pretrain_parquet import (
            FaultlessPretrainConfig, pretrain as pretrain_parquet,
        )
        fp_cfg = FaultlessPretrainConfig(
            batch_size=batch_size,
            max_steps=max_steps,
            seq_len=seq_len,
            lr=lr,
            save_every=save_every,
            save_dir="/checkpoints/pretrain",
            log_file="/logs/pretrain/pretrain.jsonl",
            device="cuda",
            dtype=torch.bfloat16,
            resume_from=resume_from or None,
            max_shards=max_shards or None,
            eos_id=eos_id,
        )
        logs = pretrain_parquet(model, fp_cfg)
    else:
        # Original streaming pipeline (legacy / fallback)
        from src.tokenizer.math_tokenizer import MathTokenizer
        from src.training.pretrain import (
            PretrainConfig, pretrain, build_pretrain_dataset,
        )
        tokenizer = MathTokenizer()
        pretrain_cfg = PretrainConfig(
            batch_size=batch_size,
            max_steps=max_steps,
            seq_len=seq_len,
            lr=lr,
            save_every=save_every,
            save_dir="/checkpoints/pretrain",
            log_file="/logs/pretrain/pretrain.jsonl",
            device="cuda",
            dtype=torch.bfloat16,
        )
        texts = build_pretrain_dataset(pretrain_cfg)
        logs = pretrain(model, pretrain_cfg, tokenizer, texts)

    CKPT_VOL.commit()
    LOGS_VOL.commit()
    print(f"Pretrain complete. Final loss: {logs[-1]['loss']:.4f}")


# ---------------------------------------------------------------------------
# Stage: sft
# ---------------------------------------------------------------------------

@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    cpu=CPU,
    memory=MEM,
    volumes={
        "/data": DATA_VOL,
        "/checkpoints": CKPT_VOL,
        "/logs": LOGS_VOL,
    },
    timeout=3600 * 12,
    retries=modal.Retries(max_retries=2),
)
def run_sft(
    batch_size: int = 8,
    max_steps: int = 5000,
    seq_len: int = 2048,
    lr: float = 1e-5,
    save_every: int = 1000,
    pretrained_ckpt: str = "/checkpoints/pretrain/final.pt",
):
    """Run SFT on A100."""
    import os
    os.makedirs("/checkpoints/sft", exist_ok=True)
    os.makedirs("/logs/sft", exist_ok=True)

    import torch
    torch.backends.cudnn.benchmark = True

    from src.model.config import ModelConfig
    from src.model.init import init_and_verify
    from src.model.rwkv7 import build_model
    from src.tokenizer.math_tokenizer import MathTokenizer
    from src.training.optimizer import AdamWConfig, build_optimizer, WarmupCosineLR
    from src.training.sft import SFTConfig, sft, build_sft_examples
    from src.training.optimizer import load_checkpoint

    cfg = ModelConfig()
    model = init_and_verify(cfg)
    model = model.to(dtype=torch.bfloat16)
    model = model.cuda()

    # Load pretrain checkpoint
    if Path(pretrained_ckpt).exists():
        load_checkpoint(pretrained_ckpt, model, map_location="cuda")
        print(f"Loaded pretrained from {pretrained_ckpt}")

    tokenizer = MathTokenizer()

    sft_cfg = SFTConfig(
        batch_size=batch_size,
        max_steps=max_steps,
        seq_len=seq_len,
        lr=lr,
        save_every=save_every,
        save_dir="/checkpoints/sft",
        log_file="/logs/sft/sft.jsonl",
        device="cuda",
        dtype=torch.bfloat16,
    )

    import wandb
    wandb.init(
        project="rwkv7-math-llm",
        name="sft",
        config={"batch_size": batch_size, "max_steps": max_steps, "lr": lr},
    )

    logs = sft(model, sft_cfg, tokenizer)

    CKPT_VOL.commit()
    LOGS_VOL.commit()
    print(f"SFT complete. Final loss: {logs[-1]['loss']:.4f}")


# ---------------------------------------------------------------------------
# Stage: grpo
# ---------------------------------------------------------------------------

@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    cpu=CPU,
    memory=MEM,
    volumes={
        "/data": DATA_VOL,
        "/checkpoints": CKPT_VOL,
        "/logs": LOGS_VOL,
    },
    timeout=3600 * 24,
    retries=modal.Retries(max_retries=2),
)
def run_grpo(
    group_size: int = 8,
    batch_size: int = 4,
    max_steps: int = 3000,
    seq_len: int = 1024,
    lr: float = 5e-6,
    save_every: int = 500,
    kl_coef: float = 0.04,
    clip_eps: float = 0.2,
    sft_ckpt: str = "/checkpoints/sft/final.pt",
):
    """Run GRPO on A100."""
    import os
    os.makedirs("/checkpoints/grpo", exist_ok=True)
    os.makedirs("/logs/grpo", exist_ok=True)

    import torch
    torch.backends.cudnn.benchmark = True

    from src.model.config import ModelConfig
    from src.model.init import init_and_verify
    from src.model.rwkv7 import build_model
    from src.tokenizer.math_tokenizer import MathTokenizer
    from src.training.grpo import GRPOConfig, grpo
    from src.training.optimizer import load_checkpoint

    cfg = ModelConfig()
    model = init_and_verify(cfg)
    model = model.to(dtype=torch.bfloat16)
    model = model.cuda()

    ref_model = init_and_verify(cfg)
    ref_model = ref_model.to(dtype=torch.bfloat16)
    ref_model = ref_model.cuda()

    # Load SFT checkpoint
    if Path(sft_ckpt).exists():
        load_checkpoint(sft_ckpt, model, map_location="cuda")
        load_checkpoint(sft_ckpt, ref_model, map_location="cuda")
        ref_model.eval()
        print(f"Loaded SFT from {sft_ckpt}")

    tokenizer = MathTokenizer()

    grpo_cfg = GRPOConfig(
        group_size=group_size,
        batch_size=batch_size,
        max_steps=max_steps,
        seq_len=seq_len,
        lr=lr,
        kl_coef=kl_coef,
        clip_eps=clip_eps,
        save_every=save_every,
        save_dir="/checkpoints/grpo",
        device="cuda",
        dtype=torch.bfloat16,
    )

    import wandb
    wandb.init(
        project="rwkv7-math-llm",
        name="grpo",
        config={
            "group_size": group_size, "batch_size": batch_size,
            "max_steps": max_steps, "lr": lr,
            "kl_coef": kl_coef, "clip_eps": clip_eps,
        },
    )

    logs = grpo(model, grpo_cfg, tokenizer, ref_model=ref_model)

    CKPT_VOL.commit()
    LOGS_VOL.commit()
    print(f"GRPO complete. Final stats: {logs[-1]}")


# ---------------------------------------------------------------------------
# Stage: eval
# ---------------------------------------------------------------------------

@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    cpu=CPU,
    memory=MEM,
    volumes={
        "/data": DATA_VOL,
        "/checkpoints": CKPT_VOL,
        "/logs": LOGS_VOL,
    },
    timeout=3600 * 6,
)
def run_eval(
    ckpt_path: str = "/checkpoints/grpo/final.pt",
    eval_split: str = "test",
    n_samples: int = 16,
    max_new_tokens: int = 512,
):
    """Evaluate on GSM8K test set."""
    import os
    os.makedirs("/logs/eval", exist_ok=True)

    import torch
    import json

    from src.model.config import ModelConfig
    from src.model.init import init_and_verify
    from src.model.rwkv7 import build_model
    from src.training.optimizer import load_checkpoint
    from src.tokenizer.math_tokenizer import MathTokenizer
    from src.eval.gsm8k_eval import evaluate_gsm8k
    from src.data.gsm8k import GSM8KDataset

    cfg = ModelConfig()
    model = init_and_verify(cfg)
    model = model.to(dtype=torch.bfloat16)
    model = model.cuda()

    if Path(ckpt_path).exists():
        load_checkpoint(ckpt_path, model, map_location="cuda")
        print(f"Loaded from {ckpt_path}")
    else:
        print(f"WARNING: checkpoint {ckpt_path} not found, using random init")

    tokenizer = MathTokenizer()
    dataset = GSM8KDataset(split=eval_split)

    results = evaluate_gsm8k(
        model, tokenizer, dataset,
        max_new_tokens=max_new_tokens,
        n_samples=n_samples,
        method="majority" if n_samples > 1 else "greedy",
    )

    results_file = f"/logs/eval/{Path(ckpt_path).stem}_{eval_split}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== GSM8K {eval_split} ===")
    print(f"Accuracy: {results['accuracy']*100:.2f}% ({results['correct']}/{results['total']})")
    print(f"Results saved to {results_file}")

    LOGS_VOL.commit()
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(stage: str = "pretrain", **kwargs):
    """Run a training stage."""
    if stage == "pretrain":
        run_pretrain(**kwargs)
    elif stage == "sft":
        run_sft(**kwargs)
    elif stage == "grpo":
        run_grpo(**kwargs)
    elif stage == "eval":
        run_eval(**kwargs)
    elif stage == "pretok":
        # Delegate to pretok/pretokenize.py on a 32-CPU Modal VM
        from pretok.pretokenize import upload_pretokenized
        upload_pretokenized.remote(
            max_openr1=kwargs.get("max_openr1", 50_000),
            max_physics=kwargs.get("max_physics", 30_000),
            max_synthetic=kwargs.get("max_synthetic", 2_000),
        )
    else:
        print(f"Unknown stage: {stage}")
        print("Available: pretrain, sft, grpo, eval, pretok")


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "pretrain"
    main(stage)
