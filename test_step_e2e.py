"""Smoke test: ensure the fixed pretrain_step runs without unpacking errors."""
import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from src.model.config import ModelConfig
from src.model.init import init_and_verify
from src.training.pretrain_parquet import FaultlessPretrainConfig, pretrain_step, build_optimizer_and_sched

cfg_m = ModelConfig()
model = init_and_verify(cfg_m)

cfg = FaultlessPretrainConfig(
    seq_len=64,
    batch_size=1,
    max_steps=10,
)
optimizer, scheduler, _ = build_optimizer_and_sched(model, cfg)

# Fake one batch
batch = {
    'input_ids': np.random.randint(0, cfg_m.vocab_size, size=(64,), dtype=np.int64),
    'labels':    np.random.randint(0, cfg_m.vocab_size, size=(64,), dtype=np.int64),
}
loss, grad_norm = pretrain_step(model, batch, optimizer, scheduler, cfg, step=0)
print(f"loss={loss:.4f}  grad_norm={grad_norm:.4f}")
print("OK")
