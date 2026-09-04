"""Verify the dtype fix: model in bf16 should keep bf16 dtype throughout WKV."""
import sys
sys.path.insert(0, '.')

import torch
from src.model.config import ModelConfig
from src.model.init import init_and_verify
import torch.nn.functional as F

cfg_m = ModelConfig()
model = init_and_verify(cfg_m)
model = model.to(dtype=torch.bfloat16)  # simulate the L40S training setup

T = 32
# Random token batch
input_ids = torch.randint(0, cfg_m.vocab_size, (1, T))
labels = torch.randint(0, cfg_m.vocab_size, (1, T))

print("Testing forward pass in bf16...")
logits = model(input_ids)
print(f"logits dtype: {logits.dtype}, shape: {logits.shape}")

shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()

loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
)
print(f"loss dtype: {loss.dtype}, loss: {loss.item():.4f}")
print("OK")
