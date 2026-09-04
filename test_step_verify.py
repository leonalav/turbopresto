import sys
import traceback
import numpy as np

sys.path.insert(0, 'src')

from src.model.config import ModelConfig
from src.model.init import init_and_verify

cfg = ModelConfig()
print(f"ModelConfig: vocab_size={cfg.vocab_size}, d_model={cfg.d_model}, n_layers={cfg.n_layers}")
model = init_and_verify(cfg)

# Simulate batch from iter_batches
T = 64
batch = {
    'input_ids': np.random.randint(0, cfg.vocab_size, size=(T,), dtype=np.int64),
    'labels':    np.random.randint(0, cfg.vocab_size, size=(T,), dtype=np.int64),
}

# Simulate pretrain_step (with the fix)
import torch
import torch.nn.functional as F

input_ids = torch.from_numpy(batch['input_ids']).long()
labels = torch.from_numpy(batch['labels']).long()

# The fix: unsqueeze to add batch dim
input_ids = input_ids.unsqueeze(0)
labels = labels.unsqueeze(0)

print('input_ids shape:', input_ids.shape)
try:
    logits = model(input_ids)
    print('logits shape:', logits.shape)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    print('shift_logits shape:', shift_logits.shape)
    print('shift_labels shape:', shift_labels.shape)

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    print('loss:', loss.item())
    print('OK')
except Exception as e:
    print('ERROR:', type(e).__name__, e)
    traceback.print_exc()
