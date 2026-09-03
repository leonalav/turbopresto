"""Initialization schemes for RWKV-7 model.

Initialization strategy (verified by /imo-mathematician and /ipho-physicist):

1. Linear projections (r, k, v, output, FFN key/value): orthogonal init
   - Maintains input/output variance through deep network
   - Critical for stable state in RWKV recurrence

2. Embeddings: scaled normal init with std = 1/sqrt(d_model)
   - Keeps initial activations O(1)

3. Time-shift mixing scalars (x_r, x_w, ...): init to 0.5
   - Initial state: blend 50/50 with previous token
   - Matches canonical RWKV-7 initialization

4. Decay-related (w0, w1, w2): small init
   - w0 = 0 (initial decay raw)
   - w1, w2 = small LoRA

5. LoRA matrices (w1, w2, a1, a2, ...): zeros for w2, normal for w1
   - LoRA convention: down-projection normal, up-projection zero
   - Initial pass-through

6. r_k (per-head r*k boost): ones init

7. LayerNorms: weight=1, bias=0

8. GroupNorm (ln_x): weight=1, bias=0

9. Output gate (g1, g2): zero init (gates start as 0)
   - Gates open gradually during training

All initialization must produce zero NaN/Inf.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.model.config import ModelConfig
from src.model.rwkv7 import (
    RWKV7Block,
    RWKV7Model,
    RWKV7ChannelMix,
    RWKV7TimeMix,
)


def _orthogonal_(w: torch.Tensor, gain: float = 1.0) -> None:
    """In-place orthogonal initialization."""
    nn.init.orthogonal_(w, gain=gain)


def _normal_(w: torch.Tensor, std: float = 0.02) -> None:
    """In-place normal initialization."""
    nn.init.normal_(w, mean=0.0, std=std)


def _zero_(w: torch.Tensor) -> None:
    nn.init.zeros_(w)


def _constant_(w: torch.Tensor, val: float) -> None:
    nn.init.constant_(w, val)


def init_time_mixing_scalars(att: RWKV7TimeMix) -> None:
    """Init x_r, x_w, x_k, x_v, x_a, x_g to 0.5 (canonical)."""
    for name in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"]:
        _constant_(getattr(att, name), 0.5)


def init_decay_lora(att: RWKV7TimeMix) -> None:
    """Init decay LoRA: w0 ~ small init for stable decay, w1 normal, w2 zero."""
    _zero_(att.w0)
    _normal_(att.w1, std=0.02)
    _zero_(att.w2)


def init_aaa_lora(att: RWKV7TimeMix) -> None:
    """Init in-context lr LoRA: a0 zero, a1 normal, a2 zero."""
    _zero_(att.a0)
    _normal_(att.a1, std=0.02)
    _zero_(att.a2)


def init_mv_lora(att: RWKV7TimeMix) -> None:
    """Init value-residual mixing LoRA: v0 = 0, v1 normal, v2 = 0."""
    _zero_(att.v0)
    _normal_(att.v1, std=0.02)
    _zero_(att.v2)


def init_gate_lora(att: RWKV7TimeMix) -> None:
    """Init output gate LoRA: g1 normal, g2 = 0 (gates start closed)."""
    _normal_(att.g1, std=0.02)
    _zero_(att.g2)


def init_kk_rk(att: RWKV7TimeMix) -> None:
    """Init k_k = 1, k_a = 0.5, r_k = 1."""
    _constant_(att.k_k, 1.0)
    _constant_(att.k_a, 0.5)
    _constant_(att.r_k, 1.0)


def init_projections(att: RWKV7TimeMix) -> None:
    """Init receptance, key, value, output projections orthogonally."""
    # Slight gain for stability with depth
    _orthogonal_(att.receptance.weight, gain=1.0)
    _orthogonal_(att.key.weight, gain=1.0)
    _orthogonal_(att.value.weight, gain=1.0)
    _orthogonal_(att.output.weight, gain=1.0)


def init_ln_x(att: RWKV7TimeMix) -> None:
    """Init ln_x (GroupNorm) weight=1, bias=0."""
    _constant_(att.ln_x.weight, 1.0)
    _zero_(att.ln_x.bias)


def init_time_mix(att: RWKV7TimeMix) -> None:
    """Full TimeMix initialization."""
    init_time_mixing_scalars(att)
    init_decay_lora(att)
    init_aaa_lora(att)
    init_mv_lora(att)
    init_gate_lora(att)
    init_kk_rk(att)
    init_projections(att)
    init_ln_x(att)


def init_channel_mix(ffn: RWKV7ChannelMix) -> None:
    """Init channel-mix (squared-ReLU FFN): key/value orthogonal, x_k=0.5."""
    _constant_(ffn.x_k, 0.5)
    _orthogonal_(ffn.key.weight, gain=1.0)
    _orthogonal_(ffn.value.weight, gain=1.0)


def init_block(block: RWKV7Block) -> None:
    """Init a full RWKV-7 block."""
    # LayerNorms
    if block.ln0 is not None:
        _constant_(block.ln0.weight, 1.0)
        _zero_(block.ln0.bias)
    _constant_(block.ln1.weight, 1.0)
    _zero_(block.ln1.bias)
    _constant_(block.ln2.weight, 1.0)
    _zero_(block.ln2.bias)

    init_time_mix(block.att)
    init_channel_mix(block.ffn)


def init_embedding(emb: nn.Embedding) -> None:
    """Init embedding with std = 1/sqrt(d_model)."""
    std = 1.0 / math.sqrt(emb.embedding_dim)
    _normal_(emb.weight, std=std)


def init_ln_out(ln: nn.LayerNorm) -> None:
    _constant_(ln.weight, 1.0)
    _zero_(ln.bias)


def init_model(model: RWKV7Model) -> None:
    """Full model initialization.

    This is the canonical RWKV-7 init scheme, adapted for our config.
    """
    init_embedding(model.emb)
    for block in model.blocks:
        init_block(block)
    init_ln_out(model.ln_out)
    if model.head is not None:
        _orthogonal_(model.head.weight, gain=1.0)


def check_no_nan_inf(model: nn.Module) -> tuple[int, int]:
    """Return (n_nan, n_inf) across all parameters.

    Per /ipho-physicist: initialization must produce no NaN/Inf.
    """
    n_nan = 0
    n_inf = 0
    for p in model.parameters():
        if torch.isnan(p).any():
            n_nan += int(torch.isnan(p).sum().item())
        if torch.isinf(p).any():
            n_inf += int(torch.isinf(p).sum().item())
    return n_nan, n_inf


def init_and_verify(cfg: ModelConfig) -> RWKV7Model:
    """Initialize model and assert no NaN/Inf produced."""
    from src.model.rwkv7 import build_model
    model = build_model(cfg)
    init_model(model)
    n_nan, n_inf = check_no_nan_inf(model)
    assert n_nan == 0, f"Found {n_nan} NaN values in init"
    assert n_inf == 0, f"Found {n_inf} Inf values in init"
    return model


if __name__ == "__main__":
    cfg = ModelConfig(vocab_size=128, n_layers=2, d_model=64, d_state=32)
    model = init_and_verify(cfg)
    print(f"Init OK. {model.num_parameters():,} params, no NaN/Inf.")
    # Test forward
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(idx)
    print(f"Forward OK: out shape {out.shape}")
    assert not torch.isnan(out).any(), "NaN in output!"
    assert not torch.isinf(out).any(), "Inf in output!"
    print("All assertions passed.")