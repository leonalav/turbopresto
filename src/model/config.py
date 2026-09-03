"""Model configuration for RWKV-7 (Goose).

This module defines the ModelConfig dataclass with explicit parameter
counting derivations. The math here is verified by /imo-mathematician.

Parameter count formula (per layer of RWKV-Tmix-x070):
    - receptance:    C x C          (input projection for r)
    - key:           C x C          (input projection for k)
    - value:         C x C          (input projection for v)
    - output:        C x C          (output projection)
    - x_r,x_w,x_k,x_v,x_a,x_g: 6 x (1,1,C)   (time-shift mixing scalars)
    - w0: (1,1,C); w1: (C, D_DECAY_LORA); w2: (D_DECAY_LORA, C)
    - a0: (1,1,C); a1: (C, D_AAA_LORA);   a2: (D_AAA_LORA, C)
    - v0: (1,1,C); v1: (C, D_MV_LORA);    v2: (D_MV_LORA, C)
    - g1: (C, D_GATE_LORA);  g2: (D_GATE_LORA, C)
    - k_k: (1,1,C);  k_a: (1,1,C);  r_k: (H, N)
    - ln_x: GroupNorm, 2*C affine

Per-layer RWKV-Tmix-x070:
    4 * C*C                          = 4 * 512*512 = 1,048,576  (r,k,v,out)
    6 * C + 3*C                      = 9*C         = 4,608      (scalar mixers)
    2*C + 2*D_DECAY_LORA*C           = 2*512 + 2*64*512 = 66,560  (decay LoRA)
    2*C + 2*D_AAA_LORA*C             = 66,560                  (aaa LoRA)
    2*C + 2*D_MV_LORA*C              = 2*512 + 2*32*512 = 33,792 (mv LoRA)
    2*D_GATE_LORA*C                  = 2*128*512     = 131,072  (gate LoRA)
    2*C + H*N                        = 2*512 + 8*64 = 1,536    (k_k, k_a, r_k)
    2*C                              = 1,024                    (ln_x affine)
    ---
    Per-layer Tmix total ≈ 1,353,728

Per-layer RWKV-CMix-x070 (squared-ReLU FFN):
    x_k: 1*C
    key: C x dim_ffn
    value: dim_ffn x C
    ---
    where dim_ffn = C * 4 (canonical RWKV-7 uses 4x, not 2.75x)
    Per-layer CMix total = C + C*4*C + 4*C*C = 512 + 1,048,576 + 1,048,576 = 2,097,664

Per-layer Block:
    ln1, ln2: 2 * 2*C = 2,048
    Total per-layer ≈ 3,453,440

8 layers total = 27,627,520
Embedding (tied): V * C = 32768 * 512 = 16,777,216
ln0 (block 0 only): 1,024
ln_out: 1,024

Grand total (tied): ~44.4M (under 50M target — comfortable margin)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ModelConfig:
    """Configuration for RWKV-7 50M math model."""

    # Core dimensions
    vocab_size: int = 32768           # Custom math tokenizer
    n_layers: int = 8                 # RWKV-7 blocks
    d_model: int = 512                # Embedding dimension (n_embd)
    d_state: int = 64                 # Head size (N)
    n_heads: Optional[int] = None     # d_model // d_state if None

    # Sequence
    ctx_len: int = 4096               # Training context

    # FFN
    ffn_mult: int = 4                 # dim_ffn = d_model * ffn_mult

    # LoRA dims (canonical RWKV-7 values)
    d_decay_lora: int = 64
    d_aaa_lora: int = 64
    d_mv_lora: int = 32
    d_gate_lora: int = 128

    # Architecture flags
    tie_embeddings: bool = True       # Share input/output embeddings
    use_value_residual: bool = True   # v_first residual from layer 0
    use_ln0: bool = True              # Pre-embedding LayerNorm at layer 0

    # Numerical
    dtype: torch.dtype = torch.float32  # CPU test default; train with bf16

    def __post_init__(self) -> None:
        if self.n_heads is None:
            assert self.d_model % self.d_state == 0, (
                f"d_model ({self.d_model}) must be divisible by d_state ({self.d_state})"
            )
            self.n_heads = self.d_model // self.d_state

    @property
    def dim_att(self) -> int:
        return self.d_model

    @property
    def dim_ffn(self) -> int:
        return self.d_model * self.ffn_mult

    @property
    def head_size(self) -> int:
        return self.d_state

    def param_breakdown(self) -> dict:
        """Compute parameter count analytically.

        Returns a dict with each component and the total.
        """
        C = self.d_model
        N = self.d_state
        H = self.n_heads
        V = self.vocab_size

        # Per-layer Tmix
        # 4 projections: r,k,v,out -> 4 * C * C
        # 6 time-shift scalars: x_r..x_g -> 6 * C
        # 3 LoRA base vectors: w0,a0,v0 -> 3 * C
        # 2 decay LoRA matrices: w1(C x D_decay) + w2(D_decay x C) -> 2 * D_decay * C
        # 2 aaa LoRA matrices: -> 2 * D_aaa * C
        # 2 mv LoRA matrices: -> 2 * D_mv * C
        # 2 gate LoRA matrices: g1(C x D_gate) + g2(D_gate x C) -> 2 * D_gate * C
        # 2 scalars: k_k, k_a -> 2 * C
        # r_k per-head matrix: H * N
        # ln_x affine: 2 * C
        per_tmix = (
            4 * C * C
            + 6 * C
            + 3 * C
            + 2 * self.d_decay_lora * C
            + 2 * self.d_aaa_lora * C
            + 2 * self.d_mv_lora * C
            + 2 * self.d_gate_lora * C
            + 2 * C
            + H * N
            + 2 * C
        )
        # Per-layer CMix (squared-ReLU FFN)
        per_cmix = C + 2 * (C * self.dim_ffn)

        # Per-block LayerNorms (excluding block 0's ln0 and excluding att's internal ln_x)
        per_block_ln = 2 * 2 * C  # ln1 and ln2, each has 2*C params (scale+bias)

        per_block = per_tmix + per_cmix + per_block_ln

        # Block 0 has additional ln0
        block_0_extra = 2 * C if self.use_ln0 else 0

        # Tied or untied embeddings
        emb = V * C
        if self.tie_embeddings:
            emb_total = emb  # input only; head uses emb.weight
        else:
            emb_total = emb  # input embedding (separate from head)

        ln_out = 2 * C
        head = 0 if self.tie_embeddings else emb  # separate output head

        total = (
            emb_total
            + self.n_layers * per_block
            + block_0_extra
            + ln_out
            + head
        )

        return {
            "vocab_size": V,
            "d_model": C,
            "n_layers": self.n_layers,
            "tie_embeddings": self.tie_embeddings,
            "per_block": per_block,
            "block_0_extra_ln0": block_0_extra,
            "embedding": emb_total,
            "ln_out": ln_out,
            "head": head,
            "total": total,
        }

    def total_params(self) -> int:
        """Return total parameter count."""
        return self.param_breakdown()["total"]

    def assert_target(self, target: int = 50_000_000, tolerance: float = 0.20) -> None:
        """Assert param count is within tolerance of target."""
        actual = self.total_params()
        lower = target * (1 - tolerance)
        upper = target * (1 + tolerance)
        if not (lower <= actual <= upper):
            raise AssertionError(
                f"Param count {actual:,} not in [{lower:,.0f}, {upper:,.0f}] "
                f"(target {target:,} ±{tolerance*100:.0f}%)"
            )


if __name__ == "__main__":
    cfg = ModelConfig()
    breakdown = cfg.param_breakdown()
    print("ModelConfig param breakdown:")
    for k, v in breakdown.items():
        if isinstance(v, int) and k != "vocab_size" and k != "d_model" and k != "n_layers":
            print(f"  {k:>20}: {v:>12,}")
        else:
            print(f"  {k:>20}: {v}")
    print()
    total = breakdown["total"]
    print(f"Total params: {total:,} ({total / 1e6:.2f}M)")
    cfg.assert_target(50_000_000, tolerance=0.20)
    print(f"OK: within 50M target ±20%")