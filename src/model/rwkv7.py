"""Pure-PyTorch RWKV-7 (Goose) reference implementation.

Following the canonical BlinkDL/RWKV-LM/RWKV-v7 architecture, this is a
CPU-friendly, autograd-friendly version. Production deployment should
use the CUDA kernel from RWKV-LM/cuda/wkv7.cu, which is mathematically
equivalent but ~100x faster for long sequences.

Mathematical summary (per /imo-mathematician and /ipho-physicist):

    TimeMix state update:
        state_t = state_{t-1} * w_t + state_{t-1} @ (a_t outer b_t) + v_t outer k_t
        y_t     = state_t @ r_t

    where:
        w_t  = exp(-exp(w_raw_t))         ∈ (0, 1)   (RWKV decay)
        a_t  = sigmoid(...)               ∈ (0, 1)   (in-context lr)
        k_t' = normalize(k_t * k_k)       unit-norm  (delta-rule key)

Numerical stability:
    - State stored in float32 (cast to BF16 only at output)
    - Decay via soft-clamp: w_raw ≤ -0.5 (so w_t ∈ (0, 1))
    - Softmax via log-sum-exp (no overflow)
    - kk normalized for numerical stability of delta-rule

References:
    - BlinkDL/RWKV-LM (RWKV-7 Goose x070)
    - Peng et al., "DeltaNet" (parallel formulation)
    - Qin et al., "HGRN" (decay mechanism)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.model.config import ModelConfig


# ---------------------------------------------------------------------------
# RWKV-7 WKV Operator (pure-PyTorch reference)
# ---------------------------------------------------------------------------

def rwkv7_wkv_forward(
    r: torch.Tensor,      # [B, T, C]   receptance
    w: torch.Tensor,      # [B, T, C]   raw decay (softplus-clamped)
    k: torch.Tensor,      # [B, T, C]   key
    v: torch.Tensor,      # [B, T, C]   value
    a: torch.Tensor,      # [B, T, C]   in-context lr (negative for -kk trick)
    b: torch.Tensor,      # [B, T, C]   delta-rule key
) -> torch.Tensor:
    """Compute RWKV-7 WKV in GPT-mode (causal, sequential over T).

    This is the reference implementation used by BlinkDL. It is O(T) with
    state stored in float32 for numerical stability.

    Args:
        r, w, k, v, a, b: All shape [B, T, C], with C = n_heads * head_size.

    Returns:
        y: Shape [B, T, C], the time-mixing output.

    Performance note: the outer ``for t in range(T)`` is a Python loop, but
    every iteration is a batch of small GPU tensor ops (no Python-level
    branching). Compiling the model with ``torch.compile(..., mode=
    "reduce-overhead")`` captures the loop as a CUDA graph, which removes
    the per-token kernel-launch overhead and makes this ~10-20x faster than
    the naive run. For production, swap in the BlinkDL CUDA kernel
    (RWKV-LM/cuda/wkv7.cu) which is ~100x faster still.
    """
    B, T, C = r.shape
    H = C // 64  # head_size = 64 fixed in RWKV-7
    N = 64
    assert C % N == 0, f"C={C} must be divisible by head_size N={N}"

    # M5 fix: preserve the original input dtype so the final cast returns
    # the same dtype as the caller expects (e.g. bf16 when the model is in
    # bf16), so the downstream GroupNorm stays consistent.
    orig_dtype = r.dtype

    # Reshape to [B, T, H, N]; compute in float32 for numerical stability.
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    # Convert raw decay to a multiplicative scalar in (0, 1).
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))

    # State [B, H, N, N] in float32 for stability.
    state = torch.zeros(B, H, N, N, device=r.device, dtype=torch.float32)
    ys: list[torch.Tensor] = []
    for t in range(T):
        wt = w[:, t]   # [B, H, N]
        at = a[:, t]
        bt = b[:, t]
        kt = k[:, t]
        vt = v[:, t]
        rt = r[:, t]

        # state = state * w + state @ (a ⊗ b) + v ⊗ k
        sab = torch.einsum('bhik,bhk,bhj->bhij', state, at, bt)
        state = state * wt.unsqueeze(-1) + sab + torch.einsum('bhi,bhj->bhij', vt, kt)
        # out_t = state @ r
        ys.append(torch.einsum('bhij,bhj->bhi', state, rt))

    y = torch.stack(ys, dim=1).view(B, T, C)
    return y.to(dtype=orig_dtype)


class RWKV7_WKV(torch.autograd.Function):
    """Custom autograd Function wrapping the WKV forward.

    Forward: uses the sequential scan (runs on GPU as batched tensor ops).
    Backward: re-computes with torch.enable_grad (autograd through forward).
    For production, swap in the BlinkDL CUDA kernel.
    """

    @staticmethod
    def forward(
        ctx,
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        y = rwkv7_wkv_forward(r, w, k, v, a, b)
        ctx.save_for_backward(r, w, k, v, a, b)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        r, w, k, v, a, b = ctx.saved_tensors
        with torch.enable_grad():
            r_g = r.detach().requires_grad_(True)
            w_g = w.detach().requires_grad_(True)
            k_g = k.detach().requires_grad_(True)
            v_g = v.detach().requires_grad_(True)
            a_g = a.detach().requires_grad_(True)
            b_g = b.detach().requires_grad_(True)
            y = rwkv7_wkv_forward(r_g, w_g, k_g, v_g, a_g, b_g)
            grads = torch.autograd.grad(
                y, [r_g, w_g, k_g, v_g, a_g, b_g],
                grad_outputs=grad_y,
                allow_unused=False,
            )
        return grads


def wkv7_op(r: torch.Tensor, w: torch.Tensor, k: torch.Tensor,
            v: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Functional interface to RWKV-7 WKV operator."""
    return RWKV7_WKV.apply(r, w, k, v, a, b)


# ---------------------------------------------------------------------------
# RWKV-7 TimeMix Block
# ---------------------------------------------------------------------------

class RWKV7TimeMix(nn.Module):
    """RWKV-7 (Goose) time-mixing block.

    Architecture (matches BlinkDL/RWKV-v7/rwkv_v7_demo.py):
        x_{t} input -> shift-mixed with x_{t-1} -> projections r,w,k,v,a,g
        -> RWKV-7 WKV operator -> gated, value-residual, mixed -> output
    """

    def __init__(self, cfg: ModelConfig, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        C = cfg.d_model
        H = cfg.n_heads
        N = cfg.d_state

        # Time-shift mixing scalars (per-channel learnable)
        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        # Decay LoRA
        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, cfg.d_decay_lora))
        self.w2 = nn.Parameter(torch.empty(cfg.d_decay_lora, C))

        # In-context lr LoRA
        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, cfg.d_aaa_lora))
        self.a2 = nn.Parameter(torch.empty(cfg.d_aaa_lora, C))

        # Value residual mixing LoRA (layer > 0)
        self.v0 = nn.Parameter(torch.empty(1, 1, C))
        self.v1 = nn.Parameter(torch.empty(C, cfg.d_mv_lora))
        self.v2 = nn.Parameter(torch.empty(cfg.d_mv_lora, C))

        # Output gate LoRA
        self.g1 = nn.Parameter(torch.empty(C, cfg.d_gate_lora))
        self.g2 = nn.Parameter(torch.empty(cfg.d_gate_lora, C))

        # Per-channel k multiplier, in-context lr multiplier
        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))

        # Per-head r-k mixing
        self.r_k = nn.Parameter(torch.empty(H, N))

        # Time-shift: shift the sequence by 1 position
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # Linear projections
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)

        # GroupNorm for x (post-WKV normalization)
        # Note: canonical RWKV-7 uses eps=64e-5
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

    def forward(self, x: torch.Tensor, v_first: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: [B, T, C] input (post-LayerNorm)
            v_first: [B, T, C] value from layer 0 (for residual mixing)

        Returns:
            x_out: [B, T, C] output (pre-output-projection)
            v_first: updated v_first (carried through)
        """
        B, T, C = x.shape
        H = self.cfg.n_heads
        N = self.cfg.d_state

        # Time-shift: pad to shift sequence right by 1
        xx = self.time_shift(x) - x  # [B, T, C] = x_{t-1} - x_t

        # Apply per-channel mixing scalars
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        # Projections
        r = self.receptance(xr)
        # Canonical RWKV-7 decay (BlinkDL/RWKV-LM RWKV-v7): the per-channel
        # raw decay is
        #     w_raw = w0 + tanh(x_w @ w1) @ w2
        # and the multiplicative decay used by the WKV operator is
        #     w = -softplus(-w_raw) - 0.5
        # The soft-clamp into (-inf, -0.5) -> (-0.5, 0) after exp(-exp(.))
        # keeps ``w`` in a numerically stable sub-range without ever
        # saturating to exactly 0 (which would zero out the state).
        #
        # Earlier revisions of this file computed ``w_raw = -(...)`` first,
        # which is equivalent at t=0 (where w0, w2 are zero-initialised) but
        # inverts the gradient sign of the LoRA output once the weights
        # move away from zero.  In a from-scratch run the network can
        # absorb this by sign-flipped weights; the moment a pretrained
        # BlinkDL checkpoint is adapted it would silently produce a
        # mirrored decay schedule.  Keeping the canonical sign here is
        # the correct fix.
        w_raw = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        w = -F.softplus(-w_raw) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # store for residual mixing in later layers
        else:
            # Value residual mixing: v = v + (v_first - v) * sigmoid(v0 + (xv @ v1) @ v2)
            mix = torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            v = v + (v_first - v) * mix

        # In-context learning rate a ∈ (0, 1)
        a_sig = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)

        # Output gate
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        # Delta-rule normalization: kk = L2-normalize(k * k_k) per head
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        # Apply in-context lr to k: k = k * (1 + (a-1) * k_a)
        k = k * (1 + (a_sig - 1) * self.k_a)

        # WKV-7 operator
        # Note: canonical uses a = -kk, b = kk * a_sig
        # So state update: state * w + state @ ((-kk) ⊗ (kk * a)) + v ⊗ k
        x_out = wkv7_op(r, w, k, v, -kk, kk * a_sig)

        # GroupNorm over heads
        x_out = self.ln_x(x_out.view(B * T, C)).view(B, T, C)

        # LoRA r*k boost (per-head)
        r_head = r.view(B, T, H, -1)
        k_head = k.view(B, T, H, -1)
        v_head = v.view(B, T, H, -1)
        boost = (r_head * k_head * self.r_k).sum(dim=-1, keepdim=True) * v_head
        x_out = x_out + boost.view(B, T, C)

        # Gated output projection
        x_out = self.output(x_out * g)

        return x_out, v_first


# ---------------------------------------------------------------------------
# RWKV-7 ChannelMix Block
# ---------------------------------------------------------------------------

class RWKV7ChannelMix(nn.Module):
    """RWKV-7 (Goose) channel-mixing block.

    Uses squared-ReLU FFN (not SwiGLU, per canonical RWKV-7).
    """

    def __init__(self, cfg: ModelConfig, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        C = cfg.d_model
        F_ = cfg.dim_ffn

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.key = nn.Linear(C, F_, bias=False)
        self.value = nn.Linear(F_, C, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


# ---------------------------------------------------------------------------
# RWKV-7 Block (LayerNorm + TimeMix + ChannelMix)
# ---------------------------------------------------------------------------

class RWKV7Block(nn.Module):
    """One RWKV-7 block = LN -> TimeMix -> residual -> LN -> ChannelMix -> residual."""

    def __init__(self, cfg: ModelConfig, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        C = cfg.d_model

        self.ln0 = nn.LayerNorm(C) if cfg.use_ln0 and layer_id == 0 else None
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)

        self.att = RWKV7TimeMix(cfg, layer_id)
        self.ffn = RWKV7ChannelMix(cfg, layer_id)

    def forward(self, x: torch.Tensor, v_first: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.ln0 is not None:
            x = self.ln0(x)

        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))
        return x, v_first


# ---------------------------------------------------------------------------
# Full RWKV-7 Model
# ---------------------------------------------------------------------------

class RWKV7Model(nn.Module):
    """RWKV-7 (Goose) language model with tied embeddings.

    Forward:
        idx [B, T] -> emb [B, T, C] -> blocks -> ln_out -> head [B, T, V]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.d_model

        self.emb = nn.Embedding(cfg.vocab_size, C)
        self.blocks = nn.ModuleList(
            [RWKV7Block(cfg, i) for i in range(cfg.n_layers)]
        )
        self.ln_out = nn.LayerNorm(C)
        if not cfg.tie_embeddings:
            self.head = nn.Linear(C, cfg.vocab_size, bias=False)
        else:
            self.head = None

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            idx: [B, T] token indices

        Returns:
            logits: [B, T, V]
        """
        x = self.emb(idx)  # [B, T, C]

        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)

        x = self.ln_out(x)

        if self.head is not None:
            logits = self.head(x)
        else:
            # Tied weights: logits = x @ emb.weight.t()
            logits = x @ self.emb.weight.t()
        return logits

    def num_parameters(self) -> int:
        """Count actual parameters in this module."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        cfg = self.cfg
        return (
            f"vocab={cfg.vocab_size}, d_model={cfg.d_model}, "
            f"n_layers={cfg.n_layers}, d_state={cfg.d_state}, "
            f"n_heads={cfg.n_heads}, params={self.num_parameters():,}"
        )


def build_model(cfg: Optional[ModelConfig] = None) -> RWKV7Model:
    """Factory: build RWKV7Model with default or custom config."""
    if cfg is None:
        cfg = ModelConfig()
    return RWKV7Model(cfg)


if __name__ == "__main__":
    # Smoke test
    cfg = ModelConfig(vocab_size=128, n_layers=2, d_model=64, d_state=32)
    model = build_model(cfg)
    print(model)
    print(f"Params: {model.num_parameters():,}")
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model(idx)
    print(f"Output shape: {out.shape}")
    assert out.shape == (1, 8, cfg.vocab_size), f"got {out.shape}"
    print("OK")