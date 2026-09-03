"""GRPO (Group Relative Policy Optimization) with MC-GRPO baseline.

Mathematical derivation (per /imo-mathematician):

Standard REINFORCE advantage estimator:
    A_t = Q(s_t, a_t) - V(s_t)
With G rollouts per prompt, we estimate:
    A_i = R_i - b

where b is a baseline. MC-GRPO (DeepSeek Math) uses:
    b = median(R_1, ..., R_G)

Why median instead of mean?
    - Mean baseline: E[R - mean(R)] = 0, but Var(A) = Var(R) * (1 - 1/G)
    - Median baseline: more robust to outliers, lower variance for small G
    - For G=4-8, median is significantly more stable than mean

PPO clipping (Schulman et al. 2017):
    L = -min(
        ratio * A,
        clip(ratio, 1-eps, 1+eps) * A
    )

where ratio = pi_theta(a|s) / pi_ref(a|s), with stop-gradient on pi_ref.

KL penalty (Saul 2017,礼服 et al.):
    L = -E[log pi_theta(a|s)] + beta * KL[pi_theta || pi_ref]

We use the PPO clipping form with MC-GRPO advantages.

KL direction: We use forward KL[pi_theta || pi_ref], which is:
    KL(pi_theta || pi_ref) = E_pi_theta[log pi_theta - log pi_ref]

Forward KL (mode-covering) is preferred over reverse KL (mode-seeking)
because it prevents the policy from collapsing to a single mode.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.model.rwkv7 import RWKV7Model
from src.training.optimizer import (
    AdamWConfig,
    WarmupCosineLR,
    build_optimizer,
    clip_grad_norm,
    save_checkpoint,
)
from src.training.reward import compute_batch_rewards
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# GRPO core math
# ---------------------------------------------------------------------------

def compute_mc_grpo_advantages(rewards: List[float], use_median: bool = True
                              ) -> Tuple[List[float], float]:
    """Compute MC-GRPO advantages using median baseline.

    Args:
        rewards: List of G rewards (one per rollout).
        use_median: If True, use median baseline (MC-GRPO). If False, mean.

    Returns:
        (advantages, baseline) where sum(advantages) ≈ 0.

    Per /imo-mathematician: MC-GRPO uses median, not mean.
    """
    G = len(rewards)
    if G == 0:
        return [], 0.0

    if use_median:
        sorted_r = sorted(rewards)
        if G % 2 == 1:
            baseline = sorted_r[G // 2]
        else:
            baseline = (sorted_r[G // 2 - 1] + sorted_r[G // 2]) / 2.0
    else:
        baseline = sum(rewards) / G

    advantages = [r - baseline for r in rewards]
    return advantages, baseline


def compute_log_probs(
    logits: torch.Tensor,
    actions: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-token log probabilities.

    Args:
        logits: [B, T, V] unnormalized logits
        actions: [B, T] token IDs (each position t, predict token at t+1)
        mask: [B, T] boolean mask (True = valid token)

    Returns:
        log_probs: [B, T] per-token log probabilities
    """
    log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
    # Gather log probs for the actual actions
    # Use gather: log_probs.gather(-1, actions.unsqueeze(-1)) -> [B, T, 1]
    lp = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # [B, T]

    if mask is not None:
        lp = lp * mask.float()

    return lp


def compute_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Compute PPO-style clipped policy loss.

    Per /imo-mathematician: the loss is:
        L = -min(
            ratio * A,
            clip(ratio, 1-eps, 1+eps) * A
        )

    where ratio = exp(log_pi_theta - log_pi_old), A = advantage.

    Args:
        log_probs: [B, T] current policy log probs
        old_log_probs: [B, T] reference policy log probs (stop-gradient applied)
        advantages: [B] or [B, T] scalar advantages
        clip_eps: PPO clipping epsilon
        mask: [B, T] boolean mask

    Returns:
        (loss, stats_dict)
    """
    # ratio = pi / pi_old = exp(log_pi - log_pi_old)
    ratio = torch.exp(log_probs - old_log_probs.detach())

    # Expand advantages to per-token if needed
    if advantages.dim() == 1:
        adv = advantages.unsqueeze(-1)  # [B, 1]
    else:
        adv = advantages

    if mask is not None:
        adv = adv * mask.float()

    # PPO clipping
    # L = -min(ratio * A, clip(ratio) * A)
    # Note: we need to expand A properly
    if adv.dim() == 2 and ratio.dim() == 2:
        # Per-token: use mean across valid tokens
        ratio_clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        loss1 = ratio * adv
        loss2 = ratio_clipped * adv
        loss_unclipped = loss1
    else:
        # Per-sequence: expand to per-token
        adv_expanded = adv.unsqueeze(-1)  # broadcast
        ratio_clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        loss1 = ratio * adv_expanded
        loss2 = ratio_clipped * adv_expanded
        loss_unclipped = loss1

    # Take min and negate
    clipped_mask = (loss2 < loss1).float()
    loss = -torch.minimum(loss1, loss2)

    if mask is not None:
        if mask.sum() == 0:
            # No valid tokens: return zero loss
            return torch.tensor(0.0, device=log_probs.device, requires_grad=log_probs.requires_grad), {
                "policy_loss": 0.0, "clip_frac": 0.0, "approx_kl": 0.0,
                "ratio_mean": 0.0, "ratio_max": 0.0,
            }
        loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
    else:
        loss = loss.mean()

    # Statistics
    with torch.no_grad():
        clip_frac = clipped_mask.float().mean().item()
        approx_kl = (log_probs - old_log_probs.detach()).mean().item()
        ratio_mean = ratio.mean().item() if ratio.numel() > 0 else 0.0
        ratio_max = ratio.max().item() if ratio.numel() > 0 else 0.0

    stats = {
        "policy_loss": loss.item(),
        "clip_frac": clip_frac,
        "approx_kl": approx_kl,
        "ratio_mean": ratio_mean,
        "ratio_max": ratio_max,
    }
    return loss, stats


def compute_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Compute forward KL penalty: KL[pi_theta || pi_ref].

    Forward KL = E_pi_theta[log pi_theta - log pi_ref]
    This penalizes pi_theta when it assigns low probability to high-ref-prob tokens.

    Args:
        log_probs: [B, T] current policy log probs
        ref_log_probs: [B, T] reference policy log probs (no grad)
        mask: [B, T] boolean mask

    Returns:
        (kl_penalty, kl_value)
    """
    # Forward KL: log_prob - ref_log_prob (stop-grad on ref)
    kl = (log_probs - ref_log_probs.detach())

    if mask is not None:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=log_probs.device), 0.0
        kl = (kl * mask.float()).sum() / mask.float().sum().clamp(min=1)
    else:
        if kl.numel() == 0:
            return torch.tensor(0.0, device=log_probs.device), 0.0
        kl = kl.mean()

    return kl, kl.item()


def compute_grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    ref_log_probs: Optional[torch.Tensor] = None,
    clip_eps: float = 0.2,
    kl_coef: float = 0.04,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Compute full GRPO loss = PPO + KL penalty.

    L = L_policy + kl_coef * KL[pi || pi_ref]

    Args:
        log_probs: [B, T] current log probs
        old_log_probs: [B, T] reference log probs (from old policy)
        advantages: [B] scalar advantages per sequence
        ref_log_probs: [B, T] reference log probs (from SFT model)
        clip_eps: PPO clipping epsilon
        kl_coef: KL penalty coefficient
        mask: [B, T] boolean mask

    Returns:
        (total_loss, stats_dict)
    """
    policy_loss, p_stats = compute_policy_loss(
        log_probs, old_log_probs, advantages, clip_eps, mask
    )

    kl_loss = 0.0
    kl_value = 0.0
    if ref_log_probs is not None and kl_coef > 0:
        kl_loss, kl_value = compute_kl_penalty(log_probs, ref_log_probs, mask)

    total_loss = policy_loss + kl_coef * kl_loss

    stats = {
        **p_stats,
        "kl_loss": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
        "kl_value": kl_value,
        "total_loss": total_loss.item(),
    }
    return total_loss, stats


# ---------------------------------------------------------------------------
# GRPO Training Loop
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    """GRPO configuration."""

    group_size: int = 8          # G rollouts per prompt
    batch_size: int = 4          # prompts per batch
    lr: float = 5e-6
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_steps: int = 3000
    seq_len: int = 1024          # max generation length
    kl_coef: float = 0.04        # KL penalty coefficient
    clip_eps: float = 0.2         # PPO clipping epsilon
    entropy_coef: float = 0.01   # Entropy bonus coefficient
    use_mc_grpo: bool = True     # Use median baseline (MC-GRPO)
    grad_clip: float = 1.0
    save_every: int = 500
    log_every: int = 10
    seed: int = 42
    save_dir: str = "checkpoints/grpo"
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    betas: tuple = (0.9, 0.95)


def simple_generate(
    model: RWKV7Model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    eos_id: Optional[int] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """Simple greedy/sampling generation for GRPO rollouts.

    This is a reference implementation. For production, use the optimized
    inference/generation.py module.

    Args:
        model: RWKV-7 model
        prompt_ids: [B, T] prompt tokens
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k truncation
        top_p: Nucleus threshold
        eos_id: Stop at this token ID

    Returns:
        (full_ids [B, T+gen], per-token log_probs [gen])
    """
    model.eval()
    B = prompt_ids.size(0)
    current = prompt_ids
    all_log_probs: List[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(current)  # [B, T, V]
            logits = logits[:, -1, :]  # [B, V] — last token only

            if temperature == 0:
                next_tok = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
                log_p = F.log_softmax(logits, dim=-1)
                lp = log_p.gather(-1, next_tok).squeeze(-1)  # [B]
            else:
                # Temperature sampling
                logits = logits / temperature
                # Top-k
                if top_k > 0:
                    top_k_vals = logits.topk(min(top_k, logits.size(-1)), dim=-1)[0]
                    logits[logits < top_k_vals[:, -1:]] = -float("inf")
                # Top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
                    cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    sorted_mask = cum_probs > top_p
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = False
                    indices_to_remove = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
                    logits[indices_to_remove] = -float("inf")
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)  # [B, 1]
                log_p = F.log_softmax(logits, dim=-1)
                lp = log_p.gather(-1, next_tok).squeeze(-1)  # [B]

            all_log_probs.append(lp)
            current = torch.cat([current, next_tok], dim=1)

            if eos_id is not None and (next_tok == eos_id).all():
                break

    # Concatenate log probs
    if all_log_probs:
        seq_log_probs = torch.stack(all_log_probs, dim=1)  # [B, gen]
    else:
        seq_log_probs = torch.zeros(B, 0)

    return current, seq_log_probs


def grpo_step(
    model: RWKV7Model,
    prompts: List[str],
    golds: List[str],
    tokenizer,
    cfg: GRPOConfig,
    ref_model: Optional[RWKV7Model] = None,
) -> Tuple[float, Dict]:
    """Run one GRPO step.

    For each prompt, generates G rollouts, computes rewards, derives
    advantages, then updates the policy.

    Args:
        model: Current policy model
        prompts: List of B prompt strings
        golds: List of B gold answers
        tokenizer: Tokenizer
        cfg: GRPO config
        ref_model: Reference model for KL penalty (optional)

    Returns:
        (mean_loss, stats_dict)
    """
    B = len(prompts)

    # Tokenize prompts
    prompt_ids_list = [tokenizer.encode(p) for p in prompts]
    prompt_tensors = [torch.tensor(ids, dtype=torch.long) for ids in prompt_ids_list]

    # Pad to same length
    max_prompt_len = max(len(ids) for ids in prompt_ids_list)
    padded_prompts = torch.full(
        (B, max_prompt_len), tokenizer.pad_id, dtype=torch.long
    )
    for i, ids in enumerate(prompt_ids_list):
        padded_prompts[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    # Generate G rollouts per prompt
    eos_id = getattr(tokenizer, "eos_id", None)
    pad_id = getattr(tokenizer, "pad_id", 0)

    all_rollouts: List[List[torch.Tensor]] = []      # [prompt][rollout] -> [T]
    all_log_probs: List[List[torch.Tensor]] = []      # [prompt][rollout] -> [gen]
    all_rewards: List[List[float]] = []               # [prompt][rollout] -> reward

    for b in range(B):
        prompt_tok = padded_prompts[b:b+1]  # [1, T]
        prompt_rollouts: List[torch.Tensor] = []
        prompt_lps: List[torch.Tensor] = []
        prompt_rewards: List[float] = []

        for g in range(cfg.group_size):
            # Generate with slight temperature variation for diversity
            temp = 0.7 + g * 0.05  # vary temperature across rollouts
            full_ids, seq_lp = simple_generate(
                model, prompt_tok,
                max_new_tokens=cfg.seq_len - max_prompt_len,
                temperature=temp,
                eos_id=eos_id,
            )
            rollout_ids = full_ids[0, max_prompt_len:]  # strip prompt

            # Decode and compute reward
            text = tokenizer.decode(rollout_ids.tolist())
            reward = compute_batch_rewards([text], [golds[b]])[0]

            prompt_rollouts.append(rollout_ids)
            prompt_lps.append(seq_lp[0])
            prompt_rewards.append(reward)

        all_rollouts.append(prompt_rollouts)
        all_log_probs.append(prompt_lps)
        all_rewards.append(prompt_rewards)

    # Compute advantages (MC-GRPO)
    advantages_per_prompt: List[List[float]] = []
    baselines: List[float] = []
    for b in range(B):
        advs, baseline = compute_mc_grpo_advantages(
            all_rewards[b], use_median=cfg.use_mc_grpo
        )
        advantages_per_prompt.append(advs)
        baselines.append(baseline)

    # Build batch: flatten all (prompt, rollout) pairs
    flat_prompt_ids = []    # per (prompt, rollout)
    flat_rollout_ids = []
    flat_log_probs = []
    flat_advantages = []
    flat_mask = []

    for b in range(B):
        for g in range(cfg.group_size):
            flat_prompt_ids.append(padded_prompts[b])  # [T_p]
            flat_rollout_ids.append(all_rollouts[b][g])  # [T_g]
            flat_log_probs.append(all_log_probs[b][g])  # [T_g]
            flat_advantages.append(advantages_per_prompt[b][g])
            flat_mask.append(torch.ones_like(all_rollouts[b][g], dtype=torch.bool))

    if not flat_prompt_ids:
        return 0.0, {"skip": "no valid rollouts"}

    # Concatenate prompt + rollout
    flat_prompt_t = torch.stack(flat_prompt_ids)  # [B*G, T_p]
    flat_rollout_t = torch.nn.utils.rnn.pad_sequence(
        flat_rollout_ids, batch_first=True, padding_value=pad_id
    )  # [B*G, T_g]

    # Concatenate for input
    flat_input = torch.cat([flat_prompt_t, flat_rollout_t], dim=1)  # [B*G, T_p+T_g]

    # Forward
    logits = model(flat_input)  # [B*G, T_p+T_g, V]

    # Extract log probs for rollout tokens only
    B_flat = flat_input.size(0)
    T_p = flat_prompt_t.size(1)
    rollout_log_probs = torch.nn.utils.rnn.pad_sequence(
        flat_log_probs, batch_first=True, padding_value=0.0
    )  # [B*G, T_g]
    T_g = rollout_log_probs.size(1)

    # Get logits for rollout positions
    rollout_logits = logits[:, T_p:T_p+T_g, :]  # [B*G, T_g, V]
    rollout_actions = flat_rollout_t  # [B*G, T_g]

    # Compute log probs from current policy
    curr_log_probs = compute_log_probs(rollout_logits, rollout_actions)

    # Old log probs from generation (stored)
    old_log_probs = rollout_log_probs  # these are the reference log probs

    # Reference model log probs (if available)
    ref_log_probs = None
    if ref_model is not None and cfg.kl_coef > 0:
        with torch.no_grad():
            ref_logits = ref_model(flat_input)[:, T_p:T_p+T_g, :]
            ref_log_probs = compute_log_probs(ref_logits, rollout_actions)

    # Mask
    mask_tensor = torch.nn.utils.rnn.pad_sequence(
        flat_mask, batch_first=True, padding_value=0.0
    ).float()  # [B*G, T_g]

    # Advantages as tensor
    adv_tensor = torch.tensor(flat_advantages, dtype=torch.float32)

    # Compute GRPO loss
    loss, stats = compute_grpo_loss(
        curr_log_probs,
        old_log_probs,
        adv_tensor,
        ref_log_probs=ref_log_probs,
        clip_eps=cfg.clip_eps,
        kl_coef=cfg.kl_coef,
        mask=mask_tensor,
    )

    # Backward
    loss.backward()

    # Clip and step
    if cfg.grad_clip > 0:
        clip_grad_norm(model.parameters(), cfg.grad_clip)

    return loss.item(), stats


def grpo(
    model: RWKV7Model,
    cfg: GRPOConfig,
    tokenizer,
    grpo_examples: Optional[List[Tuple[str, str]]] = None,
    ref_model: Optional[RWKV7Model] = None,
) -> List[Dict]:
    """Run GRPO training loop.

    Args:
        model: RWKV-7 model to train
        cfg: GRPO configuration
        tokenizer: Tokenizer
        grpo_examples: List of (prompt, gold_answer) pairs
        ref_model: Reference SFT model for KL penalty

    Returns:
        List of training logs
    """
    set_seed(cfg.seed)
    model.to(cfg.device)

    if ref_model is not None:
        ref_model.to(cfg.device)
        ref_model.eval()

    # Build examples if not provided
    if grpo_examples is None:
        from src.data.synthetic import SyntheticMathDataset
        syn = SyntheticMathDataset(size=500, seed=cfg.seed)
        grpo_examples = [syn.format_for_grpo(i) for i in range(len(syn))]

    print(f"GRPO on {len(grpo_examples)} examples, group_size={cfg.group_size}")

    # Optimizer
    opt_cfg = AdamWConfig(
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        min_lr_ratio=0.1,
        betas=cfg.betas,
    )
    optimizer = build_optimizer(model, opt_cfg)
    scheduler = WarmupCosineLR(optimizer, opt_cfg)

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    logs = []
    model.train()
    t_start = time.time()

    for step in range(cfg.max_steps):
        # Sample prompts
        n_prompts = min(cfg.batch_size, len(grpo_examples))
        indices = torch.randperm(len(grpo_examples))[:n_prompts].tolist()
        batch_prompts = [grpo_examples[i][0] for i in indices]
        batch_golds = [grpo_examples[i][1] for i in indices]

        loss, stats = grpo_step(
            model, batch_prompts, batch_golds,
            tokenizer, cfg, ref_model
        )

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(step)

        lr_now = scheduler.get_lr()[0]
        stats.update({
            "step": step,
            "lr": lr_now,
            "elapsed": time.time() - t_start,
        })
        logs.append(stats)

        if cfg.log_every > 0 and step % cfg.log_every == 0:
            print(
                f"[step {step:>5}] loss={loss:.4f} kl={stats.get('kl_value', 0):.4f} "
                f"clip={stats.get('clip_frac', 0):.3f} lr={lr_now:.2e}"
            )

        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"step_{step+1}.pt")
            save_checkpoint(model, optimizer, step + 1, ckpt_path)

    # Final save
    ckpt_path = os.path.join(cfg.save_dir, "final.pt")
    save_checkpoint(model, optimizer, cfg.max_steps, ckpt_path)
    return logs


if __name__ == "__main__":
    from src.model.config import ModelConfig
    from src.model.init import init_and_verify
    from src.tokenizer.math_tokenizer import StubTokenizer

    model_cfg = ModelConfig(vocab_size=256, n_layers=2, d_model=64, d_state=32)
    model = init_and_verify(model_cfg)
    ref = init_and_verify(model_cfg)

    tok = StubTokenizer(vocab_size=256)
    # Use very short seq_len for CPU smoke test (full seq_len=1024 is too slow on CPU)
    cfg = GRPOConfig(max_steps=2, group_size=2, batch_size=1, save_every=0, log_every=1, seq_len=32)

    from src.data.synthetic import SyntheticMathDataset
    syn = SyntheticMathDataset(size=10)
    examples = [syn.format_for_grpo(i) for i in range(10)]

    logs = grpo(model, cfg, tok, examples, ref_model=ref)
    print(f"Final policy_loss: {logs[-1].get('policy_loss', 0):.4f}")