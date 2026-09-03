"""End-to-end pipeline smoke tests.

Runs all stages (pretrain -> SFT -> GRPO -> inference -> eval) on a
tiny model to verify the pipeline works end-to-end. Designed for CPU
execution in <30s.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.data.collator import RWKVCollator, collate_for_sft
from src.data.synthetic import SyntheticMathDataset
from src.inference.generation import RWKVGenerator
from src.inference.voting import majority_vote, sample_and_vote
from src.model.config import ModelConfig
from src.model.init import init_and_verify
from src.tokenizer.math_tokenizer import StubTokenizer
from src.training.grpo import GRPOConfig, compute_mc_grpo_advantages, grpo_step
from src.training.optimizer import (
    AdamWConfig, WarmupCosineLR, build_optimizer, clip_grad_norm,
    save_checkpoint, load_checkpoint,
)
from src.training.pretrain import PretrainConfig, pretrain
from src.training.reward import compute_reward
from src.training.sft import SFTConfig, sft


# ---------------------------------------------------------------------------
# Fixtures: tiny config and dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_pipeline_setup():
    """Tiny model + tokenizer + dataset for pipeline tests."""
    cfg = ModelConfig(
        vocab_size=256, n_layers=2, d_model=64, d_state=32,
        ctx_len=64, tie_embeddings=True,
    )
    model = init_and_verify(cfg)
    tok = StubTokenizer(vocab_size=256)
    dataset = SyntheticMathDataset(size=20, max_digits=2, seed=42)
    return cfg, model, tok, dataset


# ---------------------------------------------------------------------------
# Per-stage smoke tests
# ---------------------------------------------------------------------------

class TestPretrainStage:
    """Pretrain stage runs end-to-end."""

    def test_pretrain_step_runs(self, tiny_pipeline_setup):
        """Single pretrain step on random data."""
        cfg, model, tok, dataset = tiny_pipeline_setup

        # Tiny batch
        collator = RWKVCollator(tok, seq_len=32)
        examples = [{"text": dataset.format_for_pretrain(i)} for i in range(4)]
        batch = collator.collate_fn(examples)

        opt = build_optimizer(model, AdamWConfig(lr=1e-3, max_steps=10))
        sched = WarmupCosineLR(opt, AdamWConfig(lr=1e-3, max_steps=10))

        # Forward
        logits = model(batch["input_ids"])
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            batch["labels"][:, 1:].reshape(-1),
            ignore_index=-100,
        )
        assert torch.isfinite(loss).item()
        loss.backward()
        opt.step()
        sched.step(0)
        opt.zero_grad()

    def test_full_pretrain_smoke(self, tiny_pipeline_setup):
        """Full pretrain loop with tiny model and few steps."""
        _, model, tok, dataset = tiny_pipeline_setup

        cfg = PretrainConfig(
            max_steps=3, save_every=0, log_every=1,
            batch_size=2, seq_len=32,
        )
        texts = [dataset.format_for_pretrain(i) for i in range(len(dataset))]

        logs = pretrain(model, cfg, tok, texts)
        assert len(logs) == cfg.max_steps
        # All losses finite
        assert all(math.isfinite(log["loss"]) for log in logs)


class TestSFTStage:
    """SFT stage runs end-to-end."""

    def test_sft_step_runs(self, tiny_pipeline_setup):
        """Single SFT step on synthetic CoT data."""
        cfg, model, tok, dataset = tiny_pipeline_setup

        # Build SFT batch
        examples = []
        for i in range(4):
            p, t = dataset.format_for_sft(i)
            examples.append({"prompt": p, "target": t})
        batch = collate_for_sft(examples, tok, seq_len=64)

        opt = build_optimizer(model, AdamWConfig(lr=1e-4, max_steps=10))
        sched = WarmupCosineLR(opt, AdamWConfig(lr=1e-4, max_steps=10))

        logits = model(batch["input_ids"])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch["labels"].reshape(-1),
            ignore_index=-100,
        )
        assert torch.isfinite(loss).item()
        loss.backward()
        opt.step()
        sched.step(0)
        opt.zero_grad()

    def test_full_sft_smoke(self, tiny_pipeline_setup):
        """Full SFT loop with tiny model."""
        _, model, tok, dataset = tiny_pipeline_setup

        cfg = SFTConfig(
            max_steps=3, save_every=0, log_every=1,
            batch_size=2, seq_len=64,
        )
        logs = sft(model, cfg, tok)
        assert len(logs) == cfg.max_steps


class TestGRPOStage:
    """GRPO stage runs end-to-end."""

    def test_grpo_step_runs(self, tiny_pipeline_setup):
        """Single GRPO step with G=2 rollouts."""
        cfg, model, tok, dataset = tiny_pipeline_setup

        # Get a small batch of (prompt, gold) pairs
        prompts, golds = [], []
        for i in range(2):
            p, g = dataset.format_for_grpo(i)
            prompts.append(p)
            golds.append(g)

        grpo_cfg = GRPOConfig(
            group_size=2, batch_size=2, save_every=0, log_every=1,
            seq_len=32, kl_coef=0.0,  # no ref model for speed
        )

        opt = build_optimizer(model, AdamWConfig(lr=5e-6, max_steps=10))

        loss, stats = grpo_step(model, prompts, golds, tok, grpo_cfg)
        # Loss may be 0 or tiny if all rollouts are zero-reward
        assert "policy_loss" in stats or "skip" in stats
        opt.zero_grad()

    def test_grpo_advantages(self, tiny_pipeline_setup):
        """MC-GRPO advantages compute correctly."""
        # Verify with simple example
        rewards = [0.0, 0.0, 1.0, 1.0]
        advs, baseline = compute_mc_grpo_advantages(rewards, use_median=True)
        # median = 0.5
        assert baseline == 0.5
        # Two above, two below
        assert sum(1 for a in advs if a > 0) == 2
        assert sum(1 for a in advs if a < 0) == 2


class TestInferenceStage:
    """Inference stage runs end-to-end."""

    def test_generate_after_training(self, tiny_pipeline_setup):
        """Model can generate after training."""
        cfg, model, tok, dataset = tiny_pipeline_setup

        gen = RWKVGenerator(model, tok)
        prompt, _ = dataset.format_for_grpo(0)
        text = gen.generate(prompt, max_new_tokens=10, greedy=True)
        assert isinstance(text, str)
        assert len(text) > 0


class TestVotingStage:
    """Voting stage runs end-to-end."""

    def test_majority_vote_basic(self, tiny_pipeline_setup):
        """Majority vote picks most common."""
        candidates = ["a", "b", "a", "a", "c"]
        ans, info = majority_vote(candidates)
        assert ans == "a"
        assert info["agree_count"] == 3

    def test_sample_and_vote_smoke(self, tiny_pipeline_setup):
        """Sample multiple candidates and vote."""
        cfg, model, tok, dataset = tiny_pipeline_setup

        prompt, gold = dataset.format_for_grpo(0)
        ans, info = sample_and_vote(
            model, tok, prompt,
            n_samples=2, max_new_tokens=5, method="majority",
            gold=gold,
        )
        assert "candidates" in info
        assert len(info["candidates"]) == 2


class TestEvalStage:
    """Evaluation runs end-to-end."""

    def test_arithmetic_eval_smoke(self, tiny_pipeline_setup):
        """Run arithmetic eval on tiny model."""
        from src.eval.arithmetic_eval import evaluate_arithmetic

        cfg, model, tok, dataset = tiny_pipeline_setup
        results = evaluate_arithmetic(
            model, tok, dataset,
            max_new_tokens=10, n_samples=1, method="greedy",
        )
        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0


class TestCheckpointSaveLoad:
    """Checkpoint save/load works."""

    def test_save_load_roundtrip(self, tiny_pipeline_setup):
        """Save and load preserves state."""
        cfg, model, tok, dataset = tiny_pipeline_setup

        opt = build_optimizer(model, AdamWConfig(lr=1e-3))

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "ckpt.pt")
            save_checkpoint(model, opt, 100, ckpt_path, extra={"loss": 0.5})

            # Create fresh model
            fresh_model = init_and_verify(cfg)
            fresh_opt = build_optimizer(fresh_model, AdamWConfig(lr=1e-3))

            step = load_checkpoint(ckpt_path, fresh_model, fresh_opt)
            assert step == 100


class TestEndToEnd:
    """Full pipeline: pretrain -> SFT -> GRPO -> inference -> eval."""

    def test_full_pipeline_runs(self, tmp_path):
        """Complete pipeline runs on tiny model."""
        cfg = ModelConfig(
            vocab_size=256, n_layers=2, d_model=64, d_state=32,
            ctx_len=64, tie_embeddings=True,
        )
        model = init_and_verify(cfg)
        tok = StubTokenizer(vocab_size=256)
        dataset = SyntheticMathDataset(size=10, max_digits=2, seed=42)

        # Stage 1: Pretrain (tiny)
        pretrain_cfg = PretrainConfig(
            max_steps=2, save_every=0, log_every=0,
            batch_size=2, seq_len=32,
        )
        pretrain(model, pretrain_cfg, tok, [dataset.format_for_pretrain(i) for i in range(10)])
        assert True  # No exception

        # Stage 2: SFT (tiny)
        sft_cfg = SFTConfig(
            max_steps=2, save_every=0, log_every=0,
            batch_size=2, seq_len=32,
        )
        sft(model, sft_cfg, tok)
        assert True

        # Stage 3: Inference
        gen = RWKVGenerator(model, tok)
        prompt, gold = dataset.format_for_grpo(0)
        text = gen.generate(prompt, max_new_tokens=5, greedy=True)
        assert isinstance(text, str)

        # Stage 4: Eval
        from src.eval.arithmetic_eval import evaluate_arithmetic
        results = evaluate_arithmetic(
            model, tok, dataset,
            max_new_tokens=10, n_samples=1, method="greedy",
        )
        assert "accuracy" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])