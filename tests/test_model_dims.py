"""Dimensional tests for RWKV-7 model.

Verifies:
- Parameter count matches analytical formula
- Forward/backward shapes
- State shape
- Real-config 50M target
"""

from __future__ import annotations

import pytest
import torch

from src.model.config import ModelConfig
from src.model.rwkv7 import build_model


class TestParamCount:
    """Parameter count must match analytic formula."""

    def test_analytic_count_tiny(self, tiny_config):
        """Tiny config: hand-verify count."""
        breakdown = tiny_config.param_breakdown()
        # Embedding: 128 * 64 = 8192
        assert breakdown["embedding"] == 128 * 64
        # Tied: head = 0
        assert breakdown["head"] == 0
        # ln_out: 2*64 = 128
        assert breakdown["ln_out"] == 128

    def test_actual_count_matches_analytic(self, tiny_config, tiny_model):
        """PyTorch parameter count must match ModelConfig formula."""
        analytic = tiny_config.total_params()
        actual = tiny_model.num_parameters()
        assert analytic == actual, f"Analytic {analytic} != actual {actual}"

    def test_param_count_tied_vs_untied(self):
        """Tied embeddings save V*C params."""
        cfg_tied = ModelConfig(vocab_size=128, n_layers=1, d_model=64, d_state=32, tie_embeddings=True)
        cfg_untied = ModelConfig(vocab_size=128, n_layers=1, d_model=64, d_state=32, tie_embeddings=False)
        diff = cfg_untied.total_params() - cfg_tied.total_params()
        # Untied should have V*C more params
        assert diff == 128 * 64

    def test_param_count_scales_with_layers(self):
        """Doubling layers adds 2x per-block params."""
        cfg1 = ModelConfig(vocab_size=128, n_layers=1, d_model=64, d_state=32)
        cfg2 = ModelConfig(vocab_size=128, n_layers=2, d_model=64, d_state=32)
        # Difference should be per_block
        per_block = cfg1.param_breakdown()["per_block"]
        assert cfg2.total_params() - cfg1.total_params() == per_block

    @pytest.mark.slow
    def test_50m_target_real_config(self, real_config):
        """Real config: must be in [40M, 60M] (50M ±20%)."""
        real_config.assert_target(50_000_000, tolerance=0.20)


class TestForwardShapes:
    """Forward pass shapes."""

    def test_forward_output_shape(self, tiny_model, tiny_config):
        """[B, T] -> [B, T, V]."""
        B, T = 2, 16
        idx = torch.randint(0, tiny_config.vocab_size, (B, T))
        out = tiny_model(idx)
        assert out.shape == (B, T, tiny_config.vocab_size)

    def test_forward_batch_1(self, tiny_model, tiny_config):
        """Batch size 1 works."""
        idx = torch.randint(0, tiny_config.vocab_size, (1, 8))
        out = tiny_model(idx)
        assert out.shape == (1, 8, tiny_config.vocab_size)

    def test_forward_long_seq(self, tiny_model, tiny_config):
        """Long sequence (full ctx_len) works."""
        T = tiny_config.ctx_len
        idx = torch.randint(0, tiny_config.vocab_size, (1, T))
        out = tiny_model(idx)
        assert out.shape == (1, T, tiny_config.vocab_size)

    def test_logits_finite(self, tiny_model, tiny_config):
        """No NaN/Inf in logits."""
        idx = torch.randint(0, tiny_config.vocab_size, (2, 16))
        out = tiny_model(idx)
        assert torch.isfinite(out).all(), "NaN or Inf in logits"


class TestBackwardShapes:
    """Backward pass shapes match."""

    def test_gradient_shapes(self, tiny_model, tiny_config):
        """All param gradients have correct shape (except unused layer-0 LoRAs)."""
        idx = torch.randint(0, tiny_config.vocab_size, (2, 16))
        target = torch.randint(0, tiny_config.vocab_size, (2, 16))
        logits = tiny_model(idx)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, tiny_config.vocab_size), target.reshape(-1)
        )
        loss.backward()
        # Every parameter must have a gradient of correct shape
        # EXCEPT: v0, v1, v2 in layer 0 are not used (value residual from layer 0
        # would mix with itself, so canonical RWKV-7 doesn't apply the mix at layer 0)
        unused_in_layer_0 = {"v0", "v1", "v2"}
        for name, p in tiny_model.named_parameters():
            if p.grad is None:
                # OK only if it's an unused layer-0 LoRA
                parts = name.split(".")
                if len(parts) >= 3 and parts[-2] == "att" and parts[-1] in unused_in_layer_0:
                    assert "blocks.0.att." in name, f"Unexpected missing grad for {name}"
                    continue
                raise AssertionError(f"No grad for {name}")
            assert p.grad.shape == p.shape, (
                f"Shape mismatch for {name}: param {p.shape}, grad {p.grad.shape}"
            )

    def test_gradient_finite(self, tiny_model, tiny_config):
        """All gradients are finite (no NaN/Inf)."""
        idx = torch.randint(0, tiny_config.vocab_size, (2, 16))
        target = torch.randint(0, tiny_config.vocab_size, (2, 16))
        logits = tiny_model(idx)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, tiny_config.vocab_size), target.reshape(-1)
        )
        loss.backward()
        unused_in_layer_0 = {"v0", "v1", "v2"}
        for name, p in tiny_model.named_parameters():
            if p.grad is None:
                if "blocks.0.att." in name and name.split(".")[-1] in unused_in_layer_0:
                    continue
                raise AssertionError(f"Missing grad for {name}")
            assert torch.isfinite(p.grad).all(), f"NaN/Inf in grad of {name}"


class TestArchitectureInvariants:
    """Architecture-specific shape invariants."""

    def test_head_size_divides_d_model(self):
        """d_model must be divisible by d_state (head_size)."""
        cfg = ModelConfig(d_model=512, d_state=64)
        assert cfg.d_model % cfg.d_state == 0
        assert cfg.n_heads == 8

    def test_n_heads_computed(self):
        """n_heads auto-computed if not given."""
        cfg = ModelConfig(d_model=512, d_state=64)
        assert cfg.n_heads == 8

    def test_dim_ffn_scaled(self):
        """dim_ffn = d_model * ffn_mult."""
        cfg = ModelConfig(d_model=512, ffn_mult=4)
        assert cfg.dim_ffn == 2048

    def test_dim_ffn_float_accepted(self):
        """C3 fix: float ffn_mult is accepted (YAML sets 2.75)."""
        cfg = ModelConfig(d_model=512, ffn_mult=2.75)
        assert cfg.dim_ffn == int(512 * 2.75)  # 1408

    def test_dim_ffn_rejects_string(self):
        """C3 fix: non-numeric ffn_mult raises TypeError, not silent failure."""
        with pytest.raises(TypeError):
            ModelConfig(d_model=512, ffn_mult="2.75")  # type: ignore[arg-type]