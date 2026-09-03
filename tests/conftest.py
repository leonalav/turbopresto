"""Pytest fixtures for test suite.

Tiny config used across all tests for CPU speed:
    vocab_size=128, n_layers=2, d_model=64, d_state=32
    -> ~180K params, forward in <1s on CPU
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.config import ModelConfig
from src.model.init import init_and_verify
from src.model.rwkv7 import build_model


@pytest.fixture(scope="session")
def tiny_config() -> ModelConfig:
    """Tiny config for CPU-runnable tests."""
    return ModelConfig(
        vocab_size=128,
        n_layers=2,
        d_model=64,
        d_state=32,
        ctx_len=64,
        tie_embeddings=True,
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_config: ModelConfig):
    """Tiny initialized model for tests."""
    model = init_and_verify(tiny_config)
    return model


@pytest.fixture(scope="session")
def real_config() -> ModelConfig:
    """Real production config (must hit ~50M)."""
    return ModelConfig()  # defaults


@pytest.fixture
def device() -> str:
    """Use CPU for tests (CUDA not available)."""
    return "cpu"


@pytest.fixture
def dtype() -> torch.dtype:
    """Use float32 for tests (BF16 only on CUDA)."""
    return torch.float32