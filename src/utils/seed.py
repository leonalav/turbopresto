"""Reproducibility: seeding utilities.

Per /ipho-physicist: reproducibility is critical for verifying experiments.
All random sources must be seeded deterministically.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


_DEFAULT_SEED = 42


def set_seed(seed: int = _DEFAULT_SEED, deterministic: bool = False) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: The seed value
        deterministic: If True, use deterministic algorithms (slower).
            Set to False by default for performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Set PYTHONHASHSEED for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_seed() -> int:
    """Get current random seed (or default)."""
    return os.environ.get("PYTHONHASHSEED", _DEFAULT_SEED)


if __name__ == "__main__":
    set_seed(42)
    a = torch.rand(3)
    set_seed(42)
    b = torch.rand(3)
    print(f"Same seed -> same tensor: {torch.allclose(a, b)}")