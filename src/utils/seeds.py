"""Seed control for reproducibility.

"""
from __future__ import annotations

import os
import random

import numpy as np

PROJECT_SEEDS = (11, 22, 33)


def set_seed(seed: int) -> None:
    """Seed every random number generator the project uses."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # torch is optional for the pure-data scripts
        pass


def seed_worker(worker_id: int) -> None:  # pragma: no cover - used by DataLoader
    """Seed function for ``torch.utils.data.DataLoader(worker_init_fn=...)``."""
    worker_seed = (np.random.get_state()[1][0] + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
