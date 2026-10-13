"""Seed control for reproducibility.

Every script in this project calls ``set_seed(seed)`` before doing anything
random.  The three project seeds are 11, 22 and 33; results are reported as
mean ± spread over them (brief, Section 6: "Seeds and repeated runs").

Why each line matters:
* ``random``      – Python's own RNG (used e.g. when sub-sampling negatives).
* ``numpy``       – array shuffles, sklearn/xgboost internals that read numpy's RNG.
* ``torch``       – weight initialisation, dropout masks, DataLoader shuffling.
* ``PYTHONHASHSEED`` – makes set/dict iteration order stable across runs.
* ``use_deterministic_algorithms`` – forces PyTorch to pick deterministic kernels
  where they exist; on CPU this is cheap and removes a source of run-to-run noise.
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
