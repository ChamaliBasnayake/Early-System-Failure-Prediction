"""Load a trained run (checkpoint + config) for XAI / serving."""
from __future__ import annotations

import torch

from src.models.lstm import load_state_dict_compat
from src.models.train import build_model
from src.utils.paths import RESULTS


def load_run(run_id: str):
    ck = torch.load(RESULTS / "checkpoints" / f"{run_id}.pt", map_location="cpu", weights_only=False)
    model = build_model(ck["config"], ck["vocab"], ck["n_scalars"]); load_state_dict_compat(model, ck["state_dict"]); model.eval()
    return model, ck
