"""Load checkpoints once; score a featurised window at request time (optionally with Monte-Carlo dropout)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from app.backend import settings
from src.models.lstm import load_state_dict_compat
from src.models.train import build_model
from src.utils.metrics import threshold_at_far


class ModelService:
    def __init__(self):
        self.models: dict[int, dict] = {}
        for N, run in settings.MODEL_RUNS.items():
            path = settings.CHECKPOINT_DIR / f"{run}.pt"
            if not path.exists(): continue
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = build_model(ck["config"], ck["vocab"], ck["n_scalars"]); load_state_dict_compat(model, ck["state_dict"]); model.eval()
            thresholds = self._thresholds(run)
            self.models[N] = {"run": run, "model": model, "scalar_cols": ck["scalar_cols"], "scaler": ck["scaler"], "thresholds": thresholds, "n_params": ck.get("n_params")}
        if not self.models: raise RuntimeError(f"No checkpoints found for {settings.MODEL_RUNS} in {settings.CHECKPOINT_DIR}")

    @staticmethod
    def _thresholds(run: str) -> dict:
        p = settings.SCORES_DIR / f"{run}_validation.parquet"
        if not p.exists(): return {}
        s = pd.read_parquet(p); neg = s.loc[s.label == 0, "score"].values
        return {str(far): float(threshold_at_far(neg, far)) for far in settings.FAR_LEVELS}

    def resolve(self, N: int) -> tuple[int, dict]:
        if N in self.models: return N, self.models[N]
        closest = min(self.models, key=lambda k: abs(k - N)); return closest, self.models[closest]

    @staticmethod
    def _tensors(chan, scalars_vec):
        tl, ml, cl, dl = chan; n = len(tl)
        T = torch.tensor([tl], dtype=torch.long) if n else torch.zeros(1, 1, dtype=torch.long)
        M = torch.tensor([ml], dtype=torch.long) if n else torch.zeros(1, 1, dtype=torch.long)
        C = torch.log1p(torch.tensor([cl], dtype=torch.float32)) if n else torch.zeros(1, 1)
        D = torch.log1p(torch.clamp(torch.tensor([dl], dtype=torch.float32), min=0)) if n else torch.zeros(1, 1)
        return (T, M, C, D, torch.tensor([n]))

    def score(self, win: dict, N: int, mc_samples: int | None = None) -> dict:
        used_N, m = self.resolve(N)
        mu, sd = m["scaler"]; x = np.array([win["scalars"][c] for c in m["scalar_cols"]], dtype=np.float32)
        S = torch.from_numpy(((x - mu) / sd).astype(np.float32)).unsqueeze(0)
        loc = self._tensors(win["local"], S); glo = self._tensors(win["global"], S)
        mc = settings.MC_SAMPLES if mc_samples is None else mc_samples
        with torch.no_grad():
            mean, std = m["model"].predict(loc, glo, S, mc_samples=mc)
        score = float(mean[0]); spread = float(std[0]) if std is not None else None
        thr = m["thresholds"]; decisions = {far: bool(score > t) for far, t in thr.items()}
        return {"score": score, "mc_spread": spread, "mc_samples": mc, "lead_time_used": used_N, "lead_time_requested": N, "run": m["run"],
                "thresholds": thr, "alarm_at_far": decisions, "n_params": m["n_params"]}
