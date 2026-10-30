"""Load checkpoints once; score a featurised window at request time."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from app.backend import settings
from src.models.lstm import load_state_dict_compat
from src.models.train import build_model
from src.utils.metrics import threshold_at_far
from src.xai.integrated_gradients import integrated_gradients


class ModelService:
    def __init__(self):
        self.models: dict[int, dict] = {}; self._encoder = None; self.encoder_status = "not needed yet"
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

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Frozen MiniLM encoder, loaded on first use (deployment behaviour for messages never seen in training).
        If sentence-transformers is unavailable, new texts get a zero vector and the response says so."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(settings.EMBED_MODEL, device="cpu"); self.encoder_status = f"{settings.EMBED_MODEL} (CPU)"
            except Exception as e:  # pragma: no cover
                self._encoder = False; self.encoder_status = f"unavailable ({type(e).__name__}); unseen messages embedded as zero vectors"
        if self._encoder is False: return np.zeros((len(texts), 384), dtype=np.float32)
        return self._encoder.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=False)

    def ensure_texts(self, m: dict, win: dict) -> int:
        """For the semantic model: append embeddings of the window's never-seen texts. Returns rows added."""
        model = m["model"]; new = win.get("new_texts_by_id") or {}
        if not new or not hasattr(model, "add_texts"): return 0
        return model.add_texts(new, self._encode)

    def score(self, win: dict, N: int, mc_samples: int | None = None) -> dict:
        used_N, m = self.resolve(N); added = self.ensure_texts(m, win)
        mu, sd = m["scaler"]; x = np.array([win["scalars"][c] for c in m["scalar_cols"]], dtype=np.float32)
        S = torch.from_numpy(((x - mu) / sd).astype(np.float32)).unsqueeze(0)
        loc = self._tensors(win["local"], S); glo = self._tensors(win["global"], S)
        mc = settings.MC_SAMPLES if mc_samples is None else mc_samples
        with torch.no_grad():
            mean, std = m["model"].predict(loc, glo, S, mc_samples=mc)
        score = float(mean[0]); spread = float(std[0]) if std is not None else None
        thr = m["thresholds"]; decisions = {far: bool(score > t) for far, t in thr.items()}
        return {"score": score, "mc_spread": spread, "mc_samples": mc, "lead_time_used": used_N, "lead_time_requested": N, "run": m["run"],
                "thresholds": thr, "alarm_at_far": decisions, "n_params": m["n_params"], "new_texts_embedded": added, "encoder": self.encoder_status}


    def explain(self, win: dict, N: int, steps: int = 50) -> dict:
        """Integrated Gradients for one featurised window; attributions are mapped back to the raw local lines."""
        used_N, m = self.resolve(N); self.ensure_texts(m, win)
        mu, sd = m["scaler"]; x = np.array([win["scalars"][c] for c in m["scalar_cols"]], dtype=np.float32)
        S = torch.from_numpy(((x - mu) / sd).astype(np.float32)).unsqueeze(0)
        loc = self._tensors(win["local"], S); glo = self._tensors(win["global"], S)
        ig = integrated_gradients(m["model"], loc, glo, S, steps)
        tl, ml, cl, dl = win["local"]; tg, mg, cg, dg = win["global"]
        items_local = [{"item": i, "template_index": int(tl[i]), "masked_id": int(ml[i]), "count": int(cl[i]), "attribution": float(a)} for i, a in enumerate(ig["attr_local"])]
        items_global = [{"item": i, "template_index": int(tg[i]), "masked_id": int(mg[i]), "count": int(cg[i]), "attribution": float(a)} for i, a in enumerate(ig["attr_global"])]
        scal = sorted([{"feature": c, "attribution": float(a)} for c, a in zip(m["scalar_cols"], ig["attr_scalars"])], key=lambda d: -abs(d["attribution"]))
        return {"run": m["run"], "lead_time_used": used_N, "score": ig["score"], "baseline_score": ig["baseline_score"], "logit": ig["logit"], "baseline_logit": ig["baseline_logit"],
                "completeness_gap": ig["completeness_gap"], "steps": steps, "items_local": items_local, "items_global": items_global, "scalars": scal,
                "sum_local": float(sum(d["attribution"] for d in items_local)), "sum_global": float(sum(d["attribution"] for d in items_global)), "sum_scalars": float(sum(d["attribution"] for d in scal))}
