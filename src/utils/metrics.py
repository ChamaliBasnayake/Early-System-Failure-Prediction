"""Evaluation metrics used by EVERY model (classical, neural, LLM). NumPy only.

"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAR_LEVELS = (0.001, 0.01)      # 1.4 and 14 alarms per unit per day


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    if y.sum() == 0: return float("nan")
    o = np.argsort(-s, kind="stable"); y = y[o]
    tp = np.cumsum(y); k = np.arange(1, len(y) + 1)
    return float((tp / k * y).sum() / y.sum())


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0: return float("nan")
    r = pd.Series(s).rank(method="average").values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def threshold_at_far(s_neg: np.ndarray, far: float) -> float:
    """Score threshold such that at most `far` of negative windows score STRICTLY ABOVE it.
    Decisions use `score > threshold`, so tied scores (e.g. a constant model) never fire."""
    s_neg = np.sort(np.asarray(s_neg, dtype=float))
    if len(s_neg) == 0: return float("inf")
    k = int(np.floor(far * len(s_neg)))
    return float(s_neg[-1]) if k <= 0 else float(s_neg[-k - 1])


def at_threshold(y: np.ndarray, s: np.ndarray, thr: float, incident_ids=None) -> dict:
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float); pred = s > thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    out = {"threshold": thr, "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
           "f1": 2 * tp / max(2 * tp + fp + fn, 1), "far": fp / max(int((y == 0).sum()), 1), "alarms_per_unit_day": fp / max(int((y == 0).sum()), 1) * 1440}
    if incident_ids is not None:
        inc = pd.Series(np.asarray(incident_ids))[y == 1]; caught = pd.Series(pred)[y == 1]
        per = caught.groupby(inc.values).any()
        out["incident_recall"] = float(per.mean()) if len(per) else float("nan"); out["n_incidents"] = int(len(per))
    return out


def evaluate(y, s, incident_ids=None, thresholds: dict | None = None, far_levels=FAR_LEVELS) -> dict:
    """Full metric set. If `thresholds` (chosen on validation) is given, use them; otherwise derive from these negatives."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    res = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()) if len(y) else float("nan"),
           "pr_auc": average_precision(y, s), "roc_auc": roc_auc(y, s), "thresholds": {}}
    for far in far_levels:
        thr = thresholds[str(far)] if thresholds else threshold_at_far(s[y == 0], far)
        res["thresholds"][str(far)] = thr
        m = at_threshold(y, s, thr, incident_ids)
        for k, v in m.items():
            if k != "threshold": res[f"{k}@far{far}"] = v
    return res


def flat(res: dict) -> dict:
    return {k: v for k, v in res.items() if not isinstance(v, dict)}
