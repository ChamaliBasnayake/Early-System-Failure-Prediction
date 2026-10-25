"""Phase 3, step 0 — tabular features from windows (for the classical baselines and the confound baseline).

Feature groups (all computable at prediction time t; nothing after t is used):
  time      hour-of-day (sin, cos, US/Pacific), day-of-week one-hot, weekend flag           → confound baseline
  volume    n_local, n_global, log1p of both, n_units_active_global, distinct templates      → confound baseline
  counts    template-count vectors, local and global (vocab = train templates + UNK)         → content models
  summary   UNK counts/fractions, non-target alerts, other units' failure lines, Δt stats,
            burstiness (max repeat count)                                                    → content models
  history   FAILURE HISTORY known at t: hours since the unit's last target failure (capped at 7 d),
            unit failures in the last 6 h / 24 h / 7 d, hours since the last system-wide incident,
            system incidents in the last 6 h / 24 h, mass incidents in the last 24 h,
            non-target alerts on the unit in the last 24 h                                   → history models
  Content vs. history is a deliberate ablation (Q3): failures cluster in space and time, and the
  question is how much log CONTENT adds beyond that clustering.
Outputs: data/interim/features.parquet (one row per window; meta columns kept for joining),
         data/interim/feature_groups.json
Usage:   python -m src.features.build_features [--sample]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.data.precursor_diagnostics import unit_of
from src.utils.paths import DATA_INTERIM

SEC = np.int64(1_000_000_000); HOUR = 3600 * SEC
META = ["unit", "t_ns", "anchor", "unit_episode_id", "incident_id", "category", "split", "exclusion", "window_end"]


def count_matrix(seqs: pd.Series, cnts: pd.Series, vocab: int) -> np.ndarray:
    M = np.zeros((len(seqs), vocab), dtype=np.float32)
    for i, (s, c) in enumerate(zip(seqs, cnts)):
        if len(s): np.add.at(M[i], np.asarray(s, dtype=np.int64), np.asarray(c, dtype=np.float32))
    return M


def _count_before(times_sorted: np.ndarray, t: np.ndarray, horizon: np.int64) -> np.ndarray:
    hi = np.searchsorted(times_sorted, t, side="left"); lo = np.searchsorted(times_sorted, t - horizon, side="left")
    return (hi - lo).astype(np.float32)


def _hours_since(times_sorted: np.ndarray, t: np.ndarray, cap_h: float = 168.0) -> np.ndarray:
    hi = np.searchsorted(times_sorted, t, side="left")
    out = np.full(len(t), cap_h, dtype=np.float32); m = hi > 0
    out[m] = np.minimum((t[m] - times_sorted[hi[m] - 1]) / HOUR, cap_h)
    return out


def history_features(win: pd.DataFrame, ue: pd.DataFrame, mass: pd.DataFrame, nontarget: pd.DataFrame, level: str) -> dict:
    """Failure-history features at time t (strictly before t)."""
    t = win["t_ns"].values.astype(np.int64); units = win["unit"].values
    ue_t = ue["start_ns"].values.astype(np.int64)            # unit_episodes.parquet stores start_ns (int64 ns, UTC)
    ue_u = ue["unit"].values
    by_unit = {u: np.sort(ue_t[ue_u == u]) for u in np.unique(ue_u)}
    sys_t = np.sort(ue_t)
    mass_t = np.sort(pd.to_datetime(mass["start"], utc=True).values.astype("datetime64[ns]").astype(np.int64)) if len(mass) else np.array([], dtype=np.int64)
    nt_t = pd.to_datetime(nontarget["timestamp"], utc=True).values.astype("datetime64[ns]").astype(np.int64); nt_u = unit_of(nontarget["node"], level).values
    nt_by_unit = {u: np.sort(nt_t[nt_u == u]) for u in np.unique(nt_u)}
    empty = np.array([], dtype=np.int64)
    h = {k: np.zeros(len(t), dtype=np.float32) for k in ["hrs_since_unit_fail", "unit_fails_6h", "unit_fails_24h", "unit_fails_7d", "unit_nontarget_24h"]}
    for u in np.unique(units):
        m = units == u; arr = by_unit.get(u, empty); nta = nt_by_unit.get(u, empty)
        h["hrs_since_unit_fail"][m] = _hours_since(arr, t[m]); h["unit_fails_6h"][m] = _count_before(arr, t[m], 6 * HOUR)
        h["unit_fails_24h"][m] = _count_before(arr, t[m], 24 * HOUR); h["unit_fails_7d"][m] = _count_before(arr, t[m], 168 * HOUR)
        h["unit_nontarget_24h"][m] = _count_before(nta, t[m], 24 * HOUR)
    h["hrs_since_sys_fail"] = _hours_since(sys_t, t, 168.0); h["sys_fails_6h"] = _count_before(sys_t, t, 6 * HOUR); h["sys_fails_24h"] = _count_before(sys_t, t, 24 * HOUR)
    h["mass_24h"] = _count_before(mass_t, t, 24 * HOUR)
    return h


def build(win: pd.DataFrame, vocab: int, history: dict | None = None) -> tuple[pd.DataFrame, dict]:
    f = win[META + [c for c in win.columns if c.startswith("label_")]].copy()
    end = pd.to_datetime(win["window_end"], utc=True).dt.tz_convert("America/Los_Angeles")
    hour = end.dt.hour + end.dt.minute / 60
    time_cols = {"hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24), "weekend": (end.dt.dayofweek >= 5).astype(int)}
    for d in range(7): time_cols[f"dow_{d}"] = (end.dt.dayofweek == d).astype(int)
    vol_cols = {"n_local": win["n_local"], "n_global": win["n_global"], "log_n_local": np.log1p(win["n_local"]), "log_n_global": np.log1p(win["n_global"]),
                "n_units_active_global": win["n_units_active_global"], "distinct_local": win["tpl_local"].map(lambda s: len(set(s))),
                "distinct_global": win["tpl_global"].map(lambda s: len(set(s)))}
    summ_cols = {"n_unk_local": win["n_unk_local"], "n_unk_global": win["n_unk_global"],
                 "frac_unk_local": win["n_unk_local"] / win["n_local"].clip(lower=1), "frac_unk_global": win["n_unk_global"] / win["n_global"].clip(lower=1),
                 "has_nontarget_alert_local": win["has_nontarget_alert_local"].astype(int), "n_nontarget_alerts_global": win["n_nontarget_alerts_global"],
                 "n_target_lines_global": win["n_target_lines_global"],
                 "max_repeat_local": win["cnt_local"].map(lambda c: max(c) if len(c) else 0), "max_repeat_global": win["cnt_global"].map(lambda c: max(c) if len(c) else 0),
                 "mean_dt_local": win["dt_local"].map(lambda d: float(np.mean(d)) if len(d) else 3600.0), "mean_dt_global": win["dt_global"].map(lambda d: float(np.mean(d)) if len(d) else 3600.0),
                 "min_dt_global": win["dt_global"].map(lambda d: float(np.min(d[1:])) if len(d) > 1 else 3600.0)}
    hist_cols = history or {}
    for cols in (time_cols, vol_cols, summ_cols, hist_cols):
        for k, v in cols.items(): f[k] = np.asarray(v, dtype=np.float32)
    CL = count_matrix(win["tpl_local"], win["cnt_local"], vocab); CG = count_matrix(win["tpl_global"], win["cnt_global"], vocab)
    cl_names = [f"cl_{i}" for i in range(vocab)]; cg_names = [f"cg_{i}" for i in range(vocab)]
    f = pd.concat([f, pd.DataFrame(np.log1p(CL), columns=cl_names, index=f.index), pd.DataFrame(np.log1p(CG), columns=cg_names, index=f.index)], axis=1)
    groups = {"time": list(time_cols), "volume": list(vol_cols), "summary": list(summ_cols), "history": list(hist_cols),
              "counts_local": cl_names, "counts_global": cg_names}
    groups["confound"] = groups["time"] + groups["volume"]
    groups["content"] = groups["time"] + groups["volume"] + groups["summary"] + cl_names + cg_names
    groups["history_only"] = groups["time"] + groups["history"]
    groups["full"] = groups["content"] + groups["history"]
    return f, groups


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", action="store_true")
    ap.add_argument("--tag", default="", help="windows tag (e.g. system, tc); reads windows_<tag>, writes features_<tag>")
    ap.add_argument("--unit", default=None); ap.add_argument("--templates", default="", help="templates csv suffix, e.g. _tc")
    args = ap.parse_args(); sfx = "_sample" if args.sample else ""; tsfx = sfx + (f"_{args.tag}" if args.tag else "")
    import yaml
    from src.utils.paths import CONFIGS
    level = args.unit or yaml.safe_load((CONFIGS / "windows.yaml").read_text())["unit_level"]
    win = pd.read_parquet(DATA_INTERIM / f"windows{tsfx}.parquet")
    templates = pd.read_csv(DATA_INTERIM / f"templates{sfx}{args.templates}.csv"); vocab = int(templates["template_index"].max()) + 1
    ue = pd.read_parquet(DATA_INTERIM / f"unit_episodes{tsfx}.parquet"); mass = pd.read_parquet(DATA_INTERIM / f"mass_incidents{sfx}.parquet")
    nontarget = pd.read_parquet(DATA_INTERIM / f"nontarget_alerts{sfx}.parquet")
    hist = history_features(win, ue, mass, nontarget, level)
    f, groups = build(win, vocab, hist)
    f.to_parquet(DATA_INTERIM / f"features{tsfx}.parquet", index=False)
    (DATA_INTERIM / f"feature_groups{tsfx}.json").write_text(json.dumps(groups))
    print(f"[features] {f.shape[0]:,} windows × {len(groups['full'])} features (vocab {vocab}) → features{tsfx}.parquet")


if __name__ == "__main__":
    main()
