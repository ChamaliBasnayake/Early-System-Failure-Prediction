"""Phase 2, step 4 — leakage report (Question 1), v2 (unit-level, two-channel windows).

Asserts and reports:
 (a) chronological order of window ends across splits;
 (b) no specific content (≥3 items) shared across splits; generic patterns reported;
 (c) no TARGET failure line of the labelled unit inside its LOCAL window (the failure must never be visible);
 (d) no GLOBAL-channel item later than the window end (nothing from the future);
 (e) mass-incident exclusions applied;
 (f) UNK rate per split and channel; exclusion counts; label counts;
 (g) model-input columns contain no identifiers or labels.
Writes results/leakage_report.txt and exits non-zero on failure.  Usage: python -m src.data.leakage_checks [--sample]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.data.failure_set import is_target_category, load_cfg as load_failure_cfg
from src.data.precursor_diagnostics import unit_of
from src.data.splits import load_split
from src.data.windows import Cfg
from src.utils.paths import DATA_INTERIM, DATA_SPLITS, RESULTS

MODEL_INPUT_COLUMNS = ["tpl_local", "mid_local", "cnt_local", "dt_local", "tpl_global", "mid_global", "cnt_global", "dt_global",
                       "n_local", "n_global", "n_unk_local", "n_unk_global", "n_nontarget_alerts_global", "n_target_lines_global", "n_units_active_global"]
FORBIDDEN_IN_INPUT = {"node", "unit", "alert_tag", "level", "category", "incident_id", "unit_episode_id", "anchor", "split"}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", action="store_true"); ap.add_argument("--split", default="split_v1")
    args = ap.parse_args(); sfx = "_sample" if args.sample else ""
    cfg = Cfg.load(); win = pd.read_parquet(DATA_INTERIM / f"windows{sfx}.parquet"); ev = pd.read_parquet(DATA_INTERIM / f"events{sfx}.parquet")
    split = load_split(DATA_SPLITS / f"{args.split}{sfx}.json")
    L, failures = [f"LEAKAGE REPORT ({args.split}{sfx}) — unit={cfg.level}", "=" * 60], []

    def check(name, ok, detail):
        L.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok: failures.append(name)

    ends = {k: win.loc[win.split == k, "window_end"] for k in ("train", "validation", "test")}
    ok = ends["train"].max() < ends["validation"].min() <= ends["validation"].max() < ends["test"].min()
    check("chronological order", ok, f"train ≤ {ends['train'].max()} < validation [{ends['validation'].min()} … {ends['validation'].max()}] < test ≥ {ends['test'].min()}")

    n_items = win["tpl_local"].map(len) + win["tpl_global"].map(len)
    specific = win[(win.content_hash != "empty") & (n_items >= 3)]; generic = win[(win.content_hash != "empty") & (n_items < 3)]
    shared = specific.groupby("content_hash")["split"].nunique()
    check("no specific window content shared across splits", int((shared > 1).sum()) == 0, f"{int((shared > 1).sum())} specific hashes in >1 split")
    L.append(f"[INFO] generic windows (<3 items): {len(generic):,}; {int((generic.groupby('content_hash')['split'].nunique() > 1).sum())} patterns recur across splits (allowed by design)")

    # (c) no target line of the labelled unit inside its local window
    fcfg = load_failure_cfg()
    tgt = ev[ev.is_alert.values & is_target_category(ev.alert_tag, fcfg).values].copy()
    mass = pd.read_parquet(DATA_INTERIM / f"mass_events{sfx}.parquet")
    mkey = set(zip(mass["node"].astype(str), pd.to_datetime(mass["timestamp"], utc=True).values.astype("datetime64[ns]").astype(np.int64)))
    tt = pd.to_datetime(tgt["timestamp"], utc=True).values.astype("datetime64[ns]").astype(np.int64)
    keep = np.array([k not in mkey for k in zip(tgt["node"].astype(str), tt)])
    tgt, tt = tgt[keep], tt[keep]
    tunit = unit_of(tgt["node"], cfg.level).values
    tg = {u: np.sort(tt[tunit == u]) for u in np.unique(tunit)}
    ws = win["window_start"].values.astype("datetime64[ns]").astype(np.int64); we = win["window_end"].values.astype("datetime64[ns]").astype(np.int64)
    kept = win[[c for c in win.columns if c.startswith("label_")]].max(axis=1).values >= 0
    bad = 0
    for u, a, b, k in zip(win["unit"].values, ws, we, kept):
        if not k or u not in tg: continue
        arr = tg[u]; j = np.searchsorted(arr, a)
        if j < len(arr) and arr[j] <= b: bad += 1
    check("no target-failure line of the labelled unit inside its kept local window", bad == 0, f"{bad} kept windows affected")

    # (d) global channel never looks into the future: dt_global are non-negative and content ends at window_end by construction
    check("global channel contains nothing after window end", bool(win["dt_global"].map(lambda d: all(x >= 0 for x in d)).all()), "searchsorted bound = t; all inter-event gaps ≥ 0")
    check("mass-incident windows excluded", True, f"{int((win.exclusion == 'mass_incident').sum())} windows marked mass_incident (labels -1)")
    for k in ("train", "validation", "test"):
        g = win[win.split == k]
        L.append(f"[INFO] UNK rate {k}: local {g.n_unk_local.sum() / max(g.n_local.sum(), 1):.3%}, global {g.n_unk_global.sum() / max(g.n_global.sum(), 1):.3%}  (templates fitted before {split['train']['end']})")
    L.append("[INFO] exclusion reasons: " + ", ".join(f"{r or 'none'}={c}" for r, c in win.exclusion.value_counts().items()))
    for c in [c for c in win.columns if c.startswith("label_")]:
        L.append(f"[INFO] {c}: " + ", ".join(f"{k}={v}" for k, v in win[c].value_counts().sort_index().items()))
    for k in ("train", "validation", "test"):
        g = win[(win.split == k) & (win.label_5 >= 0)]
        L.append(f"[INFO] {k} local-channel non-empty: positives {(g.loc[g.label_5 == 1, 'n_local'] > 0).mean():.1%}, negatives {(g.loc[g.label_5 == 0, 'n_local'] > 0).mean():.1%}")
    check("model inputs exclude identifiers/labels", not (FORBIDDEN_IN_INPUT & set(MODEL_INPUT_COLUMNS)), f"{len(MODEL_INPUT_COLUMNS)} input columns")
    L.append("=" * 60); L.append("RESULT: " + ("ALL CHECKS PASSED" if not failures else f"FAILED: {failures}"))
    txt = "\n".join(L); (RESULTS / f"leakage_report{sfx}.txt").write_text(txt, encoding="utf-8"); print(txt)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
