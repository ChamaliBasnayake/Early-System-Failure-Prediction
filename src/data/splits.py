"""Phase 2, step 1 — the single chronological split (and two rolling-origin variants).

Outputs: data/splits/split_v1.json (primary), split_rollA.json, split_rollB.json,
Usage:   python -m src.data.splits [--sample]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.utils.paths import DATA_INTERIM, DATA_SPLITS


def _midnight_after(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts.tz_convert("UTC").normalize() + pd.Timedelta(days=1))


def make_split(ep: pd.DataFrame, data_start: pd.Timestamp, data_end: pd.Timestamp,
               q_train: float, q_val: float, name: str) -> dict:
    inc = ep.groupby("incident_id")["start"].min().sort_values().reset_index(drop=True)
    n = len(inc)
    method = (f"chronological; cut dates = next UTC midnight after the incident at the {q_train:.0%} / {q_val:.0%} "
              f"quantile of target incidents (support only; no window contents or labels)")
    t_train_end = _midnight_after(inc.iloc[max(int(q_train * n) - 1, 0)])
    t_val_end = _midnight_after(inc.iloc[max(int(q_val * n) - 1, 0)])
    if not (data_start < t_train_end < t_val_end < data_end):
        # Only happens on tiny inputs (the committed sample): fall back to splitting the TIME SPAN by the same proportions.
        span = data_end - data_start
        t_train_end = data_start + span * q_train
        t_val_end = data_start + span * q_val
        method = (f"chronological; FALLBACK — cut dates at {q_train:.0%} / {q_val:.0%} of the time span "
                  f"(incident-quantile cuts were not strictly inside the data span; expected only on the sample)")
        print(f"[splits] {name}: incident-quantile cuts invalid on this input → time-span fallback")
    per = {"train": (data_start, t_train_end), "validation": (t_train_end, t_val_end), "test": (t_val_end, data_end)}
    counts = {}
    for k, (a, b) in per.items():
        m = (inc >= a) & (inc < b)
        counts[k] = int(m.sum())
    return {
        "name": name,
        "method": method,
        "train": {"start": per["train"][0].isoformat(), "end": per["train"][1].isoformat()},
        "validation": {"start": per["validation"][0].isoformat(), "end": per["validation"][1].isoformat()},
        "test": {"start": per["test"][0].isoformat(), "end": per["test"][1].isoformat()},
        "incidents_per_split": counts,
    }


def load_split(path: Path) -> dict:
    d = json.loads(path.read_text())
    for k in ("train", "validation", "test"):
        d[k]["start"] = pd.Timestamp(d[k]["start"]); d[k]["end"] = pd.Timestamp(d[k]["end"])
    return d


def assign_split(ts: pd.Series, split: dict) -> pd.Series:
    """Map timestamps to 'train' / 'validation' / 'test' / None."""
    out = pd.Series([None] * len(ts), index=ts.index, dtype=object)
    for k in ("train", "validation", "test"):
        m = (ts >= split[k]["start"]) & (ts < split[k]["end"])
        out[m] = k
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    sfx = "_sample" if args.sample else ""
    ep = pd.read_parquet(DATA_INTERIM / f"failure_episodes{sfx}.parquet")
    ep["start"] = pd.to_datetime(ep["start"], utc=True)
    pf = pq.ParquetFile(DATA_INTERIM / f"bgl_parsed{sfx}.parquet")
    tmin, tmax = None, None
    for b in pf.iter_batches(columns=["timestamp"], batch_size=1_000_000):
        s = b.column(0).to_pandas()
        tmin = s.min() if tmin is None else min(tmin, s.min()); tmax = s.max() if tmax is None else max(tmax, s.max())
    data_start, data_end = pd.Timestamp(tmin).tz_convert("UTC"), pd.Timestamp(tmax).tz_convert("UTC") + pd.Timedelta(seconds=1)

    specs = [("split_v1", 0.60, 0.80), ("split_rollA", 0.50, 0.70), ("split_rollB", 0.65, 0.85)]
    DATA_SPLITS.mkdir(parents=True, exist_ok=True)

    for name, qt, qv in specs:
        d = make_split(ep, data_start, data_end, qt, qv, name)
        (DATA_SPLITS / f"{name}{sfx}.json").write_text(json.dumps(d, indent=2))

        # per-category incident support per split (primary split only gets the full table)
        sp = assign_split(ep["start"], load_split(DATA_SPLITS / f"{name}{sfx}.json"))
        tab = (ep.assign(split=sp).groupby(["first_category", "split"])["incident_id"].nunique()
                 .unstack(fill_value=0).reindex(columns=["train", "validation", "test"], fill_value=0)
                 .sort_values("train", ascending=False))

        print(f"[splits] {name}: " + ", ".join(f"{k}={d['incidents_per_split'][k]}" for k in ("train", "validation", "test")))

    print(f"[splits] JSON split files → {DATA_SPLITS}")


if __name__ == "__main__":
    main()
