"""Prepare raw-log inputs for the web app demo

Writes to results/demo/: for the chosen run and N —
  tp_<unit>.log    the highest-scoring TRUE positive (a real failure followed)      → expected alarm
  fn_<unit>.log    the lowest-scoring positive (a failure the model missed)         → the required incorrect prediction
  fp_hard_<unit>.log  highest-scoring HARD negative (alert-rich, no failure)        → incorrect prediction
  neg_<unit>.log   a quiet uniform negative
  invalid.txt      not a log at all                                                  → must be rejected
  thunderbird_ood.log  2,000 lines of a DIFFERENT system (Loghub Thunderbird sample) → out-of-distribution warning
Each .log contains the SYSTEM lines of [t-60 min, t] (the global channel), so the app rebuilds both channels itself.
Usage: python scripts/make_demo_inputs.py --run lstm_v1_N5_s11 [--N 5]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import DATA_INTERIM, DATA_RAW, RESULTS  # noqa: E402

TB_URL = "https://raw.githubusercontent.com/logpai/loghub/master/Thunderbird/Thunderbird_2k.log"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--N", type=int, default=5); args = ap.parse_args()
    out = RESULTS / "demo"; out.mkdir(exist_ok=True); lab = f"label_{args.N}"
    win = pd.read_parquet(DATA_INTERIM / "windows.parquet", columns=["unit", "t_ns", "split", lab, "anchor", "has_nontarget_alert_local", "n_local", "n_global", "category"])
    sc = pd.read_parquet(RESULTS / "scores" / f"{args.run}_validation.parquet"); df = win.merge(sc[["unit", "t_ns", "score"]], on=["unit", "t_ns"]); df = df[(df.split == "validation") & (df[lab] >= 0)]
    picks = {"tp": df[df[lab] == 1].sort_values("score", ascending=False).iloc[0], "fn": df[df[lab] == 1].sort_values("score").iloc[0],
             "fp_hard": df[(df[lab] == 0) & df.has_nontarget_alert_local].sort_values("score", ascending=False).iloc[0],
             "neg": df[(df[lab] == 0) & (df.anchor == "uniform") & (df.n_global > 20)].sample(1, random_state=11).iloc[0]}
    # stream the raw log once; collect lines inside each [t-60min, t]
    ranges = {k: (int(r.t_ns) - 60 * 60 * 10**9, int(r.t_ns)) for k, r in picks.items()}; buf = {k: [] for k in picks}
    with (DATA_RAW / "BGL.log").open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split(maxsplit=2)
            if len(parts) < 2 or not parts[1].isdigit(): continue
            t = int(parts[1]) * 10**9
            for k, (a, b) in ranges.items():
                if a < t <= b: buf[k].append(line)
    md = [f"# Demo inputs — {args.run}, N={args.N} (from VALIDATION)\n", "| file | midplane | label | score | anchor | category | lines in window | expected app behaviour |", "|---|---|---|---|---|---|---|---|"]
    for k, r in picks.items():
        fn = out / f"{k}_{r.unit}.log"; fn.write_text("".join(buf[k]), encoding="utf-8")
        exp = {"tp": "alarm (true positive)", "fn": "no/low alarm although a failure followed (false negative)", "fp_hard": "alarm although no failure followed (false positive)", "neg": "no alarm"}[k]
        md.append(f"| {fn.name} | {r.unit} | {int(r[lab])} | {r.score:.3f} | {r.anchor} | {r.category or '—'} | {len(buf[k])} | {exp} |")
    (out / "invalid.txt").write_text("hello world\nthis is not a log file\n12345\n", encoding="utf-8")
    try:
        urllib.request.urlretrieve(TB_URL, out / "thunderbird_ood.log"); md.append("| thunderbird_ood.log | — | — | — | — | — | 2000 | out-of-distribution: different system, high unknown-template rate |")
    except Exception as e:
        md.append(f"| thunderbird_ood.log | download failed: {e} |")
    md.append("\nFor the demo: paste a file's content, select the midplane shown here, lead time N, and compare the app's decision with the label.")
    (out / "README.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md)); print(f"[demo] → {out}")


if __name__ == "__main__":
    main()
