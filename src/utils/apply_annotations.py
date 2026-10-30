"""Apply reviewed human annotations to experiments/log.csv (idempotent; never deletes the automatic text).

experiments/annotations.csv columns: match, kind, outcome, learned
  match   = run_id (exact) or a prefix of run_id (kind=prefix) or a substring of purpose/key_hyperparams (kind=contains)
  outcome = optional new outcome (success | unsuccessful | failed | control)
  learned = the human statement; appended as " | note: ..." unless already present
Rows of the Phase 3 random search (25 XGBoost configurations per N) get a computed note: rank within the sweep and the
distance to the sweep's best configuration, so every row states what was learned from that configuration.
Usage: python -m src.utils.apply_annotations            (then git diff experiments/log.csv to review)
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.utils.runlog import ALLOWED_OUTCOMES, COLUMNS, LOG_PATH

ANN = Path("experiments/annotations.csv")


def main() -> None:
    rows = list(csv.DictReader(LOG_PATH.open(encoding="utf-8")))
    ann = list(csv.DictReader(ANN.open(encoding="utf-8")))
    hits = 0
    for a in ann:
        for r in rows:
            k = a.get("kind", "exact")
            ok = (r["run_id"] == a["match"]) if k == "exact" else (r["run_id"].startswith(a["match"]) if k == "prefix" else (a["match"] in r["purpose"] or a["match"] in r["key_hyperparams"] or a["match"] in r["run_id"]))
            if not ok: continue
            note = "note: " + a["learned"].strip()
            if note not in r["learned"]: r["learned"] = (r["learned"] + " | " if r["learned"] else "") + note
            if a.get("outcome"):
                assert a["outcome"] in ALLOWED_OUTCOMES, a["outcome"]; r["outcome"] = a["outcome"]
            hits += 1
    # ---- computed notes for the Phase 3 baseline invocations. Every invocation of src/models/baselines.py logs the same purpose
    # strings, so invocations are separated by their trivial-reference row for N=1 and identified by matching the best
    # XGBoost PR-AUC per N against the result files (baselines_validation<tag>.csv).
    import pandas as pd
    from src.utils.paths import RESULTS
    tags = {"": "primary (chronological split, train-only templates, 60-min windows)", "_system": "CONTROL unit = whole system (Phase 3b)", "_w6h": "window-length study: 6-hour local window (Phase 3c)",
            "_w24h": "window-length study: 24-hour local window (Phase 3c)", "_randomsplit": "PROTOCOL CONTROL random window split (Phase 6) — never used for decisions", "_tc": "PROTOCOL CONTROL transductive templates (Phase 6) — never used for decisions"}
    best_by_tag = {}
    for tag in tags:
        p = RESULTS / f"baselines_validation{tag}.csv"
        if p.exists():
            b = pd.read_csv(p); best_by_tag[tag] = {int(N): round(float(v), 4) for N, v in b[b.model == "xgboost"].groupby("N")["pr_auc"].max().items()}
    p3 = sorted([r for r in rows if r["phase"] == "3"], key=lambda r: r["date_time"]); blocks, cur = [], []
    for r in p3:
        if r["model_family"] == "trivial" and r["key_hyperparams"].startswith("N=1") and cur: blocks.append(cur); cur = []
        cur.append(r)
    if cur: blocks.append(cur)
    for blk in blocks:
        xgb = [r for r in blk if r["model_family"] == "xgboost" and "max_depth" in r["key_hyperparams"]]
        best = {}
        for r in xgb:
            N = int(r["key_hyperparams"].split(";")[0].split("=")[1]); best[N] = max(best.get(N, 0), round(float(r["val_pr_auc"] or 0), 4))
        label = next((tags[t] for t, bb in best_by_tag.items() if all(abs(bb.get(N, -1) - v) < 1e-4 for N, v in best.items())), "unidentified invocation")
        for N in sorted(best):
            rs = [r for r in xgb if r["key_hyperparams"].startswith(f"N={N};")]; vals = sorted((float(r["val_pr_auc"] or 0) for r in rs), reverse=True)
            for r in rs:
                v = float(r["val_pr_auc"] or 0); rank = vals.index(v) + 1
                note = f"note: [{label}] random-search configuration {rank}/{len(rs)} for N={N}; " + ("BEST of this sweep → kept as the baseline model." if rank == 1 else f"{vals[0] - v:.4f} PR-AUC below the sweep's best → rejected.")
                if "random-search configuration" not in r["learned"]: r["learned"] = (r["learned"] + " | " if r["learned"] else "") + note
        for r in blk:
            if r["model_family"] != "xgboost" and "note:" not in r["learned"] and label.startswith(("CONTROL", "PROTOCOL", "window-length")):
                r["learned"] = (r["learned"] + " | " if r["learned"] else "") + f"note: [{label}] — comparison only; see docs/phase6_findings.md / results/baselines_validation*.csv."
            elif r["model_family"] != "xgboost" and "note:" not in r["learned"]:
                r["learned"] = (r["learned"] + " | " if r["learned"] else "") + f"note: [{label}]."
    for r in rows:                                           # test / GenAI rows
        if r["phase"] == "7" and "note:" not in r["learned"]: r["learned"] += " | note: single test read (tag final-eval-v1), thresholds frozen from validation; interpreted in docs/phase7_findings.md."
        if r["phase"] == "10" and "note:" not in r["learned"]: r["learned"] += " | note: prompt-selection run on validation; p2 chosen by the pre-registered rule; see docs/phase10_findings.md."
    with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); w.writerows(rows)
    auto_only = sum(1 for r in rows if "note:" not in r["learned"])
    print(f"[annotate] {hits} annotation hits; {len(rows)} rows; {auto_only} rows still without a human/computed note")
    for r in rows:
        if "note:" not in r["learned"]: print("   no note:", r["phase"], r["run_id"][:50], "|", r["purpose"][:50])


if __name__ == "__main__":
    main()
