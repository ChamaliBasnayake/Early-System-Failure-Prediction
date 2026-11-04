"""Error analysis of the final ANN by input type and data quality (Question 3.a), on VALIDATION scores (mean of 3 seeds).

Usage: python -m src.models.error_buckets --run-prefix sem_final --N 5
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.utils.metrics import evaluate
from src.utils.paths import DATA_INTERIM, RESULTS


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--run-prefix", default="sem_final"); ap.add_argument("--N", type=int, default=5); ap.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33]); args = ap.parse_args()
    N = args.N; lab = f"label_{N}"
    win = pd.read_parquet(DATA_INTERIM / "windows.parquet", columns=["unit", "t_ns", "split", lab, "anchor", "category", "incident_id", "n_local", "n_global", "n_unk_global", "n_unk_local", "has_nontarget_alert_local", "n_units_active_global"])
    win = win[(win.split == "validation") & (win[lab] >= 0)].reset_index(drop=True)
    sc = []
    for s in args.seeds:
        p = RESULTS / "scores" / f"{args.run_prefix}_N{N}_s{s}_validation.parquet"
        if p.exists(): sc.append(pd.read_parquet(p)[["unit", "t_ns", "score"]].rename(columns={"score": f"s{s}"}))
    if not sc: raise SystemExit("no validation score files found")
    df = win
    for d in sc: df = df.merge(d, on=["unit", "t_ns"], how="left")
    df["score"] = df[[c for c in df.columns if c.startswith("s") and c[1:].isdigit()]].mean(axis=1); df = df.dropna(subset=["score"])
    thr = json.loads((RESULTS / "thresholds_final.json").read_text()).get(f"{args.run_prefix}_s11|N={N}", {}).get("0.01") or float(np.quantile(df.loc[df[lab] == 0, "score"], 0.99))
    df["unk_rate"] = df.n_unk_global / df.n_global.clip(lower=1); df["hour"] = pd.to_datetime(df.t_ns, utc=True).dt.hour
    df["bucket_local"] = np.where(df.n_local == 0, "local channel EMPTY", "local channel has lines")
    df["bucket_unk"] = pd.cut(df.unk_rate, [-0.01, 0.0, 0.1, 0.5, 1.0], labels=["UNK 0 %", "UNK 0–10 %", "UNK 10–50 %", "UNK > 50 %"])
    df["bucket_vol"] = pd.qcut(df.n_global.rank(method="first"), 4, labels=["global volume Q1 (quietest)", "Q2", "Q3", "Q4 (busiest)"])
    df["bucket_hour"] = pd.cut(df.hour, [-1, 5, 11, 17, 23], labels=["00–05 UTC", "06–11", "12–17", "18–23"])

    def table(col) -> pd.DataFrame:
        out = []
        for b, g in df.groupby(col, observed=True):
            pos = int((g[lab] == 1).sum()); neg = int((g[lab] == 0).sum())
            r = evaluate(g[lab].values, g.score.values, g.incident_id.values, thresholds={"0.001": thr, "0.01": thr}) if pos and neg else {}
            out.append({"bucket": str(b), "windows": len(g), "positives": pos, "prevalence": round(pos / max(len(g), 1), 4), "PR-AUC": round(r.get("pr_auc", float("nan")), 4), "PR-AUC / prevalence": round(r.get("pr_auc", float("nan")) / max(pos / max(len(g), 1), 1e-9), 1) if pos else float("nan"),
                        "ROC-AUC": round(r.get("roc_auc", float("nan")), 3), "recall @ frozen 1 % thr": round(r.get("recall@far0.01", float("nan")), 3), "FAR @ frozen 1 % thr": round(float((g.loc[g[lab] == 0, "score"] > thr).mean()), 4) if neg else float("nan"),
                        "incident recall": round(r.get("incident_recall@far0.01", float("nan")), 3)})
        return pd.DataFrame(out)

    L = [f"# Error analysis by input type and data quality — {args.run_prefix} (mean of seeds {args.seeds}), N={N}, validation\n",
         f"Frozen 1 % FAR threshold {thr:.4f}. 'PR-AUC / prevalence' is the lift over a constant predictor within the bucket (1.0 = chance).\n"]
    for title, col in (("Local channel empty vs not", "bucket_local"), ("Unknown-template rate of the global channel", "bucket_unk"), ("Global log volume in the window", "bucket_vol"), ("Hour of day", "bucket_hour")):
        t = table(col); L.append(f"## {title}\n\n" + t.to_markdown(index=False) + "\n"); t.assign(factor=col).to_csv(RESULTS / "error_buckets.csv", mode="a" if title != "Local channel empty vs not" else "w", header=title == "Local channel empty vs not", index=False)
    neg = df[df[lab] == 0]; L.append("## Negatives: false-alarm rate at the frozen threshold by anchor type\n\n" + pd.DataFrame([{"anchor": a, "windows": len(g), "FAR": round(float((g.score > thr).mean()), 4), "mean score": round(float(g.score.mean()), 4)} for a, g in neg.groupby("anchor")]).to_markdown(index=False) + "\n")
    pos = df[df[lab] == 1]; cat = pos.groupby("category").apply(lambda g: pd.Series({"incidents": g.incident_id.nunique(), "windows": len(g), "mean score": round(float(g.score.mean()), 4), "incident recall @ thr": round(float(g.groupby("incident_id").score.max().gt(thr).mean()), 3), "share with empty local channel": round(float((g.n_local == 0).mean()), 3)}), include_groups=False)
    L.append("## Positives by failure category\n\n" + cat.to_markdown() + "\n")
    L.append("## Reading\n- If the EMPTY-local bucket has PR-AUC/prevalence near 1 and the non-empty bucket well above 1, the model's whole signal lives in windows where the midplane itself logged something; the empty bucket is the irreducible-error regime (no input).\n"
             "- A FAR that is lower on hard negatives than on uniform negatives means the model is not an alert-chaser.\n- UNK buckets on validation are small (2.8 % UNK overall).\n")
    (RESULTS / "error_buckets.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L[:6])); print(f"→ {RESULTS / 'error_buckets.md'}")


if __name__ == "__main__":
    main()
