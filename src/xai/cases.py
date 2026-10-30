"""Incorrect-prediction case studies (Question 5.b): what do the explanations say when the model is wrong?

Picks from VALIDATION, for the chosen run and N: (1) the highest-scoring hard-negative false positive,
(2) an ordinary (uniform-negative) false positive, (3) the lowest-scoring positive (false negative), plus one
true positive for contrast. For each: score, the top positively and negatively attributed items in each channel
with their template text, the scalar attributions, and a line for your own interpretation.
Writes results/xai_cases.md.   Usage: python -m src.xai.cases --run lstm_v1_N5_s11 [--N 5]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.models.seq_data import WindowDataset, collate, load_windows
from src.utils.paths import DATA_INTERIM, RESULTS
from src.xai.integrated_gradients import integrated_gradients
from src.xai.loader import load_run


def describe(ig, df_row, mm, tpl, cols, k=6) -> list[str]:
    L = [f"- score **{ig['score']:.3f}** (baseline/empty-window score {ig['baseline_score']:.3f}; completeness gap {ig['completeness_gap']:+.3f} logits)"]
    for ch in ("local", "global"):
        a = ig[f"attr_{ch}"]; mids = df_row[f"mid_{ch}"]; cnts = df_row[f"cnt_{ch}"]; tpls = df_row[f"tpl_{ch}"]
        if len(a) == 0: L.append(f"- {ch} channel: empty"); continue
        L.append(f"- {ch} channel ({len(a)} items): top pushing TOWARDS failure →")
        for j in np.argsort(-a)[:k]:
            if a[j] <= 0: break
            L.append(f"    - {a[j]:+.3f}  ×{cnts[j]}  `{str(mm.get(int(mids[j]), tpl.get(int(tpls[j]), '?')))[:110]}`" + ("  (UNK template)" if tpls[j] == 1 else ""))
        L.append(f"  top pushing AWAY →")
        for j in np.argsort(a)[:3]:
            if a[j] >= 0: break
            L.append(f"    - {a[j]:+.3f}  ×{cnts[j]}  `{str(mm.get(int(mids[j]), tpl.get(int(tpls[j]), '?')))[:110]}`")
    s = ig["attr_scalars"]; order = np.argsort(-np.abs(s))[:5]
    L.append("- scalar context: " + ", ".join(f"{cols[j]} {s[j]:+.3f}" for j in order))
    L.append("- **Interpretation (write this yourself):** ")
    return L


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--N", type=int, default=5); ap.add_argument("--steps", type=int, default=50); args = ap.parse_args()
    model, ck = load_run(args.run)
    df, X, cols = load_windows("", args.N, "validation", ck["config"]["model"]["scalar_groups"]); ds = WindowDataset(df, X, args.N, 256, 256, scaler=ck["scaler"])
    scores = pd.read_parquet(RESULTS / "scores" / f"{args.run}_validation.parquet"); df = df.merge(scores[["unit", "t_ns", "score"]], on=["unit", "t_ns"], how="left")
    mm = pd.read_parquet(DATA_INTERIM / "masked_messages.parquet").set_index("masked_id")["text"].to_dict(); tpl = pd.read_csv(DATA_INTERIM / "templates.csv").set_index("template_index")["template"].to_dict()
    thr = float(np.quantile(df.loc[df[f"label_{args.N}"] == 0, "score"], 0.99)); lab = f"label_{args.N}"
    picks = {"Case 1 — false positive on a HARD negative (alert-rich window, no failure followed)": df[(df[lab] == 0) & df["has_nontarget_alert_local"]].sort_values("score", ascending=False).head(1),
             "Case 2 — false positive on an ordinary negative": df[(df[lab] == 0) & (df["anchor"] == "uniform")].sort_values("score", ascending=False).head(1),
             "Case 3 — false negative (a real failure the model scored lowest)": df[df[lab] == 1].sort_values("score").head(1),
             "Case 4 — false negative WITH log activity on the midplane (lowest-scoring positive whose local channel is not empty)": df[(df[lab] == 1) & (df["n_local"] > 0)].sort_values("score").head(1),
             "Contrast — true positive (highest-scoring real failure)": df[df[lab] == 1].sort_values("score", ascending=False).head(1)}
    L = [f"# Incorrect-prediction case studies — {args.run}, N={args.N} (auto-generated; interpretations to be written by hand)\n",
         f"1 % FAR threshold on validation: {thr:.4f}. Attributions: Integrated Gradients ({args.steps} steps) relative to an empty window.\n"]
    for title, sub in picks.items():
        if sub.empty: L.append(f"## {title}\n(no window in this category)\n"); continue
        i = int(sub.index[0]); r = df.loc[i]; loc, glo, S, y = collate([ds[i]]); ig = integrated_gradients(model, loc, glo, S, args.steps)
        verdict = ("INCORRECT at the 1 % FAR threshold" if (int(r[lab]) == 1) != (float(r["score"]) > thr) else "correct at the 1 % FAR threshold") + f" (score {float(r['score']):.3f} vs {thr:.3f})"
        L.append(f"## {title}\n**{verdict}.** midplane **{r['unit']}**, window end {pd.to_datetime(r['t_ns'], utc=True)}, label {int(r[lab])}, anchor {r['anchor']}, category {r['category'] or '—'}, "
                 f"local items {len(r['tpl_local'])} ({r['n_local']} raw lines), global items {len(r['tpl_global'])} ({r['n_global']} raw lines), UNK local {r['n_unk_local']} / global {r['n_unk_global']}\n")
        L += describe(ig, r, mm, tpl, cols); L.append("")
    L.append("## Limitations of the method (state these in the report)\n- Attribution measures sensitivity along one path from an EMPTY baseline; it is not a causal claim.\n"
             "- Correlated items share attribution; a line can be important yet receive little attribution if a similar line already explains the change.\n"
             "- A large share of attribution on the scalar context means the model leans on time/volume, which the perturbation test will also show.")
    (RESULTS / "xai_cases.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L[:40])); print(f"[xai] → {RESULTS / 'xai_cases.md'}")


if __name__ == "__main__":
    main()
