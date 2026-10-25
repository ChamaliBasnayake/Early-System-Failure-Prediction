"""Phase 3 analysis — what does the XGBoost content model actually use?

Reads results/models/xgboost_content_N{N}.joblib (and the full model), takes gain-based importances,
maps count features back to template text (cl_i = local channel, cg_i = global channel) and writes
results/feature_importance.md. Also reports, for the top local/global templates, the mean log-count
in positive vs negative validation windows so the DIRECTION of each effect is visible.
Usage: python -m src.models.feature_importance [--tag w24h] [--N 5] [--top 25]
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src.utils.paths import DATA_INTERIM, RESULTS


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--N", type=int, nargs="*", default=[1, 5, 15, 30])
    ap.add_argument("--top", type=int, default=25); ap.add_argument("--sample", action="store_true"); args = ap.parse_args()
    sfx = ("_sample" if args.sample else "") + (f"_{args.tag}" if args.tag else "")
    f = pd.read_parquet(DATA_INTERIM / f"features{sfx}.parquet"); groups = json.loads((DATA_INTERIM / f"feature_groups{sfx}.json").read_text())
    tpl = pd.read_csv(DATA_INTERIM / ("templates_sample.csv" if args.sample else "templates.csv")).set_index("template_index")["template"]
    name_of = lambda c: (f"[local] {tpl.get(int(c[3:]), 'UNK' if c.endswith('_1') else 'PAD')[:90]}" if c.startswith("cl_") else
                         f"[global] {tpl.get(int(c[3:]), 'UNK' if c.endswith('_1') else 'PAD')[:90]}" if c.startswith("cg_") else c)
    L = [f"# Feature importance — XGBoost content model (tag='{args.tag or 'primary'}')\n",
         "Gain-based importance; direction = mean feature value in positive vs negative VALIDATION windows.\n"]
    for N in args.N:
        for variant, cols in (("content", groups["content"]), ("full", groups["full"])):
            path = RESULTS / "models" / (f"xgboost_{variant}_N{N}{sfx}.joblib" if variant == "content" else f"xgboost_N{N}{sfx}.joblib")
            if not path.exists(): continue
            model = joblib.load(path); booster = model.get_booster()
            gain = booster.get_score(importance_type="gain")
            imp = pd.Series({cols[int(k[1:])]: v for k, v in gain.items()}).sort_values(ascending=False)
            va = f[(f.split == "validation") & (f[f"label_{N}"] >= 0)]; pos, neg = va[va[f"label_{N}"] == 1], va[va[f"label_{N}"] == 0]
            rows = [(name_of(c), round(v, 1), round(float(pos[c].mean()), 3), round(float(neg[c].mean()), 3)) for c, v in imp.head(args.top).items()]
            L.append(f"## N={N} min — {variant} model ({len(imp)} features used)\n")
            L.append(pd.DataFrame(rows, columns=["feature", "gain", "mean in positives", "mean in negatives"]).to_markdown(index=False) + "\n")
            share = {g: float(imp[imp.index.isin(groups[g])].sum() / imp.sum()) for g in ("time", "volume", "summary", "history", "counts_local", "counts_global") if g in groups}
            L.append("Share of total gain by feature group: " + ", ".join(f"{k} {v:.1%}" for k, v in share.items()) + "\n")
    (RESULTS / f"feature_importance{sfx}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40])); print(f"[importance] → {RESULTS / f'feature_importance{sfx}.md'}")


if __name__ == "__main__":
    main()
