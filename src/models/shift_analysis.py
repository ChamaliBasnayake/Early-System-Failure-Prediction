"""Post-hoc DESCRIPTIVE analysis of the validation→test shift (run AFTER final-eval-v1; trains nothing, changes nothing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.paths import DATA_INTERIM, RESULTS

N = 5


def main() -> None:
    f = pd.read_parquet(DATA_INTERIM / "features.parquet"); w = pd.read_parquet(DATA_INTERIM / "windows.parquet", columns=["unit", "t_ns", "split", f"label_{N}", "incident_id", "category", "anchor", "n_unk_global", "n_global", "n_local"])
    f = f.merge(w[["unit", "t_ns", "incident_id", "category", "anchor", "n_unk_global", "n_global", "n_local"]], on=["unit", "t_ns"], how="left", suffixes=("", "_w"))
    rows = []
    for split in ("train", "validation", "test"):
        g = f[(f.split == split) & (f[f"label_{N}"] >= 0)]; pos = g[g[f"label_{N}"] == 1]; neg = g[g[f"label_{N}"] == 0]
        inc_per_unit = pos.groupby("unit")["incident_id"].nunique(); top5 = inc_per_unit.sort_values(ascending=False).head(5)
        rows.append({"split": split, "windows": len(g), "prevalence": round(float(g[f"label_{N}"].mean()), 4), "positive incidents": pos.incident_id.nunique(), "failing units": pos.unit.nunique(),
                     "incidents per failing unit": round(float(inc_per_unit.mean()), 2), "share of incidents on top-5 units": round(float(top5.sum() / max(inc_per_unit.sum(), 1)), 2),
                     "positives with unit failure in prev 24 h": round(float((pos.unit_fails_24h > 0).mean()), 3), "negatives with unit failure in prev 24 h": round(float((neg.unit_fails_24h > 0).mean()), 3),
                     "positives with unit failure in prev 7 d": round(float((pos.unit_fails_7d > 0).mean()), 3), "negatives with unit failure in prev 7 d": round(float((neg.unit_fails_7d > 0).mean()), 3),
                     "UNK rate (global lines)": round(float(g.n_unk_global.sum() / max(g.n_global.sum(), 1)), 3), "median global lines/hour": float(g.n_global.median()), "positives with empty local channel": round(float((pos.n_local == 0).mean()), 3),
                     "top categories": ", ".join(f"{c} {n}" for c, n in pos.groupby("category")["incident_id"].nunique().sort_values(ascending=False).head(4).items())})
    df = pd.DataFrame(rows).set_index("split").T
    L = [f"# Validation → test shift (descriptive, post-hoc; N={N})\n", "Computed after `final-eval-v1`; no model was trained or changed. Explains why failure-history features were useless on validation but strong on test, and why the ANN (time+volume context only, no history inputs) did not transfer.\n", df.to_markdown() + "\n"]
    # lift of the single history feature as a rule: P(pos | unit failed in prev 7 d) / P(pos)
    for split in ("validation", "test"):
        g = f[(f.split == split) & (f[f"label_{N}"] >= 0)]; p = g[f"label_{N}"].mean(); p_h = g.loc[g.unit_fails_7d > 0, f"label_{N}"].mean()
        L.append(f"- {split}: P(failure | unit failed in previous 7 d) = {p_h:.4f} vs base rate {p:.4f} → lift {p_h / p:.1f}×; such windows are {(g.unit_fails_7d > 0).mean():.1%} of all windows.")
    (RESULTS / "test_eval" / "shift_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print(df.to_markdown()); print(f"→ {RESULTS / 'test_eval' / 'shift_analysis.md'}")


if __name__ == "__main__":
    main()
