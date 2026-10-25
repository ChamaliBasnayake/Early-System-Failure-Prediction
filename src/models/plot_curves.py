"""Plot learning curves from results/curves/*.json (one figure per run, plus an overlay for runs sharing a prefix).
Usage: python -m src.models.plot_curves --runs lstm_v1_N5_s11 lstm_v1_N5_s22 lstm_v1_N5_s33 --out lstm_v1_N5
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.paths import RESULTS


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True); args = ap.parse_args()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for r in args.runs:
        d = json.loads((RESULTS / "curves" / f"{r}.json").read_text()); c = d["curves"]; ep = [x["epoch"] for x in c]
        ax[0].plot(ep, [x["train_loss"] for x in c], "--", label=f"{r} train"); ax[0].plot(ep, [x["val_loss"] for x in c], "-", label=f"{r} val")
        ax[1].plot(ep, [x["val_pr_auc"] for x in c], "-o", ms=3, label=r)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("weighted BCE loss"); ax[0].set_title("Loss"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation PR-AUC"); ax[1].set_title("Validation PR-AUC"); ax[1].legend(fontsize=7)
    fig.tight_layout(); (RESULTS / "figures").mkdir(exist_ok=True); out = RESULTS / "figures" / f"curves_{args.out}.png"; fig.savefig(out, dpi=150); print(f"[curves] → {out}")


if __name__ == "__main__":
    main()
