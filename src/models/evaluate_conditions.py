"""Phase 6 (validation) / Phase 7 (test, ONCE) — evaluate every model under every condition on one split.

Usage: python -m src.models.evaluate_conditions --split validation
       python -m src.models.evaluate_conditions --split test        # Phase 7 only
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.models.seq_data import WindowDataset, collate, load_windows
from src.utils.metrics import evaluate, flat, threshold_at_far
from src.utils.paths import CONFIGS, DATA_INTERIM, RESULTS
from src.utils.runlog import LOG_PATH, log_run
from src.models.lstm import load_state_dict_compat
from src.models.train import build_model

NS = yaml.safe_load((CONFIGS / "windows.yaml").read_text())["lead_times_min"]
SEEDS = (11, 22, 33)


# ----------------------------------------------------------------------------- scoring helpers
def load_run(run_id: str):
    """Load a trained run (checkpoint + config); same logic as src/xai/loader.py, duplicated so Phase 6 has no XAI dependency."""
    ck = torch.load(RESULTS / "checkpoints" / f"{run_id}.pt", map_location="cpu", weights_only=False)
    model = build_model(ck["config"], ck["vocab"], ck["n_scalars"]); load_state_dict_compat(model, ck["state_dict"]); model.eval()
    return model, ck


def score_nn(run_id: str, split: str, N: int) -> tuple[pd.DataFrame, float]:
    model, ck = load_run(run_id)
    df, X, _ = load_windows("", N, split, ck["config"]["model"]["scalar_groups"]); ds = WindowDataset(df, X, N, 256, 256, scaler=ck["scaler"])
    dl = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate); t0 = time.time(); out = []
    with torch.no_grad():
        for loc, glo, S, y in dl: out.append(torch.sigmoid(model(loc, glo, S)).numpy())
    ms = (time.time() - t0) / len(ds) * 1000
    return df.assign(score=np.concatenate(out), label=df[f"label_{N}"].values), ms


def score_xgb(kind: str, split: str, N: int) -> tuple[pd.DataFrame, float]:
    f = pd.read_parquet(DATA_INTERIM / "features.parquet"); groups = json.loads((DATA_INTERIM / "feature_groups.json").read_text())
    cols = {"xgboost": groups["full"], "xgboost_content": groups["content"], "xgboost_history_only": groups["history_only"]}[kind]
    model = joblib.load(RESULTS / "models" / (f"xgboost_N{N}.joblib" if kind == "xgboost" else f"{kind}_N{N}.joblib"))
    g = f[(f.split == split) & (f[f"label_{N}"] >= 0)].reset_index(drop=True); t0 = time.time(); s = model.predict_proba(g[cols].values.astype(np.float32))[:, 1]
    return g.assign(score=s, label=g[f"label_{N}"].values), (time.time() - t0) / len(g) * 1000


def score_confound(split: str, N: int) -> tuple[pd.DataFrame, float]:
    f = pd.read_parquet(DATA_INTERIM / "features.parquet"); groups = json.loads((DATA_INTERIM / "feature_groups.json").read_text())
    model = joblib.load(RESULTS / "models" / f"confound_lr_N{N}.joblib"); g = f[(f.split == split) & (f[f"label_{N}"] >= 0)].reset_index(drop=True)
    t0 = time.time(); s = model.predict_proba(g[groups["confound"]].values.astype(np.float32))[:, 1]
    return g.assign(score=s, label=g[f"label_{N}"].values), (time.time() - t0) / len(g) * 1000


def train_time(run_id_or_family: str, N: int | None = None) -> float:
    rows = list(csv.DictReader(LOG_PATH.open(encoding="utf-8")))
    cand = [r for r in rows if r["run_id"] == run_id_or_family] or [r for r in rows if r["model_family"] == run_id_or_family and (N is None or r["key_hyperparams"] == f"N={N}" or r["key_hyperparams"].startswith(f"N={N};"))]
    if not cand: return float("nan")
    try: return float(max(cand, key=lambda r: r["date_time"])["train_time_s"] or "nan")
    except ValueError: return float("nan")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=["validation", "test"], required=True); ap.add_argument("--final", default="sem_final"); ap.add_argument("--initial", default="lstm_v1")
    args = ap.parse_args(); split = args.split
    out_dir = RESULTS / ("test_eval" if split == "test" else "validation_eval"); out_dir.mkdir(exist_ok=True)
    thr_path = RESULTS / "thresholds_final.json"
    if split == "test":
        if not thr_path.exists(): raise SystemExit("Run --split validation first: thresholds must be frozen before the test evaluation.")
        if (out_dir / "final_test_results.json").exists(): raise SystemExit("final_test_results.json already exists. The test set is evaluated ONCE. If a bug was fixed, document it in the log and delete the file deliberately.")
        thresholds = json.loads(thr_path.read_text())
    else:
        thresholds = {}
    rows, scored = [], {}
    protocol = {"split": split, "split_file": "data/splits/split_v1.json", "templates": "Drain3 fitted on train only (317 templates; UNK=1)", "windows": "unit=midplane, W_local=W_global=60 min, H=10 min, offsets 0..8",
                "negatives": "uniform (unit, minute) within split period; hard negatives 30 s after non-target alerts", "far_levels": [0.001, 0.01], "seeds": list(SEEDS), "thresholds_frozen_from": "validation"}

    def add(name: str, family: str, N: int, df: pd.DataFrame, infer_ms: float, run_id: str | None = None, seed=None):
        key = f"{name}|N={N}"; thr = thresholds.get(key) if split == "test" else None
        res = evaluate(df["label"].values, df["score"].values, df["incident_id"].values, thresholds=thr)
        if split == "validation": thresholds[key] = res["thresholds"]
        scored[(name, N)] = df
        rows.append({"model": name, "family": family, "N": N, "seed": seed, **flat(res), "infer_ms_per_window": round(infer_ms, 4), "train_time_s": train_time(run_id or family, N)})
        print(f"  {name:28s} N={N:2d}  PR-AUC {res['pr_auc']:.4f}  ROC {res['roc_auc']:.3f}  recall@1% {res['recall@far0.01']:.3f}  inc-recall@1% {res.get('incident_recall@far0.01', float('nan')):.3f}")

    for N in NS:
        print(f"\n[{split}] N={N}")
        fx = pd.read_parquet(DATA_INTERIM / "features.parquet"); g = fx[(fx.split == split) & (fx[f"label_{N}"] >= 0)]
        add("trivial", "trivial", N, g.assign(score=0.5, label=g[f"label_{N}"].values), 0.0)
        df, ms = score_confound(split, N); add("confound_lr", "confound_lr", N, df, ms)
        for kind in ("xgboost", "xgboost_content", "xgboost_history_only"):
            if (RESULTS / "models" / (f"xgboost_N{N}.joblib" if kind == "xgboost" else f"{kind}_N{N}.joblib")).exists():
                df, ms = score_xgb(kind, split, N); add(kind, kind, N, df, ms)
        for seed in SEEDS:                                                   # initial ANN
            rid = f"{args.initial}_N{N}_s{seed}"
            if (RESULTS / "checkpoints" / f"{rid}.pt").exists(): df, ms = score_nn(rid, split, N); add(f"{args.initial}_s{seed}", "lstm", N, df, ms, rid, seed)
        for seed in SEEDS:                                                   # final ANN
            rid = f"{args.final}_N{N}_s{seed}"
            if (RESULTS / "checkpoints" / f"{rid}.pt").exists(): df, ms = score_nn(rid, split, N); add(f"{args.final}_s{seed}", "lstm_semantic", N, df, ms, rid, seed)
    res = pd.DataFrame(rows); res.to_csv(out_dir / f"results_{split}.csv", index=False)

    # ---- comparison table (mean ± sd over seeds for the ANNs)
    def agg(fam):
        sub = res[(res.model == fam) | res.model.str.match(rf"^{fam}_s\d+$")]      # exact family, or its seed runs — never the ablations
        return sub.groupby("N").agg(pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"), roc=("roc_auc", "mean"), recall_1=("recall@far0.01", "mean"), inc_recall_1=("incident_recall@far0.01", "mean"),
                                    recall_01=("recall@far0.001", "mean"), infer_ms=("infer_ms_per_window", "mean"), train_s=("train_time_s", "mean"), n_seeds=("model", "size"))
    L = [f"# {'FINAL TEST EVALUATION (run once)' if split == 'test' else 'Validation evaluation'} — all models, all conditions\n", "## Protocol\n", "```\n" + json.dumps(protocol, indent=1) + "\n```\n", "## Comparison table (primary metric PR-AUC; mean ± sd over seeds where available)\n"]
    tab = []
    for name, fam in (("trivial", "trivial"), ("confound_lr", "confound_lr"), ("xgboost_history_only", "xgboost_history_only"), ("xgboost_content", "xgboost_content"), ("xgboost", "xgboost"), (f"{args.initial} (initial ANN)", args.initial), (f"{args.final} (final ANN)", args.final)):
        a = agg(fam)
        for N, r in a.iterrows():
            tab.append({"model": name, "N": N, "PR-AUC": f"{r.pr_auc:.4f}" + (f" ± {r.pr_auc_sd:.4f}" if r.n_seeds > 1 and not np.isnan(r.pr_auc_sd) else ""), "ROC-AUC": f"{r.roc:.3f}", "recall@0.1%FAR": f"{r.recall_01:.3f}", "recall@1%FAR": f"{r.recall_1:.3f}",
                        "incident recall@1%": f"{r.inc_recall_1:.3f}", "train s": "" if np.isnan(r.train_s) else f"{r.train_s:.0f}", "infer ms/window": f"{r.infer_ms:.3f}", "seeds": int(r.n_seeds)})
    L.append(pd.DataFrame(tab).to_markdown(index=False) + "\n")

    # ---- earliness curve
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for fam, lab in (("xgboost", "XGBoost (content+history)"), (args.initial, "initial ANN (template LSTM)"), (args.final, "final ANN (semantic LSTM)"), ("confound_lr", "confound (time+volume)")):
        a = agg(fam)
        if len(a): ax[0].plot(a.index, a.pr_auc, "-o", label=lab); ax[1].plot(a.index, a.inc_recall_1, "-o", label=lab)
    tr = agg("trivial"); ax[0].plot(tr.index, tr.pr_auc, "k--", label="trivial (prevalence)")
    ax[0].set_xlabel("lead time N (min)"); ax[0].set_ylabel("PR-AUC"); ax[0].set_title(f"Earliness ({split})"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("lead time N (min)"); ax[1].set_ylabel("incident recall @ 1 % FAR"); ax[1].axvline(1.5, color="grey", ls=":", lw=1); ax[1].axvline(3, color="grey", ls=":", lw=1); ax[1].set_title("Operational: Desh thresholds 1.5 / 3 min dotted")
    fig.tight_layout(); fig.savefig(out_dir / "earliness.png", dpi=140); plt.close(fig); L.append(f"![](earliness.png)\n")

    # ---- per-category (final ANN, seed-averaged scores) and hard negatives, at the frozen 1 % threshold
    L.append("## Final ANN: per-category incident recall and hard-negative false-alarm rate (1 % FAR threshold)\n")
    cat_rows, hn_rows = [], []
    for N in NS:
        dfs = [scored[k] for k in scored if k[0].startswith(args.final) and k[1] == N]
        if not dfs: continue
        d = dfs[0].copy(); d["score"] = np.mean([x["score"].values for x in dfs], axis=0)
        thr = np.mean([thresholds[f"{args.final}_s{s}|N={N}"]["0.01"] for s in SEEDS if f"{args.final}_s{s}|N={N}" in thresholds])
        pos = d[d.label == 1]
        for cat, g in pos.groupby("category"):
            if not str(cat).strip(): continue
            inc = g.groupby("incident_id")["score"].max(); cat_rows.append({"N": N, "category": cat, "cell": f"{(inc > thr).mean():.2f} ({len(inc)})"})
        neg = d[d.label == 0]; hard = neg[neg.has_nontarget_alert_local]; ordinary = neg[~neg.has_nontarget_alert_local]
        hn_rows.append({"N": N, "ordinary negatives": len(ordinary), "FAR ordinary": round(float((ordinary.score > thr).mean()), 4), "hard negatives": len(hard), "FAR hard": round(float((hard.score > thr).mean()), 4) if len(hard) else None})
    if cat_rows: L.append("Cells: incident recall at the 1 % FAR threshold (number of incidents in this split).\n\n" + pd.DataFrame(cat_rows).pivot(index="category", columns="N", values="cell").fillna("—").to_markdown() + "\n")
    L.append("Hard negatives (alert-rich windows not followed by a failure) vs ordinary negatives:\n\n" + pd.DataFrame(hn_rows).to_markdown(index=False) + "\n")

    # ---- held-out category and machine-disjoint (if the retrained checkpoints exist), N=5
    N = 5; base_key = [k for k in scored if k[0] == f"{args.final}_s11" and k[1] == N]
    if base_key:
        base = scored[base_key[0]]; L.append("## Generalisation to UNSEEN failure types (leave-one-category-out, N=5, seed 11)\n")
        rows_c = []
        for cat in ("KERNMC", "KERNRTSP", "KERNSOCK", "KERNDTLB"):
            rid = f"{args.final}_no{cat}_N{N}_s11"
            if not (RESULTS / "checkpoints" / f"{rid}.pt").exists(): continue
            d, _ = score_nn(rid, split, N); negs = d[d.label == 0]; thr_c = threshold_at_far(negs.score.values, 0.01) if split == "validation" else thresholds.get(f"{rid}|N={N}", {}).get("0.01", threshold_at_far(negs.score.values, 0.01))
            if split == "validation": thresholds[f"{rid}|N={N}"] = {"0.01": thr_c}
            pos_c = d[(d.label == 1) & (d.category == cat)]; inc_c = pos_c.groupby("incident_id")["score"].max()
            thr_b = thresholds[f"{args.final}_s11|N={N}"]["0.01"]; pos_b = base[(base.label == 1) & (base.category == cat)]; inc_b = pos_b.groupby("incident_id")["score"].max()
            rows_c.append({"held-out category": cat, "incidents": len(inc_c), "incident recall@1% — model NEVER saw this category": round(float((inc_c > thr_c).mean()), 3), "incident recall@1% — all-category model": round(float((inc_b > thr_b).mean()), 3),
                           "PR-AUC (category positives vs all negatives) — held-out model": round(evaluate(np.r_[np.ones(len(pos_c)), np.zeros(len(negs))], np.r_[pos_c.score.values, negs.score.values])["pr_auc"], 4),
                           "PR-AUC — all-category model": round(evaluate(np.r_[np.ones(len(pos_b)), np.zeros(len(base[base.label == 0]))], np.r_[pos_b.score.values, base[base.label == 0].score.values])["pr_auc"], 4),
                           "ROC — held-out model": round(evaluate(np.r_[np.ones(len(pos_c)), np.zeros(len(negs))], np.r_[pos_c.score.values, negs.score.values])["roc_auc"], 3)})
        if rows_c: L.append(pd.DataFrame(rows_c).to_markdown(index=False) + "\n")
        rid = f"{args.final}_rackout_N{N}_s11"
        if (RESULTS / "checkpoints" / f"{rid}.pt").exists():
            d, _ = score_nn(rid, split, N); ck = torch.load(RESULTS / "checkpoints" / f"{rid}.pt", map_location="cpu", weights_only=False); held = set(ck.get("held_racks", []))
            d["held"] = d.unit.str.split("-").str[0].isin(held); L.append(f"## Generalisation to UNSEEN machines (racks held out of training: {sorted(held)}; N=5, seed 11)\n")
            rr = []; b = base.copy(); b["held"] = b.unit.str.split("-").str[0].isin(held)
            for model_name, dd in (("rack-held-out model", d), ("all-rack model (sem_final seed 11)", b)):
                for name, part in (("held-out racks (never in training of the held-out model)", dd[dd.held]), ("seen racks", dd[~dd.held])):
                    if (part.label == 1).sum() == 0: continue
                    r = evaluate(part.label.values, part.score.values, part.incident_id.values)
                    rr.append({"model": model_name, "windows from": name, "windows": len(part), "positives": int(part.label.sum()), "PR-AUC": round(r["pr_auc"], 4), "ROC-AUC": round(r["roc_auc"], 3), "incident recall@1%": round(r.get("incident_recall@far0.01", float("nan")), 3)})
            L.append(pd.DataFrame(rr).to_markdown(index=False) + "\n")

    # ---- protocol controls
    L.append("## Protocol controls (comparison only — never used for any model decision)\n")
    tc_rows = []
    for N in NS:                                               # transductive-template XGBoost on THIS split (the control that matters on test, where UNK = 61 %)
        p_tc = RESULTS / "models" / f"xgboost_N{N}_tc.joblib"
        if not (p_tc.exists() and (DATA_INTERIM / "features_tc.parquet").exists()): continue
        f = pd.read_parquet(DATA_INTERIM / "features_tc.parquet"); groups = json.loads((DATA_INTERIM / "feature_groups_tc.json").read_text())
        fh = pd.read_parquet(DATA_INTERIM / "features.parquet", columns=["split", f"label_{N}", "n_unk_global", "n_global"]); fh = fh[(fh.split == split) & (fh[f"label_{N}"] >= 0)]
        g = f[(f.split == split) & (f[f"label_{N}"] >= 0)]; sc = joblib.load(p_tc).predict_proba(g[groups["full"]].values.astype(np.float32))[:, 1]
        r_tc = evaluate(g[f"label_{N}"].values, sc, g["incident_id"].values); r_h = res[(res.model == "xgboost") & (res.N == N)].iloc[0]
        tc_rows.append({"N": N, "XGBoost, honest templates (train-only Drain): PR-AUC": round(r_h.pr_auc, 4), "ROC": round(r_h.roc_auc, 3), "XGBoost, transductive templates (Drain on all data): PR-AUC": round(r_tc["pr_auc"], 4), "ROC ": round(r_tc["roc_auc"], 3),
                        "UNK rate of this split under honest templates": round(float(fh["n_unk_global"].sum() / max(fh["n_global"].sum(), 1)), 3)})
    if tc_rows: L.append(f"Transductive-template control scored on the {split} split (same windows, features rebuilt from templates that have seen all data):\n\n" + pd.DataFrame(tc_rows).to_markdown(index=False) + "\n")
    ctl = []
    for tag, label in (("", "honest: chronological split, templates fitted on train only"), ("_randomsplit", "CONTROL: random window split (leaky protocol of most prior work)"), ("_tc", "CONTROL: transductive templates (Drain fitted on all data)")):
        p = RESULTS / f"baselines_validation{tag}.csv"
        if p.exists():
            b = pd.read_csv(p); best = b[b.model == "xgboost"].groupby("N")["pr_auc"].max(); roc = b[b.model == "xgboost"].sort_values("pr_auc").groupby("N").tail(1).set_index("N")["roc_auc"]
            for N in best.index: ctl.append({"protocol": label, "N": N, "XGBoost best PR-AUC (validation)": round(best[N], 4), "ROC-AUC": round(roc[N], 3)})
    if ctl: L.append("XGBoost best validation PR-AUC under each protocol (read from the Phase 3/6 result files):\n\n" + pd.DataFrame(ctl).pivot(index="protocol", columns="N", values="XGBoost best PR-AUC (validation)").to_markdown() + "\n")
    for tag, label in (("_rollA", "rolling origin A (cuts 50 % / 70 %)"), ("_rollB", "rolling origin B (cuts 65 % / 85 %)")):
        p = RESULTS / f"baselines_validation{tag}.csv"
        if p.exists():
            b = pd.read_csv(p); best = b[b.model == "xgboost"].groupby("N")["pr_auc"].max(); L.append(f"Rolling origin — {label}: XGBoost best validation PR-AUC by N: " + ", ".join(f"N={N}: {v:.4f}" for N, v in best.items()) + "\n")

    (out_dir / f"conditions_{split}.md").write_text("\n".join(L), encoding="utf-8")
    if split == "validation":
        thr_path.write_text(json.dumps(thresholds, indent=1)); print(f"[eval] thresholds frozen → {thr_path}")
    else:
        (out_dir / "final_test_results.json").write_text(json.dumps({"protocol": protocol, "rows": rows, "thresholds_used": thresholds}, indent=1, default=str))
        for r in rows:
            log_run(run_id=f"TEST_{r['model']}_N{r['N']}", phase=7, model_family=r["family"], purpose="FINAL TEST EVALUATION (single run)", changed_vs_previous="none — frozen models and thresholds",
                    key_hyperparams=f"N={r['N']};model={r['model']}", seed=r["seed"] or 0, split="test", val_pr_auc=round(r["pr_auc"], 4), val_recall_at_far=round(r["recall@far0.01"], 4), val_f1=round(r["f1@far0.01"], 4),
                    infer_ms_per_window=r["infer_ms_per_window"], outcome="success", learned=f"auto: test PR-AUC {r['pr_auc']:.4f}, ROC {r['roc_auc']:.3f}, incident recall@1% {r.get('incident_recall@far0.01', float('nan')):.3f}")
        print(f"[eval] TEST RESULTS WRITTEN → {out_dir}. Commit them now: git add results/test_eval && git commit -m 'Phase 7: final test evaluation (single run)'")
    print(f"[eval] → {out_dir / f'conditions_{split}.md'}")


if __name__ == "__main__":
    main()
