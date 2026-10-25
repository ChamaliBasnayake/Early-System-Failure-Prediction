"""Phase 3 — baselines, evaluated on VALIDATION only, every configuration logged.

  trivial    : constant score (base rate) → PR-AUC equals prevalence, recall at any FAR = 0 (reference row)
  confound   : logistic regression on TIME + VOLUME features only (no message content). If a content
               model cannot beat this clearly, it has learned schedule, not symptoms.
  xgboost    : gradient boosting on the FULL feature set (content + failure history); 25 random
               configurations from a fixed search space (same budget as the neural family); selected on
               validation PR-AUC. The best configuration is then re-fitted on two ABLATION feature sets —
               content only and history only — to measure how much log CONTENT adds beyond the known
               clustering of failures in space and time (Q3 error analysis / Q2.b expectation).
For each lead time N the best model is saved to results/models/, metrics to results/baselines_validation.csv,
and thresholds at the FAR levels are frozen in results/thresholds_baselines.json for Phase 7.
Usage: python -m src.models.baselines [--sample] [--n_configs 25] [--N 5] [--tag system] [--tag tc] [--random-split-control]
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from src.utils.metrics import evaluate, flat
from src.utils.paths import CONFIGS, DATA_INTERIM, RESULTS
from src.utils.runlog import log_run
from src.utils.seeds import set_seed

SEARCH_SPACE = {"max_depth": [3, 5, 7], "learning_rate": [0.03, 0.1, 0.3], "n_estimators": [200, 500],
                "subsample": [0.7, 1.0], "colsample_bytree": [0.5, 1.0], "min_child_weight": [1, 5], "scale_pos_weight": ["ratio", 1.0]}


def sample_configs(n: int, rng: np.random.Generator) -> list[dict]:
    seen, out = set(), []
    while len(out) < n:
        c = {k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()}
        key = tuple(sorted((k, str(v)) for k, v in c.items()))
        if key not in seen: seen.add(key); out.append(c)
    return out


def load(sfx: str, random_split_seed: int | None = None):
    f = pd.read_parquet(DATA_INTERIM / f"features{sfx}.parquet"); groups = json.loads((DATA_INTERIM / f"feature_groups{sfx}.json").read_text())
    if random_split_seed is not None:
        # PROTOCOL CONTROL ONLY: re-assign windows to train/validation/test at random (the leaky protocol of most
        # prior work), keeping the same split sizes. Never used for any model decision.
        rng = np.random.default_rng(random_split_seed); sizes = f["split"].value_counts()
        perm = rng.permutation(len(f)); lab = np.empty(len(f), dtype=object); i = 0
        for k in ("train", "validation", "test"):
            lab[perm[i:i + sizes[k]]] = k; i += sizes[k]
        f = f.assign(split=lab)
    return f, groups


def split_xy(f: pd.DataFrame, cols: list[str], N: int, split: str):
    g = f[(f.split == split) & (f[f"label_{N}"] >= 0)]
    return g[cols].values.astype(np.float32), g[f"label_{N}"].values.astype(int), g["incident_id"].values, g


OUTCOME = "success"


def run_one(name: str, model, Xtr, ytr, Xva, yva, inc_va, N: int, cfg_desc: str, seed: int, phase: int = 3) -> tuple[dict, float, float]:
    t0 = time.time(); model.fit(Xtr, ytr); train_s = time.time() - t0
    t1 = time.time(); s = model.predict_proba(Xva)[:, 1]; infer_ms = (time.time() - t1) / max(len(Xva), 1) * 1000
    res = evaluate(yva, s, inc_va)
    log_run(phase=phase, model_family=name, purpose=f"baseline {name}, N={N}", changed_vs_previous=cfg_desc, config_path="src/models/baselines.py",
            key_hyperparams=f"N={N};{cfg_desc}", seed=seed, split="validation", val_pr_auc=round(res["pr_auc"], 4),
            val_recall_at_far=round(res["recall@far0.01"], 4), val_f1=round(res["f1@far0.01"], 4), train_time_s=round(train_s, 1),
            infer_ms_per_window=round(infer_ms, 4), outcome=OUTCOME,
            learned=f"auto: val PR-AUC {res['pr_auc']:.4f}, recall@1%FAR {res['recall@far0.01']:.3f}, incident recall {res.get('incident_recall@far0.01', float('nan')):.3f}")
    return res, train_s, infer_ms


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", action="store_true"); ap.add_argument("--n_configs", type=int, default=25)
    ap.add_argument("--N", type=int, nargs="*", default=None); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", default="", help="features tag (system, tc, …); results are written with the same tag")
    ap.add_argument("--random-split-control", action="store_true", help="PROTOCOL CONTROL: random window split instead of chronological")
    args = ap.parse_args(); sfx = "_sample" if args.sample else ""; set_seed(args.seed); rng = np.random.default_rng(args.seed)
    in_sfx = sfx + (f"_{args.tag}" if args.tag else "")
    f, groups = load(in_sfx, random_split_seed=args.seed if args.random_split_control else None)
    sfx = in_sfx + ("_randomsplit" if args.random_split_control else "")       # output suffix
    global OUTCOME
    if args.random_split_control or args.tag == "tc":
        OUTCOME = "control"; print(f"[baselines] *** PROTOCOL CONTROL ({'random split' if args.random_split_control else 'transductive templates'}) — comparison only ***")
    Ns = args.N or yaml.safe_load((CONFIGS / "windows.yaml").read_text())["lead_times_min"]
    rows, thresholds, best_cfgs = [], {}, {}
    (RESULTS / "models").mkdir(exist_ok=True)
    for N in Ns:
        Xtr, ytr, _, _ = split_xy(f, groups["full"], N, "train"); Xva, yva, inc_va, _ = split_xy(f, groups["full"], N, "validation")
        print(f"\n[N={N}] train {len(ytr):,} windows ({ytr.sum()} pos) | validation {len(yva):,} ({yva.sum()} pos)")
        # trivial
        s = np.full(len(yva), ytr.mean()); res = evaluate(yva, s, inc_va)
        rows.append({"N": N, "model": "trivial", "config": "constant=base rate", **flat(res), "train_time_s": 0.0, "infer_ms_per_window": 0.0})
        log_run(phase=3, model_family="trivial", purpose=f"reference, N={N}", changed_vs_previous="constant score", key_hyperparams=f"N={N}", seed=args.seed,
                split="validation", val_pr_auc=round(res["pr_auc"], 4), val_recall_at_far=0, val_f1=0, outcome="control", learned="PR-AUC of a constant equals prevalence")
        # confound
        ci = [groups["full"].index(c) for c in groups["confound"]]
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        res, ts, im = run_one("confound_lr", lr, Xtr[:, ci], ytr, Xva[:, ci], yva, inc_va, N, "time+volume features only", args.seed)
        rows.append({"N": N, "model": "confound_lr", "config": "time+volume", **flat(res), "train_time_s": ts, "infer_ms_per_window": im})
        joblib.dump(lr, RESULTS / "models" / f"confound_lr_N{N}{sfx}.joblib")
        print(f"  trivial PR-AUC {rows[-2]['pr_auc']:.4f} | confound PR-AUC {res['pr_auc']:.4f}, recall@1%FAR {res['recall@far0.01']:.3f}")
        # xgboost search
        ratio = float((ytr == 0).sum() / max(ytr.sum(), 1)); best = None
        for i, c in enumerate(sample_configs(args.n_configs, rng)):
            cc = dict(c); cc["scale_pos_weight"] = ratio if cc["scale_pos_weight"] == "ratio" else 1.0
            model = XGBClassifier(**{k: (float(v) if k in ("learning_rate", "subsample", "colsample_bytree", "scale_pos_weight") else int(v)) for k, v in cc.items()},
                                  objective="binary:logistic", eval_metric="aucpr", tree_method="hist", n_jobs=0, random_state=args.seed, verbosity=0)
            desc = ";".join(f"{k}={cc[k] if k != 'scale_pos_weight' else ('ratio' if c[k] == 'ratio' else 1)}" for k in SEARCH_SPACE)
            res, ts, im = run_one("xgboost", model, Xtr, ytr, Xva, yva, inc_va, N, desc, args.seed)
            rows.append({"N": N, "model": "xgboost", "config": desc, **flat(res), "train_time_s": ts, "infer_ms_per_window": im})
            print(f"  xgb {i + 1:2d}/{args.n_configs}: PR-AUC {res['pr_auc']:.4f}  recall@1% {res['recall@far0.01']:.3f}  ({ts:.0f}s)  {desc}")
            if best is None or res["pr_auc"] > best[0]: best = (res["pr_auc"], model, desc, res)
        joblib.dump(best[1], RESULTS / "models" / f"xgboost_N{N}{sfx}.joblib"); best_cfgs[str(N)] = best[2]; thresholds[str(N)] = best[3]["thresholds"]
        print(f"  BEST xgb N={N}: PR-AUC {best[0]:.4f}  config {best[2]}")
        # ablations with the best configuration: content only / history only
        for abl in ("content", "history_only"):
            ci = [groups["full"].index(c) for c in groups[abl]]
            model = XGBClassifier(**best[1].get_params())
            res, ts, im = run_one(f"xgboost_{abl}", model, Xtr[:, ci], ytr, Xva[:, ci], yva, inc_va, N, f"best config; features={abl}", args.seed)
            rows.append({"N": N, "model": f"xgboost_{abl}", "config": f"best config; features={abl}", **flat(res), "train_time_s": ts, "infer_ms_per_window": im})
            joblib.dump(model, RESULTS / "models" / f"xgboost_{abl}_N{N}{sfx}.joblib")
            print(f"  ablation {abl:13s}: PR-AUC {res['pr_auc']:.4f}  ROC-AUC {res['roc_auc']:.3f}  recall@1% {res['recall@far0.01']:.3f}  incident recall@1% {res.get('incident_recall@far0.01', float('nan')):.3f}")
    df = pd.DataFrame(rows); df.to_csv(RESULTS / f"baselines_validation{sfx}.csv", index=False)
    (RESULTS / f"thresholds_baselines{sfx}.json").write_text(json.dumps({"xgboost": thresholds, "best_configs": best_cfgs, "note": "chosen on validation; frozen for Phase 7"}, indent=2))
    summ = df.groupby(["N", "model"]).agg(best_pr_auc=("pr_auc", "max"), best_roc_auc=("roc_auc", "max"), median_pr_auc=("pr_auc", "median"),
                                          recall_1pct=("recall@far0.01", "max"), n_configs=("config", "size")).reset_index()
    print("\n" + summ.to_string(index=False)); print(f"[baselines] → {RESULTS / f'baselines_validation{sfx}.csv'}")


if __name__ == "__main__":
    main()
