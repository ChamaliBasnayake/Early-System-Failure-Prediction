"""Phase 4/5 — train a sequence model from a YAML config; evaluate on VALIDATION every epoch.

Saves: results/checkpoints/<name>_N<N>_s<seed>.pt, results/curves/<run_id>.json (per-epoch train loss,
validation loss, validation PR-AUC), results/scores/<name>_N<N>_s<seed>_validation.parquet (per-window
scores for error analysis / XAI), and one experiment-log row with what was changed and what was learned.
Usage: python -m src.models.train --config configs/lstm_v1.yaml [--seed 22] [--N 5] [--tag w24h]
       [--set train.lr=0.0003 --set model.hidden=128]   (override any config key; recorded in the log)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.models.lstm import TwoChannelLSTM, count_parameters
from src.models.seq_data import WindowDataset, collate, load_windows
from src.utils.metrics import evaluate
from src.utils.paths import DATA_INTERIM, RESULTS
from src.utils.runlog import log_run
from src.utils.seeds import set_seed


def build_model(cfg: dict, vocab: int, n_scalars: int) -> nn.Module:
    m = cfg["model"]
    if m["type"] in ("lstm",):
        return TwoChannelLSTM(vocab, n_scalars, m["embed_dim"], m["hidden"], m["layers"], m["dropout"], m["layernorm"], m["pooling"])
    if m["type"] == "cnn":
        from src.models.cnn import TwoChannelCNN
        return TwoChannelCNN(vocab, n_scalars, m["embed_dim"], m.get("filters", 64), m.get("kernels", [3, 5, 7]), m["dropout"], m["layernorm"])
    if m["type"] == "lstm_semantic":
        from src.models.semantic import TwoChannelSemanticLSTM
        return TwoChannelSemanticLSTM(vocab, n_scalars, m["hidden"], m["layers"], m["dropout"], m["layernorm"], m["pooling"], m.get("proj_dim", 64), m["embedding_file"])
    raise ValueError(m["type"])


def make_optimizer(cfg: dict, params):
    t = cfg["train"]
    if t["optimizer"] == "adam": return torch.optim.Adam(params, lr=t["lr"], weight_decay=t["weight_decay"])
    if t["optimizer"] == "adamw": return torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    if t["optimizer"] == "sgd": return torch.optim.SGD(params, lr=t["lr"], momentum=t["momentum"], weight_decay=t["weight_decay"])
    raise ValueError(t["optimizer"])


def run_epoch(model, loader, loss_fn, opt=None):
    train = opt is not None; model.train(train); tot, n, scores, ys = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for loc, glo, S, y in loader:
            logit = model(loc, glo, S); loss = loss_fn(logit, y)
            if train: opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(y); n += len(y); scores.append(torch.sigmoid(logit).detach().numpy()); ys.append(y.numpy())
    return tot / max(n, 1), np.concatenate(scores), np.concatenate(ys)


def apply_overrides(cfg: dict, sets: list[str]) -> dict:
    for s in sets or []:
        k, v = s.split("=", 1); d = cfg
        parts = k.split(".")
        for p in parts[:-1]: d = d[p]
        old = d[parts[-1]]; d[parts[-1]] = type(old)(v) if not isinstance(old, bool) else v.lower() in ("1", "true", "yes")
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--N", type=int, default=None); ap.add_argument("--tag", default=""); ap.add_argument("--sample", action="store_true")
    ap.add_argument("--set", action="append", default=[]); ap.add_argument("--purpose", default=""); ap.add_argument("--changed", default="")
    ap.add_argument("--phase", type=int, default=4); args = ap.parse_args()
    cfg = apply_overrides(yaml.safe_load(Path(args.config).read_text()), args.set)
    if args.seed is not None: cfg["train"]["seed"] = args.seed
    if args.N is not None: cfg["train"]["N"] = args.N
    t = cfg["train"]; N, seed = t["N"], t["seed"]; set_seed(seed); torch.set_num_threads(max(1, torch.get_num_threads()))
    sfx = ("_sample" if args.sample else "") + (f"_{args.tag}" if args.tag else "")
    name = cfg["name"] + (f"_{args.tag}" if args.tag else "")

    tr_df, tr_X, cols = load_windows(sfx, N, "train", cfg["model"]["scalar_groups"]); va_df, va_X, _ = load_windows(sfx, N, "validation", cfg["model"]["scalar_groups"])
    tr_ds = WindowDataset(tr_df, tr_X, N, t["max_local"], t["max_global"]); va_ds = WindowDataset(va_df, va_X, N, t["max_local"], t["max_global"], scaler=tr_ds.scaler)
    g = torch.Generator(); g.manual_seed(seed)
    tr_dl = DataLoader(tr_ds, batch_size=t["batch_size"], shuffle=True, collate_fn=collate, num_workers=t["num_workers"], generator=g)
    va_dl = DataLoader(va_ds, batch_size=512, shuffle=False, collate_fn=collate, num_workers=t["num_workers"])
    vocab = int(pd.read_csv(DATA_INTERIM / ("templates_sample.csv" if args.sample else "templates.csv"))["template_index"].max()) + 1
    model = build_model(cfg, vocab, tr_X.shape[1]); n_params = count_parameters(model)
    n_pos, n_neg = float(tr_ds.y.sum()), float(len(tr_ds.y) - tr_ds.y.sum())
    pw = {"ratio": n_neg / n_pos, "sqrt": (n_neg / n_pos) ** 0.5, "1": 1.0}[str(t["pos_weight"])]
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw)); opt = make_optimizer(cfg, model.parameters())
    print(f"[train] {name} N={N} seed={seed} | train {len(tr_ds):,} ({int(n_pos)} pos) val {len(va_ds):,} | params {n_params:,} | pos_weight {pw:.2f} | scalars {cols}")

    curves, best, best_state, bad, t0 = [], -1.0, None, 0, time.time()
    for ep in range(1, t["epochs"] + 1):
        tl, _, _ = run_epoch(model, tr_dl, loss_fn, opt)
        vl, vs, vy = run_epoch(model, va_dl, loss_fn)
        res = evaluate(vy, vs, va_ds.incident); curves.append({"epoch": ep, "train_loss": tl, "val_loss": vl, "val_pr_auc": res["pr_auc"], "val_roc_auc": res["roc_auc"]})
        print(f"  ep {ep:2d}  train loss {tl:.4f}  val loss {vl:.4f}  val PR-AUC {res['pr_auc']:.4f}  ROC {res['roc_auc']:.3f}  recall@1% {res['recall@far0.01']:.3f}  ({time.time() - t0:.0f}s)")
        if res["pr_auc"] > best: best, best_state, bad, best_res, best_scores = res["pr_auc"], {k: v.clone() for k, v in model.state_dict().items()}, 0, res, vs
        else:
            bad += 1
            if bad >= t["patience"]: print(f"  early stop at epoch {ep}"); break
    train_s = time.time() - t0; model.load_state_dict(best_state)
    t1 = time.time(); _ = run_epoch(model, va_dl, loss_fn); infer_ms = (time.time() - t1) / len(va_ds) * 1000

    run_id = f"{name}_N{N}_s{seed}"
    for d in ("checkpoints", "curves", "scores"): (RESULTS / d).mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "config": cfg, "vocab": vocab, "n_scalars": tr_X.shape[1], "scalar_cols": cols, "scaler": tr_ds.scaler, "n_params": n_params}, RESULTS / "checkpoints" / f"{run_id}.pt")
    (RESULTS / "curves" / f"{run_id}.json").write_text(json.dumps({"run_id": run_id, "config": cfg, "curves": curves, "best": {k: v for k, v in best_res.items() if not isinstance(v, dict)}}, indent=1))
    pd.DataFrame({"unit": va_df["unit"], "t_ns": va_df["t_ns"], "label": vy, "score": best_scores, "incident_id": va_df["incident_id"], "anchor": va_df["anchor"]}).to_parquet(RESULTS / "scores" / f"{run_id}_validation.parquet", index=False)
    hp = f"N={N};type={cfg['model']['type']};emb={cfg['model']['embed_dim']};hidden={cfg['model']['hidden']};layers={cfg['model']['layers']};drop={cfg['model']['dropout']};pool={cfg['model']['pooling']};opt={t['optimizer']};lr={t['lr']};bs={t['batch_size']};wd={t['weight_decay']};posw={t['pos_weight']};Wl={t['max_local']}"
    best_ep = max(curves, key=lambda c: c["val_pr_auc"])["epoch"]
    log_run(run_id=run_id, phase=args.phase, model_family=cfg["model"]["type"], purpose=args.purpose or f"{name} N={N}", changed_vs_previous=args.changed or ("; ".join(args.set) if args.set else "config as file"),
            config_path=args.config, key_hyperparams=hp, seed=seed, split="validation", val_pr_auc=round(best_res["pr_auc"], 4),
            val_recall_at_far=round(best_res["recall@far0.01"], 4), val_f1=round(best_res["f1@far0.01"], 4), train_time_s=round(train_s, 1), infer_ms_per_window=round(infer_ms, 4),
            outcome="success", learned=f"auto: best epoch {best_ep}/{len(curves)}, val PR-AUC {best_res['pr_auc']:.4f}, ROC {best_res['roc_auc']:.3f}, params {n_params:,}; curves in results/curves/{run_id}.json")
    print(f"[train] best val PR-AUC {best:.4f} at epoch {best_ep}; checkpoint + curves + scores saved as {run_id}")


if __name__ == "__main__":
    main()
