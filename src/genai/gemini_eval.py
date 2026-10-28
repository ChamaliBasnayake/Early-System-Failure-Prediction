"""Phase 10 (validation: prompt selection) / Phase 7 (test, ONCE) — Gemini as a zero-/few-shot failure predictor.

Protocol
  * A fixed, seeded, balanced sample of windows per split (results/genai/sample_<split>.parquet): up to 150 positives
    stratified by failure category + 100 uniform negatives + 50 hard negatives, at N=5 by default.
  * Every window is rendered as text from the SAME two channels the ANN sees (masked/templated lines, repeat counts,
    relative times); nothing the ANN cannot see is given to the LLM, and nothing the LLM sees is hidden from the ANN.
  * Four prompt designs (src/genai/prompts/): zero-shot, domain-context, few-shot (examples from TRAIN only), structured.
    Temperature 0, JSON output {"probability", "reason"}. Every response is cached (results/genai/cache/) so re-runs are
    free and the test run is reproducible; model name, date, tokens and latency are recorded.
  * The final ANN (mean of 3 seeds) and the best XGBoost are scored on the IDENTICAL sample so the comparison is fair.
  * Prompt selection uses --split validation only. --split test runs ONLY the prompt named in
    results/genai/selected_prompt.json, and refuses to run twice.
Key: environment variable GEMINI_API_KEY (never in code or config).
Usage:
  python -m src.genai.gemini_eval --split validation                 # all prompts, 300 windows
  python -m src.genai.gemini_eval --split validation --prompts p1 --limit 20   # cheap dry run first
  python -m src.genai.gemini_eval --select p2                        # freeze the chosen prompt (after reading the report)
  python -m src.genai.gemini_eval --split test                       # Phase 7 only
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.evaluate_conditions import load_run
from src.models.seq_data import WindowDataset, collate, load_windows
from src.utils.metrics import evaluate, flat
from src.utils.paths import DATA_INTERIM, RESULTS
from src.utils.runlog import log_run

MODEL_NAME = "models/gemini-2.5-flash"
PROMPT_DIR = Path(__file__).parent / "prompts"
OUT = RESULTS / "genai"; CACHE = OUT / "cache"
H = 10


# ----------------------------------------------------------------------------- sample
def make_sample(split: str, N: int, seed: int, n_pos=150, n_uni=100, n_hard=50) -> pd.DataFrame:
    p = OUT / f"sample_{split}_N{N}.parquet"
    if p.exists(): return pd.read_parquet(p)
    win = pd.read_parquet(DATA_INTERIM / "windows.parquet"); w = win[(win.split == split) & (win[f"label_{N}"] >= 0)]
    rng = np.random.default_rng(seed)
    pos = w[w[f"label_{N}"] == 1]; cats = pos.category.value_counts()
    quota = {c: max(5, int(round(n_pos * n / len(pos)))) for c, n in cats.items()}
    picks = [g.sample(min(len(g), quota[c]), random_state=int(rng.integers(1e9))) for c, g in pos.groupby("category")]
    pos_s = pd.concat(picks); pos_s = pos_s.sample(min(len(pos_s), n_pos), random_state=seed) if len(pos_s) > n_pos else pos_s
    neg = w[w[f"label_{N}"] == 0]; hard = neg[neg.has_nontarget_alert_local]; uni = neg[~neg.has_nontarget_alert_local]
    s = pd.concat([pos_s, uni.sample(min(n_uni, len(uni)), random_state=seed), hard.sample(min(n_hard, len(hard)), random_state=seed)])
    s = s.sample(frac=1, random_state=seed).reset_index(drop=True); s["label"] = s[f"label_{N}"]
    s["key"] = s["unit"].astype(str) + "|" + s["t_ns"].astype(str); s.to_parquet(p, index=False); return s


# ----------------------------------------------------------------------------- rendering
class Renderer:
    def __init__(self, max_lines: int = 60):
        mm = pd.read_parquet(DATA_INTERIM / "masked_messages.parquet"); self.text = dict(zip(mm.masked_id.astype(int), mm.text)); self.max_lines = max_lines

    def channel(self, mids, cnts, dts) -> str:
        mids, cnts, dts = list(mids)[-self.max_lines:], list(cnts)[-self.max_lines:], list(dts)[-self.max_lines:]
        if not mids: return "(no log lines in this hour)"
        t = np.cumsum([0.0] + [float(x) for x in dts[1:]]); out = []
        for m, c, ti in zip(mids, cnts, t):
            out.append(f"+{int(ti)}s  {self.text.get(int(m), '<unknown message type>')}" + (f"   x{int(c)}" if int(c) > 1 else ""))
        return "\n".join(out)

    def fill(self, template: str, row: pd.Series, N: int, examples: str = "") -> str:
        end = pd.Timestamp(row.window_end)
        return template.format(N=N, N_plus_H=N + H, max_lines=self.max_lines, examples=examples,
                               local=self.channel(row.mid_local, row.cnt_local, row.dt_local), **{"global": self.channel(row.mid_global, row.cnt_global, row.dt_global)},
                               n_local=row.n_local, k_local=len(set(row.mid_local)), n_unk_local=row.n_unk_local, n_global=row.n_global, n_units_active_global=row.n_units_active_global,
                               n_nontarget_alerts_global=row.n_nontarget_alerts_global, has_nontarget_alert_local="yes" if row.has_nontarget_alert_local else "no", hour=end.hour, dow=end.day_name())

    def examples(self, N: int, seed: int, k_each: int = 2) -> str:
        win = pd.read_parquet(DATA_INTERIM / "windows.parquet"); tr = win[(win.split == "train") & (win[f"label_{N}"] >= 0)]
        ex = pd.concat([tr[(tr[f"label_{N}"] == 1) & (tr.n_local > 0)].sample(k_each, random_state=seed), tr[(tr[f"label_{N}"] == 0) & (tr.n_local > 0)].sample(k_each, random_state=seed)]).sample(frac=1, random_state=seed)
        old = self.max_lines; self.max_lines = 15; parts = []
        for i, (_, r) in enumerate(ex.iterrows(), 1):
            lab = f"YES — a {r.category} failure started {N}–{N + H} min later" if r[f"label_{N}"] == 1 else "NO — no kernel failure followed"
            parts.append(f"--- Example {i} (midplane lines) ---\n{self.channel(r.mid_local, r.cnt_local, r.dt_local)}\n--- Example {i} (machine-wide, most recent 15) ---\n{self.channel(r.mid_global, r.cnt_global, r.dt_global)}\nAnswer: {lab}\n")
        self.max_lines = old; return "\n".join(parts)


# ----------------------------------------------------------------------------- Gemini
class Gemini:
    def __init__(self, rpm: int):
        import google.generativeai as genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key: raise SystemExit("GEMINI_API_KEY is not set (user environment variable). Open a NEW terminal after setting it.")
        genai.configure(api_key=key); self.min_gap = 60.0 / max(rpm, 1); self.last = 0.0
        self.model = genai.GenerativeModel(MODEL_NAME, generation_config={"temperature": 0.0, "response_mime_type": "application/json", "max_output_tokens": 4096})

    def ask(self, prompt: str) -> dict:
        for attempt in range(6):
            wait = self.min_gap - (time.time() - self.last)
            if wait > 0: time.sleep(wait)
            try:
                t0 = time.time(); r = self.model.generate_content(prompt); self.last = time.time()
                u = getattr(r, "usage_metadata", None); txt = r.text if r.candidates else ""
                return {"text": txt, "latency_s": round(time.time() - t0, 3), "in_tokens": getattr(u, "prompt_token_count", None), "out_tokens": getattr(u, "candidates_token_count", None),
                        "thinking_tokens": getattr(u, "thoughts_token_count", None), "model": MODEL_NAME, "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
            except Exception as e:  # rate limit / transient
                msg = str(e); self.last = time.time()
                if attempt == 5: return {"text": "", "error": msg[:300], "latency_s": None, "in_tokens": None, "out_tokens": None, "thinking_tokens": None, "model": MODEL_NAME, "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
                time.sleep(min(60, 5 * 2 ** attempt)); print(f"    retry {attempt + 1}: {msg[:80]}")


def parse_prob(text: str) -> tuple[float, str]:
    try:
        j = json.loads(text); p = float(j.get("probability")); return min(max(p, 0.0), 1.0), str(j.get("reason", ""))[:300]
    except Exception:
        m = re.search(r'"?probability"?\s*[:=]\s*([01](?:\.\d+)?|\.\d+)', text)
        return (min(max(float(m.group(1)), 0.0), 1.0) if m else float("nan")), text[:300].replace("\n", " ")


# ----------------------------------------------------------------------------- reference models on the same sample
def ann_scores(sample: pd.DataFrame, N: int, run_prefix="sem_final", seeds=(11, 22, 33)) -> np.ndarray:
    out = []
    for s in seeds:
        rid = f"{run_prefix}_N{N}_s{s}"
        if not (RESULTS / "checkpoints" / f"{rid}.pt").exists(): continue
        model, ck = load_run(rid); df, X, _ = load_windows("", N, sample.split.iloc[0], ck["config"]["model"]["scalar_groups"])
        df["key"] = df["unit"].astype(str) + "|" + df["t_ns"].astype(str); idx = df.set_index("key").index.get_indexer(sample.key)
        assert (idx >= 0).all(), "sample windows not found in windows table"
        ds = WindowDataset(df.iloc[idx].reset_index(drop=True), X[idx], N, 256, 256, scaler=ck["scaler"]); dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate); sc = []
        with torch.no_grad():
            for loc, glo, S, y in dl: sc.append(torch.sigmoid(model(loc, glo, S)).numpy())
        out.append(np.concatenate(sc))
    return np.mean(out, axis=0) if out else np.full(len(sample), np.nan)


def xgb_scores(sample: pd.DataFrame, N: int) -> np.ndarray:
    p = RESULTS / "models" / f"xgboost_N{N}.joblib"
    if not p.exists(): return np.full(len(sample), np.nan)
    f = pd.read_parquet(DATA_INTERIM / "features.parquet"); groups = json.loads((DATA_INTERIM / "feature_groups.json").read_text())
    f["key"] = f["unit"].astype(str) + "|" + f["t_ns"].astype(str); g = f.set_index("key").loc[sample.key]
    return joblib.load(p).predict_proba(g[groups["full"]].values.astype(np.float32))[:, 1]


def metrics_on_sample(y, s, inc) -> dict:
    ok = ~np.isnan(s); y, s, inc = np.asarray(y)[ok], np.asarray(s)[ok], np.asarray(inc)[ok]
    r = evaluate(y, s, inc, far_levels=(0.01, 0.10)); r["n_scored"] = int(ok.sum())
    pred = s >= 0.5; r["balanced_acc@0.5"] = 0.5 * ((pred & (y == 1)).sum() / max((y == 1).sum(), 1) + (~pred & (y == 0)).sum() / max((y == 0).sum(), 1))
    return r


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=["validation", "test"]); ap.add_argument("--N", type=int, default=5); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--prompts", nargs="*", default=None, help="prompt ids, e.g. p1 p2 (default: all on validation; the selected one on test)")
    ap.add_argument("--limit", type=int, default=None, help="only the first K sample windows (dry run / budget)"); ap.add_argument("--rpm", type=int, default=10, help="requests per minute (free tier: 10)")
    ap.add_argument("--max-lines", type=int, default=60); ap.add_argument("--select", default=None, help="freeze the chosen prompt id for the test run and exit")
    args = ap.parse_args(); OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True); sel_path = OUT / "selected_prompt.json"
    if args.select:
        sel_path.write_text(json.dumps({"prompt": args.select, "selected_on": "validation", "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "N": args.N})); print(f"selected prompt {args.select} → {sel_path}"); return
    if not args.split: raise SystemExit("--split is required")
    split, N = args.split, args.N
    all_prompts = sorted(p.stem for p in PROMPT_DIR.glob("p*.txt")); short = {p.split("_")[0]: p for p in all_prompts}
    if split == "test":
        if not sel_path.exists(): raise SystemExit("No selected prompt. Run validation, read the report, then: python -m src.genai.gemini_eval --select pX")
        if (OUT / f"gemini_test_N{N}.md").exists(): raise SystemExit("Test evaluation already exists — the test set is evaluated ONCE.")
        prompts = [short[json.loads(sel_path.read_text())["prompt"]]]
    else:
        prompts = [short[p] if p in short else p for p in (args.prompts or list(short))]
    sample = make_sample(split, N, args.seed)
    if args.limit: sample = sample.head(args.limit)
    print(f"[gemini] {split} N={N}: {len(sample)} windows ({int(sample.label.sum())} pos, {int((sample.label == 0).sum())} neg, {int(sample.has_nontarget_alert_local[sample.label == 0].sum())} hard neg); prompts {prompts}")
    rend = Renderer(args.max_lines); examples = rend.examples(N, args.seed); gem = Gemini(args.rpm)
    rows_summary = []; scores = {}
    for pid in prompts:
        tpl = (PROMPT_DIR / f"{pid}.txt").read_text(encoding="utf-8"); recs = []
        for i, (_, r) in enumerate(sample.iterrows()):
            prompt = rend.fill(tpl, r, N, examples); h = hashlib.md5((MODEL_NAME + pid + prompt).encode()).hexdigest(); cp = CACHE / f"{h}.json"
            if cp.exists(): resp = json.loads(cp.read_text(encoding="utf-8"))
            else:
                resp = gem.ask(prompt); resp["prompt_chars"] = len(prompt); cp.write_text(json.dumps(resp), encoding="utf-8")
            prob, reason = parse_prob(resp.get("text", ""))
            recs.append({"key": r.key, "unit": r.unit, "window_end": str(r.window_end), "label": int(r.label), "category": r.category, "incident_id": r.incident_id, "hard_negative": bool(r.has_nontarget_alert_local and r.label == 0),
                         "prompt": pid, "probability": prob, "reason": reason, "error": resp.get("error", ""), **{k: resp.get(k) for k in ("latency_s", "in_tokens", "out_tokens", "thinking_tokens", "model", "date")}})
            if (i + 1) % 25 == 0: print(f"  {pid}: {i + 1}/{len(sample)}  (cached+new)")
        d = pd.DataFrame(recs); d.to_csv(OUT / f"gemini_{split}_N{N}_{pid}.csv", index=False); scores[pid] = d.probability.values
        m = metrics_on_sample(d.label, d.probability, d.incident_id)
        rows_summary.append({"model": f"Gemini 2.5 Flash — {pid}", **{k: v for k, v in flat(m).items()}, "unparsable": int(d.probability.isna().sum()), "errors": int((d.error != "").sum()),
                             "mean latency s": round(d.latency_s.dropna().mean(), 2) if d.latency_s.notna().any() else None, "in tokens": int(d.in_tokens.dropna().sum()), "out tokens": int(d.out_tokens.dropna().sum()), "thinking tokens": int(d.thinking_tokens.dropna().sum()) if d.thinking_tokens.notna().any() else 0})
        print(f"  {pid}: ROC {m['roc_auc']:.3f}  PR-AUC(sample) {m['pr_auc']:.3f}  recall@10%FAR {m['recall@far0.1']:.3f}  bal.acc {m['balanced_acc@0.5']:.3f}  unparsable {int(d.probability.isna().sum())}")
    # reference models on the identical windows
    for name, sc in (("final ANN (semantic LSTM, mean of 3 seeds)", ann_scores(sample, N)), ("XGBoost (best validation config)", xgb_scores(sample, N))):
        if np.isnan(sc).all(): continue
        m = metrics_on_sample(sample.label, sc, sample.incident_id); rows_summary.append({"model": name, **flat(m)}); scores[name] = sc
    summ = pd.DataFrame(rows_summary); summ.to_csv(OUT / f"gemini_{split}_N{N}_summary.csv", index=False)
    keep = ["model", "n_scored", "roc_auc", "pr_auc", "recall@far0.01", "recall@far0.1", "incident_recall@far0.1", "balanced_acc@0.5", "unparsable", "errors", "mean latency s", "in tokens", "out tokens", "thinking tokens"]
    L = [f"# Gemini evaluation — {split}, N={N}, {len(sample)} windows ({int(sample.label.sum())} positives / {int((sample.label == 0).sum())} negatives, of which {int(sample.has_nontarget_alert_local[sample.label == 0].sum())} hard)\n",
         f"Model `{MODEL_NAME}`, temperature 0, JSON output, {args.max_lines} most recent lines per channel; responses cached in results/genai/cache. "
         "The sample is balanced, so PR-AUC here is NOT comparable with the full-split PR-AUC elsewhere; compare models WITHIN this table only (all scored on the identical windows). "
         "FAR thresholds are set on the sample's own negatives.\n",
         summ[[c for c in keep if c in summ.columns]].to_markdown(index=False) + "\n"]
    # agreement + qualitative
    if scores:
        L.append("## Pairwise Spearman correlation of scores on the sample\n"); sdf = pd.DataFrame(scores); L.append(sdf.corr(method="spearman").round(2).to_markdown() + "\n")
    for pid in prompts:
        d = pd.read_csv(OUT / f"gemini_{split}_N{N}_{pid}.csv"); L.append(f"## {pid}: probability distribution and example reasons\n")
        L.append("Probability histogram by label (counts):\n\n" + pd.crosstab(pd.cut(d.probability, [-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]), d.label).to_markdown() + "\n")
        for lab, title in ((1, "Positives (failure followed)"), (0, "Negatives")):
            g = d[d.label == lab].sort_values("probability", ascending=(lab == 1))
            L.append(f"**{title} — most {'confident miss' if lab == 1 else 'confident false alarm'} first:**\n"); L += [f"- p={r.probability:.2f} [{r.category or 'neg'}{' hard' if r.hard_negative else ''}] {r.reason}" for _, r in g.head(3).iterrows()]
            L.append("")
    rate = {"in": 0.30, "out": 2.50}  # USD per 1M tokens, Gemini 2.5 Flash list price — CHECK ai.google.dev/pricing on the run day and record it here
    tot_in = int(summ["in tokens"].dropna().sum()) if "in tokens" in summ else 0; tot_out = int((summ["out tokens"].fillna(0) + summ["thinking tokens"].fillna(0)).sum()) if "out tokens" in summ else 0
    L.append(f"## Cost\n\nTokens this run: {tot_in:,} input, {tot_out:,} output incl. thinking → ≈ ${tot_in / 1e6 * rate['in'] + tot_out / 1e6 * rate['out']:.2f} at list price (${rate['in']}/M in, ${rate['out']}/M out; verify on run day).\n")
    if split == "validation": L.append("## Decision\n\nChoose ONE prompt on this table (ROC-AUC first, then recall@10 % FAR, then unparsable rate) and freeze it: `python -m src.genai.gemini_eval --select pX`. The test run uses only that prompt.\n")
    (OUT / f"gemini_{split}_N{N}.md").write_text("\n".join(L), encoding="utf-8"); print(f"[gemini] → {OUT / f'gemini_{split}_N{N}.md'}")
    for _, r in summ[summ.model.str.startswith("Gemini")].iterrows():
        pid = r.model.split("— ")[-1]
        log_run(run_id=f"gemini_{pid}_{split}_N{N}", phase=7 if split == "test" else 10, model_family="gemini", purpose=f"{'FINAL TEST' if split == 'test' else 'prompt selection'} — {pid}", changed_vs_previous="prompt design",
                key_hyperparams=f"N={N};model={MODEL_NAME};prompt={pid};temp=0;max_lines={args.max_lines};sample={len(sample)}", seed=args.seed, split=split, val_pr_auc=round(r.pr_auc, 4), val_recall_at_far=round(r["recall@far0.1"], 4), val_f1=round(r["f1@far0.1"], 4),
                infer_ms_per_window=round((r["mean latency s"] or 0) * 1000, 1), outcome="success" if r.errors == 0 else "failed", learned=f"auto: ROC {r.roc_auc:.3f} on balanced sample; unparsable {int(r.unparsable)}")


if __name__ == "__main__":
    main()
