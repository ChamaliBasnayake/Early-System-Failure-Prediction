"""Phase 10 (validation: prompt selection) / Phase 7 (test, ONCE) — Gemini as a zero-/few-shot failure predictor.

Usage:
  python -m src.genai.gemini_eval --list-models                      # which models this key can use
  python -m src.genai.gemini_eval --split validation                 # all prompts, 300 windows
  python -m src.genai.gemini_eval --split validation --prompts p1 --limit 20   # cheap dry run first
  python -m src.genai.gemini_eval --select p2                        # freeze the chosen prompt
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

MODEL_NAME = os.environ.get("GEMINI_MODEL", "models/gemini-3.8-flash")   # override with --model or the GEMINI_MODEL env var; recorded in every cached response
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


# ----------------------------------------------------------------------------- Gemini (plain REST, Interactions API)
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
JSON_SCHEMA = {"type": "object", "properties": {"probability": {"type": "number"}, "reason": {"type": "string"}}, "required": ["probability", "reason"]}


class QuotaExhausted(RuntimeError):
    pass


def _output_text(j: dict) -> str:
    if j.get("output_text"): return j["output_text"]
    parts = [c.get("text", "") for st in j.get("steps", []) if st.get("type") == "model_output" for c in st.get("content", []) if c.get("type") == "text"]
    return "".join(parts)


class Gemini:
    """Direct HTTPS calls to the Interactions API (no SDK: the google-genai client hung on this machine while curl answered in seconds).
    Thinking level explicit (default 'low'), seed fixed, JSON schema enforced when the API accepts it."""

    def __init__(self, rpm: int, model_name: str, thinking: str = "low", seed: int = 11):
        import httpx
        key = os.environ.get("GEMINI_API_KEY")
        if not key: raise SystemExit("GEMINI_API_KEY is not set (user environment variable). Open a NEW terminal after setting it.")
        self.http = httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0), headers={"x-goog-api-key": key, "Content-Type": "application/json"})
        self.name = model_name.replace("models/", ""); self.thinking = thinking; self.seed = seed; self.min_gap = 60.0 / max(rpm, 1); self.last = 0.0; self.use_schema = True

    @staticmethod
    def _post(http, body: dict):
        r = http.post(API_URL, json=body); return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else {"error": {"message": r.text[:300]}})

    @staticmethod
    def probe(models: list[str], thinking: str) -> None:
        import httpx
        http = httpx.Client(timeout=60.0, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"})
        for m in models:
            for lvl in dict.fromkeys([thinking, "minimal", "low"]):
                t0 = time.time()
                try:
                    code, j = Gemini._post(http, {"model": m.replace("models/", ""), "input": 'Answer with JSON only: {"probability": 0.5, "reason": "probe"}', "generation_config": {"thinking_level": lvl, "max_output_tokens": 256}})
                except Exception as e: code, j = 0, {"error": {"message": str(e)[:120]}}
                if code == 200:
                    u = j.get("usage", {}); print(f"  {m:28s} thinking={lvl:8s} {time.time() - t0:6.1f} s   in={u.get('total_input_tokens')} out={u.get('total_output_tokens')} think={u.get('total_thought_tokens')}   {_output_text(j)[:50]!r}", flush=True); break
                msg = j.get("error", {}).get("message", str(j))[:110]; print(f"  {m:28s} thinking={lvl:8s} {time.time() - t0:6.1f} s   HTTP {code}: {msg}", flush=True)
                if "thinking_level" not in msg.lower(): break

    @staticmethod
    def list_models() -> None:
        import httpx
        r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, timeout=60.0)
        for m in r.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []): print(f"  {m['name']:45s} {m.get('displayName', '')}")

    def ask(self, prompt: str) -> dict:
        for attempt in range(5):
            wait = self.min_gap - (time.time() - self.last)
            if wait > 0: time.sleep(wait)
            body = {"model": self.name, "input": prompt, "generation_config": {"thinking_level": self.thinking, "seed": self.seed, "max_output_tokens": 4096}}
            if self.use_schema: body["response_format"] = JSON_SCHEMA
            print("    → request sent (timeout 120 s) ...", end="", flush=True); t0 = time.time()
            try: code, j = self._post(self.http, body)
            except Exception as e: code, j = 0, {"error": {"message": f"{type(e).__name__}: {str(e)[:150]}"}}
            self.last = time.time(); print(f" {time.time() - t0:.1f} s  HTTP {code}", flush=True)
            if code == 200:
                u = j.get("usage", {})
                return {"text": _output_text(j), "latency_s": round(time.time() - t0, 3), "in_tokens": u.get("total_input_tokens"), "out_tokens": u.get("total_output_tokens"), "thinking_tokens": u.get("total_thought_tokens"),
                        "model": self.name, "thinking_level": self.thinking, "status": j.get("status"), "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
            msg = j.get("error", {}).get("message", str(j))[:300]; low = msg.lower()
            if code == 400 and "response_format" in low and self.use_schema: self.use_schema = False; print("    (API rejected response_format; continuing with JSON requested in the prompt)"); continue
            if code == 400 and "thinking_level" in low and self.thinking != "low": print(f"    (thinking_level={self.thinking} rejected; falling back to low)"); self.thinking = "low"; continue
            if code == 404 or "not found" in low or "no longer available" in low: raise SystemExit(f"Model {self.name} is not available to this key: {msg}\nRun --list-models / --probe and pass --model <name>")
            if code in (401, 403): raise SystemExit(f"Key problem: {msg}")
            if code == 429 or "quota" in low or "resource_exhausted" in low:
                if attempt >= 2: raise QuotaExhausted(msg)
                print(f"    quota/rate limit — waiting 65 s ({msg[:70]})"); time.sleep(65); continue
            if attempt == 4: return {"text": "", "error": f"HTTP {code}: {msg}", "latency_s": None, "in_tokens": None, "out_tokens": None, "thinking_tokens": None, "model": self.name, "thinking_level": self.thinking, "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
            time.sleep(min(60, 8 * 2 ** attempt)); print(f"    retry {attempt + 1}: HTTP {code} {msg[:80]}")


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
    ap.add_argument("--model", default=MODEL_NAME, help=f"Gemini model name (default {MODEL_NAME})"); ap.add_argument("--list-models", action="store_true", help="print the models this key can use and exit")
    ap.add_argument("--probe", action="store_true", help="time one tiny request against candidate models and exit")
    ap.add_argument("--thinking", default="low", choices=["minimal", "low", "medium", "high"], help="thinking level (3.8-flash supports low/medium/high; default low)")
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens on the run day (from ai.google.dev/pricing)"); ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens on the run day")
    args = ap.parse_args(); OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True); sel_path = OUT / "selected_prompt.json"
    if args.list_models: Gemini.list_models(); return
    if args.probe: Gemini.probe([args.model, "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "gemini-2.5-flash"], args.thinking); return
    model_name = args.model
    if args.select:
        sel_path.write_text(json.dumps({"prompt": args.select, "selected_on": "validation", "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "N": args.N, "model": model_name, "thinking": args.thinking})); print(f"selected prompt {args.select} → {sel_path}"); return
    if not args.split: raise SystemExit("--split is required")
    split, N = args.split, args.N
    all_prompts = sorted(p.stem for p in PROMPT_DIR.glob("p*.txt")); short = {p.split("_")[0]: p for p in all_prompts}
    if split == "test":
        if not sel_path.exists(): raise SystemExit("No selected prompt. Run validation, then: python -m src.genai.gemini_eval --select pX")
        if (OUT / f"gemini_test_N{N}.md").exists(): raise SystemExit("Test evaluation already exists — the test set is evaluated ONCE.")
        sel = json.loads(sel_path.read_text()); prompts = [short[sel["prompt"]]]
        if sel.get("model") and sel["model"] != model_name: raise SystemExit(f"Prompt was selected with {sel['model']}; the test run must use the same model (pass --model {sel['model']}).")
        if sel.get("thinking") and sel["thinking"] != args.thinking: raise SystemExit(f"Prompt was selected with thinking={sel['thinking']}; pass --thinking {sel['thinking']}.")
    else:
        prompts = [short[p] if p in short else p for p in (args.prompts or list(short))]
    sample = make_sample(split, N, args.seed)
    if args.limit: sample = sample.head(args.limit)
    print(f"[gemini] {split} N={N}: {len(sample)} windows ({int(sample.label.sum())} pos, {int((sample.label == 0).sum())} neg, {int(sample.has_nontarget_alert_local[sample.label == 0].sum())} hard neg); prompts {prompts}")
    rend = Renderer(args.max_lines); examples = rend.examples(N, args.seed); gem = Gemini(args.rpm, model_name, args.thinking, args.seed); print(f"[gemini] model {model_name}, thinking={args.thinking}, seed={args.seed}")
    rows_summary = []; scores = {}
    for pid in prompts:
        tpl = (PROMPT_DIR / f"{pid}.txt").read_text(encoding="utf-8"); recs = []; print(f"[gemini] prompt {pid}: {len(sample)} windows, ~{60 / max(args.rpm, 1):.0f} s pacing per new call", flush=True)
        for i, (_, r) in enumerate(sample.iterrows()):
            prompt = rend.fill(tpl, r, N, examples); h = hashlib.md5((model_name + args.thinking + pid + prompt).encode()).hexdigest(); cp = CACHE / f"{h}.json"
            if cp.exists(): resp = json.loads(cp.read_text(encoding="utf-8"))
            else:
                try: resp = gem.ask(prompt)
                except QuotaExhausted as e:
                    pd.DataFrame(recs).to_csv(OUT / f"gemini_{split}_N{N}_{pid}_partial.csv", index=False)
                    raise SystemExit(f"\nQuota exhausted after {len(recs)} windows of {pid} ({e}).\nEverything answered so far is cached — rerun the SAME command later (or after enabling billing) and it resumes.")
                resp["prompt_chars"] = len(prompt)
                if not resp.get("error"): cp.write_text(json.dumps(resp), encoding="utf-8")        # errors are never cached
            prob, reason = parse_prob(resp.get("text", ""))
            print(f"  {pid} {i + 1:3d}/{len(sample)}  label={int(r.label)}  p={prob:.2f}  {resp.get('latency_s') or 0:.1f}s  in={resp.get('in_tokens')} out={resp.get('out_tokens')} think={resp.get('thinking_tokens')}  {'(cached)' if resp.get('latency_s') is None or cp.stat().st_mtime < time.time() - 5 else ''}{(' ERROR ' + resp['error'][:60]) if resp.get('error') else ''}", flush=True)
            recs.append({"key": r.key, "unit": r.unit, "window_end": str(r.window_end), "label": int(r.label), "category": r.category, "incident_id": r.incident_id, "hard_negative": bool(r.has_nontarget_alert_local and r.label == 0),
                         "prompt": pid, "probability": prob, "reason": reason, "error": resp.get("error", ""), **{k: resp.get(k) for k in ("latency_s", "in_tokens", "out_tokens", "thinking_tokens", "model", "date")}})
        d = pd.DataFrame(recs); d.to_csv(OUT / f"gemini_{split}_N{N}_{pid}.csv", index=False); scores[pid] = d.probability.values
        m = metrics_on_sample(d.label, d.probability, d.incident_id)
        inv = metrics_on_sample(d.label, 1.0 - d.probability.values, d.incident_id)       # DIAGNOSTIC ONLY: does the LLM rank with the wrong sign?
        rows_summary.append({"model": f"Gemini — {pid}", **{k: v for k, v in flat(m).items()}, "roc_auc of (1-p) [diagnostic]": round(inv["roc_auc"], 3), "unparsable": int(d.probability.isna().sum()), "errors": int((d.error != "").sum()),
                             "mean latency s": round(d.latency_s.dropna().mean(), 2) if d.latency_s.notna().any() else None, "in tokens": int(d.in_tokens.dropna().sum()), "out tokens": int(d.out_tokens.dropna().sum()), "thinking tokens": int(d.thinking_tokens.dropna().sum()) if d.thinking_tokens.notna().any() else 0})
        print(f"  {pid}: ROC {m['roc_auc']:.3f}  PR-AUC(sample) {m['pr_auc']:.3f}  recall@10%FAR {m['recall@far0.1']:.3f}  bal.acc {m['balanced_acc@0.5']:.3f}  unparsable {int(d.probability.isna().sum())}")
    # reference models on the identical windows
    for name, sc in (("final ANN (semantic LSTM, mean of 3 seeds)", ann_scores(sample, N)), ("XGBoost (best validation config)", xgb_scores(sample, N))):
        if np.isnan(sc).all(): continue
        m = metrics_on_sample(sample.label, sc, sample.incident_id); rows_summary.append({"model": name, **flat(m)}); scores[name] = sc
    summ = pd.DataFrame(rows_summary); summ.to_csv(OUT / f"gemini_{split}_N{N}_summary.csv", index=False)
    keep = ["model", "n_scored", "roc_auc", "roc_auc of (1-p) [diagnostic]", "pr_auc", "recall@far0.01", "recall@far0.1", "incident_recall@far0.1", "balanced_acc@0.5", "unparsable", "errors", "mean latency s", "in tokens", "out tokens", "thinking tokens"]
    L = [f"# Gemini evaluation — {split}, N={N}, {len(sample)} windows ({int(sample.label.sum())} positives / {int((sample.label == 0).sum())} negatives, of which {int(sample.has_nontarget_alert_local[sample.label == 0].sum())} hard)\n",
         f"Model `{model_name}`, thinking level `{args.thinking}`, seed {args.seed}, run {dt.date.today().isoformat()}, JSON output, {args.max_lines} most recent lines per channel; responses cached in results/genai/cache. "
         "The sample is balanced, so PR-AUC here is NOT comparable with the full-split PR-AUC elsewhere; compare models WITHIN this table only (all scored on the identical windows). "
         "FAR thresholds are set on the sample's own negatives. The sample's negatives are one-third hard negatives (alert-rich windows), against ~10 % in the population, so a model that fires on alerts is penalised more here than it would be in deployment; the same sample design is used on test and every model is scored on the identical windows. "
         "The column `roc_auc of (1-p)` is a diagnostic only (is the LLM's ranking informative but inverted?); it is never used as a model.\n",
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
    tot_in = int(summ["in tokens"].dropna().sum()) if "in tokens" in summ else 0; tot_out = int((summ["out tokens"].fillna(0) + summ["thinking tokens"].fillna(0)).sum()) if "out tokens" in summ else 0
    cost = (f"≈ ${tot_in / 1e6 * args.price_in + tot_out / 1e6 * args.price_out:.2f} at ${args.price_in}/M input and ${args.price_out}/M output (ai.google.dev/pricing, {dt.date.today().isoformat()})"
            if args.price_in is not None and args.price_out is not None else "pass --price-in/--price-out (USD per 1M tokens from ai.google.dev/pricing on the run day) to print the cost")
    L.append(f"## Cost\n\nTokens this run: {tot_in:,} input, {tot_out:,} output incl. thinking; {cost}.\n")
    if split == "validation": L.append("## Decision\n\nChoose ONE prompt on this table (ROC-AUC first, then recall@10 % FAR, then unparsable rate) and freeze it: `python -m src.genai.gemini_eval --select pX`. The test run uses only that prompt.\n")
    (OUT / f"gemini_{split}_N{N}.md").write_text("\n".join(L), encoding="utf-8"); print(f"[gemini] → {OUT / f'gemini_{split}_N{N}.md'}")
    for _, r in summ[summ.model.str.startswith("Gemini")].iterrows():
        pid = r.model.split("— ")[-1]
        log_run(run_id=f"gemini_{pid}_{split}_N{N}", phase=7 if split == "test" else 10, model_family="gemini", purpose=f"{'FINAL TEST' if split == 'test' else 'prompt selection'} — {pid}", changed_vs_previous="prompt design",
                key_hyperparams=f"N={N};model={model_name};thinking={args.thinking};seed={args.seed};prompt={pid};max_lines={args.max_lines};sample={len(sample)}", seed=args.seed, split=split, val_pr_auc=round(r.pr_auc, 4), val_recall_at_far=round(r["recall@far0.1"], 4), val_f1=round(r["f1@far0.1"], 4),
                infer_ms_per_window=round((r["mean latency s"] or 0) * 1000, 1), outcome="success" if r.errors == 0 else "failed", learned=f"auto: ROC {r.roc_auc:.3f} on balanced sample; unparsable {int(r.unparsable)}")


if __name__ == "__main__":
    main()
