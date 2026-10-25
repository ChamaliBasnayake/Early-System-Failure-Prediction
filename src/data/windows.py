"""Phase 2, step 3 (v2) — unit-level, two-channel labelled windows with an enforced lead-time gap.

Unit of prediction: configurable hardware level (default MIDPLANE = 512 nodes, the unit BGL
allocates jobs by). Motivation: results/precursor_diagnostics.md — single nodes are silent
before their own failures (90–97 % empty), the system is not (≈90 % active, elevated volume).

For a unit u and window end t:
  LOCAL  channel : events of u          in [t - W_local,  t]
  GLOBAL channel : events of the system in [t - W_global, t]
  label_N(u, t) = 1  if a TARGET unit-episode of u starts in [t+N, t+N+H)
                = 0  otherwise
                = -1 EXCLUDED: a target episode of u overlaps the local window (failure visible);
                     u had an episode end within post_episode_exclusion before t (reboot chatter);
                     a target episode of u starts in (t, t+N) (inside the gap);
                     a MASS-incident line of u falls in [t-W, t+N+H]; the span crosses a split cut.
  The GLOBAL channel may legitimately contain OTHER units' failure lines (they are observable at
  time t); it never contains anything after t.

Window ends: pos (start - N - k for each N, k), uniform (unit, time) negatives — the deployment
process — and hard negatives 30 s after a non-target alert on the unit. Exact-duplicate windows are
removed within a split and (for ≥3-item local content) across splits. Positives carry
incident_id / category for per-incident metrics.

Outputs: data/interim/windows.parquet, results/windows_summary.md, results/class_distribution.csv
Usage:   python -m src.data.windows [--sample] [--split split_v1] [--unit system --tag system] [--events _tc --tag tc]
         python -m src.data.windows --local-window 360 --tag w6h      (window-length study)
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from src.data.failure_set import is_target_category, load_cfg as load_failure_cfg
from src.data.precursor_diagnostics import unit_of
from src.data.splits import assign_split, load_split
from src.utils.paths import CONFIGS, DATA_INTERIM, DATA_SPLITS, RESULTS
from src.utils.seeds import set_seed

SEC = np.int64(1_000_000_000)
MIN = 60 * SEC


@dataclass
class Cfg:
    level: str; Wl: int; Wg: int; max_local: int; max_global: int; Ns: list; H: int; offsets: list
    ep_gap_s: int; post_excl: int; uniform_negatives: dict; hard_negatives: dict; seed: int; recent_excl: int = 60

    @classmethod
    def load(cls) -> "Cfg":
        y = yaml.safe_load((CONFIGS / "windows.yaml").read_text())
        return cls(level=y["unit_level"], Wl=y["local_window_min"], Wg=y["global_window_min"],
                   max_local=y["max_events_local"], max_global=y["max_events_global"], Ns=list(y["lead_times_min"]),
                   H=y["horizon_minutes"], offsets=list(y["positive_offsets_min"]), ep_gap_s=y["unit_episode_gap_s"],
                   post_excl=y["post_episode_exclusion_min"], uniform_negatives=y["uniform_negatives"],
                   hard_negatives=y["hard_negatives"], seed=y["seed"], recent_excl=y.get("recent_failure_exclusion_min", 60))


def _ns(ts) -> np.ndarray:
    return pd.to_datetime(pd.Series(ts), utc=True).values.astype("datetime64[ns]").astype(np.int64)


def _cluster(ts_sorted: np.ndarray, gap_ns: np.int64) -> np.ndarray:
    gaps = np.diff(ts_sorted, prepend=ts_sorted[0] - gap_ns - 1)
    return np.cumsum(gaps > gap_ns)


# ----------------------------------------------------------------------------- unit episodes
def unit_episodes(ep: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    """Collapse node episodes to UNIT episodes (same unit within ep_gap_s). Keeps incident_id / category of the first."""
    e = ep.copy(); e["unit"] = unit_of(e["node"], cfg.level).values
    e["start_ns"] = _ns(e["start"]); e["end_ns"] = _ns(e["end"])
    e = e.sort_values(["unit", "start_ns"])
    e["uep"] = e.groupby("unit")["start_ns"].transform(lambda s: _cluster(s.values, np.int64(cfg.ep_gap_s) * SEC))
    ue = (e.groupby(["unit", "uep"]).agg(start_ns=("start_ns", "min"), end_ns=("end_ns", "max"),
                                         incident_id=("incident_id", "first"), category=("first_category", "first"),
                                         n_nodes=("node", "nunique"), n_node_episodes=("start_ns", "size"))
            .reset_index().drop(columns="uep"))
    ue["unit_episode_id"] = np.arange(len(ue))
    return ue


# ----------------------------------------------------------------------------- candidates
def make_candidates(ue: pd.DataFrame, all_units: np.ndarray, nontarget: pd.DataFrame, split: dict, cfg: Cfg,
                    rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for N in cfg.Ns:
        for k in cfg.offsets:
            rows.append(pd.DataFrame({"unit": ue["unit"].values, "t_ns": ue["start_ns"].values - (N + k) * MIN, "anchor": "pos",
                                      "unit_episode_id": ue["unit_episode_id"].values, "incident_id": ue["incident_id"].values,
                                      "category": ue["category"].values}))
    for k, n in cfg.uniform_negatives.items():           # uniform over (unit, time) within the split period = deployment process
        a, b = pd.Timestamp(split[k]["start"]).value, pd.Timestamp(split[k]["end"]).value
        rows.append(pd.DataFrame({"unit": rng.choice(all_units, size=n), "t_ns": rng.integers(a, b, size=n) // MIN * MIN,
                                  "anchor": "uniform", "unit_episode_id": -1, "incident_id": "", "category": ""}))
    nt_unit = unit_of(nontarget["node"], cfg.level).values; nt_t = _ns(nontarget["timestamp"])
    nt_split = assign_split(pd.to_datetime(nontarget["timestamp"], utc=True), split).values
    for k, n in cfg.hard_negatives.items():
        idx = np.flatnonzero(nt_split == k)
        if len(idx) == 0: continue
        pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
        rows.append(pd.DataFrame({"unit": nt_unit[pick], "t_ns": nt_t[pick] + 30 * SEC, "anchor": "hard",
                                  "unit_episode_id": -1, "incident_id": "", "category": ""}))
    cand = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["unit", "t_ns"]).reset_index(drop=True)
    return cand


# ----------------------------------------------------------------------------- labels
def label_windows(cand: pd.DataFrame, ue: pd.DataFrame, mass_events: pd.DataFrame, split: dict, cfg: Cfg) -> pd.DataFrame:
    W, H, PE = cfg.Wl * MIN, cfg.H * MIN, cfg.post_excl * MIN
    R = min(cfg.recent_excl, cfg.Wl) * MIN          # recent-failure guard (never longer than the local window)
    maxN = max(cfg.Ns) * MIN
    cand = cand.copy()
    ep_by_unit = {u: (g["start_ns"].values, g["end_ns"].values) for u, g in ue.groupby("unit")}
    me_by_unit = {}
    if len(mass_events):
        mu = unit_of(mass_events["node"], cfg.level).values; mt = _ns(mass_events["timestamp"])
        for u in np.unique(mu): me_by_unit[u] = np.sort(mt[mu == u])
    cuts = np.array([pd.Timestamp(split["train"]["end"]).value, pd.Timestamp(split["validation"]["end"]).value])
    labels = {N: np.zeros(len(cand), dtype=np.int8) for N in cfg.Ns}
    reason = np.array([""] * len(cand), dtype=object)
    for i, (u, t) in enumerate(zip(cand["unit"].values, cand["t_ns"].values)):
        excl = ""
        if u in me_by_unit:
            me = me_by_unit[u]; j = np.searchsorted(me, t - W)
            if j < len(me) and me[j] <= t + maxN + H: excl = "mass_incident"
        starts, ends = ep_by_unit.get(u, (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
        if not excl and len(starts):
            if np.any((starts <= t) & (ends >= t - R)): excl = "recent_failure"      # failure within the last R min
            elif np.any((ends < t) & (ends >= t - PE)): excl = "post_episode"
        if not excl and np.any((cuts > t - W) & (cuts <= t + maxN + H)): excl = "crosses_split_cut"
        reason[i] = excl
        for N in cfg.Ns:
            Nns = N * MIN
            if excl: labels[N][i] = -1
            elif len(starts) and np.any((starts > t) & (starts < t + Nns)): labels[N][i] = -1     # inside the gap
            elif len(starts) and np.any((starts >= t + Nns) & (starts < t + Nns + H)): labels[N][i] = 1
            else: labels[N][i] = 0
    for N in cfg.Ns: cand[f"label_{N}"] = labels[N]
    cand["exclusion"] = reason
    cand["split"] = assign_split(pd.to_datetime(cand["t_ns"], utc=True), split).values
    return cand


# ----------------------------------------------------------------------------- content
def _collapse(tpl, mid, ts, cap):
    if len(tpl) == 0: return [], [], [], [], "empty"
    change = np.r_[True, mid[1:] != mid[:-1]]; idx = np.flatnonzero(change)
    rep = np.diff(np.r_[idx, len(tpl)]); ctpl, cmid, cts = tpl[idx], mid[idx], ts[idx]
    dt = np.r_[0, np.diff(cts)] / SEC
    if len(ctpl) > cap: ctpl, cmid, rep, dt = ctpl[-cap:], cmid[-cap:], rep[-cap:], dt[-cap:]
    h = hashlib.md5(cmid.tobytes() + rep.astype(np.int32).tobytes()).hexdigest()
    return ctpl.tolist(), cmid.tolist(), rep.astype(np.int32).tolist(), np.round(dt, 3).tolist(), h


def extract_content(cand: pd.DataFrame, events: pd.DataFrame, nontarget_mask: np.ndarray, target_mask: np.ndarray, cfg: Cfg) -> pd.DataFrame:
    Wl, Wg = cfg.Wl * MIN, cfg.Wg * MIN
    ev_t = _ns(events["timestamp"]); ev_tpl = events["template_index"].values.astype(np.int32)
    ev_mid = events["masked_id"].values.astype(np.int32) if "masked_id" in events.columns else ev_tpl.copy()
    ev_unit = unit_of(events["node"], cfg.level).values
    codes, uniques = pd.factorize(pd.Series(ev_unit))
    # local index: sorted by (unit, time)
    o = np.lexsort((ev_t, codes))
    lt, lcodes, ltpl, lmid, lnt, ltg = ev_t[o], codes[o], ev_tpl[o], ev_mid[o], nontarget_mask[o], target_mask[o]
    uc, first = np.unique(lcodes, return_index=True); last = np.r_[first[1:], len(lcodes)]
    span = {uniques[c]: (a, b) for c, a, b in zip(uc, first, last)}
    # global index: sorted by time
    g = np.argsort(ev_t, kind="stable")
    gt, gtpl, gmid, gnt, gtg, gunit = ev_t[g], ev_tpl[g], ev_mid[g], nontarget_mask[g], target_mask[g], codes[g]

    cols = {k: [] for k in ["tpl_local", "mid_local", "cnt_local", "dt_local", "tpl_global", "mid_global", "cnt_global", "dt_global",
                            "n_local", "n_global", "n_unk_local", "n_unk_global", "has_nontarget_alert_local",
                            "n_nontarget_alerts_global", "n_target_lines_global", "n_units_active_global", "content_hash"]}
    for u, t in zip(cand["unit"].values, cand["t_ns"].values):
        a, b = span.get(u, (0, 0))
        lo = a + np.searchsorted(lt[a:b], t - Wl, side="left"); hi = a + np.searchsorted(lt[a:b], t, side="right")
        tl, ml, cl, dl, hl = _collapse(ltpl[lo:hi], lmid[lo:hi], lt[lo:hi], cfg.max_local)
        glo = np.searchsorted(gt, t - Wg, side="left"); ghi = np.searchsorted(gt, t, side="right")
        tg, mg, cg, dg, hg = _collapse(gtpl[glo:ghi], gmid[glo:ghi], gt[glo:ghi], cfg.max_global)
        cols["tpl_local"].append(tl); cols["mid_local"].append(ml); cols["cnt_local"].append(cl); cols["dt_local"].append(dl)
        cols["tpl_global"].append(tg); cols["mid_global"].append(mg); cols["cnt_global"].append(cg); cols["dt_global"].append(dg)
        cols["n_local"].append(int(hi - lo)); cols["n_global"].append(int(ghi - glo))
        cols["n_unk_local"].append(int((ltpl[lo:hi] == 1).sum())); cols["n_unk_global"].append(int((gtpl[glo:ghi] == 1).sum()))
        cols["has_nontarget_alert_local"].append(bool(lnt[lo:hi].any()))
        cols["n_nontarget_alerts_global"].append(int(gnt[glo:ghi].sum())); cols["n_target_lines_global"].append(int(gtg[glo:ghi].sum()))
        cols["n_units_active_global"].append(int(len(np.unique(gunit[glo:ghi]))) if ghi > glo else 0)
        cols["content_hash"].append("empty" if (hl == "empty" and hg == "empty") else hashlib.md5((hl + hg).encode()).hexdigest())
    out = cand.copy()
    for k, v in cols.items(): out[k] = v
    out["window_start"] = pd.to_datetime(out["t_ns"] - Wl, utc=True); out["window_end"] = pd.to_datetime(out["t_ns"], utc=True)
    return out


# ----------------------------------------------------------------------------- dedup
def dedup(win: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    order = {"train": 0, "validation": 1, "test": 2}
    win = win[win["split"].notna()].copy(); win["_o"] = win["split"].map(order); win = win.sort_values(["_o", "t_ns"])
    before = len(win)
    specific = (win["content_hash"] != "empty") & ((win["tpl_local"].map(len) + win["tpl_global"].map(len)) >= 3)
    first_split = win[specific].groupby("content_hash")["_o"].min()
    cross = specific & (win["content_hash"].map(first_split) < win["_o"])
    win = win[~cross]
    key = win["split"] + "|" + win["unit"].astype(str) + "|" + win["content_hash"]      # identical content on the SAME unit within a split
    dup_count = key.map(key.value_counts())
    keep = ~((win["content_hash"] != "empty") & key.duplicated())
    win = win[keep].copy(); win["dup_count"] = dup_count.loc[win.index].values
    return win.drop(columns="_o").reset_index(drop=True), {"before": before, "removed_cross_split": int(cross.sum()),
                                                            "removed_within_split": int((~keep).sum()), "after": len(win)}


# ----------------------------------------------------------------------------- summary
def summarise(win: pd.DataFrame, ue: pd.DataFrame, cfg: Cfg, ded: dict, out_md, out_csv) -> None:
    L = ["# Windows summary v2 (auto-generated by src/data/windows.py)\n",
         f"Unit = {cfg.level}; W_local={cfg.Wl} min (cap {cfg.max_local}), W_global={cfg.Wg} min (cap {cfg.max_global}); N={cfg.Ns} min; H={cfg.H} min; "
         f"positive offsets {cfg.offsets}; unit-episode gap {cfg.ep_gap_s} s; post-episode exclusion {cfg.post_excl} min; seed {cfg.seed}.",
         f"Unit episodes: {len(ue):,} (from node episodes), incidents {ue.incident_id.nunique():,}, units with ≥1 episode {ue.unit.nunique():,}.",
         f"Dedup: {ded['before']:,} → {ded['after']:,} windows (cross-split {ded['removed_cross_split']:,}, within-split {ded['removed_within_split']:,}).\n",
         "## Class distribution per split and lead time (Q1)\n"]
    rows = []
    for sp in ("train", "validation", "test"):
        g = win[win["split"] == sp]
        for N in cfg.Ns:
            lab = g[f"label_{N}"]
            rows.append(dict(split=sp, N=N, windows=len(g), positives=int((lab == 1).sum()), negatives=int((lab == 0).sum()),
                             excluded=int((lab == -1).sum()), hard_negatives=int(((lab == 0) & g["has_nontarget_alert_local"]).sum()),
                             positive_incidents=g.loc[lab == 1, "incident_id"].nunique(), positive_unit_episodes=g.loc[lab == 1, "unit_episode_id"].nunique(),
                             pos_local_nonempty=round(float((g.loc[lab == 1, "n_local"] > 0).mean()), 3) if (lab == 1).any() else None,
                             neg_local_nonempty=round(float((g.loc[lab == 0, "n_local"] > 0).mean()), 3) if (lab == 0).any() else None))
    cd = pd.DataFrame(rows); cd.to_csv(out_csv, index=False); L.append(cd.to_markdown(index=False) + "\n")
    L.append("## Exclusion reasons\n" + win["exclusion"].replace("", "none").value_counts().to_frame("windows").to_markdown() + "\n")
    L.append("## Anchors per split\n" + pd.crosstab(win["split"], win["anchor"]).to_markdown() + "\n")
    lab5 = win["label_5"]
    cls = np.where(lab5 == 1, "positive", np.where(lab5 == 0, "negative", "excluded"))
    for ch in ("local", "global"):
        d = (win.assign(cls=cls).groupby(["split", "cls"])[f"n_{ch}"]
                .agg(mean="mean", median="median", p90=lambda s: s.quantile(.9), nonempty=lambda s: (s > 0).mean()).round(2))
        L.append(f"## {ch} channel: raw event counts by split × class (N=5)\n" + d.to_markdown() + "\n")
    L.append("## UNK share of raw events by split × channel\n" +
             pd.DataFrame({"local": win.groupby("split")["n_unk_local"].sum() / win.groupby("split")["n_local"].sum().clip(lower=1),
                           "global": win.groupby("split")["n_unk_global"].sum() / win.groupby("split")["n_global"].sum().clip(lower=1)}).round(4).to_markdown() + "\n")
    pos = win[lab5 == 1]
    L.append("## Positive incidents per category and split (N=5)\n" + pos.groupby(["category", "split"])["incident_id"].nunique().unstack(fill_value=0).to_markdown() + "\n")
    out_md.write_text("\n".join(L), encoding="utf-8"); print(cd.to_string(index=False)); print(f"[windows] → {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", action="store_true"); ap.add_argument("--split", default="split_v1")
    ap.add_argument("--unit", default=None, help="override unit_level from configs/windows.yaml (e.g. system)")
    ap.add_argument("--events", default="", help="events file suffix, e.g. _tc for the transductive-control templates")
    ap.add_argument("--tag", default="", help="output tag, e.g. system or tc; outputs become windows_<tag>.parquet etc.")
    ap.add_argument("--local-window", type=int, default=None, help="override local_window_min (W study: 60, 360, 1440)")
    ap.add_argument("--global-window", type=int, default=None, help="override global_window_min")
    args = ap.parse_args(); sfx = "_sample" if args.sample else ""; out_sfx = sfx + (f"_{args.tag}" if args.tag else "")
    cfg = Cfg.load(); set_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    if args.unit: cfg.level = args.unit
    if args.local_window: cfg.Wl = args.local_window
    if args.global_window: cfg.Wg = args.global_window
    split = load_split(DATA_SPLITS / f"{args.split}{sfx}.json")
    events = pd.read_parquet(DATA_INTERIM / f"events{sfx}{args.events}.parquet")
    ep = pd.read_parquet(DATA_INTERIM / f"failure_episodes{sfx}.parquet")
    mass_events = pd.read_parquet(DATA_INTERIM / f"mass_events{sfx}.parquet")
    nontarget = pd.read_parquet(DATA_INTERIM / f"nontarget_alerts{sfx}.parquet")
    fcfg = load_failure_cfg()
    target_mask = (events["is_alert"].values & is_target_category(events["alert_tag"], fcfg).values)
    nontarget_mask = (events["is_alert"].values & ~is_target_category(events["alert_tag"], fcfg).values)
    print(f"[windows] unit={cfg.level}; {len(events):,} events, {len(ep):,} node episodes, {len(mass_events):,} mass lines, {len(nontarget):,} non-target alerts")
    ue = unit_episodes(ep, cfg)
    all_units = np.unique(unit_of(events["node"], cfg.level).values)
    print(f"[windows] {len(ue):,} unit episodes on {ue.unit.nunique():,} of {len(all_units):,} units")
    cand = make_candidates(ue, all_units, nontarget, split, cfg, rng)
    print(f"[windows] {len(cand):,} candidates: " + ", ".join(f"{k}={v}" for k, v in cand.anchor.value_counts().items()))
    cand = label_windows(cand, ue, mass_events, split, cfg)
    win = extract_content(cand, events, nontarget_mask, target_mask, cfg)
    win, ded = dedup(win)
    win.to_parquet(DATA_INTERIM / f"windows{out_sfx}.parquet", index=False)
    ue.to_parquet(DATA_INTERIM / f"unit_episodes{out_sfx}.parquet", index=False)
    summarise(win, ue, cfg, ded, RESULTS / f"windows_summary{out_sfx}.md", RESULTS / f"class_distribution{out_sfx}.csv")


if __name__ == "__main__":
    main()
