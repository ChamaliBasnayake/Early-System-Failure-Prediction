"""Phase 1, step 4 — apply the FAILURE DEFINITION and build failure episodes.

The most consequential decision in the project, made explicitly before any
modelling. It is a label definition, so it may use the whole log (declared in Q1).

Findings that shaped it (results/category_evidence.md):
  * all 41 BGL alert categories carry level FATAL/FAILURE, so the alert tag alone
    marks a fatal event;
  * APP* categories are user-application errors logged by every node of a job —
    not node failures → hard-negative pool;
  * several KERN* categories are MASS-SYNCHRONOUS: e.g. KERNSTOR = 63,491 node
    episodes but only 8 system-wide incidents (median 1,874 nodes in the same
    minute). Those are partition/job events, not node faults.

Rule (configs/failure_definition.yaml):
  1. candidate = alert line whose category matches `target_category_patterns` (^KERN)
     and is not in `exclude_categories`;
  2. candidates are clustered system-wide per category with `incident_gap_s`; a
     cluster touching more than `mass_incident_min_nodes` distinct nodes is a MASS
     INCIDENT → saved separately, removed from the target set, and windows that
     overlap it are excluded in Phase 2 (neither healthy nor node-failing);
  3. remaining target events on the same node within `burst_gap_s` form one EPISODE
     (start, end, first/dominant category, category set);
  4. support is counted in INCIDENTS (system-wide clusters of target episodes with
     `incident_gap_s`); a category is evaluable at ≥ `min_incidents_per_category`.

Outputs (data/interim/):
  failure_episodes.parquet   target node-episodes, with incident_id
  mass_incidents.parquet     excluded mass-synchronous incidents (one row per incident)
  mass_events.parquet        the individual alert lines belonging to mass incidents (node, timestamp, category)
  nontarget_alerts.parquet   APP*/LINK*/… alert lines (hard-negative pool)
  results/failure_set_summary.md

Usage:  python -m src.data.failure_set [--sample]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from src.utils.paths import CONFIGS, DATA_INTERIM, RESULTS

DEFAULT_CFG = {
    "target_category_patterns": ["^KERN"],
    "exclude_categories": [],
    "incident_gap_s": 60,
    "mass_incident_min_nodes": 32,
    "burst_gap_s": 300,
    "support_gap_s": 300,
    "min_incidents_per_category": 30,
}


def load_cfg() -> dict:
    p = CONFIGS / "failure_definition.yaml"
    if not p.exists():
        p.write_text(yaml.safe_dump(DEFAULT_CFG, sort_keys=False))
        print(f"[failure_set] wrote default config → {p}")
    return {**DEFAULT_CFG, **yaml.safe_load(p.read_text())}


def is_target_category(cat: pd.Series, cfg: dict) -> pd.Series:
    pat = re.compile("|".join(f"(?:{p})" for p in cfg["target_category_patterns"]))
    m = cat.astype(str).str.match(pat)
    if cfg["exclude_categories"]:
        m &= ~cat.astype(str).isin(cfg["exclude_categories"])
    return m


def _cluster_ids(ts: pd.Series, gap_s: float) -> pd.Series:
    """Consecutive timestamps (already sorted) closer than gap_s share an id."""
    gaps = ts.diff().dt.total_seconds()
    return (gaps.isna() | (gaps > gap_s)).cumsum()


def split_mass_incidents(ev: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (node_level_events, mass_incidents_table, mass_events_lines)."""
    ev = ev.sort_values(["alert_tag", "timestamp"]).copy()
    ev["cluster"] = ev.groupby("alert_tag", observed=True)["timestamp"].transform(
        lambda s: _cluster_ids(s, cfg["incident_gap_s"]))
    size = ev.groupby(["alert_tag", "cluster"], observed=True)["node"].transform("nunique")
    mass_mask = size > cfg["mass_incident_min_nodes"]
    mass = (ev[mass_mask].groupby(["alert_tag", "cluster"], observed=True)
              .agg(start=("timestamp", "min"), end=("timestamp", "max"),
                   n_nodes=("node", "nunique"), n_events=("timestamp", "size"))
              .reset_index().rename(columns={"alert_tag": "category"}).drop(columns="cluster"))
    mass_events = ev[mass_mask][["timestamp", "node", "alert_tag"]].rename(columns={"alert_tag": "category"})
    return ev[~mass_mask].drop(columns="cluster"), mass, mass_events


def build_episodes(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alerts = df[df["is_alert"]].copy()
    alerts["timestamp"] = pd.to_datetime(alerts["timestamp"], utc=True)
    tgt = is_target_category(alerts["alert_tag"], cfg)
    nontarget = alerts[~tgt].copy()
    cand = alerts[tgt]
    node_ev, mass, mass_events = split_mass_incidents(cand, cfg)
    if node_ev.empty:
        return node_ev.iloc[0:0], mass, nontarget, mass_events

    ev = node_ev.sort_values(["node", "timestamp"])
    ev = ev.assign(episode_id=ev.groupby("node", observed=True)["timestamp"]
                   .transform(lambda s: _cluster_ids(s, cfg["burst_gap_s"])))
    ev["episode_key"] = ev["node"].astype(str) + "#" + ev["episode_id"].astype(str)

    ep = (ev.groupby("episode_key")
            .agg(node=("node", "first"), start=("timestamp", "first"), end=("timestamp", "last"),
                 first_category=("alert_tag", "first"),
                 dominant_category=("alert_tag", lambda s: s.value_counts().idxmax()),
                 categories=("alert_tag", lambda s: "|".join(sorted(set(s.astype(str))))),
                 n_categories=("alert_tag", "nunique"),
                 component=("component", "first"), n_events=("timestamp", "size"))
            .reset_index(drop=True))
    ep["duration_s"] = (ep["end"] - ep["start"]).dt.total_seconds()
    # incident id: system-wide clustering of episodes per first category (for support counting / per-incident metrics)
    ep = ep.sort_values(["first_category", "start"])
    ep["incident_id"] = (ep["first_category"].astype(str) + "#" +
                         ep.groupby("first_category", observed=True)["start"]
                           .transform(lambda s: _cluster_ids(s, cfg["support_gap_s"])).astype(str))
    return ep.sort_values("start").reset_index(drop=True), mass, nontarget, mass_events


def summarise(ep: pd.DataFrame, mass: pd.DataFrame, nontarget: pd.DataFrame, cfg: dict, out_md: Path) -> None:
    L = ["# Failure set summary (auto-generated by src/data/failure_set.py)\n",
         f"Rule: alert line AND category matches {cfg['target_category_patterns']} (excluded: {cfg['exclude_categories'] or 'none'}); "
         f"mass incidents = > {cfg['mass_incident_min_nodes']} nodes within {cfg['incident_gap_s']} s (same category) → excluded; "
         f"node episodes: burst gap {cfg['burst_gap_s']} s; incidents for support: {cfg['support_gap_s']} s; "
         f"evaluable at ≥ {cfg['min_incidents_per_category']} incidents.\n",
         f"- Target failure EVENTS (lines): **{int(ep.n_events.sum()):,}**",
         f"- Target node EPISODES: **{len(ep):,}**",
         f"- Target INCIDENTS (system-wide): **{ep.incident_id.nunique():,}**",
         f"- Nodes with ≥1 target episode: **{ep.node.nunique():,}**",
         f"- Episodes with >1 category: **{int((ep.n_categories > 1).sum()):,}**; "
         f"median duration {ep.duration_s.median():.0f} s, 90th pct {ep.duration_s.quantile(.9):.0f} s",
         f"- MASS incidents excluded: **{len(mass):,}** covering {int(mass.n_events.sum()):,} alert lines",
         f"- NON-target alert lines (hard-negative pool): **{len(nontarget):,}** across {nontarget.alert_tag.nunique()} categories\n",
         "## Target support per first category\n"]
    per = (ep.groupby("first_category", observed=True)
             .agg(incidents=("incident_id", "nunique"), episodes=("start", "size"),
                  events=("n_events", "sum"), nodes=("node", "nunique"))
             .sort_values("incidents", ascending=False))
    per["status"] = ["evaluable" if i >= cfg["min_incidents_per_category"] else "insufficient support" for i in per.incidents]
    L.append(per.to_markdown() + "\n")
    L.append("## Mass incidents excluded (per category)\n")
    if len(mass):
        mm = mass.groupby("category").agg(incidents=("start", "size"), lines=("n_events", "sum"),
                                          median_nodes=("n_nodes", "median"), max_nodes=("n_nodes", "max")).sort_values("lines", ascending=False)
        L.append(mm.to_markdown() + "\n")
    L.append("## Most common category combinations within a target episode\n" +
             ep.categories.value_counts().head(10).to_frame("episodes").to_markdown() + "\n")
    monthly = ep.set_index("start").resample("1ME").agg(episodes=("node", "size"), incidents=("incident_id", "nunique"))
    L.append("## Target episodes and incidents per month\n" + monthly.to_markdown() + "\n")
    L.append("## Non-target alert categories (hard-negative pool)\n" +
             nontarget.alert_tag.value_counts().to_frame("lines").to_markdown() + "\n")
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(f"[failure_set] {len(ep):,} target episodes / {ep.incident_id.nunique():,} incidents; "
          f"{len(mass):,} mass incidents excluded; {len(nontarget):,} non-target lines → {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg()
    sfx = "_sample" if args.sample else ""
    df = pd.read_parquet(DATA_INTERIM / f"bgl_parsed{sfx}.parquet",
                         columns=["timestamp", "node", "alert_tag", "level", "component", "is_alert"])
    ep, mass, nontarget, mass_events = build_episodes(df, cfg)
    ep.to_parquet(DATA_INTERIM / f"failure_episodes{sfx}.parquet", index=False)
    mass.to_parquet(DATA_INTERIM / f"mass_incidents{sfx}.parquet", index=False)
    mass_events.to_parquet(DATA_INTERIM / f"mass_events{sfx}.parquet", index=False)
    nontarget.to_parquet(DATA_INTERIM / f"nontarget_alerts{sfx}.parquet", index=False)
    summarise(ep, mass, nontarget, cfg, RESULTS / f"failure_set_summary{sfx}.md")


if __name__ == "__main__":
    main()
