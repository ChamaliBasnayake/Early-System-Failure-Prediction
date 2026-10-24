"""Phase 2, step 2 — log templates with Drain3, FITTED ON THE TRAINING PERIOD ONLY.

Why: Le & Zhang (2022) and Xiao et al. (2026) show that fitting the parser on all
data leaks future vocabulary into training. Here the miner sees only messages with
timestamp < train_end. Validation/test messages are matched against the frozen
miner; unmatched messages become UNK, and the UNK rate per split is a reported
protocol field.

BGL has a documented software upgrade in October 2005 (idoproxydb "$Name: V1R1M2_254/255")
after which the kernel emits new RAS message types; ~60 % of test-period lines belong to
templates unseen in training. So, in addition to the frozen template index, every distinct
MASKED MESSAGE TEXT is stored with an id. The text is available at inference time in
deployment too, and lets a pretrained (inductive) text encoder embed unseen messages.

Index convention (used by every model):  PAD = 0,  UNK = 1,  templates = 2 … K+1.

Outputs (data/interim/):
  drain_miner.pkl          frozen Drain3 state (fitted on train only)
  templates.csv            template_index, drain_cluster_id, size_in_train, template text
  masked_messages.parquet  masked_id, text, n_lines, first_split   (ALL distinct masked messages)
  events.parquet           timestamp, node, template_index, masked_id, is_alert, alert_tag
  results/template_report.md
Usage:  python -m src.data.templates [--sample] [--transductive-control]
        --transductive-control fits Drain on ALL periods (the leaky protocol used by most prior work);
        outputs get the suffix _tc and must only be used as a labelled control, never for model selection.
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from src.data.splits import assign_split, load_split
from src.utils.paths import CONFIGS, DATA_INTERIM, DATA_SPLITS, RESULTS

PAD, UNK = 0, 1
WILD = "<*>"

# Ordered masking rules — every variable part becomes the same wildcard so that hex/number/path
# variants of one logging statement collapse into one template (standard Drain practice).
MASKS = [
    (re.compile(r"R\d{2}-M\d-N[0-9A-Fa-f]+(?:-[A-Z]:J\d{2}-U\d{2})?"), WILD),      # node ids
    (re.compile(r"\bbglio\d+\b"), WILD),                                             # I/O hosts
    (re.compile(r"\$Name:\s*\S+\s*\$"), WILD),                                       # software version tags
    (re.compile(r"0x[0-9a-fA-F]+"), WILD),                                           # 0x hex
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), WILD),                              # IPv4
    (re.compile(r"(?<![\w.])\.?/[\w./\-]+"), WILD),                                  # paths (/a/b, ./a/b)
    (re.compile(r"\b[\w\-]+\.(?:rts|cnk|out|exe|so|cpp|cc|c|h|txt|dat|log|properties|rpm|conf)\b"), WILD),  # file names
    (re.compile(r"\b(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{6,}\b"), WILD),  # hex words ≥6 chars with letters AND digits (0244c20c); keeps C1, dead, add
    (re.compile(r"\b\d+\b"), WILD),                                                  # plain numbers
]


def mask(msg: str) -> str:
    for rx, tok in MASKS:
        msg = rx.sub(tok, msg)
    return re.sub(r"\s+", " ", msg).strip()


def new_miner() -> TemplateMiner:
    cfg = TemplateMinerConfig()
    cfg.load(str(CONFIGS / "drain3.ini"))
    cfg.profiling_enabled = False
    return TemplateMiner(config=cfg)


def fit(parquet: Path, train_end: pd.Timestamp | None) -> TemplateMiner:
    miner = new_miner()
    pf = pq.ParquetFile(parquet)
    n_seen = 0
    for b in pf.iter_batches(columns=["timestamp", "message"], batch_size=500_000):
        df = b.to_pandas()
        if train_end is not None:
            df = df[pd.to_datetime(df["timestamp"], utc=True) < train_end]
        if df.empty:
            continue
        for m in df["message"].map(mask).unique():     # distinct messages per batch: same clustering, ~10x faster
            miner.add_log_message(m)
        n_seen += len(df)
        print(f"\r  fitted on {n_seen:,} lines, {len(miner.drain.clusters):,} templates", end="", flush=True)
    print()
    return miner


def assign(parquet: Path, miner: TemplateMiner, out: Path, index_of: dict[int, int], split: dict) -> pd.DataFrame:
    """Match every line against the frozen miner; also id every distinct masked text. Returns the masked-message table."""
    pf = pq.ParquetFile(parquet)
    writer = None
    tpl_cache: dict[str, int] = {}
    masked_id: dict[str, int] = {}
    masked_count: dict[str, int] = {}
    masked_first: dict[str, str] = {}
    order = {"train": 0, "validation": 1, "test": 2}
    total = unk = 0
    for b in pf.iter_batches(columns=["timestamp", "node", "message", "is_alert", "alert_tag"], batch_size=500_000):
        df = b.to_pandas()
        masked = df["message"].map(mask)
        sp = assign_split(pd.to_datetime(df["timestamp"], utc=True), split)
        tids, mids = [], []
        for m, s in zip(masked, sp):
            tid = tpl_cache.get(m)
            if tid is None:
                c = miner.match(m, full_search_strategy="fallback")
                tid = index_of[c.cluster_id] if c is not None else UNK
                tpl_cache[m] = tid
            tids.append(tid)
            mid = masked_id.get(m)
            if mid is None:
                mid = len(masked_id); masked_id[m] = mid; masked_first[m] = s or "none"
            elif s is not None and order.get(s, 9) < order.get(masked_first[m], 9):
                masked_first[m] = s
            masked_count[m] = masked_count.get(m, 0) + 1
            mids.append(mid)
        df["template_index"] = pd.Series(tids, index=df.index, dtype="int32")
        df["masked_id"] = pd.Series(mids, index=df.index, dtype="int32")
        total += len(df); unk += int((df["template_index"] == UNK).sum())
        table = pa.Table.from_pandas(df[["timestamp", "node", "template_index", "masked_id", "is_alert", "alert_tag"]], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out, table.schema, compression="zstd")
        writer.write_table(table)
        print(f"\r  assigned {total:,} lines ({unk / max(total, 1):.2%} UNK so far, {len(masked_id):,} distinct masked texts)", end="", flush=True)
    writer.close(); print()
    mm = pd.DataFrame({"masked_id": list(masked_id.values()), "text": list(masked_id.keys())})
    mm["n_lines"] = mm["text"].map(masked_count); mm["first_split"] = mm["text"].map(masked_first)
    return mm.sort_values("masked_id").reset_index(drop=True)


def report(events_parquet: Path, templates: pd.DataFrame, mm: pd.DataFrame, split: dict, out_md: Path, label: str) -> None:
    ev = pd.read_parquet(events_parquet, columns=["timestamp", "template_index"])
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True)
    L = [f"# Template report — {label} (auto-generated by src/data/templates.py)\n",
         f"{len(templates):,} templates; PAD=0, UNK=1; {len(mm):,} distinct masked message texts over the whole log.\n",
         "## UNK rate per split (protocol field)\n", "| split | lines | UNK lines | UNK rate |", "|---|---:|---:|---:|"]
    for k in ("train", "validation", "test"):
        m = (ev["timestamp"] >= split[k]["start"]) & (ev["timestamp"] < split[k]["end"])
        n = int(m.sum()); u = int((ev.loc[m, "template_index"] == UNK).sum())
        L.append(f"| {k} | {n:,} | {u:,} | {u / max(n, 1):.3%} |")
    L.append("\n## Log evolution: distinct masked message texts by the split in which they FIRST appear\n")
    fs = mm.groupby("first_split").agg(texts=("masked_id", "size"), lines=("n_lines", "sum")).reindex(["train", "validation", "test"]).fillna(0).astype(int)
    L.append(fs.to_markdown() + "\n")
    L.append("## Top 20 message texts that first appear AFTER the training period\n")
    new = mm[mm.first_split.isin(["validation", "test"])].sort_values("n_lines", ascending=False).head(20)
    L.append(new[["first_split", "n_lines", "text"]].assign(text=lambda d: d.text.str.slice(0, 140)).to_markdown(index=False) + "\n")
    L.append("## Top 25 templates by training-period size\n")
    L.append(templates.sort_values("size_in_train", ascending=False).head(25)[["template_index", "size_in_train", "template"]]
             .assign(template=lambda d: d.template.str.slice(0, 140)).to_markdown(index=False))
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(f"[templates] report → {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--split", default="split_v1")
    ap.add_argument("--transductive-control", action="store_true",
                    help="fit Drain on ALL periods (leaky protocol of most prior work); outputs suffixed _tc; CONTROL ONLY")
    args = ap.parse_args()
    sfx = "_sample" if args.sample else ""
    tc = "_tc" if args.transductive_control else ""
    split = load_split(DATA_SPLITS / f"{args.split}{sfx}.json")
    parquet = DATA_INTERIM / f"bgl_parsed{sfx}.parquet"

    miner = fit(parquet, None if args.transductive_control else split["train"]["end"])
    clusters = sorted(miner.drain.clusters, key=lambda c: c.cluster_id)
    index_of = {c.cluster_id: i + 2 for i, c in enumerate(clusters)}          # 2 … K+1
    templates = pd.DataFrame({"template_index": [index_of[c.cluster_id] for c in clusters],
                              "drain_cluster_id": [c.cluster_id for c in clusters],
                              "size_in_train": [c.size for c in clusters],
                              "template": [c.get_template() for c in clusters]})
    templates.to_csv(DATA_INTERIM / f"templates{sfx}{tc}.csv", index=False)
    with (DATA_INTERIM / f"drain_miner{sfx}{tc}.pkl").open("wb") as f:
        pickle.dump({"miner": miner, "index_of": index_of,
                     "fitted_before": None if args.transductive_control else split["train"]["end"].isoformat(),
                     "transductive_control": args.transductive_control}, f)
    label = "TRANSDUCTIVE CONTROL (Drain fitted on all data — leaky, control only)" if args.transductive_control else "inductive (Drain fitted on train only)"
    print(f"[templates] {len(templates):,} templates — {label}")

    mm = assign(parquet, miner, DATA_INTERIM / f"events{sfx}{tc}.parquet", index_of, split)
    mm.to_parquet(DATA_INTERIM / f"masked_messages{sfx}{tc}.parquet", index=False)
    report(DATA_INTERIM / f"events{sfx}{tc}.parquet", templates, mm, split, RESULTS / f"template_report{sfx}{tc}.md", label)


if __name__ == "__main__":
    main()
