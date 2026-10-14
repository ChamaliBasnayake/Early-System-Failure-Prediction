"""Phase 1, step 5 — build the representative sample that ships with the code.

The brief requires a sample that lets the code be tested when the full dataset
is too large to submit. We take a few contiguous time blocks (so sequences stay
realistic) that together contain several failure episodes and normal periods.

Usage:
    python -m src.data.make_sample --lines 20000
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.utils.paths import DATA_INTERIM, DATA_RAW, DATA_SAMPLE
from src.utils.seeds import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=20_000)
    ap.add_argument("--blocks", type=int, default=4, help="number of contiguous blocks")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    set_seed(args.seed)

    ep = pd.read_parquet(DATA_INTERIM / "failure_episodes.parquet")
    df = pd.read_parquet(DATA_INTERIM / "bgl_parsed.parquet", columns=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    ts = df["timestamp"].sort_values().reset_index(drop=True)

    # choose block centres: half around randomly chosen failure episodes, half at random normal times
    centres = list(ep.sample(n=args.blocks // 2, random_state=args.seed)["start"])
    centres += list(ts.sample(n=args.blocks - len(centres), random_state=args.seed))
    per_block = args.lines // args.blocks

    keep_idx: set[int] = set()
    for c in centres:
        pos = ts.searchsorted(c)
        lo, hi = max(0, pos - per_block // 2), min(len(ts), pos + per_block // 2)
        keep_idx.update(range(lo, hi))
    keep = sorted(keep_idx)

    # re-read raw lines in order and write the kept line numbers (raw file order == time order in BGL)
    DATA_SAMPLE.mkdir(parents=True, exist_ok=True)
    out = DATA_SAMPLE / "bgl_sample.log"
    want = set(keep); n = 0
    with (DATA_RAW / "BGL.log").open("r", encoding="utf-8", errors="replace") as f, out.open("w", encoding="utf-8") as g:
        for i, line in enumerate(f):
            if i in want:
                g.write(line); n += 1
    print(f"[sample] wrote {n:,} lines → {out}  (blocks centred on {len(centres)} points)")


if __name__ == "__main__":
    main()
