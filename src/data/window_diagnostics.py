"""Phase 2 diagnostic (v2) — content by class and channel, and the unseen (UNK) texts near failures.

Writes results/window_diagnostics.md.   Usage: python -m src.data.window_diagnostics [--sample] [--N 5]
"""
from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

from src.utils.paths import DATA_INTERIM, RESULTS


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", action="store_true"); ap.add_argument("--N", type=int, default=5)
    args = ap.parse_args(); sfx = "_sample" if args.sample else ""
    win = pd.read_parquet(DATA_INTERIM / f"windows{sfx}.parquet")
    mm = pd.read_parquet(DATA_INTERIM / f"masked_messages{sfx}.parquet").set_index("masked_id")
    lab = f"label_{args.N}"; win = win[win[lab] >= 0].copy()
    win["cls"] = win[lab].map({1: "positive", 0: "negative"})
    L = ["# Window diagnostics v2 (auto-generated)\n", f"Lead time N={args.N}; kept windows only.\n"]
    for ch in ("local", "global"):
        g = win.groupby(["split", "anchor"]).agg(windows=("t_ns", "size"), events=(f"n_{ch}", "sum"), unk=(f"n_unk_{ch}", "sum"),
                                                 nonempty=(f"n_{ch}", lambda s: round((s > 0).mean(), 3)))
        g["unk_rate"] = (g["unk"] / g["events"].clip(lower=1)).round(4)
        L.append(f"## {ch} channel by split × anchor\n" + g.to_markdown() + "\n")

    def top_texts(sub, ch, only_unk, k=15):
        c = Counter()
        for tpls, mids, cnts in zip(sub[f"tpl_{ch}"], sub[f"mid_{ch}"], sub[f"cnt_{ch}"]):
            for t, m, n in zip(tpls, mids, cnts):
                if (not only_unk) or t == 1: c[m] += int(n)
        return pd.DataFrame([(mm.loc[m, "text"][:110], mm.loc[m, "first_split"], n) for m, n in c.most_common(k)],
                            columns=["masked text", "first seen", "lines"])

    for sp in ("train", "validation", "test"):
        for ch in ("local", "global"):
            pos = win[(win.split == sp) & (win.cls == "positive")]; neg = win[(win.split == sp) & (win.cls == "negative")]
            L.append(f"## {sp} / {ch}: top texts in POSITIVE windows ({len(pos)})\n" + top_texts(pos, ch, False).to_markdown(index=False) + "\n")
            L.append(f"## {sp} / {ch}: top texts in NEGATIVE windows ({len(neg)})\n" + top_texts(neg, ch, False).to_markdown(index=False) + "\n")
            if sp != "train":
                L.append(f"## {sp} / {ch}: top UNSEEN (UNK) texts in POSITIVE windows\n" + top_texts(pos, ch, True).to_markdown(index=False) + "\n")
    (RESULTS / f"window_diagnostics{sfx}.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[windiag] → {RESULTS / f'window_diagnostics{sfx}.md'}")


if __name__ == "__main__":
    main()
