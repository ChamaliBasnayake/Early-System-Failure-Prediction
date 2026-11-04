"""Append an interpretation to the `learned` field of an experiment-log row (never overwrites the automatic text).

Usage: python -m src.utils.annotate_log --run lstm_v1_N5_s11 --note "Validation loss minimum at epoch 3; PR-AUC peak at epoch 10 is a single-epoch spike; ..."
       python -m src.utils.annotate_log --contains lstm --N 5 --seed 11 --phase 4 --note "..."   (filter by family/N/seed/phase)
"""
from __future__ import annotations

import argparse
import csv

from src.utils.runlog import COLUMNS, LOG_PATH


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--run", help="exact run_id, or a prefix of it (e.g. the name without the timestamp)")
    ap.add_argument("--contains", help="substring of model_family, purpose or key_hyperparams"); ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None); ap.add_argument("--phase", type=int, default=None)
    ap.add_argument("--note", required=True); args = ap.parse_args()
    rows = list(csv.DictReader(LOG_PATH.open(encoding="utf-8"))); hit = 0
    for r in rows:
        ok = True
        if args.run: ok &= (r["run_id"] == args.run) or r["run_id"].endswith(args.run) or (args.run in r["run_id"])
        if args.contains: ok &= (args.contains in r["model_family"]) or (args.contains in r["key_hyperparams"]) or (args.contains in r["purpose"])
        if args.N is not None: ok &= f"N={args.N};" in r["key_hyperparams"] or r["key_hyperparams"].startswith(f"N={args.N}")
        if args.seed is not None: ok &= str(r["seed"]) == str(args.seed)
        if args.phase is not None: ok &= str(r["phase"]) == str(args.phase)
        if ok:
            r["learned"] = (r["learned"] + " | " if r["learned"] else "") + "note: " + args.note; hit += 1
    with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); w.writerows(rows)
    print(f"[annotate] {hit} row(s) annotated in {LOG_PATH}")


if __name__ == "__main__":
    main()
