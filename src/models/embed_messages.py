"""Build frozen sentence embeddings for every distinct masked message text (Phase 5 semantic variant).

Model: sentence-transformers/all-MiniLM-L6-v2 (384-d, ~90 MB, runs on CPU; downloaded to HF_HOME on first use).
Output: data/interim/masked_embeddings_minilm.npy with row i = embedding of masked_id i (float32),
        results/embedding_report.md (coverage: texts first seen in train / validation / test).
Inductive: the encoder is pretrained on general text; nothing here is fitted on our data. Embedding texts
that first appear in validation/test is legitimate — at inference time the message text IS the input.
Usage: python -m src.models.embed_messages [--sample] [--model sentence-transformers/all-MiniLM-L6-v2]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.utils.paths import DATA_INTERIM, RESULTS


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", action="store_true"); ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=256); args = ap.parse_args(); sfx = "_sample" if args.sample else ""
    from sentence_transformers import SentenceTransformer
    mm = pd.read_parquet(DATA_INTERIM / f"masked_messages{sfx}.parquet").sort_values("masked_id")
    assert (mm["masked_id"].values == np.arange(len(mm))).all(), "masked_id must be contiguous 0..n-1"
    texts = mm["text"].fillna("").str.replace("<*>", "", regex=False).str.replace(r"\s+", " ", regex=True).str.strip().tolist()
    model = SentenceTransformer(args.model, device="cpu"); t0 = time.time()
    E = model.encode(texts, batch_size=args.batch, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    np.save(DATA_INTERIM / f"masked_embeddings_minilm{sfx}.npy", E)
    fs = mm.groupby("first_split").size().reindex(["train", "validation", "test"]).fillna(0).astype(int)
    md = ["# Message embedding report (auto-generated)\n", f"Encoder: {args.model} (frozen, 384-d, L2-normalised). {len(texts):,} distinct masked texts embedded in {time.time() - t0:.0f} s on CPU.\n",
          "Texts by the split in which they first appear (all are embedded; the encoder saw none of this data in training):\n", fs.to_frame("texts").to_markdown(), "",
          "Note: wildcard tokens `<*>` are removed before encoding so that embeddings reflect the message words only."]
    (RESULTS / f"embedding_report{sfx}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[embed] {E.shape} → masked_embeddings_minilm{sfx}.npy; report → results/embedding_report{sfx}.md")


if __name__ == "__main__":
    main()
