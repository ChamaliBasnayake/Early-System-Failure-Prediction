"""Phase 1, step 2 — parse the raw BGL log into a typed table (Parquet).

Usage:
    python -m src.data.parse_bgl                       # data/raw/BGL.log → data/interim/bgl_parsed.parquet
    python -m src.data.parse_bgl --sample              # data/sample/bgl_sample.log → data/interim/bgl_parsed_sample.parquet
    python -m src.data.parse_bgl --input path/to.log --output path/to.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.paths import DATA_INTERIM, DATA_RAW, DATA_SAMPLE, RESULTS

COLUMNS = ["alert_tag", "epoch", "date", "node", "datetime_str", "node_repeat",
           "type", "component", "level", "message"]
CHUNK_LINES = 250_000


def parse_line(line: str) -> list[str] | None:
    parts = line.rstrip("\n").split(maxsplit=9)
    if len(parts) < 9:
        return None
    if len(parts) == 9:          # message can be empty on rare lines
        parts.append("")
    return parts


def _to_frame(rows: list[list[str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
    df["timestamp"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df["is_alert"] = df["alert_tag"] != "-"
    # node hierarchy: R02-M1-N0-C:J12-U11 → rack R02, midplane R02-M1, node card R02-M1-N0
    node_parts = df["node"].str.split("-", n=3, expand=True)
    df["rack"] = node_parts[0]
    df["midplane"] = node_parts[0].str.cat(node_parts[1], sep="-", na_rep="")
    df["nodecard"] = df["midplane"].str.cat(node_parts[2], sep="-", na_rep="")
    return df.drop(columns=["date", "datetime_str", "node_repeat"])


def parse_file(input_path: Path, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    n_lines = n_bad = 0
    rows: list[list[str]] = []
    schema = None
    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_lines += 1
            parsed = parse_line(line)
            if parsed is None:
                n_bad += 1
                continue
            rows.append(parsed)
            if len(rows) >= CHUNK_LINES:
                df = _to_frame(rows)
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
                writer.write_table(table.cast(schema))
                rows = []
                print(f"\r  parsed {n_lines:,} lines", end="", flush=True)
    if rows:
        df = _to_frame(rows)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(output_path, schema, compression="zstd")
        writer.write_table(table.cast(schema))
    if writer is not None:
        writer.close()
    print()
    stats = {"input": str(input_path), "output": str(output_path),
             "lines_total": n_lines, "lines_malformed_dropped": n_bad,
             "lines_kept": n_lines - n_bad}
    stats_path = RESULTS / (output_path.stem + "_parse_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[parse] {stats['lines_kept']:,} rows written, {n_bad:,} malformed lines dropped → {output_path}")
    print(f"[parse] stats saved to {stats_path}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="parse data/sample/bgl_sample.log instead of the full log")
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.input:
        inp, out = args.input, args.output or DATA_INTERIM / (args.input.stem + ".parquet")
    elif args.sample:
        inp, out = DATA_SAMPLE / "bgl_sample.log", DATA_INTERIM / "bgl_parsed_sample.parquet"
    else:
        inp, out = DATA_RAW / "BGL.log", DATA_INTERIM / "bgl_parsed.parquet"
    parse_file(inp, out)


if __name__ == "__main__":
    main()
