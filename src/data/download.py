"""Phase 1, step 1 — download the BGL dataset from Zenodo and verify it.

The brief (Section 8, Deliverables) requires a reproducible download procedure
because the raw log exceeds 500 MB. This script is that procedure.

Usage:
    python -m src.data.download            # downloads, verifies MD5, unzips
    python -m src.data.download --force    # re-download even if present

Source:  Loghub collection, Zenodo record 8196385 (CC BY 4.0).
Cite:    Zhu et al., "Loghub: A Large Collection of System Log Datasets for
         AI-driven Log Analytics", ISSRE 2023;  Oliner & Stearley, "What
         Supercomputers Say: A Study of Five System Logs", DSN 2007.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

from src.utils.paths import DATA_RAW

URL = "https://zenodo.org/records/8196385/files/BGL.zip?download=1"
ZIP_PATH = DATA_RAW / "BGL.zip"
LOG_PATH = DATA_RAW / "BGL.log"
EXPECTED_MD5 = "4452953c470f2d95fcb32d5f6e733f7a"   # from the Zenodo file listing


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _progress(block_num: int, block_size: int, total: int) -> None:
    done = block_num * block_size
    if total > 0:
        pct = min(100, done * 100 // total)
        sys.stdout.write(f"\r  downloading… {done / 1e6:6.1f} / {total / 1e6:6.1f} MB ({pct}%)")
        sys.stdout.flush()


def download(force: bool = False) -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and not force:
        print(f"[download] {ZIP_PATH} already present, skipping download")
    else:
        print(f"[download] fetching {URL}")
        urllib.request.urlretrieve(URL, ZIP_PATH, reporthook=_progress)
        print()
    actual = md5sum(ZIP_PATH)
    if actual != EXPECTED_MD5:
        raise SystemExit(f"[download] MD5 mismatch: expected {EXPECTED_MD5}, got {actual}. "
                         "Delete data/raw/BGL.zip and re-run.")
    print(f"[download] MD5 OK ({actual})")
    return ZIP_PATH


def extract(zip_path: Path) -> Path:
    if LOG_PATH.exists():
        print(f"[extract] {LOG_PATH} already present, skipping extraction")
        return LOG_PATH
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f"[extract] archive contains: {names}")
        # The archive holds BGL.log (full) and may hold a small BGL_2k.log sample + README.
        log_members = [n for n in names if n.lower().endswith(".log") and "2k" not in n.lower()]
        if not log_members:
            raise SystemExit("[extract] no full .log file found in the archive")
        member = log_members[0]
        print(f"[extract] extracting {member} → {LOG_PATH}")
        with zf.open(member) as src, LOG_PATH.open("wb") as dst:
            for block in iter(lambda: src.read(1 << 20), b""):
                dst.write(block)
    size_mb = LOG_PATH.stat().st_size / 1e6
    print(f"[extract] done, {size_mb:.0f} MB")
    return LOG_PATH


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if the zip exists")
    args = ap.parse_args()
    extract(download(force=args.force))


if __name__ == "__main__":
    main()
