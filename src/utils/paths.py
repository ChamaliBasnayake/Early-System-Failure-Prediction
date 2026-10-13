"""Central place for project paths so scripts never hard-code them."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_SPLITS = ROOT / "data" / "splits"
DATA_SAMPLE = ROOT / "data" / "sample"
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
EXPERIMENTS = ROOT / "experiments"
DOCS = ROOT / "docs"

for _p in (DATA_RAW, DATA_INTERIM, DATA_SPLITS, DATA_SAMPLE, RESULTS, EXPERIMENTS):
    _p.mkdir(parents=True, exist_ok=True)
