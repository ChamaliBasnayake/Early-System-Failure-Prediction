"""Backend configuration (environment variables override defaults)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(ROOT / 'app' / 'backend' / 'app.db').as_posix()}")
# One checkpoint per lead time N. Missing N falls back to the closest available (reported in the response).
MODEL_RUNS = {int(k): v for k, v in (p.split(":") for p in os.getenv("MODEL_RUNS", "5:lstm_v1_N5_s11").split(","))}
CHECKPOINT_DIR = ROOT / "results" / "checkpoints"; SCORES_DIR = ROOT / "results" / "scores"; INTERIM = ROOT / "data" / "interim"
WINDOW_MIN = int(os.getenv("WINDOW_MIN", "60")); MAX_ITEMS = 256; MC_SAMPLES = int(os.getenv("MC_SAMPLES", "20"))
FAR_LEVELS = (0.001, 0.01)
MAX_LINES = int(os.getenv("MAX_LINES", "200000"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "lstm_v1 (Phase 4) — will be replaced by the Phase 5 final model")
LIMITATIONS = [
    "Trained on one system (IBM BlueGene/L at LLNL, June–September 2005). Other systems, or BlueGene/L after its October 2005 software upgrade, produce messages the model has never seen; the 'unknown template' rate shown is the warning sign.",
    "Predicts kernel-level node failures (machine check, panic, control-socket loss, TLB storm) on a MIDPLANE within the chosen lead time. Application errors, filesystem faults and partition-wide events are not targets.",
    "Validation performance is modest (PR-AUC ≈ 0.03 vs 0.01 for a constant; ROC ≈ 0.65). The score is a ranking signal, not a calibrated probability; the Monte-Carlo spread is an instability indicator, not a confidence interval.",
    "The model sees only the last 60 minutes of logs. It cannot know about maintenance actions, hardware swaps or jobs scheduled outside the log.",
    "Thresholds correspond to 1.4 and 14 false alarms per midplane per day on validation data; actual false-alarm rates in other periods may differ.",
]
