"""Backend configuration."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(ROOT / 'app' / 'backend' / 'app.db').as_posix()}")
# One checkpoint per lead time N. Missing N falls back to the closest available.
MODEL_RUNS = {int(k): v for k, v in (p.split(":") for p in os.getenv("MODEL_RUNS", "1:sem_final_N1_s11,5:sem_final_N5_s11,15:sem_final_N15_s11,30:sem_final_N30_s11").split(","))}
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")   # frozen encoder for messages never seen in training (semantic model)
CHECKPOINT_DIR = ROOT / "results" / "checkpoints"; SCORES_DIR = ROOT / "results" / "scores"; INTERIM = ROOT / "data" / "interim"
WINDOW_MIN = int(os.getenv("WINDOW_MIN", "60")); MAX_ITEMS = 256; MC_SAMPLES = int(os.getenv("MC_SAMPLES", "20"))
FAR_LEVELS = (0.001, 0.01)
MAX_LINES = int(os.getenv("MAX_LINES", "200000"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "sem_final (Phase 5 final: semantic two-channel LSTM, 101,897 params; seed 11 of 3) — evaluated once on test, tag final-eval-v1")
LIMITATIONS = [
    "Trained on one system (IBM BlueGene/L at LLNL, June–September 2005). Messages never seen in training are embedded on the fly with the same frozen text encoder; the 'unknown template' rate and 'new texts' count shown are the warning signs. On the held-out test period (Nov 2005–Jan 2006, after a software upgrade) 66 % of lines were unseen.",
    "Predicts kernel-level node failures (machine check, panic, control-socket loss, TLB storm) on a MIDPLANE within the chosen lead time. Application errors, filesystem faults and partition-wide events are not targets.",
    "FINAL TEST RESULT (single run, frozen thresholds): at 5 min lead the model's PR-AUC is 0.024 against a prevalence of 0.018 (ROC 0.60); it catches about 5 % of failure incidents at 14 alarms per midplane-day. A gradient-boosting model on template counts plus failure history reached PR-AUC 0.075 (ROC 0.79) on the same windows, mainly because failures in the test period recurred on the same midplanes — a signal this network does not receive. The score is a weak ranking signal, not a calibrated probability.",
    "The model sees only the last 60 minutes of logs. It cannot know about maintenance actions, hardware swaps or jobs scheduled outside the log.",
    "Thresholds correspond to 1.4 and 14 false alarms per midplane per day on validation data; on the test period the ordinary-window false-alarm rate was 0.7–2.2 % and 0.0–0.5 % on alert-rich windows.",
    "A frontier LLM (Gemini 3.8 Flash) evaluated on the same windows ranked below chance (ROC 0.39): it reads a silent midplane as healthy, which in this system is where most failures start.",
]
