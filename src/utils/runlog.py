"""The experiment log.

The brief (Section 6, "Experiment log") requires that *every* substantive
training or evaluation run is recorded, including failures, with what was
changed, the hyper-parameters, the validation result and what was learned.
Any number in the report must be traceable to a row here.

Usage at the end of every script::

    from src.utils.runlog import log_run
    log_run(
        phase=4, model_family="lstm", purpose="first LSTM, N=5",
        changed_vs_previous="baseline neural model", config_path="configs/lstm_v1.yaml",
        key_hyperparams="lr=1e-3;batch=128;hidden=128;layers=1;W=10;N=5",
        seed=11, split="validation",
        val_pr_auc=0.31, val_recall_at_far=0.42, val_f1=0.28,
        train_time_s=412, infer_ms_per_window=0.8,
        outcome="success",
        learned="Validation loss flattens after epoch 8; curves suggest underfitting. Next: wider hidden.",
    )

The log is a plain CSV so it can be opened in Excel and pasted into the report appendix.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_PATH = Path(__file__).resolve().parents[2] / "experiments" / "log.csv"

COLUMNS = [
    "run_id", "date_time", "phase", "model_family", "purpose", "changed_vs_previous",
    "config_path", "key_hyperparams", "seed", "split",
    "val_pr_auc", "val_recall_at_far", "val_f1",
    "train_time_s", "infer_ms_per_window", "outcome", "learned",
]

ALLOWED_OUTCOMES = {"success", "failed", "control", "pending"}
ALLOWED_SPLITS = {"train", "validation", "test"}


def _make_run_id(model_family: str, key_hyperparams: str, seed: Any) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short = key_hyperparams.replace("=", "").replace(";", "_").replace(" ", "")[:40]
    return f"{stamp}_{model_family}_{short}_s{seed}"


def ensure_log_exists(path: Path = LOG_PATH) -> None:
    """Create the CSV with headers if it does not exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()


def log_run(path: Path = LOG_PATH, **fields: Any) -> str:
    """Append one row to the experiment log and return its run_id.

    Missing optional metrics are written as empty strings. Required fields
    are checked so that an incomplete row cannot be written by accident.
    """
    required = ["phase", "model_family", "purpose", "changed_vs_previous",
                "key_hyperparams", "seed", "split", "outcome", "learned"]
    missing = [k for k in required if k not in fields or fields[k] in (None, "")]
    if missing:
        raise ValueError(f"log_run: missing required fields {missing}")
    if fields["outcome"] not in ALLOWED_OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(ALLOWED_OUTCOMES)}")
    if fields["split"] not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}")
    if fields["split"] == "test" and int(fields["phase"]) not in (7, 10):
        raise ValueError("The test split may only be logged from Phase 7 (final evaluation) "
                         "or Phase 10 (GenAI on the same test windows). Brief, Section 6.")

    ensure_log_exists(path)
    row = {c: "" for c in COLUMNS}
    row.update({k: v for k, v in fields.items() if k in COLUMNS})
    row["run_id"] = fields.get("run_id") or _make_run_id(
        fields["model_family"], fields["key_hyperparams"], fields["seed"])
    row["date_time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)
    return row["run_id"]
