"""Smoke tests for Phase 0. Run with:  pytest -q"""
import csv
from pathlib import Path

from src.utils import paths
from src.utils.runlog import COLUMNS, log_run
from src.utils.seeds import PROJECT_SEEDS, set_seed


def test_seeds_are_three_fixed_values():
    assert PROJECT_SEEDS == (11, 22, 33)


def test_set_seed_makes_numpy_reproducible():
    import numpy as np
    set_seed(11); a = np.random.rand(5)
    set_seed(11); b = np.random.rand(5)
    assert (a == b).all()


def test_log_run_writes_a_row(tmp_path: Path):
    log = tmp_path / "log.csv"
    rid = log_run(path=log, phase=0, model_family="dummy", purpose="smoke test",
                  changed_vs_previous="n/a", key_hyperparams="none", seed=11,
                  split="validation", outcome="control", learned="logging works")
    rows = list(csv.DictReader(log.open()))
    assert len(rows) == 1 and rows[0]["run_id"] == rid
    assert list(rows[0].keys()) == COLUMNS


def test_log_run_refuses_test_split_before_phase7(tmp_path: Path):
    import pytest
    with pytest.raises(ValueError):
        log_run(path=tmp_path / "log.csv", phase=4, model_family="lstm", purpose="x",
                changed_vs_previous="x", key_hyperparams="x", seed=11, split="test",
                outcome="success", learned="x")


def test_project_folders_exist():
    for p in (paths.DATA_RAW, paths.DATA_SPLITS, paths.RESULTS, paths.EXPERIMENTS):
        assert p.exists()
