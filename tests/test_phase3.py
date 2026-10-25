"""Phase 3 tests: metrics (numpy) and feature construction on a tiny synthetic window frame."""
import numpy as np
import pandas as pd

from src.utils.metrics import average_precision, at_threshold, evaluate, roc_auc, threshold_at_far
from src.features.build_features import build


def test_average_precision_and_auc_basic():
    y = np.array([1, 0, 1, 0, 0]); s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert abs(average_precision(y, s) - (1.0 + 2 / 3) / 2) < 1e-9
    assert abs(roc_auc(y, s) - 5 / 6) < 1e-9
    assert np.isnan(average_precision(np.zeros(3), np.ones(3)))


def test_threshold_at_far_and_incident_recall():
    s_neg = np.linspace(0, 1, 1000)
    thr = threshold_at_far(s_neg, 0.01)
    assert (s_neg > thr).mean() <= 0.01 + 1e-9
    assert threshold_at_far(np.full(50, 0.3), 0.01) == 0.3            # constant scores → nothing fires
    y = np.array([1, 1, 1, 0, 0]); s = np.array([0.9, 0.1, 0.95, 0.5, 0.2]); inc = np.array(["A", "A", "B", "", ""])
    m = at_threshold(y, s, 0.6, inc)
    assert m["recall"] == 2 / 3 and m["incident_recall"] == 1.0 and m["n_incidents"] == 2


def test_evaluate_uses_given_thresholds():
    y = np.array([1, 0, 0, 1]); s = np.array([0.7, 0.6, 0.1, 0.2])
    r = evaluate(y, s, thresholds={"0.001": 0.65, "0.01": 0.15})
    assert r["recall@far0.001"] == 0.5 and r["recall@far0.01"] == 1.0
    const = evaluate(np.array([1, 0, 0, 1]), np.full(4, 0.2))
    assert const["recall@far0.01"] == 0.0 and const["far@far0.01"] == 0.0


def test_build_features_shapes():
    t = pd.Timestamp("2005-08-01T12:00:00Z")
    win = pd.DataFrame({
        "unit": ["R01-M0", "R02-M1"], "t_ns": [t.value, t.value], "anchor": ["pos", "uniform"], "unit_episode_id": [0, -1],
        "incident_id": ["KERNMC#1", ""], "category": ["KERNMC", ""], "split": ["train", "train"], "exclusion": ["", ""],
        "window_end": [t, t], "label_5": [1, 0],
        "tpl_local": [[2, 3, 2], []], "cnt_local": [[1, 4, 1], []], "dt_local": [[0, 10, 5], []],
        "tpl_global": [[2, 1], [3]], "cnt_global": [[1, 2], [7]], "dt_global": [[0, 3], [0]],
        "n_local": [6, 0], "n_global": [3, 7], "n_unk_local": [0, 0], "n_unk_global": [2, 0], "has_nontarget_alert_local": [False, False],
        "n_nontarget_alerts_global": [0, 1], "n_target_lines_global": [0, 0], "n_units_active_global": [2, 1]})
    f, groups = build(win, vocab=5)
    assert f.shape[0] == 2 and len(groups["full"]) == len(groups["time"]) + len(groups["volume"]) + len(groups["summary"]) + 10
    assert abs(f.loc[0, "cl_2"] - np.log1p(2)) < 1e-6 and abs(f.loc[0, "cl_3"] - np.log1p(4)) < 1e-6 and f.loc[1, "cl_2"] == 0
    assert f.loc[0, "max_repeat_local"] == 4 and f.loc[1, "mean_dt_local"] == 3600.0
    assert set(groups["confound"]).isdisjoint(set(groups["counts_local"]))
