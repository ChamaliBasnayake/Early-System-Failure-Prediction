"""Phase 2 tests (v2): unit episodes, labels with gap/horizon, exclusions, two-channel content, dedup — synthetic data."""
import numpy as np
import pandas as pd

from src.data.windows import Cfg, dedup, extract_content, label_windows, make_candidates, unit_episodes

T0 = pd.Timestamp("2005-08-01T00:00:00Z")
CFG = Cfg(level="midplane", Wl=10, Wg=10, max_local=256, max_global=256, Ns=[1, 5, 15, 30], H=10, offsets=[0, 2, 4, 6, 8],
          ep_gap_s=300, post_excl=20, uniform_negatives={"train": 30, "validation": 10, "test": 10},
          hard_negatives={"train": 2, "validation": 2, "test": 2}, seed=11)
SPLIT = {"train": {"start": T0, "end": T0 + pd.Timedelta(hours=1)},
         "validation": {"start": T0 + pd.Timedelta(hours=1), "end": T0 + pd.Timedelta(hours=1, minutes=40)},
         "test": {"start": T0 + pd.Timedelta(hours=1, minutes=40), "end": T0 + pd.Timedelta(hours=4)}}
NA, NA2, NB = "R01-M0-N0-C:J01-U01", "R01-M0-N3-C:J05-U11", "R02-M1-N0-C:J01-U01"   # NA, NA2 share midplane R01-M0
S = T0 + pd.Timedelta(hours=3); MT = T0 + pd.Timedelta(hours=3, minutes=30); NT = T0 + pd.Timedelta(minutes=5)


def _data():
    ts = pd.date_range(T0, T0 + pd.Timedelta(hours=4), freq="10s", inclusive="left", tz="UTC")
    ev = pd.concat([
        pd.DataFrame({"timestamp": ts, "node": NA, "template_index": np.repeat(np.arange(2, 7), len(ts) // 5 + 1)[:len(ts)], "masked_id": np.repeat(np.arange(2, 7), len(ts) // 5 + 1)[:len(ts)], "is_alert": False, "alert_tag": "-"}),
        pd.DataFrame({"timestamp": ts, "node": NB, "template_index": 3, "masked_id": 3, "is_alert": False, "alert_tag": "-"}),
        pd.DataFrame({"timestamp": [NT, S, S + pd.Timedelta(seconds=20), MT], "node": [NA, NA, NA2, NB], "template_index": [7, 8, 8, 9], "masked_id": [7, 8, 8, 9],
                      "is_alert": True, "alert_tag": ["APPSEV", "KERNMC", "KERNMC", "KERNSTOR"]})], ignore_index=True)
    ep = pd.DataFrame({"node": [NA, NA2], "start": [S, S + pd.Timedelta(seconds=20)], "end": [S, S + pd.Timedelta(seconds=20)],
                       "first_category": ["KERNMC", "KERNMC"], "incident_id": ["KERNMC#1", "KERNMC#1"]})
    mass = pd.DataFrame({"timestamp": [MT], "node": [NB], "category": ["KERNSTOR"]})
    nontarget = pd.DataFrame({"timestamp": [NT], "node": [NA], "alert_tag": ["APPSEV"]})
    return ev, ep, mass, nontarget


def _build():
    ev, ep, mass, nontarget = _data()
    ue = unit_episodes(ep, CFG)
    units = np.array(["R01-M0", "R02-M1"])
    cand = make_candidates(ue, units, nontarget, SPLIT, CFG, np.random.default_rng(11))
    lab = label_windows(cand, ue, mass, SPLIT, CFG)
    tmask = ev["is_alert"].values & ev["alert_tag"].astype(str).str.startswith("KERN").values & (ev["alert_tag"] != "KERNSTOR").values
    ntmask = ev["is_alert"].values & ~ev["alert_tag"].astype(str).str.startswith("KERN").values
    return ev, ue, lab, extract_content(lab, ev, ntmask, tmask, CFG)


def test_unit_episodes_merge_nodes_of_same_midplane():
    ev, ep, mass, nontarget = _data()
    ue = unit_episodes(ep, CFG)
    assert len(ue) == 1 and ue.unit.iloc[0] == "R01-M0" and ue.n_nodes.iloc[0] == 2


def test_labels_gap_and_horizon():
    _, _, lab, _ = _build()
    pos = lab[lab.anchor == "pos"]
    def L(k):
        r = pos[pos.t_ns == int((S - pd.Timedelta(minutes=k)).value)].iloc[0]; return {N: int(r[f"label_{N}"]) for N in CFG.Ns}
    assert L(5) == {1: 1, 5: 1, 15: -1, 30: -1}
    assert L(13) == {1: 0, 5: 1, 15: -1, 30: -1}
    assert L(21) == {1: 0, 5: 0, 15: 1, 30: -1}
    assert L(30) == {1: 0, 5: 0, 15: 0, 30: 1}


def test_two_channels_and_hard_negative():
    _, _, lab, win = _build()
    h = win[win.anchor == "hard"].iloc[0]
    assert (h[[f"label_{N}" for N in CFG.Ns]] == 0).all() and h.has_nontarget_alert_local and 7 in h.tpl_local
    # local channel of unit R01-M0 contains node NA's lines; global channel also contains NB's template 3
    assert 3 in h.tpl_global and h.n_global > h.n_local
    # window ending 1 min before the failure must not contain the failure line (8) in its LOCAL channel
    w = win[(win.anchor == "pos") & (win.t_ns == int((S - pd.Timedelta(minutes=1)).value))].iloc[0]
    assert 8 not in w.tpl_local and 8 not in w.tpl_global      # the failure is after t, so not in global either


def test_exclusions():
    ev, ep, mass, nontarget = _data(); ue = unit_episodes(ep, CFG)
    extra = pd.DataFrame({"unit": ["R01-M0", "R01-M0", "R02-M1"], "t_ns": [int((S + pd.Timedelta(minutes=3)).value), int((S + pd.Timedelta(minutes=12)).value), int((MT - pd.Timedelta(minutes=5)).value)],
                          "anchor": "uniform", "unit_episode_id": -1, "incident_id": "", "category": ""})
    lab = label_windows(extra, ue, mass, SPLIT, CFG)
    assert lab.exclusion.tolist() == ["episode_in_window", "post_episode", "mass_incident"]


def test_dedup():
    _, _, _, win = _build()
    win2, st = dedup(win)
    assert st["after"] <= st["before"] and "dup_count" in win2.columns
