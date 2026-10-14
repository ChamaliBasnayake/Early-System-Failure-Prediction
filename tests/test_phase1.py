"""Phase 1 tests: parsing and the failure-set logic on tiny synthetic input."""
from pathlib import Path

import pandas as pd

from src.data.parse_bgl import parse_file, parse_line
from src.data.failure_set import build_episodes, DEFAULT_CFG

GOOD = "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected\n"
FATAL = "KERNDTLB 1117838580 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.43.00.000000 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL data TLB error interrupt\n"
BAD = "this line is malformed\n"


def test_parse_line_keeps_message_spaces():
    parts = parse_line(GOOD)
    assert len(parts) == 10 and parts[0] == "-" and parts[9] == "instruction cache parity error corrected"


def test_parse_line_rejects_malformed():
    assert parse_line(BAD) is None


def test_parse_file_counts_and_types(tmp_path: Path, monkeypatch):
    import src.data.parse_bgl as pb
    monkeypatch.setattr(pb, "RESULTS", tmp_path)
    log = tmp_path / "x.log"; log.write_text(GOOD + FATAL + BAD + GOOD)
    out = tmp_path / "x.parquet"
    stats = parse_file(log, out)
    assert stats["lines_total"] == 4 and stats["lines_malformed_dropped"] == 1
    df = pd.read_parquet(out)
    assert len(df) == 3 and df.is_alert.sum() == 1
    assert str(df.rack.iloc[0]) == "R02" and str(df.nodecard.iloc[0]) == "R02-M1-N0"


T0 = pd.Timestamp("2005-06-03T15:43:00Z")


def _row(s, cat, node="N1", comp="KERNEL"):
    return dict(timestamp=T0 + pd.Timedelta(seconds=s), node=node, alert_tag=cat, level="FATAL", component=comp, is_alert=True)


def test_episodes_mass_incidents_and_pools():
    rows = [_row(0, "KERNMC"), _row(5, "KERNDTLB"), _row(30, "KERNDTLB"),      # N1: one episode, two categories
            _row(900, "KERNSTOR"), _row(905, "KERNSTOR"),                       # N1: second episode (gap 870 s > 300)
            _row(0, "APPSEV", comp="APP")]                                      # non-target → hard-negative pool
    rows += [_row(2000 + i % 3, "KERNSTOR", node=f"M{i}") for i in range(40)]  # 40 nodes within 3 s → mass incident
    ep, mass, nontarget = build_episodes(pd.DataFrame(rows), DEFAULT_CFG)
    assert len(ep) == 2 and list(ep.n_events) == [3, 2]
    assert ep.first_category.iloc[0] == "KERNMC" and ep.dominant_category.iloc[0] == "KERNDTLB"
    assert ep.categories.iloc[0] == "KERNDTLB|KERNMC"
    assert len(mass) == 1 and mass.n_nodes.iloc[0] == 40 and mass.category.iloc[0] == "KERNSTOR"
    assert len(nontarget) == 1 and nontarget.alert_tag.iloc[0] == "APPSEV"
    assert ep.incident_id.nunique() == 2
