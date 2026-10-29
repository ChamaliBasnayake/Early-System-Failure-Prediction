"""Turn raw BGL log lines into the model's two-channel window + scalar context, reusing the training code paths.

Same steps as training: parse (src.data.parse_bgl.parse_line) → mask + match against the FROZEN Drain miner
(src.data.templates) → unit = midplane (src.data.precursor_diagnostics.unit_of) → collapse repeats, cap 256
(same rule as src.data.windows) → scalar context in the same column order as the checkpoint, standardised with
the TRAIN scaler stored in the checkpoint.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backend import settings
from src.data.parse_bgl import parse_line
from src.data.precursor_diagnostics import unit_of
from src.data.templates import UNK, mask


@dataclass
class ParsedLog:
    df: pd.DataFrame
    n_input_lines: int
    n_malformed: int
    units: list[str] = field(default_factory=list)


class Featurizer:
    def __init__(self):
        state = pickle.load((settings.INTERIM / "drain_miner.pkl").open("rb"))
        self.miner, self.index_of = state["miner"], state["index_of"]
        mm = pd.read_parquet(settings.INTERIM / "masked_messages.parquet")
        self.masked_id = dict(zip(mm["text"], mm["masked_id"])); self.n_known_texts = len(mm)
        self.next_new_id = int(mm["masked_id"].max()) + 1; self.new_texts: dict[str, int] = {}
        self.cache: dict[str, int] = {}
        self.text_by_id = {int(v): k for k, v in self.masked_id.items()}

    def text_of(self, masked_id: int) -> str | None:
        t = self.text_by_id.get(int(masked_id))
        if t is None:
            for k, v in self.new_texts.items():
                if v == masked_id: return k
        return t

    def parse(self, text: str) -> ParsedLog:
        lines = [l for l in text.splitlines() if l.strip()]
        rows, bad = [], 0
        for l in lines:
            p = parse_line(l + "\n")
            if p is None: bad += 1; continue
            rows.append(p)
        if not rows:
            return ParsedLog(pd.DataFrame(), len(lines), bad)
        df = pd.DataFrame(rows, columns=["alert_tag", "epoch", "date", "node", "datetime_str", "node_repeat", "type", "component", "level", "message"])
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce"); df = df.dropna(subset=["epoch"])
        df["timestamp"] = pd.to_datetime(df["epoch"].astype("int64"), unit="s", utc=True)
        df["unit"] = unit_of(df["node"], "midplane").values
        df["masked"] = df["message"].map(mask)
        df["template_index"] = [self._tpl(m) for m in df["masked"]]
        df["masked_id"] = [self._mid(m) for m in df["masked"]]
        df = df.sort_values("timestamp").reset_index(drop=True)
        return ParsedLog(df, len(lines), bad, sorted(df["unit"].unique().tolist()))

    def _tpl(self, m: str) -> int:
        t = self.cache.get(m)
        if t is None:
            c = self.miner.match(m, full_search_strategy="fallback"); t = self.index_of[c.cluster_id] if c is not None else UNK; self.cache[m] = t
        return t

    def _mid(self, m: str) -> int:
        if m in self.masked_id: return int(self.masked_id[m])
        if m not in self.new_texts: self.new_texts[m] = self.next_new_id; self.next_new_id += 1
        return self.new_texts[m]

    @staticmethod
    def _collapse(tpl, mid, ts_ns, cap):
        if len(tpl) == 0: return [], [], [], []
        change = np.r_[True, mid[1:] != mid[:-1]]; idx = np.flatnonzero(change)
        rep = np.diff(np.r_[idx, len(tpl)]); dt = np.r_[0, np.diff(ts_ns[idx])] / 1e9
        tpl, mid, rep, dt = tpl[idx][-cap:], mid[idx][-cap:], rep[-cap:], dt[-cap:]
        return tpl.tolist(), mid.tolist(), rep.astype(np.int32).tolist(), np.round(dt, 3).tolist()

    def window(self, parsed: ParsedLog, unit: str, window_end: pd.Timestamp | None = None) -> dict:
        df = parsed.df
        t_end = window_end or df["timestamp"].max()
        t0 = t_end - pd.Timedelta(minutes=settings.WINDOW_MIN)
        g = df[(df["timestamp"] > t0) & (df["timestamp"] <= t_end)]
        loc = g[g["unit"] == unit]
        ts = lambda d: d["timestamp"].values.astype("datetime64[ns]").astype(np.int64)
        tl, ml, cl, dl = self._collapse(loc["template_index"].values.astype(np.int64), loc["masked_id"].values.astype(np.int64), ts(loc), settings.MAX_ITEMS)
        tg, mg, cg, dg = self._collapse(g["template_index"].values.astype(np.int64), g["masked_id"].values.astype(np.int64), ts(g), settings.MAX_ITEMS)
        end_local = t_end.tz_convert("America/Los_Angeles"); hour = end_local.hour + end_local.minute / 60
        scal = {"hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24), "weekend": int(end_local.dayofweek >= 5)}
        for d in range(7): scal[f"dow_{d}"] = int(end_local.dayofweek == d)
        scal.update({"n_local": len(loc), "n_global": len(g), "log_n_local": np.log1p(len(loc)), "log_n_global": np.log1p(len(g)),
                     "n_units_active_global": g["unit"].nunique(), "distinct_local": loc["template_index"].nunique(), "distinct_global": g["template_index"].nunique()})
        unk_local = int((loc["template_index"] == UNK).sum()); unk_global = int((g["template_index"] == UNK).sum())
        return {"unit": unit, "window_start": t0, "window_end": t_end, "local": (tl, ml, cl, dl), "global": (tg, mg, cg, dg), "scalars": scal,
                "n_local": len(loc), "n_global": len(g), "unk_rate_local": unk_local / max(len(loc), 1), "unk_rate_global": unk_global / max(len(g), 1),
                "n_new_texts": int(sum(1 for m in g["masked"] if m in self.new_texts)),
                "new_texts_by_id": {int(i): m for m, i in self.new_texts.items() if i in set(int(x) for x in np.r_[ml, mg].astype(np.int64))} if self.new_texts else {},
                "lines": self._lines_with_items(loc, ml, len(ml))}

    @staticmethod
    def _lines_with_items(loc: pd.DataFrame, mid_items: list, n_items: int, keep: int = 300) -> list[dict]:
        """Map each raw local line to the index of the collapsed item it belongs to (runs of equal masked text, last n_items runs)."""
        mids = loc["masked_id"].values
        if len(mids) == 0: return []
        change = np.r_[True, mids[1:] != mids[:-1]]; run_idx = np.cumsum(change) - 1
        n_runs = int(run_idx[-1]) + 1; first_kept = n_runs - n_items          # runs before the cap have no item
        item_of_line = run_idx - first_kept
        rows = []
        for (r, it) in list(zip(loc.itertuples(), item_of_line))[-keep:]:
            rows.append({"timestamp": r.timestamp.isoformat(), "node": r.node, "unit": r.unit, "template_index": int(r.template_index), "message": r.message, "item": int(it) if it >= 0 else None})
        return rows
