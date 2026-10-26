"""Window → tensor dataset for the sequence models.

Each window has two channels (local, global). A channel is a sequence of items
(template_index, masked_text_id, repeat_count, seconds_since_previous). Per step the
template models feed [embedding(template), log1p(count), log1p(dt)]; the semantic model
feeds [projection(pretrained_text_embedding[masked_id]), log1p(count), log1p(dt)].
Sequences are right-aligned and padded with 0 to the batch's longest length; lengths are
kept for masking/packing so padded positions never influence the encoders.
A small scalar context vector (time + volume features) is concatenated after pooling.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.paths import DATA_INTERIM

PAD, UNK = 0, 1


def load_windows(sfx: str, N: int, split: str, scalar_groups: list[str]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    win = pd.read_parquet(DATA_INTERIM / f"windows{sfx}.parquet")
    feats = pd.read_parquet(DATA_INTERIM / f"features{sfx}.parquet"); groups = json.loads((DATA_INTERIM / f"feature_groups{sfx}.json").read_text())
    cols = [c for g in scalar_groups for c in groups[g]]
    key = ["unit", "t_ns"]
    extra = [c for c in cols if c not in win.columns]            # columns already present in the window table are reused as-is
    df = win.merge(feats[key + extra], on=key, how="left", validate="one_to_one") if extra else win
    df = df[(df.split == split) & (df[f"label_{N}"] >= 0)].reset_index(drop=True)
    X = df[cols].fillna(0).values.astype(np.float32)
    return df, X, cols


class WindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scalars: np.ndarray, N: int, max_local: int, max_global: int, scaler: tuple | None = None):
        self.tl, self.ml, self.cl, self.dl = df["tpl_local"].values, df["mid_local"].values, df["cnt_local"].values, df["dt_local"].values
        self.tg, self.mg, self.cg, self.dg = df["tpl_global"].values, df["mid_global"].values, df["cnt_global"].values, df["dt_global"].values
        self.y = df[f"label_{N}"].values.astype(np.float32)
        self.incident = df["incident_id"].values; self.capl, self.capg = max_local, max_global
        if scaler is None:                                      # standardise scalars with TRAIN statistics only
            mu, sd = scalars.mean(0), scalars.std(0) + 1e-6
        else:
            mu, sd = scaler
        self.scaler = (mu, sd); self.scalars = ((scalars - mu) / sd).astype(np.float32)

    def __len__(self): return len(self.y)

    @staticmethod
    def _chan(t, m, c, d, cap):
        t = np.array(t, dtype=np.int64)[-cap:]; m = np.array(m, dtype=np.int64)[-cap:]
        c = np.array(c, dtype=np.float32)[-cap:]; d = np.array(d, dtype=np.float32)[-cap:]
        return t, m, np.log1p(c), np.log1p(np.clip(d, 0, None))

    def __getitem__(self, i):
        return (self._chan(self.tl[i], self.ml[i], self.cl[i], self.dl[i], self.capl),
                self._chan(self.tg[i], self.mg[i], self.cg[i], self.dg[i], self.capg), self.scalars[i], self.y[i])


def collate(batch):
    def pad_chan(seqs):
        L = max(1, max(len(s[0]) for s in seqs)); B = len(seqs)
        T = torch.zeros(B, L, dtype=torch.long); M = torch.zeros(B, L, dtype=torch.long); C = torch.zeros(B, L); D = torch.zeros(B, L); lens = torch.zeros(B, dtype=torch.long)
        for k, (t, m, c, d) in enumerate(seqs):
            n = len(t); lens[k] = n
            if n:                                                   # right-aligned: padding in front, last item at the end
                T[k, L - n:] = torch.from_numpy(t); M[k, L - n:] = torch.from_numpy(m); C[k, L - n:] = torch.from_numpy(c); D[k, L - n:] = torch.from_numpy(d)
        return T, M, C, D, lens
    loc = pad_chan([b[0] for b in batch]); glo = pad_chan([b[1] for b in batch])
    S = torch.from_numpy(np.stack([b[2] for b in batch])); y = torch.tensor([b[3] for b in batch])
    return loc, glo, S, y
