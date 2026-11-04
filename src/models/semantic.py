"""Semantic two-channel LSTM (Phase 5 transfer-learning variant).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.lstm import FusionHead, RNNEncoder, TwoChannelLSTM


class TwoChannelSemanticLSTM(nn.Module):
    def __init__(self, vocab: int, n_scalars: int, hidden=64, layers=1, dropout=0.3, layernorm=True, pooling="last", proj_dim=64, embedding_file="data/interim/masked_embeddings_minilm.npy"):
        super().__init__()
        E = torch.from_numpy(np.load(embedding_file).astype(np.float32))                 # (n_masked_texts, 384)
        self.table = nn.Embedding.from_pretrained(E, freeze=True)
        self.proj = nn.Linear(E.shape[1], proj_dim)
        self.local = RNNEncoder(proj_dim + 2, hidden, layers, layernorm, pooling, dropout); self.glob = RNNEncoder(proj_dim + 2, hidden, layers, layernorm, pooling, dropout)
        self.head = FusionHead(self.local.out_dim + self.glob.out_dim + n_scalars, dropout)

    @property
    def n_texts(self) -> int: return self.table.weight.shape[0]

    def add_texts(self, texts_by_id: dict[int, str], encode) -> int:
        """Deployment path: embed masked texts the table has never seen (ids ≥ n_texts) with the SAME frozen encoder and
        append them. `encode(list[str]) -> array (n, 384)`. Returns the number of rows added. Nothing is trained."""
        new = {int(i): t for i, t in texts_by_id.items() if int(i) >= self.n_texts or not torch.any(self.table.weight[int(i)] != 0)}
        if not new: return 0
        W = self.table.weight.data; top = max(new) + 1
        if top > W.shape[0]: W = torch.cat([W, torch.zeros(top - W.shape[0], W.shape[1], dtype=W.dtype)], 0)
        ids = sorted(new); W[ids] = torch.as_tensor(np.asarray(encode([new[i] for i in ids]), dtype=np.float32))
        self.table = nn.Embedding.from_pretrained(W, freeze=True); return len(ids)

    def _steps(self, M, C, D): return torch.cat([self.proj(self.table(M)), C.unsqueeze(-1), D.unsqueeze(-1)], -1)

    def forward(self, loc, glo, S):
        T, M, C, D, L = loc; Tg, Mg, Cg, Dg, Lg = glo
        return self.head(torch.cat([self.local(self._steps(M, C, D), L), self.glob(self._steps(Mg, Cg, Dg), Lg), S], -1))

    predict = TwoChannelLSTM.predict
