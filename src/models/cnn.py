"""Two-channel 1-D CNN variant (Phase 5 architecture alternative).

Per channel: [emb ‖ log1p(cnt) ‖ log1p(dt)] → LayerNorm → three parallel Conv1d (kernels 3/5/7, F filters each)
→ ReLU → masked global max-pool → concat (3F). Same fusion head as the LSTM.
Assumption: the predictive evidence is LOCAL PATTERNS of a few consecutive message types (n-grams), wherever
they occur in the window, rather than long-range order — the hypothesis Hadadi et al. (2024) found best on
their failure-prediction data.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.lstm import FusionHead, TwoChannelLSTM


class CNNEncoder(nn.Module):
    def __init__(self, d_in: int, filters: int, kernels: list[int], layernorm: bool):
        super().__init__()
        self.norm = nn.LayerNorm(d_in) if layernorm else nn.Identity()
        self.convs = nn.ModuleList([nn.Conv1d(d_in, filters, k, padding=k // 2) for k in kernels])
        self.out_dim = filters * len(kernels)

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        x = self.norm(x); B, L, _ = x.shape
        valid = (torch.arange(L, device=x.device).unsqueeze(0) >= (L - lens).unsqueeze(1))        # right-aligned validity mask
        h = x.transpose(1, 2)                                                                       # (B, d_in, L)
        outs = []
        for conv in self.convs:
            c = torch.relu(conv(h))[:, :, :L]                                                       # (B, F, L)
            c = c.masked_fill(~valid.unsqueeze(1), -1e4)
            outs.append(c.max(-1).values)
        rep = torch.cat(outs, -1)
        return rep.masked_fill((lens == 0).unsqueeze(1), 0.0)


class TwoChannelCNN(nn.Module):
    def __init__(self, vocab: int, n_scalars: int, embed_dim=32, filters=64, kernels=(3, 5, 7), dropout=0.3, layernorm=True):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.local = CNNEncoder(embed_dim + 2, filters, list(kernels), layernorm); self.glob = CNNEncoder(embed_dim + 2, filters, list(kernels), layernorm)
        self.head = FusionHead(self.local.out_dim + self.glob.out_dim + n_scalars, dropout)

    def _steps(self, T, C, D): return torch.cat([self.embed(T), C.unsqueeze(-1), D.unsqueeze(-1)], -1)

    def forward(self, loc, glo, S):
        T, M, C, D, L = loc; Tg, Mg, Cg, Dg, Lg = glo
        return self.head(torch.cat([self.local(self._steps(T, C, D), L), self.glob(self._steps(Tg, Cg, Dg), Lg), S], -1))

    predict = TwoChannelLSTM.predict
