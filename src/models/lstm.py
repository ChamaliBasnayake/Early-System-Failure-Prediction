"""Two-channel sequence classifier for failure prediction (Phase 4 v1; Phase 5 adds variants).

                 local items ─► [emb ‖ log1p(cnt) ‖ log1p(dt)] ─► LayerNorm ─► LSTM ─► pool ─┐
                                                                                             ├─► concat ─► Dropout ─► FC ─► ReLU ─► FC ─► logit
                 global items ─► [emb ‖ log1p(cnt) ‖ log1p(dt)] ─► LayerNorm ─► LSTM ─► pool ─┤
                 scalar context (time, volume) ──────────────────────────────────────────────┘

Assumptions this architecture makes about the data (Q2.a):
 * a window is an ORDERED sequence of message types whose order and spacing may matter (LSTM);
 * the same message type means the same thing in both channels (shared embedding);
 * the unit's own history and the system's state are different information sources (two encoders);
 * repeat counts and inter-arrival gaps carry information beyond the message identity (extra channels);
 * the sequence model is not expected to discover time-of-day or raw volume — they are given as scalars.
Right-aligned padding: the LAST hidden state is the state at the window end, which is what the
prediction is about; padding sits at the front and is skipped via packed sequences.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ChannelEncoder(nn.Module):
    def __init__(self, embed: nn.Embedding, hidden: int, layers: int, layernorm: bool, pooling: str, dropout: float):
        super().__init__()
        self.embed = embed; d_in = embed.embedding_dim + 2
        self.norm = nn.LayerNorm(d_in) if layernorm else nn.Identity()
        self.rnn = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.pooling = pooling; self.out_dim = hidden if pooling == "last" else 2 * hidden

    def forward(self, T, C, D, lens):
        x = self.norm(torch.cat([self.embed(T), C.unsqueeze(-1), D.unsqueeze(-1)], dim=-1))     # (B, L, d_in)
        B, L, _ = x.shape
        empty = lens == 0
        lens_c = lens.clamp(min=1)
        # sequences are right-aligned (padding in front): flip so that packing (which expects left-aligned) works,
        # then the LAST valid step after un-flipping is the window end.
        idx = torch.arange(L, device=x.device).unsqueeze(0)
        start = (L - lens_c).unsqueeze(1)
        gather = (start + idx).clamp(max=L - 1)                                                   # left-align each row
        x_la = torch.gather(x, 1, gather.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        packed = pack_padded_sequence(x_la, lens_c.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, _) = self.rnn(packed)
        if self.pooling == "last":
            rep = h[-1]                                                                           # (B, hidden) state at window end
        else:
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=L)
            mask = (idx < lens_c.unsqueeze(1)).unsqueeze(-1)
            mean = (out * mask).sum(1) / lens_c.unsqueeze(1); mx = out.masked_fill(~mask, -1e4).max(1).values
            rep = torch.cat([mean, mx], -1)
        return rep.masked_fill(empty.unsqueeze(1), 0.0)                                           # empty channel → zero vector


class TwoChannelLSTM(nn.Module):
    def __init__(self, vocab: int, n_scalars: int, embed_dim=32, hidden=64, layers=1, dropout=0.3, layernorm=True, pooling="last"):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.local = ChannelEncoder(self.embed, hidden, layers, layernorm, pooling, dropout)
        self.glob = ChannelEncoder(self.embed, hidden, layers, layernorm, pooling, dropout)
        d = self.local.out_dim + self.glob.out_dim + n_scalars
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, loc, glo, S):
        return self.head(torch.cat([self.local(*loc), self.glob(*glo), S], -1)).squeeze(-1)

    @torch.no_grad()
    def predict(self, loc, glo, S, mc_samples: int = 0):
        """Scores in [0,1]. mc_samples>0 → Monte-Carlo dropout: returns (mean, std) over stochastic passes."""
        if mc_samples <= 0:
            self.eval(); return torch.sigmoid(self.forward(loc, glo, S)), None
        self.train()                                   # dropout active; LSTM dropout only matters for layers>1
        ps = torch.stack([torch.sigmoid(self.forward(loc, glo, S)) for _ in range(mc_samples)])
        self.eval(); return ps.mean(0), ps.std(0)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
