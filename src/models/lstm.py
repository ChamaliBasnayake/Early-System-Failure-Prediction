"""Two-channel sequence classifier (Phase 4 v1). Variants: cnn.py (1-D CNN), semantic.py (pretrained text embeddings).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def left_align(x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Right-aligned (padding in front) → left-aligned, for packing."""
    B, L, _ = x.shape
    idx = torch.arange(L, device=x.device).unsqueeze(0)
    gather = ((L - lens.clamp(min=1)).unsqueeze(1) + idx).clamp(max=L - 1)
    return torch.gather(x, 1, gather.unsqueeze(-1).expand(-1, -1, x.size(-1)))


class RNNEncoder(nn.Module):
    """[step features] → LayerNorm → LSTM → pooled representation (zero vector for empty sequences)."""
    def __init__(self, d_in: int, hidden: int, layers: int, layernorm: bool, pooling: str, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_in) if layernorm else nn.Identity()
        self.rnn = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.pooling = pooling; self.out_dim = hidden if pooling == "last" else 2 * hidden

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        x = self.norm(x); B, L, _ = x.shape; empty = lens == 0; lens_c = lens.clamp(min=1)
        packed = pack_padded_sequence(left_align(x, lens), lens_c.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, _) = self.rnn(packed)
        if self.pooling == "last":
            rep = h[-1]
        else:
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=L)
            mask = (torch.arange(L, device=x.device).unsqueeze(0) < lens_c.unsqueeze(1)).unsqueeze(-1)
            rep = torch.cat([(out * mask).sum(1) / lens_c.unsqueeze(1), out.masked_fill(~mask, -1e4).max(1).values], -1)
        return rep.masked_fill(empty.unsqueeze(1), 0.0)


class FusionHead(nn.Module):
    def __init__(self, d_in: int, dropout: float, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, z): return self.net(z).squeeze(-1)


class TwoChannelLSTM(nn.Module):
    def __init__(self, vocab: int, n_scalars: int, embed_dim=32, hidden=64, layers=1, dropout=0.3, layernorm=True, pooling="last"):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.local = RNNEncoder(embed_dim + 2, hidden, layers, layernorm, pooling, dropout)
        self.glob = RNNEncoder(embed_dim + 2, hidden, layers, layernorm, pooling, dropout)
        self.head = FusionHead(self.local.out_dim + self.glob.out_dim + n_scalars, dropout)

    def _steps(self, T, C, D): return torch.cat([self.embed(T), C.unsqueeze(-1), D.unsqueeze(-1)], -1)

    def forward(self, loc, glo, S):
        T, M, C, D, L = loc; Tg, Mg, Cg, Dg, Lg = glo
        return self.head(torch.cat([self.local(self._steps(T, C, D), L), self.glob(self._steps(Tg, Cg, Dg), Lg), S], -1))

    @torch.no_grad()
    def predict(self, loc, glo, S, mc_samples: int = 0):
        """Scores in [0,1]. mc_samples>0 → Monte-Carlo dropout: (mean, std) over stochastic passes."""
        if mc_samples <= 0:
            self.eval(); return torch.sigmoid(self.forward(loc, glo, S)), None
        self.train(); ps = torch.stack([torch.sigmoid(self.forward(loc, glo, S)) for _ in range(mc_samples)]); self.eval()
        return ps.mean(0), ps.std(0)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def load_state_dict_compat(model: nn.Module, state_dict: dict) -> None:
    """Load a checkpoint saved before the Phase-5 refactor (encoders held a reference to the shared embedding,
    the head was a bare Sequential). Keys are renamed; the duplicated embedding references are dropped."""
    fixed = {}
    for k, v in state_dict.items():
        if k in ("local.embed.weight", "glob.embed.weight"):
            continue                                                   # same tensor as embed.weight
        if k.startswith("head.") and not k.startswith("head.net."):
            k = "head.net." + k[len("head."):]
        fixed[k] = v
    model.load_state_dict(fixed, strict=True)
