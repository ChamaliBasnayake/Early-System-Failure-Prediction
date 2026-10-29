"""Integrated Gradients for the two-channel sequence models (Question 5.b).

What is explained: how much each ITEM of the local and global channels (a collapsed run of one message type)
and each SCALAR context feature pushed the logit up or down, relative to a baseline window.
Baseline: the "empty" window — every item embedding replaced by the PAD embedding (what the model sees for
"nothing was logged") and scalars at 0 (= the training mean after standardisation). Attributions are summed
over the embedding dimension, so one number per item; by the completeness property they sum (approximately)
to logit(window) − logit(baseline), which is checked and reported.

Why IG rather than attention or SHAP: the LSTM has no attention weights; KernelSHAP over 2×256 items needs
thousands of forward passes per window; IG needs `steps` (50) forward+backward passes and is defined for any
differentiable input. Limitation: IG measures sensitivity along one path from the baseline — it does not
prove that a line CAUSED the prediction, and correlated items can share or hide attribution.
Works for TwoChannelLSTM, TwoChannelCNN (template embedding) and TwoChannelSemanticLSTM (projected text embedding).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _embed_fn(model: nn.Module):
    """Return (embed_items(T, M) → E, forward_from_embeddings(E_l, E_g, loc, glo, S) → logit)."""
    if hasattr(model, "table"):                                        # semantic model: frozen table → projection
        def embed(T, M): return model.proj(model.table(M))
    else:
        def embed(T, M): return model.embed(T)

    def forward(E_l, E_g, loc, glo, S):
        T, M, C, D, L = loc; Tg, Mg, Cg, Dg, Lg = glo
        xl = torch.cat([E_l, C.unsqueeze(-1), D.unsqueeze(-1)], -1); xg = torch.cat([E_g, Cg.unsqueeze(-1), Dg.unsqueeze(-1)], -1)
        return model.head(torch.cat([model.local(xl, L), model.glob(xg, Lg), S], -1))
    return embed, forward


@torch.no_grad()
def _baseline_like(model: nn.Module, E: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "table"):
        pad = model.proj(model.table.weight[:1] * 0)                   # projection of a zero text vector
        return pad.expand_as(E)
    return model.embed(torch.zeros_like(T))                           # PAD embedding


def integrated_gradients(model: nn.Module, loc, glo, S, steps: int = 50) -> dict:
    """Attributions for ONE window (batch size 1). Returns per-item scores for both channels and per-scalar scores."""
    model.eval(); embed, forward = _embed_fn(model)
    T, M, C, D, L = loc; Tg, Mg, Cg, Dg, Lg = glo
    with torch.no_grad():
        E_l, E_g = embed(T, M), embed(Tg, Mg)
        B_l, B_g, B_s = _baseline_like(model, E_l, T), _baseline_like(model, E_g, Tg), torch.zeros_like(S)
        f_x = forward(E_l, E_g, loc, glo, S).item(); f_b = forward(B_l, B_g, loc, glo, B_s).item()
    grads_l = torch.zeros_like(E_l); grads_g = torch.zeros_like(E_g); grads_s = torch.zeros_like(S)
    for k in range(1, steps + 1):
        a = k / steps
        xl = (B_l + a * (E_l - B_l)).detach().requires_grad_(True); xg = (B_g + a * (E_g - B_g)).detach().requires_grad_(True)
        xs = (B_s + a * (S - B_s)).detach().requires_grad_(True)
        out = forward(xl, xg, loc, glo, xs); out.sum().backward()
        grads_l += xl.grad; grads_g += xg.grad; grads_s += xs.grad
    att_l = ((E_l - B_l) * grads_l / steps).sum(-1)[0]; att_g = ((E_g - B_g) * grads_g / steps).sum(-1)[0]; att_s = ((S - B_s) * grads_s / steps)[0]
    # only valid (non-padding) positions are meaningful: right-aligned → last L items
    Ll, Lg_ = int(L[0]), int(Lg[0])
    att_l = att_l[-Ll:] if Ll else att_l[:0]; att_g = att_g[-Lg_:] if Lg_ else att_g[:0]
    total = float(att_l.sum() + att_g.sum() + att_s.sum())
    return {"logit": f_x, "baseline_logit": f_b, "score": float(torch.sigmoid(torch.tensor(f_x))), "baseline_score": float(torch.sigmoid(torch.tensor(f_b))),
            "attr_local": att_l.detach().numpy().astype(float), "attr_global": att_g.detach().numpy().astype(float), "attr_scalars": att_s.detach().numpy().astype(float),
            "completeness_gap": float((f_x - f_b) - total), "steps": steps}


def top_items(attr: np.ndarray, k: int = 5) -> list[int]:
    """Indices of the k items with the largest POSITIVE attribution (pushing towards 'failure')."""
    order = np.argsort(-attr); return [int(i) for i in order[:k] if attr[i] > 0]
