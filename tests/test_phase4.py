"""Phase 4 tests: collate/padding, right-aligned last state, empty channels, parameter count, MC dropout."""
import numpy as np
import pandas as pd
import torch

from src.models.lstm import TwoChannelLSTM, count_parameters
from src.models.seq_data import WindowDataset, collate


def _df():
    return pd.DataFrame({
        "tpl_local": [[2, 3, 2], [], [5]], "cnt_local": [[1, 4, 1], [], [2]], "dt_local": [[0, 10, 5], [], [0]],
        "tpl_global": [[2, 1, 4, 4], [3], []], "cnt_global": [[1, 2, 1, 9], [7], []], "dt_global": [[0, 3, 1, 1], [0], []],
        "label_5": [1, 0, 0], "incident_id": ["I1", "", ""]})


def test_dataset_collate_shapes_and_alignment():
    df = _df(); S = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 1.0]], dtype=np.float32)
    ds = WindowDataset(df, S, 5, 256, 256)
    (T, C, D, lens), (Tg, Cg, Dg, lg), Sb, y = collate([ds[i] for i in range(3)])
    assert T.shape == (3, 3) and lens.tolist() == [3, 0, 1] and lg.tolist() == [4, 1, 0]
    assert T[2].tolist() == [0, 0, 5]            # right-aligned: last position holds the last item
    assert abs(float(C[0, 1]) - np.log1p(4)) < 1e-6 and Sb.shape == (3, 2) and y.tolist() == [1.0, 0.0, 0.0]
    assert abs(float(Sb[:, 0].mean())) < 1e-5     # standardised with train stats


def test_model_forward_handles_empty_channels_and_counts_params():
    df = _df(); S = np.zeros((3, 2), dtype=np.float32); ds = WindowDataset(df, S, 5, 256, 256)
    loc, glo, Sb, y = collate([ds[i] for i in range(3)])
    torch.manual_seed(0); m = TwoChannelLSTM(vocab=10, n_scalars=2, embed_dim=8, hidden=16)
    out = m(loc, glo, Sb); assert out.shape == (3,) and torch.isfinite(out).all()
    assert count_parameters(m) > 0
    p, sd = m.predict(loc, glo, Sb, mc_samples=5); assert p.shape == (3,) and sd.shape == (3,) and (p >= 0).all() and (p <= 1).all()


def test_last_state_depends_on_last_item_not_padding():
    torch.manual_seed(0); m = TwoChannelLSTM(vocab=10, n_scalars=1, embed_dim=8, hidden=16).eval()
    def enc(seq):
        T = torch.tensor([seq]); C = torch.ones(1, len(seq)); D = torch.zeros(1, len(seq)); L = torch.tensor([len(seq)])
        return m.local(T, C, D, L)
    a = enc([0, 0, 2, 3]); L = torch.tensor([2]); b = m.local(torch.tensor([[0, 0, 2, 3]]), torch.ones(1, 4), torch.zeros(1, 4), L)
    c = m.local(torch.tensor([[2, 3]]), torch.ones(1, 2), torch.zeros(1, 2), L)
    assert torch.allclose(b, c, atol=1e-5)        # two leading PADs with len=2 ≡ the 2-item sequence
