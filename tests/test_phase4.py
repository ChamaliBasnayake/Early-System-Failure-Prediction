"""Phase 4/5 tests: collate/padding, right-aligned last state, empty channels, variants, MC dropout."""
import numpy as np
import pandas as pd
import torch

from src.models.cnn import TwoChannelCNN
from src.models.lstm import TwoChannelLSTM, count_parameters
from src.models.seq_data import WindowDataset, collate


def _df():
    return pd.DataFrame({
        "tpl_local": [[2, 3, 2], [], [5]], "mid_local": [[7, 8, 7], [], [9]], "cnt_local": [[1, 4, 1], [], [2]], "dt_local": [[0, 10, 5], [], [0]],
        "tpl_global": [[2, 1, 4, 4], [3], []], "mid_global": [[7, 11, 12, 12], [8], []], "cnt_global": [[1, 2, 1, 9], [7], []], "dt_global": [[0, 3, 1, 1], [0], []],
        "label_5": [1, 0, 0], "incident_id": ["I1", "", ""]})


def _batch():
    df = _df(); S = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 1.0]], dtype=np.float32)
    ds = WindowDataset(df, S, 5, 256, 256); return ds, collate([ds[i] for i in range(3)])


def test_dataset_collate_shapes_and_alignment():
    ds, ((T, M, C, D, lens), (Tg, Mg, Cg, Dg, lg), Sb, y) = _batch()
    assert T.shape == (3, 3) and M.shape == (3, 3) and lens.tolist() == [3, 0, 1] and lg.tolist() == [4, 1, 0]
    assert T[2].tolist() == [0, 0, 5] and M[2].tolist() == [0, 0, 9]          # right-aligned
    assert abs(float(C[0, 1]) - np.log1p(4)) < 1e-6 and Sb.shape == (3, 2) and y.tolist() == [1.0, 0.0, 0.0]
    assert abs(float(Sb[:, 0].mean())) < 1e-5


def test_lstm_forward_empty_channels_params_mc():
    ds, (loc, glo, Sb, y) = _batch()
    torch.manual_seed(0); m = TwoChannelLSTM(vocab=10, n_scalars=2, embed_dim=8, hidden=16)
    out = m(loc, glo, Sb); assert out.shape == (3,) and torch.isfinite(out).all() and count_parameters(m) > 0
    p, sd = m.predict(loc, glo, Sb, mc_samples=5); assert p.shape == (3,) and sd.shape == (3,) and (p >= 0).all() and (p <= 1).all()


def test_cnn_forward():
    ds, (loc, glo, Sb, y) = _batch()
    torch.manual_seed(0); m = TwoChannelCNN(vocab=10, n_scalars=2, embed_dim=8, filters=4)
    out = m(loc, glo, Sb); assert out.shape == (3,) and torch.isfinite(out).all()


def test_semantic_forward(tmp_path):
    import numpy as np
    from src.models.semantic import TwoChannelSemanticLSTM
    E = np.random.RandomState(0).randn(20, 12).astype(np.float32); f = tmp_path / "E.npy"; np.save(f, E)
    ds, (loc, glo, Sb, y) = _batch()
    torch.manual_seed(0); m = TwoChannelSemanticLSTM(vocab=10, n_scalars=2, hidden=16, proj_dim=8, embedding_file=str(f))
    out = m(loc, glo, Sb); assert out.shape == (3,) and torch.isfinite(out).all()
    assert not m.table.weight.requires_grad                                    # frozen pretrained table


def test_last_state_depends_on_last_item_not_padding():
    torch.manual_seed(0); m = TwoChannelLSTM(vocab=10, n_scalars=1, embed_dim=8, hidden=16).eval()
    def rep(T, L):
        T = torch.tensor([T]); n = T.shape[1]
        return m.local(m._steps(T, torch.ones(1, n), torch.zeros(1, n)), torch.tensor([L]))
    assert torch.allclose(rep([0, 0, 2, 3], 2), rep([2, 3], 2), atol=1e-5)     # leading PADs with len=2 ≡ the 2-item sequence


def test_legacy_checkpoint_keys_load():
    from src.models.lstm import load_state_dict_compat
    torch.manual_seed(0); m = TwoChannelLSTM(vocab=10, n_scalars=2, embed_dim=8, hidden=16)
    sd = m.state_dict(); legacy = {}
    for k, v in sd.items():
        legacy["head." + k[len("head.net."):] if k.startswith("head.net.") else k] = v
    legacy["local.embed.weight"] = sd["embed.weight"]; legacy["glob.embed.weight"] = sd["embed.weight"]
    m2 = TwoChannelLSTM(vocab=10, n_scalars=2, embed_dim=8, hidden=16); load_state_dict_compat(m2, legacy)
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
