"""Phase 8 tests: Integrated Gradients completeness and item deletion on a tiny model."""
import numpy as np
import pandas as pd
import torch

from src.models.lstm import TwoChannelLSTM
from src.models.seq_data import WindowDataset, collate
from src.xai.integrated_gradients import integrated_gradients, top_items
from src.xai.perturbation import delete_items


def _batch():
    df = pd.DataFrame({"tpl_local": [[2, 3, 2, 5]], "mid_local": [[7, 8, 7, 9]], "cnt_local": [[1, 4, 1, 2]], "dt_local": [[0, 10, 5, 3]],
                       "tpl_global": [[2, 1, 4, 4, 6]], "mid_global": [[7, 11, 12, 12, 13]], "cnt_global": [[1, 2, 1, 9, 1]], "dt_global": [[0, 3, 1, 1, 2]],
                       "label_5": [1], "incident_id": ["I1"]})
    ds = WindowDataset(df, np.array([[0.5, -1.0]], dtype=np.float32), 5, 256, 256, scaler=(np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)))
    return collate([ds[0]])


def test_ig_completeness_and_shapes():
    torch.manual_seed(0); m = TwoChannelLSTM(vocab=16, n_scalars=2, embed_dim=8, hidden=16)
    loc, glo, S, y = _batch(); ig = integrated_gradients(m, loc, glo, S, steps=100)
    assert ig["attr_local"].shape == (4,) and ig["attr_global"].shape == (5,) and ig["attr_scalars"].shape == (2,)
    assert abs(ig["completeness_gap"]) < 0.05 * max(1.0, abs(ig["logit"] - ig["baseline_logit"]))       # IG sums to the logit difference


def test_delete_items_keeps_alignment():
    loc, glo, S, y = _batch(); l2, g2 = delete_items(loc, glo, [0, 2], [4])
    assert int(l2[4][0]) == 2 and l2[0].tolist()[0] == [3, 5] and int(g2[4][0]) == 4 and g2[0].tolist()[0] == [2, 1, 4, 4]
    l3, g3 = delete_items(loc, glo, [0, 1, 2, 3], []); assert int(l3[4][0]) == 0


def test_top_items_only_positive():
    assert top_items(np.array([0.5, -0.2, 0.1, 0.0]), k=3) == [0, 2]


def test_semantic_add_texts_grows_frozen_table(tmp_path):
    """Deployment path: never-seen masked texts are embedded with the frozen encoder and appended; nothing trainable changes."""
    import numpy as np, torch
    from src.models.semantic import TwoChannelSemanticLSTM
    E = np.random.default_rng(0).normal(size=(10, 384)).astype(np.float32); f = tmp_path / "emb.npy"; np.save(f, E)
    m = TwoChannelSemanticLSTM(vocab=5, n_scalars=3, embedding_file=str(f)); before = m.proj.weight.clone()
    added = m.add_texts({12: "new message <*>", 10: "another"}, lambda texts: np.ones((len(texts), 384), dtype=np.float32))
    assert added == 2 and m.n_texts == 13 and not m.table.weight.requires_grad
    assert torch.allclose(m.table.weight[12], torch.ones(384)) and torch.allclose(m.table.weight[11], torch.zeros(384))
    assert torch.equal(m.proj.weight, before)
    assert m.add_texts({3: "already present"}, lambda t: np.zeros((len(t), 384))) == 0
