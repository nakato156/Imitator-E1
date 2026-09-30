"""Tests para CTCEncoder (v119) — arquitectura PORTADA de Min et al. ("A Closer
Look at Skeleton-based CSLR", ICCVW 2025) y LiftSign (CVPRW 2026): GCN multi-capa
-> TCN (K3-P2-K3-P2, P2=TLP) -> BiLSTM -> clasificador COMPARTIDO con supervisión
CTC dual (Y_s sobre la salida del TCN, Y_l sobre la salida del BiLSTM). Reemplaza
el diseño anterior (1 STGCNBlock + Transformer de 6 capas + 1 sola cabeza), que
colapsó a blank tanto en 1000 como en 5600 clips reales (ver report.md).

Carga directa de archivos (igual que test_contrastive_aligner.py) para evitar el
import circular real de src/mslm/models/__init__.py <-> src/mslm/utils/__init__.py.
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name, file_path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_package(name, dir_path):
    spec = importlib.util.spec_from_file_location(
        name, dir_path / "__init__.py", submodule_search_locations=[str(dir_path)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


models_stub = types.ModuleType("src.mslm.models")
models_stub.__path__ = [str(_ROOT / "src/mslm/models")]
sys.modules["src.mslm.models"] = models_stub

_load_package("src.mslm.models.components", _ROOT / "src/mslm/models/components")
_load_module("src.mslm.models.components.stgcn", _ROOT / "src/mslm/models/components/stgcn.py")
_load_module("src.mslm.models.components.tlp", _ROOT / "src/mslm/models/components/tlp.py")
_ctc_mod = _load_module("src.mslm.models.ctc_encoder", _ROOT / "src/mslm/models/ctc_encoder.py")
CTCEncoder = _ctc_mod.CTCEncoder


N = 4  # nodos de juguete
A = np.array(
    [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=np.float32,
)
B, T, V_SIZE, HIDDEN = 2, 32, 5, 16


def _make_encoder(**kwargs):
    return CTCEncoder(
        A=A, input_size=2, gcn_channels=(8, 8, 8), hidden_size=HIDDEN,
        lstm_layers=1, vocab_size=V_SIZE, **kwargs,
    )


def _make_batch(lengths=(32, 20)):
    x = torch.randn(B, T, N, 2)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, length:] = True
    return x, mask, torch.tensor(lengths)


def test_forward_returns_dual_log_probs_same_shape():
    model = _make_encoder()
    x, mask, _ = _make_batch()
    log_probs_s, log_probs_l, seq_lengths, aux = model(x, mask)
    assert log_probs_s.shape == log_probs_l.shape
    assert log_probs_s.shape[0] == B
    assert log_probs_s.shape[-1] == V_SIZE + 1


def test_forward_downsamples_T_by_four():
    model = _make_encoder()
    x, mask, lengths = _make_batch(lengths=(32, 20))
    log_probs_s, log_probs_l, seq_lengths, aux = model(x, mask)
    assert log_probs_s.size(1) == T // 4  # dos TLP en cascada (K3-P2-K3-P2)
    expected = ((lengths + 1) // 2 + 1) // 2
    assert torch.equal(seq_lengths, expected)


def test_classifier_is_shared_between_short_and_long_term():
    model = _make_encoder()
    classifier_params = [n for n, _ in model.named_parameters() if "classifier" in n]
    # Un solo nn.Linear (weight + bias) reusado para Y_s y Y_l, no dos cabezas.
    assert len(classifier_params) == 2


def test_log_probs_sum_to_one_after_exp():
    model = _make_encoder()
    x, mask, _ = _make_batch()
    log_probs_s, log_probs_l, _, _ = model(x, mask)
    for lp in (log_probs_s, log_probs_l):
        probs_sum = lp.exp().sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4)


def test_aux_losses_present_from_tlp():
    model = _make_encoder()
    x, mask, _ = _make_batch()
    _, _, _, aux = model(x, mask)
    assert "L_u" in aux and "L_p" in aux
    assert torch.isfinite(aux["L_u"]) and torch.isfinite(aux["L_p"])


def test_forward_with_motion_stream():
    model = _make_encoder(use_motion_stream=True)
    x, mask, lengths = _make_batch()
    log_probs_s, log_probs_l, seq_lengths, aux = model(x, mask)
    assert log_probs_l.size(1) == T // 4
    assert model.stgcn_motion_layers is not None


def test_gradients_flow_through_both_heads_to_gcn():
    model = _make_encoder(use_motion_stream=True)
    x, mask, _ = _make_batch()
    log_probs_s, log_probs_l, _, _ = model(x, mask)
    (log_probs_s.sum() + log_probs_l.sum()).backward()
    first_gcn_layer = model.stgcn_layers[0]
    assert first_gcn_layer.gconv.weight.grad is not None
    assert torch.isfinite(first_gcn_layer.gconv.weight.grad).all()
    assert model.classifier.weight.grad is not None


def test_bilstm_respects_padding_no_leak_from_longer_sample():
    """El sample corto (length=12) no debe verse afectado por el padding del
    sample largo (length=32) cuando se procesan juntos en un batch -- requiere
    pack_padded_sequence antes del BiLSTM (sin esto, la dirección hacia atrás
    del BiLSTM leakea el padding del sample largo al sample corto)."""
    torch.manual_seed(0)
    model = _make_encoder()
    model.eval()

    x_short = torch.randn(1, 12, N, 2)
    mask_short = torch.zeros(1, 12, dtype=torch.bool)

    x_batch = torch.zeros(2, 32, N, 2)
    x_batch[0, :12] = x_short[0]
    x_batch[1] = torch.randn(32, N, 2)
    mask_batch = torch.zeros(2, 32, dtype=torch.bool)
    mask_batch[0, 12:] = True

    with torch.no_grad():
        _, lp_long_alone, len_alone, _ = model(x_short, mask_short)
        _, lp_long_batch, len_batch, _ = model(x_batch, mask_batch)

    valid_len = len_alone[0].item()
    # atol holgado (no 1e-4): InstanceNorm1d dentro de TLP calcula sus stats
    # sobre TODAS las posiciones del tensor, incluyendo el padding derivado del
    # sample largo -- una fuga pequeña y ya documentada, distinta de la fuga
    # grande que causaría un BiLSTM sin pack_padded_sequence (que este test
    # SÍ detecta: sin packing el diff máximo sube de ~0.006 a >>0.1). El
    # pipeline real (train_ctc_v119.py::_encode_batch) llama siempre con B=1
    # sin padding, así que esta fuga de InstanceNorm no aplica en producción.
    assert torch.allclose(
        lp_long_alone[0, :valid_len], lp_long_batch[0, :valid_len], atol=0.01
    )
