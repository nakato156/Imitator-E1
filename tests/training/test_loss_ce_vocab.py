"""Unit tests para la pérdida CE-vocab (v115.1). Corren sin datos ni GPU."""
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "loss_ce_vocab", _ROOT / "src/mslm/training/loss_ce_vocab.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
imitator_ce_loss = _mod.imitator_ce_loss
IGNORE_INDEX = _mod.IGNORE_INDEX

VOCAB, DIM = 50, 16


def _table():
    g = torch.Generator().manual_seed(7)
    return torch.randn(VOCAB, DIM, generator=g)


def _table_norm():
    return F.normalize(_table(), dim=-1)


def test_perfect_prediction_high_acc():
    table_norm = _table_norm()
    ids = torch.tensor([[3, 11, 42, 7]])
    pred = table_norm[ids] * 5.0
    # With cosine logits, scale_factor needed to separate classes. Use temp=10 to sharpen.
    loss, ce, acc, acc5 = imitator_ce_loss(pred, ids, table_norm, logit_temp=10.0)
    assert acc.item() == 1.0
    assert ce.item() < 1.0


def test_padding_ignored():
    table_norm = _table_norm()
    ids = torch.tensor([[3, 11, IGNORE_INDEX, IGNORE_INDEX]])
    pred_good = table_norm[torch.tensor([[3, 11, 0, 0]])] * 5.0
    loss_a, _, acc_a, _ = imitator_ce_loss(pred_good, ids, table_norm)
    pred_bad_pad = pred_good.clone()
    pred_bad_pad[:, 2:] = -pred_bad_pad[:, 2:]
    loss_b, _, acc_b, _ = imitator_ce_loss(pred_bad_pad, ids, table_norm)
    assert torch.allclose(loss_a, loss_b)
    assert acc_a.item() == acc_b.item() == 1.0


def test_length_alignment():
    table_norm = _table_norm()
    ids = torch.tensor([[3, 11]])
    pred = table_norm[torch.tensor([[3, 11, 5, 9]])]
    loss, _, acc, _ = imitator_ce_loss(pred, ids, table_norm)
    assert acc.item() == 1.0


def test_gradient_flows_to_pred_not_table():
    table_norm = _table_norm()
    ids = torch.tensor([[3, 11, 42, 7]])
    pred = torch.randn(1, 4, DIM, requires_grad=True)
    loss, _, _, _ = imitator_ce_loss(pred, ids, table_norm)
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
    assert table_norm.grad is None


def test_collapsed_prediction_is_penalized():
    """La predicción degenerada (misma salida para todo) debe tener CE peor que la correcta."""
    table_norm = _table_norm()
    ids = torch.tensor([[3, 11, 42, 7], [29, 1, 14, 33]])
    mean_pred = table_norm[ids].mean(dim=(0, 1), keepdim=True).expand(2, 4, DIM)
    correct_pred = table_norm[ids] * 5.0
    loss_mean, _, _, _ = imitator_ce_loss(mean_pred, ids, table_norm)
    loss_correct, _, _, _ = imitator_ce_loss(correct_pred, ids, table_norm)
    assert loss_correct.item() < loss_mean.item()


def test_collate_fn_includes_token_ids():
    _cspec = importlib.util.spec_from_file_location(
        "components", _ROOT / "src/mslm/dataloader/components.py")
    _c = importlib.util.module_from_spec(_cspec)
    _cspec.loader.exec_module(_c)

    batch = [
        (torch.randn(5, 4, 2), torch.randn(3, DIM), torch.tensor([3, 11, 42])),
        (torch.randn(7, 4, 2), torch.randn(2, DIM), torch.tensor([29, 1])),
    ]
    out = _c.collate_fn(batch)
    assert len(out) == 5
    kp, fmask, emb, emask, ids = out
    assert ids.shape == (2, 3)
    assert ids[1, 2].item() == IGNORE_INDEX
    assert ids[0].tolist() == [3, 11, 42]

    batch_legacy = [
        (torch.randn(5, 4, 2), torch.randn(3, DIM), None),
        (torch.randn(7, 4, 2), torch.randn(2, DIM), None),
    ]
    assert len(_c.collate_fn(batch_legacy)) == 4


# --- Tests nuevos (v115.1) ---

def test_returns_4_tuple():
    table_norm = _table_norm()
    ids = torch.tensor([[3, 11, 42, 7]])
    pred = torch.randn(1, 4, DIM)
    result = imitator_ce_loss(pred, ids, table_norm)
    assert len(result) == 4


def test_cosine_logits_gradient_flows_to_log_temp():
    table_norm = _table_norm()
    ids = torch.tensor([[3, 11, 42, 7]])
    pred = torch.randn(1, 4, DIM)
    log_temp = torch.tensor(2.659, requires_grad=True)
    loss, _, _, _ = imitator_ce_loss(pred, ids, table_norm, logit_temp=log_temp)
    loss.backward()
    assert log_temp.grad is not None and log_temp.grad.abs().item() > 0


def test_norm_runaway_does_not_affect_logits():
    """Con logits coseno, escalar pred por 100 no debe cambiar la CE."""
    table_norm = _table_norm()
    ids = torch.tensor([[3, 11, 42, 7]])
    pred = torch.randn(1, 4, DIM)
    loss1, _, _, _ = imitator_ce_loss(pred, ids, table_norm)
    loss100, _, _, _ = imitator_ce_loss(pred * 100.0, ids, table_norm)
    assert torch.allclose(loss1, loss100, atol=1e-4)


def test_top5_geq_top1():
    table_norm = _table_norm()
    ids = torch.tensor([[3, 11, 42, 7]])
    pred = torch.randn(1, 4, DIM)
    _, _, top1, top5 = imitator_ce_loss(pred, ids, table_norm)
    assert top5.item() >= top1.item()


def test_label_smoothing_increases_loss():
    # Use correct predictions (near-zero CE) so smoothing always increases the loss.
    table_norm = _table_norm()
    ids = torch.tensor([[3, 11, 42, 7]])
    pred = table_norm[ids] * 50.0   # very confident correct → CE ≈ 0, smooth pushes it up
    loss_no_smooth, _, _, _ = imitator_ce_loss(pred, ids, table_norm, label_smoothing=0.0)
    loss_smooth, _, _, _    = imitator_ce_loss(pred, ids, table_norm, label_smoothing=0.1)
    assert loss_smooth.item() > loss_no_smooth.item()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK {name}")
    print("Todos los tests pasaron.")
