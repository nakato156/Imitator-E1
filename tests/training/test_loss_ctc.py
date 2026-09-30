"""Tests para ctc_loss (v119 — CTC sobre secuencia, reemplaza el objetivo contrastivo).

Import vía importlib (mismo patrón que test_loss_infonce.py) para no arrastrar
la cadena de imports pesada de src.mslm.training.__init__ (Trainer -> unsloth).
"""
import importlib.util
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "loss_ctc", _ROOT / "src/mslm/training/loss_ctc.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ctc_loss = _mod.ctc_loss

B, T, V = 4, 20, 6  # V = vocab_size (sin contar blank, que vive en el índice 0)


def _make_logits(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, V + 1, generator=g, requires_grad=True)


def test_ctc_loss_returns_scalar():
    logits = _make_logits()
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = torch.randint(1, V + 1, (B, 5), generator=torch.Generator().manual_seed(1))
    target_lengths = torch.full((B,), 5, dtype=torch.long)
    input_lengths = torch.full((B,), T, dtype=torch.long)

    loss = ctc_loss(log_probs, targets.flatten(), input_lengths, target_lengths)
    assert loss.dim() == 0


def test_ctc_loss_is_differentiable():
    logits = _make_logits()
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = torch.randint(1, V + 1, (B, 5), generator=torch.Generator().manual_seed(1))
    target_lengths = torch.full((B,), 5, dtype=torch.long)
    input_lengths = torch.full((B,), T, dtype=torch.long)

    loss = ctc_loss(log_probs, targets.flatten(), input_lengths, target_lengths)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_ctc_loss_lower_for_peaked_correct_alignment():
    """Si los logits están muy concentrados en la secuencia objetivo (repetida para
    llenar T), la pérdida debe ser mucho menor que con logits uniformes (sin señal)."""
    target_ids = [1, 2, 3]
    target_lengths = torch.tensor([3])
    input_lengths = torch.tensor([T])
    targets = torch.tensor(target_ids)

    # Logits "uniformes": ninguna clase domina.
    uniform_logits = torch.zeros(1, T, V + 1)
    uniform_log_probs = torch.log_softmax(uniform_logits, dim=-1)
    uniform_loss = ctc_loss(uniform_log_probs, targets, input_lengths, target_lengths)

    # Logits "peaked": bloques contiguos blank/1/2/3/blank (alineamiento CTC válido
    # que colapsa exactamente a [1, 2, 3], sin pasos de blank intermedios necesarios
    # porque el target no tiene repeticiones consecutivas).
    peaked_logits = torch.full((1, T, V + 1), -10.0)
    path = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [0] * 4
    for t in range(T):
        peaked_logits[0, t, path[t]] = 10.0
    peaked_log_probs = torch.log_softmax(peaked_logits, dim=-1)
    peaked_loss = ctc_loss(peaked_log_probs, targets, input_lengths, target_lengths)

    assert peaked_loss.item() < uniform_loss.item()


def test_ctc_loss_finite_when_target_longer_than_input():
    """zero_infinity=True: con target_length > input_length la pérdida debe quedar
    finita (0), no inf/nan, para no romper el training loop con un batch así."""
    logits = torch.randn(1, 2, V + 1, requires_grad=True)
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = torch.tensor([1, 2, 3, 4, 5])
    target_lengths = torch.tensor([5])
    input_lengths = torch.tensor([2])

    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    assert torch.isfinite(loss)
