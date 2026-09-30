"""Tests para TemporalLiftPooling (v119 — LiftSign §3.2.2, ablation A3).

Reemplazo aprendible de max-pooling temporal: split par/impar + predictor/updater
(esquema de lifting de wavelets) + Local Weighting gate. Dos auxiliary losses
(L_u, L_p) estabilizan el entrenamiento (ecuación 2 del paper).
"""
import importlib.util
from pathlib import Path

import torch

# Carga directa del archivo (igual que test_loss_infonce.py): importar el módulo
# por su ruta de paquete dispararía src/mslm/models/__init__.py, que arrastra
# Imitator -> Trainer -> unsloth (dependencia circular pesada para un test CPU).
_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "tlp", _ROOT / "src/mslm/models/components/tlp.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TemporalLiftPooling = _mod.TemporalLiftPooling

B, C = 2, 8


def test_output_halves_even_length():
    tlp = TemporalLiftPooling(C)
    x = torch.randn(B, 20, C)
    lengths = torch.full((B,), 20, dtype=torch.long)
    out, out_lengths, aux = tlp(x, lengths)
    assert out.shape == (B, 10, C)


def test_output_halves_odd_length():
    tlp = TemporalLiftPooling(C)
    x = torch.randn(B, 21, C)
    lengths = torch.full((B,), 21, dtype=torch.long)
    out, out_lengths, aux = tlp(x, lengths)
    assert out.shape == (B, 11, C)


def test_lengths_downsampled_match_ceil_half():
    tlp = TemporalLiftPooling(C)
    x = torch.randn(B, 20, C)
    lengths = torch.tensor([20, 7])  # uno con padding real (longitud válida 7 de 20)
    out, out_lengths, aux = tlp(x, lengths)
    assert out_lengths.tolist() == [10, 4]  # ceil(20/2)=10, ceil(7/2)=4


def test_aux_losses_present_and_finite():
    tlp = TemporalLiftPooling(C)
    x = torch.randn(B, 16, C)
    lengths = torch.full((B,), 16, dtype=torch.long)
    _, _, aux = tlp(x, lengths)
    assert "L_u" in aux and "L_p" in aux
    assert torch.isfinite(aux["L_u"])
    assert torch.isfinite(aux["L_p"])


def test_stacking_two_reduces_length_by_four():
    tlp1 = TemporalLiftPooling(C)
    tlp2 = TemporalLiftPooling(C)
    x = torch.randn(B, 40, C)
    lengths = torch.full((B,), 40, dtype=torch.long)
    x, lengths, _ = tlp1(x, lengths)
    x, lengths, _ = tlp2(x, lengths)
    assert x.shape == (B, 10, C)
    assert lengths.tolist() == [10, 10]


def test_forward_does_not_crash_when_output_length_is_one():
    """Clips cortos pueden quedar en T=1 tras el pooling (ej. T=2 de entrada,
    visto en producción con CTCEncoder sobre clips reales de pocos frames).
    nn.InstanceNorm1d exige >1 elemento espacial en modo training y crashea
    con ValueError -- la normalización del gate debe tolerar T=1."""
    tlp = TemporalLiftPooling(C)
    x = torch.randn(1, 2, C)
    lengths = torch.tensor([2])
    out, out_lengths, aux = tlp(x, lengths)
    assert out.shape == (1, 1, C)
    assert torch.isfinite(out).all()


def test_gradients_flow_to_input():
    tlp = TemporalLiftPooling(C)
    x = torch.randn(B, 20, C, requires_grad=True)
    lengths = torch.full((B,), 20, dtype=torch.long)
    out, _, aux = tlp(x, lengths)
    (out.sum() + aux["L_u"] + aux["L_p"]).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
