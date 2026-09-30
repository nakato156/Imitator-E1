"""Unit tests para SIGReg (corren sin datos ni GPU)."""
import importlib.util
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]

# Cargamos ambos módulos por ruta (funciones puras, solo torch/numpy) para no arrastrar el
# __init__ del paquete. Se inyecta _flatten_valid en sigreg sin ejecutar su import relativo.
_cm_spec = importlib.util.spec_from_file_location(
    "collapse_metrics", _ROOT / "src/mslm/training/collapse_metrics.py")
_cm = importlib.util.module_from_spec(_cm_spec)
_cm_spec.loader.exec_module(_cm)

_src = (_ROOT / "src/mslm/training/sigreg.py").read_text()
_src = _src.replace("from .collapse_metrics import _flatten_valid", "")
_ns = {"_flatten_valid": _cm._flatten_valid}
exec(compile(_src, "sigreg.py", "exec"), _ns)
sigreg_loss = _ns["sigreg_loss"]


def test_isotropic_gaussian_has_low_sigreg():
    torch.manual_seed(0)
    x = torch.randn(2000, 64)                     # ~N(0, I)
    loss = sigreg_loss(x, n_slices=256, standardize=True)
    assert float(loss) < 0.02


def test_dimensional_collapse_has_high_sigreg():
    torch.manual_seed(0)
    direction = torch.randn(64)
    x = torch.randn(2000, 1) * direction          # rango 1 (colapso dimensional)
    loss = sigreg_loss(x, n_slices=256, standardize=True)
    assert float(loss) > 0.1


def test_collapsed_much_larger_than_gaussian():
    torch.manual_seed(1)
    g = sigreg_loss(torch.randn(2000, 64), n_slices=256)
    direction = torch.randn(64)
    c = sigreg_loss(torch.randn(2000, 1) * direction, n_slices=256)
    assert float(c) > 5 * float(g)


def test_gradient_flows_to_embeddings():
    x = torch.randn(500, 32, requires_grad=True)
    loss = sigreg_loss(x, n_slices=128)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_mask_and_3d_shape():
    embs = torch.randn(4, 10, 16)
    mask = torch.zeros(4, 10, dtype=torch.bool)
    mask[:, 7:] = True
    loss = sigreg_loss(embs, mask=mask, n_slices=64)
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_isotropic_gaussian_has_low_sigreg()
    test_dimensional_collapse_has_high_sigreg()
    test_collapsed_much_larger_than_gaussian()
    test_gradient_flows_to_embeddings()
    test_mask_and_3d_shape()
    print("OK: todos los tests de sigreg pasaron")
