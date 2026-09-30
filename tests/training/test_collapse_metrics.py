"""Unit tests para las métricas de colapso (corren sin datos ni GPU)."""
import importlib.util
from pathlib import Path

import torch

# Cargamos collapse_metrics por ruta de archivo: prueba funciones puras (solo torch) y
# evita arrastrar el __init__ del paquete (Trainer/torchtune/etc.) en un test unitario.
_MOD = Path(__file__).resolve().parents[2] / "src" / "mslm" / "training" / "collapse_metrics.py"
_spec = importlib.util.spec_from_file_location("collapse_metrics", _MOD)
_cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cm)
compute_collapse_metrics = _cm.compute_collapse_metrics


def test_diverse_embeddings_have_high_effective_rank():
    torch.manual_seed(0)
    D = 64
    embs = torch.randn(8, 50, D)  # (B, T, D), alta diversidad
    m = compute_collapse_metrics(embs)
    # Embeddings isótropos -> rango efectivo cercano a D.
    assert m["effective_rank"] > 0.7 * D
    assert m["per_dim_collapsed_frac"] == 0.0
    assert abs(m["pairwise_cosine_mean"]) < 0.1  # casi ortogonales en media


def test_constant_collapse_detected_by_std_and_cosine():
    # Colapso a una constante (encoder predice la media): tras centrar, la varianza es
    # ~0, por lo que se detecta con per_dim_std/coseno (no con el effrank, invariante a escala).
    D = 64
    base = torch.randn(D)
    embs = base.repeat(8, 50, 1) + 1e-4 * torch.randn(8, 50, D)  # casi constante
    m = compute_collapse_metrics(embs, per_dim_std_eps=0.01)
    assert m["pairwise_cosine_mean"] > 0.99      # todas apuntan a la misma dirección
    assert m["per_dim_collapsed_frac"] > 0.9     # casi todas las dims con std<eps
    assert m["per_dim_std_mean"] < 0.01


def test_dimensional_collapse_has_low_effective_rank():
    # Colapso dimensional: todas las muestras viven en un subespacio de rango 1.
    D = 64
    direction = torch.randn(D)
    scales = torch.randn(8, 50, 1)               # un escalar aleatorio por muestra
    embs = scales * direction                    # (B, T, D) sobre una recta
    m = compute_collapse_metrics(embs)
    assert m["effective_rank"] < 2.0
    assert m["participation_ratio"] < 2.0


def test_mask_excludes_padding():
    D = 16
    embs = torch.zeros(2, 4, D)
    embs[:, :2] = torch.randn(2, 2, D)  # tokens válidos
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[:, 2:] = True                   # True = padding
    m = compute_collapse_metrics(embs, mask=mask)
    # Sólo se usan 4 tokens válidos; no debe contar el padding (ceros).
    assert m["mean_l2_norm"] > 0.0
    assert not torch.isnan(torch.tensor(m["effective_rank"]))


def test_effrank_ratio_vs_target():
    torch.manual_seed(1)
    D = 32
    target = torch.randn(8, 20, D)            # target diverso (como el LLM)
    collapsed = torch.randn(D).repeat(8, 20, 1)  # predicción colapsada
    m = compute_collapse_metrics(collapsed, target_embs=target)
    assert "effrank_ratio_vs_target" in m
    assert m["effrank_ratio_vs_target"] < 0.25  # diversidad muy por debajo del target


if __name__ == "__main__":
    test_diverse_embeddings_have_high_effective_rank()
    test_constant_collapse_detected_by_std_and_cosine()
    test_dimensional_collapse_has_low_effective_rank()
    test_mask_excludes_padding()
    test_effrank_ratio_vs_target()
    print("OK: todos los tests de collapse_metrics pasaron")
