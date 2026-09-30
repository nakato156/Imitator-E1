"""SIGReg — Sketched Isotropic Gaussian Regularization (del paper LeJEPA, Balestriero & LeCun).

Empuja la distribución de embeddings del encoder hacia una gaussiana isotrópica N(0, I) para
evitar el colapso. Mecánica:
  1. (opcional) estandarizar cada dimensión del embedding sobre el batch (media 0, var 1).
  2. Sketching: proyectar los embeddings sobre `n_slices` direcciones aleatorias de la esfera
     unidad -> muestras 1D. Si la distribución es N(0,I), cada proyección es ~N(0,1).
  3. Por cada slice, test de normalidad de Epps–Pulley/BHEP: distancia L2 entre la función
     característica empírica (ECF) y la de N(0,1) (= exp(-t²/2)), integrada con peso e^{-t²/2}
     vía cuadratura de Gauss–Hermite (`n_freqs` nodos). Estadístico de gradiente acotado.
  4. La pérdida es el promedio del estadístico sobre los slices.

Un colapso dimensional hace que las proyecciones sobre direcciones "vacías" se desvíen de
N(0,1) -> estadístico alto -> el gradiente empuja a re-expandir esas direcciones.

Hiperparámetro principal: el peso lambda con que se suma a la pérdida de predicción.
"""
from __future__ import annotations

import numpy as np
import torch

from .collapse_metrics import _flatten_valid

# Nodos/pesos de Gauss–Hermite "probabilista" (HermiteE): ∫ f(t) e^{-t²/2} dt ≈ Σ w_k f(t_k).
# Es exactamente el peso del test de Epps–Pulley. Se cachean por n_freqs.
_GH_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
# Direcciones de sketching cacheadas (cuando resample_slices=False).
_DIR_CACHE: dict[tuple, torch.Tensor] = {}


def _gauss_hermite(n_freqs: int):
    if n_freqs not in _GH_CACHE:
        t, w = np.polynomial.hermite_e.hermegauss(n_freqs)
        _GH_CACHE[n_freqs] = (
            torch.tensor(t, dtype=torch.float32),
            torch.tensor(w, dtype=torch.float32),
        )
    return _GH_CACHE[n_freqs]


def sigreg_loss(
    embs: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    n_slices: int = 1024,
    n_freqs: int = 17,
    standardize: bool = True,
    resample_slices: bool = True,
    gather_fn=None,
    eps: float = 1e-6,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pérdida SIGReg (escalar). `embs` (B,T,D) o (M,D); `mask` (B,T) con True=padding.

    gather_fn: opcional (p.ej. accelerator.gather) para agregar las proyecciones entre GPUs.
    """
    x = _flatten_valid(embs, mask)            # (M, D)
    M, D = x.shape
    if M < 2:
        return embs.new_zeros(())

    if standardize:
        x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + eps)

    # Direcciones aleatorias en la esfera unidad de R^D.
    key = (D, n_slices)
    if resample_slices or key not in _DIR_CACHE:
        U = torch.randn(D, n_slices, device=x.device, dtype=torch.float32, generator=generator)
        U = U / (U.norm(dim=0, keepdim=True) + eps)
        if not resample_slices:
            _DIR_CACHE[key] = U
    else:
        U = _DIR_CACHE[key].to(x.device)

    proj = x.float() @ U                      # (M, n_slices), cada columna ~ N(0,1) si isotrópico

    if gather_fn is not None:                 # agregación entre procesos (distribuido)
        proj = gather_fn(proj)

    t, w = _gauss_hermite(n_freqs)
    t = t.to(proj.device); w = w.to(proj.device)

    # ECF: media de cos/sin de (t_k * proj) sobre las muestras.
    ang = t.view(-1, 1, 1) * proj.unsqueeze(0)   # (K, M, S)
    c = ang.cos().mean(dim=1)                     # (K, S)
    s = ang.sin().mean(dim=1)                     # (K, S)

    target = torch.exp(-0.5 * t.pow(2)).view(-1, 1)   # CF de N(0,1) (real)
    per_freq = (c - target).pow(2) + s.pow(2)         # |ECF - φ_N|²   (K, S)
    stat = (w.view(-1, 1) * per_freq).sum(dim=0)      # estadístico por slice (S,)
    return stat.mean()
