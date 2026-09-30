"""Métricas de diagnóstico de colapso de embeddings (Fase 1 del experimento SIGReg).

El encoder regresa los embeddings de keypoints hacia embeddings de texto fijos del LLM.
Una solución degenerada (colapso) predice un embedding casi constante e ignora la entrada,
minimizando parcialmente MSE+coseno. Estas métricas detectan ese colapso sin necesidad de
etiquetas: operan sobre los embeddings PREDICHOS y, opcionalmente, los comparan con la
diversidad de los embeddings OBJETIVO del LLM (que son diversos por construcción).

Todas las funciones son puras (sólo torch), corren en CPU/GPU y no requieren datos en disco.
"""
from __future__ import annotations

import torch


def _flatten_valid(embs: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Aplana ``(B, T, D)`` -> ``(M, D)`` quedándose con los tokens válidos.

    Args:
        embs: embeddings ``(B, T, D)`` o ya ``(M, D)``.
        mask: máscara de padding ``(B, T)`` con ``True = padding`` (como en collate_fn).
              Si es ``None`` se consideran todos los tokens válidos.
    """
    if embs.dim() == 2:
        return embs.float()
    if embs.dim() != 3:
        raise ValueError(f"Se esperaba (B, T, D) o (M, D); se recibió {tuple(embs.shape)}")

    B, T, D = embs.shape
    flat = embs.reshape(B * T, D).float()
    if mask is None:
        return flat
    valid = (~mask).reshape(B * T)
    return flat[valid]


def _eigenvalues(x: torch.Tensor) -> torch.Tensor:
    """Autovalores de la covarianza de ``x`` (M, D) vía SVD de la matriz centrada."""
    x = x - x.mean(dim=0, keepdim=True)
    # svdvals es más estable que eig sobre la covarianza explícita.
    s = torch.linalg.svdvals(x)
    denom = max(x.shape[0] - 1, 1)
    return (s ** 2) / denom


def effective_rank(eigvals: torch.Tensor, eps: float = 1e-12) -> float:
    """Rango efectivo = exp(entropía de Shannon de los autovalores normalizados).

    Roy & Vetterli (2007). Cae hacia 1 cuando los embeddings colapsan a una dirección.
    """
    total = eigvals.sum()
    if total <= eps:
        return 0.0
    p = eigvals / total
    p = p[p > eps]
    entropy = -(p * p.log()).sum()
    return float(entropy.exp())


def participation_ratio(eigvals: torch.Tensor, eps: float = 1e-12) -> float:
    """(Σλ)² / Σλ². Medida alternativa de dimensionalidad efectiva (cae a 1 en colapso)."""
    s1 = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    if s2 <= eps:
        return 0.0
    return float((s1 ** 2) / s2)


def pairwise_cosine_mean(x: torch.Tensor, eps: float = 1e-8) -> float:
    """Coseno medio entre pares distintos de muestras. ->1 indica colapso direccional.

    Calculado en O(M·D): ``Σ_{i≠j} ĉos = ||Σ n_i||² - M`` con ``n_i`` los vectores
    normalizados, evitando la matriz M×M.
    """
    M = x.shape[0]
    if M < 2:
        return float("nan")
    n = torch.nn.functional.normalize(x, dim=-1, eps=eps)
    sum_sq = n.sum(dim=0).pow(2).sum()      # ||Σ n_i||²
    off_diag = sum_sq - M                    # restar la diagonal (cada n_i·n_i = 1)
    return float(off_diag / (M * (M - 1)))


def compute_collapse_metrics(
    embs: torch.Tensor,
    mask: torch.Tensor | None = None,
    target_embs: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    per_dim_std_eps: float = 0.01,
) -> dict[str, float]:
    """Calcula el conjunto de métricas de colapso sobre los embeddings predichos.

    Args:
        embs: embeddings predichos ``(B, T, D)`` o ``(M, D)``.
        mask: máscara de padding de ``embs`` (``True = padding``).
        target_embs: embeddings objetivo del LLM, opcional, para comparar diversidad.
        target_mask: máscara de padding de ``target_embs``.
        per_dim_std_eps: umbral por debajo del cual se considera una dimensión colapsada.

    Returns:
        dict con: effective_rank, participation_ratio, pairwise_cosine_mean,
        per_dim_std_mean, per_dim_collapsed_frac, mean_l2_norm y, si hay target,
        target_effective_rank y effrank_ratio_vs_target.

    Nota sobre los modos de colapso:
        - Colapso *dimensional* (las muestras viven en un subespacio de baja dimensión):
          lo detectan ``effective_rank`` y ``participation_ratio`` (sobre la covarianza
          centrada). Cae hacia ~1.
        - Colapso *a constante* (el encoder predice ~la media e ignora la entrada): el
          effrank es invariante a escala y NO lo detecta; se detecta con
          ``per_dim_std_mean``/``per_dim_collapsed_frac`` (varianza ~0) y
          ``pairwise_cosine_mean`` (->1). Interpretar siempre el conjunto, no una métrica.
    """
    x = _flatten_valid(embs, mask)
    out: dict[str, float] = {}

    if x.shape[0] < 2:
        return {
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "pairwise_cosine_mean": float("nan"),
            "per_dim_std_mean": 0.0,
            "per_dim_collapsed_frac": 1.0,
            "mean_l2_norm": float(x.norm(dim=-1).mean()) if x.numel() else 0.0,
        }

    eigvals = _eigenvalues(x)
    out["effective_rank"] = effective_rank(eigvals)
    out["participation_ratio"] = participation_ratio(eigvals)
    out["pairwise_cosine_mean"] = pairwise_cosine_mean(x)

    per_dim_std = x.std(dim=0)
    out["per_dim_std_mean"] = float(per_dim_std.mean())
    out["per_dim_collapsed_frac"] = float((per_dim_std < per_dim_std_eps).float().mean())
    out["mean_l2_norm"] = float(x.norm(dim=-1).mean())

    if target_embs is not None:
        t = _flatten_valid(target_embs, target_mask)
        if t.shape[0] >= 2:
            t_effrank = effective_rank(_eigenvalues(t))
            out["target_effective_rank"] = t_effrank
            out["effrank_ratio_vs_target"] = (
                out["effective_rank"] / t_effrank if t_effrank > 0 else 0.0
            )

    return out
