"""Métricas de clasificación (v120 — reconocimiento de señas aislado)."""
import torch


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """Fracción de muestras donde el target está entre las k clases de mayor logit.

    logits: [B, num_classes]. targets: [B] (índices de clase)."""
    k = min(k, logits.size(-1))
    topk = logits.topk(k, dim=-1).indices  # [B, k]
    hits = (topk == targets.unsqueeze(1)).any(dim=1)
    return hits.float().mean().item()
