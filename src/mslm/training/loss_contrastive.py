"""Pérdida contrastiva CLIP-style + métricas de retrieval para la etapa v118.

v118 separa "aprender a codificar seña" de "aprender a generar texto": el encoder
(PrefixImitator) se pre-entrena alineando el vídeo con el embedding de la frase
mediante InfoNCE simétrico (NT-Xent), SIN el LLM en el loop. La métrica de
selección es retrieval@1 — mide directamente si el embedding del vídeo i recupera
su propia frase frente a las demás, que es justo lo que la línea CE-AR (v116/v117)
no lograba (retrieval@1 = 0%, ver memoria v117-no-grounding-verdict).
"""
import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Media sobre la dimensión de secuencia ignorando posiciones de padding.

    x:            [B, N, H]
    padding_mask: [B, N]  (True = padding, como en collate_fn)
    returns:      [B, H]
    """
    valid = (~padding_mask).float().unsqueeze(-1)          # [B, N, 1]
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def clip_contrastive_loss(
    v: torch.Tensor,            # [B, d] embeddings de vídeo, L2-normalizados
    t: torch.Tensor,            # [B, d] embeddings de texto, L2-normalizados
    logit_scale: torch.Tensor,  # escalar (log-temperatura aprendible); se aplica exp()
    max_scale: float = 100.0,
) -> torch.Tensor:
    """InfoNCE simétrico (NT-Xent) entre vídeo y texto, estilo CLIP.

    El colapso es imposible por construcción: si todos los vídeos colapsan al
    mismo vector, las similitudes fuera de la diagonal igualan a la diagonal y la
    pérdida alcanza su máximo log(B) en lugar de un mínimo gratis.
    """
    scale = logit_scale.exp().clamp(max=max_scale)
    logits = scale * (v @ t.T)                              # [B, B]
    labels = torch.arange(v.size(0), device=v.device)
    loss_v2t = F.cross_entropy(logits, labels)
    loss_t2v = F.cross_entropy(logits.T, labels)
    return (loss_v2t + loss_t2v) / 2


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    v: torch.Tensor,            # [B, d] embeddings de vídeo, SIN normalizar
    t: torch.Tensor,            # [B, d] embeddings de texto, SIN normalizar
    lambda_inv: float = 25.0,
    mu_var: float = 25.0,
    nu_cov: float = 1.0,
) -> torch.Tensor:
    """VICReg cross-modal (Bardes et al.) entre vídeo y texto.

    A diferencia de InfoNCE no usa temperatura ni negativos explícitos: el
    término de invariancia alinea cada par, mientras varianza+covarianza
    previenen el colapso por construcción (embeddings constantes -> std=0 ->
    término de varianza máximo). Pensado para lotes pequeños (no necesita
    muchos negativos), el régimen donde InfoNCE entró en sobreajuste (val
    loss diverge porque logit_scale crece sin freno) en v118/v118b.
    """
    inv_loss = F.mse_loss(v, t)

    def _var_loss(x):
        std = torch.sqrt(x.var(dim=0) + 1e-4)
        return F.relu(1.0 - std).mean()

    var_loss = _var_loss(v) + _var_loss(t)

    def _cov_loss(x):
        B, d = x.shape
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (B - 1)
        return _off_diagonal(cov).pow(2).sum() / d

    cov_loss = _cov_loss(v) + _cov_loss(t)

    return lambda_inv * inv_loss + mu_var * var_loss + nu_cov * cov_loss


@torch.no_grad()
def retrieval_metrics(v: torch.Tensor, t: torch.Tensor) -> dict:
    """Métricas de retrieval vídeo→texto sobre un conjunto completo.

    v, t: [N, d] L2-normalizados, emparejados por índice (v[i] <-> t[i]).
    Devuelve R@1/R@5/R@10 (fracción 0-1), median_rank (1-indexado) y chance (1/N),
    promediando ambas direcciones (vídeo→texto y texto→vídeo).
    """
    N = v.size(0)
    sim = v @ t.T                                           # [N, N]

    def _ranks(scores: torch.Tensor) -> torch.Tensor:
        # rank (1-indexado) de la pareja correcta (diagonal) en cada fila
        diag = scores.diag().unsqueeze(1)                  # [N, 1]
        return (scores > diag).sum(dim=1) + 1              # [N]

    ranks = torch.cat([_ranks(sim), _ranks(sim.T)]).float()
    return {
        "R@1": (ranks <= 1).float().mean().item(),
        "R@5": (ranks <= 5).float().mean().item(),
        "R@10": (ranks <= 10).float().mean().item(),
        "median_rank": ranks.median().item(),
        "chance": 1.0 / N,
    }
