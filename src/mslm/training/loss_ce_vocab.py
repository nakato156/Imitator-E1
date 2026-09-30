import math
import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def ce_floor(vocab_size: int, label_smoothing: float) -> float:
    """Piso teórico de la cross-entropy con label smoothing sobre ``vocab_size`` clases.

    Con label smoothing ε, el término de suavizado de F.cross_entropy es
    ``logsumexp(z) - mean(z) >= log(V)`` para cualquier vector de logits z, por lo que
    ``CE >= ε * log(V)`` independientemente de la predicción. Una CE reportada por debajo
    de este valor es matemáticamente imposible y delata una métrica corrupta
    (p.ej. la val_ce ~0.568 de v115.1, con piso 0.1*log(262400)=1.248).
    """
    return label_smoothing * math.log(vocab_size)


def imitator_ce_loss(
    pred_embs: torch.Tensor,
    token_ids: torch.Tensor,
    embed_table_norm: torch.Tensor,
    logit_temp: "torch.Tensor | float" = 1.0,
    label_smoothing: float = 0.0,
) -> tuple:
    """Cross-entropy sobre el vocabulario con logits coseno + temperatura aprendible.

    Normaliza pred y E antes del producto punto: la norma de las predicciones no afecta
    los logits por construcción, eliminando el runaway de confianza de v115.

    Args:
        pred_embs: (B, L, D) embeddings predichos por el Imitator.
        token_ids: (B, L) IDs objetivo, padding = IGNORE_INDEX (-100).
        embed_table_norm: (V, D) tabla de embeddings pre-normalizada (F.normalize por el Trainer).
        logit_temp: temperatura como nn.Parameter (log-escala) o escalar.
        label_smoothing: suavizado de etiquetas para reducir sobreconfianza.
    Returns:
        (ce, ce_detached, token_acc_top1, token_acc_top5)
    """
    L_common = min(pred_embs.size(1), token_ids.size(1))
    pred = pred_embs[:, :L_common]
    ids  = token_ids[:, :L_common]

    pred_norm = F.normalize(pred, dim=-1)
    scale = logit_temp.exp() if isinstance(logit_temp, torch.Tensor) else logit_temp
    logits = pred_norm @ embed_table_norm.to(pred.dtype).T * scale   # (B, L, V)

    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        ids.reshape(-1).long(),
        ignore_index=IGNORE_INDEX,
        label_smoothing=label_smoothing,
    )

    with torch.no_grad():
        valid = ids != IGNORE_INDEX
        if valid.any():
            flat_logits = logits[valid]
            flat_ids    = ids[valid]
            top1 = (flat_logits.argmax(dim=-1) == flat_ids).float().mean()
            top5 = (flat_logits.topk(5, dim=-1).indices == flat_ids.unsqueeze(1)).any(1).float().mean()
        else:
            top1 = top5 = torch.zeros((), device=pred.device)

    return ce, ce.detach(), top1, top5
