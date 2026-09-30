"""CTC loss para la etapa v119 — secuencia, reemplaza el objetivo contrastivo global.

A diferencia de v118 (InfoNCE/VICReg sobre un vector por clip), aquí el encoder
emite una distribución por paso temporal y CTC aprende el alineamiento implícito
contra la secuencia de palabras, sin necesitar un vector global ni comprimir la
frase en un solo embedding.
"""
import torch
import torch.nn as nn


def ctc_loss(
    log_probs: torch.Tensor,    # [B, T, V+1] (ya log-softmax, incluye blank=0)
    targets: torch.Tensor,      # [sum(target_lengths)] concatenado, convención CTC
    input_lengths: torch.Tensor,   # [B]
    target_lengths: torch.Tensor,  # [B]
    blank: int = 0,
) -> torch.Tensor:
    log_probs_tbv = log_probs.transpose(0, 1)  # [T, B, V+1], requerido por nn.CTCLoss
    loss_fn = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")
    return loss_fn(log_probs_tbv, targets, input_lengths, target_lengths)
