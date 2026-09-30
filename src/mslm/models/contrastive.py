"""ContrastiveAligner: torre de vídeo (PrefixImitator) + torre de texto para v118.

Etapa de pre-entrenamiento contrastivo SIN LLM. El lado texto usa los embeddings
de Gemma ya cacheados en el HDF5 (batch[2]/[3]); no se carga el LLM. El encoder de
vídeo es el mismo PrefixImitator de v116/v117.

Nota: v119 YA NO es "PrefixImitator + QLoRA" (plan original, ver historial de este
comentario). El diagnóstico train/val sobre v118c (gap_factor≈1.0 en
outputs/diag_v118_train_val_gap.json: R@1 igual de malo en train que en val)
descartó sobreajuste y señaló que el cuello de botella es la formulación de la
tarea (comprimir la frase a un vector y rankearla), no la falta de una etapa
generativa con el LLM. v119 pasa a ser CTC sobre secuencia (CTCEncoder en
src/mslm/models/ctc_encoder.py, sin LLM en el loop, igual que esta etapa), ver
report.md y scripts/train/train_ctc_v119.py.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mslm.training.loss_contrastive import masked_mean


class _ProjHead(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear hacia el espacio de proyección compartido."""

    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveAligner(nn.Module):
    """Alinea el embedding del vídeo con el de la frase vía proyecciones + InfoNCE.

    Parameters
    ----------
    prefix_imitator:
        Instancia de PrefixImitator (encoder de vídeo). Produce prefix [B, K, H].
    hidden:
        Dimensión H de los embeddings (2048 para gemma-3n-E2B, = output_size del Imitator).
    proj_dim:
        Dimensión del espacio contrastivo compartido.
    init_temperature:
        Temperatura inicial; logit_scale = log(1/temp) (como CLIP, τ≈0.07).
    """

    def __init__(self, prefix_imitator: nn.Module, hidden: int = 2048,
                 proj_dim: int = 256, init_temperature: float = 0.07):
        super().__init__()
        self.encoder = prefix_imitator
        self.video_head = _ProjHead(hidden, proj_dim)
        self.text_head = _ProjHead(hidden, proj_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

    def encode_video(self, keypoints: torch.Tensor, frames_padding_mask: torch.Tensor,
                      normalize: bool = True) -> torch.Tensor:
        prefix, _ = self.encoder(keypoints, frames_padding_mask)   # [B, K, H]
        out = self.video_head(prefix.mean(dim=1))
        return F.normalize(out, dim=-1) if normalize else out

    def encode_text(self, text_embeds: torch.Tensor, text_padding_mask: torch.Tensor,
                     normalize: bool = True) -> torch.Tensor:
        pooled = masked_mean(text_embeds, text_padding_mask)       # [B, H]
        out = self.text_head(pooled)
        return F.normalize(out, dim=-1) if normalize else out
