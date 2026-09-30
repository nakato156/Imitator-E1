"""Temporal Lift Pooling (Hu et al., LiftSign CVPRW 2026 §3.2.2).

Alternativa aprendible al max-pooling temporal: split par/impar por stride-2,
un paso predictor/updater (esquema de lifting de wavelets) y un gate de "Local
Weighting" (ecuación 3). Dos pérdidas auxiliares (L_u, L_p, ecuación 2) estabilizan
el entrenamiento del módulo. Usado en la ablation A3 de v119 (ctc_v119c.toml).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLiftPooling(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        pad = kernel_size // 2
        self.predictor = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.updater = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.lw_conv = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _instance_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Equivalente a nn.InstanceNorm1d(affine=False) pero sin su validación
        de >1 elemento espacial -- con T=1 (clips cortos tras el pooling, visto
        en producción) la varianza es 0 y el resultado es simplemente 0, no un
        crash."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + eps)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """x: [B, T, C]; lengths: [B] (longitud válida, no-padding).
        Devuelve (x_downsampled [B, ceil(T/2), C], lengths_downsampled [B], aux_losses)."""
        B, T, C = x.shape
        x_c = x.transpose(1, 2)  # [B, C, T]
        if T % 2 == 1:
            x_c = F.pad(x_c, (0, 1))  # replica longitud par para el split

        x_e = x_c[:, :, 0::2]  # [B, C, ceil(T/2)]
        x_o = x_c[:, :, 1::2]  # [B, C, floor(T/2)]

        # x_e siempre tiene >= elementos que x_o (cuando T es impar, uno más).
        x_e_aligned = x_e[:, :, : x_o.size(-1)]

        d = x_o - self.dropout(self.predictor(x_e_aligned))         # detalle de alta frecuencia
        s = x_e_aligned + self.dropout(self.updater(d))             # aproximación de baja frecuencia
        if x_e.size(-1) > x_o.size(-1):
            s = torch.cat([s, x_e[:, :, -1:]], dim=-1)               # último impar sin pareja

        L_p = d.pow(2).mean()
        L_u = (s[:, :, : x_o.size(-1)] - x_o).pow(2).mean()

        gate = torch.sigmoid(self._instance_norm(self.lw_conv(s))) - 0.5
        s_lw = s + s * gate

        lengths_down = (lengths + 1) // 2
        return s_lw.transpose(1, 2), lengths_down, {"L_u": L_u, "L_p": L_p}
