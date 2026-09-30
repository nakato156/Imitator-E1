"""CTCEncoder — encoder de vídeo para v119 (CTC sobre secuencia).

Arquitectura PORTADA directamente de la línea CoSign/Min et al. ("A Closer Look
at Skeleton-based CSLR", ICCVW 2025) / LiftSign (CVPRW 2026), que reporta WER
4.6%/41.0% en Isharah con este mismo diseño: GCN multi-capa -> TCN (K3-P2-K3-P2,
P2 = pooling temporal aprendible) -> BiLSTM -> clasificador COMPARTIDO con
supervisión CTC dual (ecuación 6 de Min et al.: L = CTC(Y_s,G) + CTC(Y_l,G), con
Y_s = salida del TCN ("short-term") y Y_l = salida del BiLSTM ("long-term")).

Reemplaza el diseño anterior (1 solo STGCNBlock + Transformer de 6 capas +
1 sola cabeza CTC), que colapsó a blank tanto en 1000 como en 5600 clips reales
(ver report.md, sección v119) -- Min et al. atribuyen ese mismo síntoma (el
módulo de contexto largo no generaliza con pocos datos) a la falta de
supervisión de corto plazo, que es justo lo que aquí se añade.

P2 usa TemporalLiftPooling (LiftSign) en vez del max-pooling plano de Min et al.
-- es la mejora que LiftSign le hace a ese mismo paso del pipeline, no una
desviación.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .components.stgcn import STGCNBlock, partition_adjacency
from .components.tlp import TemporalLiftPooling


class CTCEncoder(nn.Module):
    def __init__(
        self,
        A,
        input_size: int = 111,
        gcn_channels=(32, 64, 128),
        hidden_size: int = 256,
        lstm_layers: int = 2,
        vocab_size: int = 100,
        use_motion_stream: bool = False,
    ):
        super().__init__()
        self.use_motion_stream = use_motion_stream
        A_part = partition_adjacency(A)

        def make_stack(in_ch=2):
            layers = nn.ModuleList()
            c_in = in_ch
            for c_out in gcn_channels:
                layers.append(STGCNBlock(c_in, c_out, A_part, kernel_size=3))
                c_in = 3 * c_out
            return layers, c_in

        self.stgcn_layers, static_out = make_stack()
        if use_motion_stream:
            self.stgcn_motion_layers, motion_out = make_stack()
            fuse_in = static_out + motion_out
        else:
            self.stgcn_motion_layers = None
            fuse_in = static_out

        self.linear_hidden = nn.Sequential(
            nn.Conv2d(fuse_in, hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size),
        )

        pad = 1
        self.tcn_conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=pad)
        self.tlp1 = TemporalLiftPooling(hidden_size)
        self.tcn_conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=pad)
        self.tlp2 = TemporalLiftPooling(hidden_size)

        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Clasificador COMPARTIDO entre Y_s (corto plazo) y Y_l (largo plazo) --
        # Min et al. §3.3: "we share the classifier between the feature extractor
        # and the alignment module". hidden_size//2 * 2 direcciones = hidden_size,
        # así que el mismo nn.Linear sirve para ambas salidas sin proyección extra.
        self.classifier = nn.Linear(hidden_size, vocab_size + 1)  # +1 = blank (índice 0)

    @staticmethod
    def _motion_stream(x: torch.Tensor) -> torch.Tensor:
        """Diferencias frame-a-frame (CoSign/LiftSign §3.2.1); primer frame con padding cero."""
        diff = x[:, 1:] - x[:, :-1]
        pad = torch.zeros_like(x[:, :1])
        return torch.cat([pad, diff], dim=1)

    @staticmethod
    def _run_stack(layers, x):
        for layer in layers:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor, frames_padding_mask: torch.Tensor):
        """x: [B, T, N, 2]; frames_padding_mask: [B, T] (True = padding).
        Devuelve (log_probs_short [B,T',V+1], log_probs_long [B,T',V+1],
        seq_lengths [B], aux_losses dict)."""
        lengths = (~frames_padding_mask).sum(dim=1)

        static_in = x.permute(0, 3, 1, 2)  # [B, 2, T, N]
        feats = self._run_stack(self.stgcn_layers, static_in)
        if self.use_motion_stream:
            motion_in = self._motion_stream(x).permute(0, 3, 1, 2)
            feats_m = self._run_stack(self.stgcn_motion_layers, motion_in)
            feats = torch.cat([feats, feats_m], dim=1)

        feats = self.linear_hidden(feats)            # [B, hidden, T, N]
        feats = feats.mean(dim=-1)                    # [B, hidden, T]

        feats = F.relu(self.tcn_conv1(feats))
        feats_bt = feats.permute(0, 2, 1).contiguous()  # [B, T, hidden]
        feats_bt, lengths, aux1 = self.tlp1(feats_bt, lengths)

        feats = feats_bt.permute(0, 2, 1).contiguous()
        feats = F.relu(self.tcn_conv2(feats))
        feats_bt = feats.permute(0, 2, 1).contiguous()
        feats_short, lengths, aux2 = self.tlp2(feats_bt, lengths)  # [B, T', hidden]

        aux_losses = {"L_u": aux1["L_u"] + aux2["L_u"], "L_p": aux1["L_p"] + aux2["L_p"]}

        logits_short = self.classifier(feats_short)
        log_probs_short = F.log_softmax(logits_short, dim=-1)

        packed = pack_padded_sequence(
            feats_short, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bilstm(packed)
        feats_long, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=feats_short.size(1)
        )

        logits_long = self.classifier(feats_long)
        log_probs_long = F.log_softmax(logits_long, dim=-1)

        return log_probs_short, log_probs_long, lengths, aux_losses
