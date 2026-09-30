"""IsolatedSignClassifier — clasificador de señas aisladas (v120, dataset1: 64
glosas x 50 ejemplos c/u).

Reusa el backbone GCN ya portado para CTCEncoder (`STGCNBlock` +
`partition_adjacency`, ver components/stgcn.py): mismo stack de 3 capas que
`CTCEncoder.make_stack` (gcn_channels=[32,64,128]). No hay BiLSTM ni CTC: con
1 palabra (glosa) por clip no hay secuencia que alinear, solo una clase por
clip -- mean-pool sobre joints y tiempo + clasificador lineal.

Modelo chico a propósito: "Less is More" (Huamani-malca & Bejarano, PUCP/LSP)
encuentra que modelos chicos generalizan mejor en datasets chicos de la misma
lengua y régimen de datos (3200 clips, 64 clases).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.stgcn import STGCNBlock, make_channel_norm_2d, partition_adjacency


class IsolatedSignClassifier(nn.Module):
    def __init__(
        self,
        A,
        input_size: int = 111,
        gcn_channels=(32, 64, 128),
        hidden_size: int = 128,
        num_classes: int = 64,
        norm_type: str = "batch",
        norm_groups: int = 16,
        temporal_head: bool = False,
        use_motion_stream: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.temporal_head = temporal_head
        self.use_motion_stream = use_motion_stream
        A_part = partition_adjacency(A)

        def make_stack():
            layers = nn.ModuleList()
            c_in = 2
            for c_out in gcn_channels:
                layers.append(
                    STGCNBlock(
                        c_in,
                        c_out,
                        A_part,
                        kernel_size=3,
                        norm_type=norm_type,
                        norm_groups=norm_groups,
                    )
                )
                c_in = 3 * c_out
            return layers, c_in

        self.stgcn_layers, static_out = make_stack()
        if use_motion_stream:
            self.stgcn_motion_layers, motion_out = make_stack()
            fuse_in = static_out + motion_out
        else:
            self.stgcn_motion_layers = None
            fuse_in = static_out

        # Conv -> BN -> ReLU (NO Conv -> ReLU -> BN, que usa CTCEncoder.linear_hidden):
        # ahí se promedia solo sobre joints, dejando T para el TCN/BiLSTM
        # posterior, así que el promedio global de BatchNorm2d (media cero
        # sobre B,T,N) nunca coincide con ese pooling parcial. Aquí se
        # promedia sobre (T,N) A LA VEZ -- exactamente las dims que
        # BatchNorm2d normaliza a media cero -- así que terminar en BN
        # anula matemáticamente el pooled feature para cualquier input
        # (verificado: gradiente de classifier.weight y de todo el GCN
        # backbone exactamente 0.0, el modelo solo podía aprender un sesgo
        # constante por clase). ReLU después de BN rompe esa cancelación.
        self.linear_hidden = nn.Sequential(
            nn.Conv2d(fuse_in, hidden_size, kernel_size=1),
            make_channel_norm_2d(hidden_size, norm_type, norm_groups),
            nn.ReLU(),
        )
        if temporal_head:
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.GroupNorm(1, hidden_size),
                nn.ReLU(),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.GroupNorm(1, hidden_size),
                nn.ReLU(),
            )
            self.temporal_attention = nn.Conv1d(hidden_size, 1, kernel_size=1)
        else:
            self.temporal_conv = None
            self.temporal_attention = None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    @staticmethod
    def _motion_stream(x: torch.Tensor) -> torch.Tensor:
        diff = x[:, 1:] - x[:, :-1]
        return torch.cat([torch.zeros_like(x[:, :1]), diff], dim=1)

    @staticmethod
    def _run_stack(layers, x):
        for layer in layers:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, N, 2], SIN padding -- cada clip se procesa a su longitud
        real (mismo motivo que `_encode_batch` en train_ctc_v119.py: batchear
        con padding+máscara filtra entre samples porque el cero de padding deja
        de ser exactamente cero tras la primera capa STGCN, contaminando el
        frame límite en las capas siguientes; verificado numéricamente incluso
        en float64). El training loop llama una vez por clip (B=1) y concatena
        logits. Devuelve logits [B, num_classes]."""
        feats = self._run_stack(self.stgcn_layers, x.permute(0, 3, 1, 2))
        if self.use_motion_stream:
            motion = self._motion_stream(x).permute(0, 3, 1, 2)
            motion_feats = self._run_stack(self.stgcn_motion_layers, motion)
            feats = torch.cat([feats, motion_feats], dim=1)
        feats = self.linear_hidden(feats)         # [B, hidden, T, N]
        if self.temporal_head:
            temporal = feats.mean(dim=3)          # [B, hidden, T]
            temporal = self.temporal_conv(temporal)
            weights = F.softmax(self.temporal_attention(temporal), dim=-1)
            feats = (temporal * weights).sum(dim=-1)
        else:
            feats = feats.mean(dim=(2, 3))         # [B, hidden]
        return self.classifier(self.dropout(feats))
