import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .attention import MultiheadAttentionRoPE

class TransformerEncoderLayerRoPE(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_first, norm_first, **kwargs):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=norm_first
        )
        factory_kwargs = {"device": kwargs.get("device"), "dtype": kwargs.get("dtype")}
        self.self_attn = MultiheadAttentionRoPE(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            use_rotary=True,
            **factory_kwargs
        )