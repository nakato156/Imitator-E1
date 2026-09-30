import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import gcd


def make_channel_norm_2d(channels: int, norm_type: str = "batch", norm_groups: int = 16):
    """Construye una normalización 2D sin cambiar el layout [B,C,T,N].

    BatchNorm queda como default para preservar los modelos existentes. GroupNorm
    es la opción segura cuando el pipeline procesa cada clip con B=1.
    """
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "group":
        groups = gcd(channels, max(1, int(norm_groups)))
        return nn.GroupNorm(groups, channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"norm_type desconocido: {norm_type!r}")

def partition_adjacency(A:np.ndarray) -> np.ndarray:
    """
    Particiona la matriz de adyacencia base A en K=3 subconjuntos para ST-GCN:
      A0: self-loops (identidad)
      A1: conexiones directas (A)
      A2: conexiones a 2 saltos (A^2), excluyendo las de A1 y A0
    Args:
        A (np.ndarray): Matriz de adyacencia base (J, J)
    Returns:
        np.ndarray de forma (K=3, J, J) con las particiones A0, A1, A2
    """
    J = A.shape[0]
    # A0: self-loops
    A0 = np.eye(J, dtype=A.dtype)

    # A1: conexiones directas
    A1 = (A > 0).astype(A.dtype)
    
    # A2: vecinos a 2 saltos
    A2_temp = (A1 @ A1) > 0
    
    # Quita conexiones de A1 y A0
    A2 = np.logical_and(A2_temp, np.logical_not(A1 + A0))
    A2 = A2.astype(A.dtype)
    
    return np.stack([A0, A1, A2])

class STGCNBlock(nn.Module):
    """
    Single ST-GCN block as described in SignFormer-GCN (based on Yan et al. ST-GCN).
    Args:
        in_channels (int): Number of input feature channels (C_m).
        out_channels (int): Number of output feature channels per temporal branch (C_i).
        A (np.ndarray or tensor): Adjacency matrix of shape (K, N, N) for K partitions.
        kernel_size (int): Temporal kernel size for all TCNs (odd number).
        stride (int): Temporal stride.
    Output:
        Tensor of shape (B, 3*out_channels, T_out, N)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        kernel_size=3,
        stride=1,
        norm_type="batch",
        norm_groups=16,
    ):
        super().__init__()
        # Adjacency partitions: A_k for k=0..K-1
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))  # (K, N, N)
        self.K, self.N, _ = A.shape
        # 1x1 Graph conv weights: map in_channels -> out_channels * K
        self.gconv = nn.Conv2d(in_channels, out_channels * self.K, kernel_size=1)
        self.bn1 = make_channel_norm_2d(out_channels * self.K, norm_type, norm_groups)
        # Temporal conv branches (3 branches)
        pad = (kernel_size - 1) // 2
        self.tcn1 = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1))
        self.tcn2 = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1))
        self.tcn3 = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1))
        self.bn2 = make_channel_norm_2d(out_channels * 3, norm_type, norm_groups)
        self.relu = nn.ReLU()
        
    def normalize_adj(self, A_k: torch.Tensor) -> torch.Tensor:
        A_k = A_k + torch.eye(A_k.shape[0], device=A_k.device)
        deg = A_k.sum(-1)
        deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
        D = torch.diag(deg_inv_sqrt)
        return D @ A_k @ D
    
    def forward(self, x):
        """
        x: Tensor of shape (B, C_m, T, N)
        returns: Tensor of shape (B, 3*out_channels, T_out, N)
        """
        B, C, T, N = x.shape
        # 1) Graph convolution
        # Linear projection to K partitions
        x_gc = self.gconv(x)                              # (B, K*out, T, N)
        x_gc = self.bn1(x_gc)
        # reshape for partitioned multiplication
        x_gc = x_gc.view(B, self.K, -1, T, N)             # (B, K, out, T, N)

        out = 0
        for k in range(self.K):
            A_k = self.normalize_adj(self.A[k])   # safe normalization
            xk = x_gc[:, k]
            # propagate across nodes: einsum over node dimension
            xk = torch.einsum('bctn,nm->bctm', xk, A_k)
            out = out + xk
        out = self.relu(out)

        # 2) Temporal convolutions (3 parallel branches)
        f1 = self.tcn1(out)
        f2 = self.tcn2(out)
        f3 = self.tcn3(out)
        # concat along channel dim
        y = torch.cat((f1, f2, f3), dim=1)                # (B, 3*out, T_out, N)
        y = self.bn2(y)
        y = self.relu(y)
        return y
