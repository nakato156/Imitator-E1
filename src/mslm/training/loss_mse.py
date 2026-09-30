import torch
import torch.nn as nn
import torch.nn.functional as F

def imitator_loss(pred_embs: torch.Tensor, target_embs: torch.Tensor, embedding_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Args:
        pred_embs: Tensor of shape (batch_size, seq_len, emb_dim)
        target_embs: Tensor of shape (batch_size, seq_len, emb_dim)
        embedding_mask: Tensor of shape (batch_size, seq_len) indicating valid positions (True for valid, False for invalid).
                        If None, all positions are considered valid.
    Returns:
        loss: Tensor of shape (batch_size, seq_len)
    """
    # align the lengths of the sequences
    L_common = min(pred_embs.size(1), target_embs.size(1))
    pred_embs     = pred_embs   [:, :L_common]
    target_embs   = target_embs [:, :L_common]
    embedding_mask = embedding_mask[:, :L_common]

    # Compute the L2 loss
    mse_per_element = F.mse_loss(pred_embs, target_embs, reduction='none')

    mse_per_token = mse_per_element.mean(dim=-1)

    valid = ~embedding_mask

    loss = (mse_per_token * valid).sum() / valid.sum()

    return loss