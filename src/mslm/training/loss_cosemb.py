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
    _, _, D = pred_embs.shape

    L_common = min(pred_embs.size(1), target_embs.size(1))
    pred_embs     = pred_embs   [:, :L_common].reshape(-1, D)
    target_embs   = target_embs [:, :L_common].reshape(-1, D)
    embedding_mask = embedding_mask[:, :L_common].reshape(-1)

    pred_norm = F.normalize(pred_embs, dim=-1)
    target_norm = F.normalize(target_embs, dim=-1)

    labels = torch.ones(pred_norm.size(0), device=pred_embs.device)

    # Compute the L2 loss
    cos_losses = F.cosine_embedding_loss(
        pred_norm, target_norm, labels,
        reduction='none'
    )

    valid_loss = cos_losses[embedding_mask]

    return valid_loss.mean()