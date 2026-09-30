import torch
import torch.nn.functional as F

def imitator_loss(pred_embs: torch.Tensor, target_embs: torch.Tensor, embedding_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Args:
        pred_embs: Tensor of shape (batch_size, seq_len, emb_dim)
        target_embs: Tensor of shape (batch_size, seq_len, emb_dim)
        embedding_mask: Tensor of shape (batch_size, seq_len) indicating valid positions (True for valid, False for invalid).
                        If None, all positions are considered valid.
    Returns:
        Scalar loss
    """
    # align the lengths of the sequences
    L_common = min(pred_embs.size(1), target_embs.size(1))
    pred_embs     = pred_embs   [:, :L_common]
    target_embs   = target_embs [:, :L_common]
    embedding_mask = embedding_mask[:, :L_common]
    
    valid = (~embedding_mask).float()

    # Compute the MSE Loss
    mse_per_token = F.mse_loss(pred_embs, target_embs, reduction='none').mean(dim=-1)
    masked_loss_mse = (mse_per_token * valid).sum() / valid.sum()

    # Compute the Cosine Similarity Loss
    pred_norm = F.normalize(pred_embs, dim=-1)
    target_norm = F.normalize(target_embs, dim=-1)

    loss_cossim = 1 - F.cosine_similarity(pred_norm, target_norm, dim=-1)
    masked_loss_cossim = (loss_cossim * valid).sum() / valid.sum()

    beta = (masked_loss_mse / (masked_loss_cossim + 1e-6)).detach()

    loss_total = masked_loss_mse + (beta * masked_loss_cossim)

    return (loss_total, masked_loss_mse, masked_loss_cossim)