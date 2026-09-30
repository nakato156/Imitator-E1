import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def build_ar_labels(token_ids: torch.Tensor, k_prefix: int) -> torch.Tensor:
    """Build autoregressive labels for the v116 soft-prefix pipeline.

    Masks the first K-1 prefix positions with IGNORE_INDEX so that the loss
    only supervises positions K-1 onward.  Position K-1 (last prefix slot)
    predicts token_ids[:, 0] — the first text token — which was silently
    dropped in the original Bug-1 implementation.

    Args:
        token_ids: [B, L] — ground-truth token IDs (-100 for padding, kept as-is)
        k_prefix:  K — number of soft-prefix tokens prepended to inputs_embeds

    Returns:
        labels: [B, K-1+L] = [B, K+L-1] — aligned with logits from Gemma forward
    """
    B = token_ids.size(0)
    device = token_ids.device
    prefix_mask = torch.full((B, k_prefix - 1), IGNORE_INDEX, dtype=torch.long, device=device)
    return torch.cat([prefix_mask, token_ids], dim=1)


def imitator_ar_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k_prefix: int,
) -> tuple:
    """Autoregressive CE loss for the v116 soft-prefix pipeline.

    Args:
        logits:   [B, K+L-1, V] — logits from Gemma forward (prefix + text positions)
        labels:   [B, K+L-1]   — -100 for prefix positions and padding; token IDs elsewhere
        k_prefix: int           — number of prefix tokens (for documentation only;
                                  the labels tensor already masks them with -100)

    Returns:
        (ce, ce_detached, top1_acc, top5_acc) — same 4-tuple contract as imitator_ce_loss
    """
    V = logits.size(-1)

    ce = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1).long(),
        ignore_index=IGNORE_INDEX,
    )

    with torch.no_grad():
        valid = labels != IGNORE_INDEX
        if valid.any():
            flat_logits = logits[valid]          # (N_valid, V)
            flat_ids    = labels[valid].long()   # (N_valid,)
            top1 = (flat_logits.argmax(dim=-1) == flat_ids).float().mean()
            top5 = (flat_logits.topk(5, dim=-1).indices == flat_ids.unsqueeze(1)).any(1).float().mean()
        else:
            top1 = top5 = torch.zeros((), device=logits.device)

    return ce, ce.detach(), top1, top5
