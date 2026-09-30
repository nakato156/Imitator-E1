import torch
import torch.nn.functional as F


def prefix_text_infonce(
    prefix: torch.Tensor,           # [B, K, H] — output of PrefixImitator
    text_embeds: torch.Tensor,      # [B, L-1, H] — bridge.embed_tokens(token_ids[:,:-1])
    text_ids_input: torch.Tensor,   # [B, L-1] — original token ids with -100 for padding
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE (NT-Xent) between pooled prefix and pooled text embeddings.

    Penalizes prefix collapse: if all prefixes point in the same direction, the
    off-diagonal similarities equal the diagonal and the loss reaches its maximum
    log(B), making collapse the worst-case outcome instead of a free minimum.
    """
    B = prefix.size(0)

    # Pool prefix: mean over K tokens → [B, H], L2-normalize
    v = F.normalize(prefix.mean(dim=1), dim=-1)

    # Pool text: masked mean (exclude padding positions) → [B, H], L2-normalize
    mask = (text_ids_input != -100).float().unsqueeze(-1)   # [B, L-1, 1]
    t = (text_embeds * mask).sum(1) / mask.sum(1).clamp(min=1)
    t = F.normalize(t, dim=-1)

    # Symmetric InfoNCE
    sim = v @ t.T / temperature                             # [B, B]
    labels = torch.arange(B, device=sim.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
