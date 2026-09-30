import torch
from torch.nn.utils.rnn import pad_sequence

# Mismo valor que loss_ce_vocab.IGNORE_INDEX (no se importa para no arrastrar
# la cadena de imports de training/ a los workers del DataLoader).
IGNORE_INDEX = -100

def collate_fn(batch):
    """
    batch: List of tuples (keypoints, embeddings[, token_ids])
      - keypoints: [T_i, K, D]
      - embeddings: [N_i, E]
      - token_ids: [N_i] (opcional, experimento CE-vocab)
    returns:
      keypoints_padded  → [B, T_max, K, D]
      frames_mask       → [B, T_max]     (True = padding)
      embeddings_padded → [B, N_max, E]
      embeddings_mask   → [B, N_max]     (True = padding)
      token_ids_padded  → [B, N_max]     (padding = IGNORE_INDEX; sólo si vienen token_ids)
    """
    keypoints_list  = [item[0] for item in batch]
    embeddings_list = [item[1] for item in batch]

    # Normaliza embeddings a [N, E]
    for i in range(len(embeddings_list)):
        emb = embeddings_list[i]
        if emb.dim() == 3 and emb.size(0) == 1:
            embeddings_list[i] = emb.squeeze(0)
        elif emb.dim() != 2:
            raise ValueError(f"Embedding at index {i} has invalid shape {emb.shape}. Expected [N, E] or [1, N, E].")

    frame_lengths = torch.tensor([kp.size(0) for kp in keypoints_list],  dtype=torch.long)
    token_lengths = torch.tensor([emb.size(0) for emb in embeddings_list], dtype=torch.long)

    keypoints_padded  = pad_sequence(keypoints_list,  batch_first=True, padding_value=0.0)  # [B, T_max, K, D]
    embeddings_padded = pad_sequence(embeddings_list, batch_first=True, padding_value=0.0)  # [B, N_max, E]

    B, T_max, K, D = keypoints_padded.shape
    _, N_max, E    = embeddings_padded.shape

    arange_frames = torch.arange(T_max).unsqueeze(0).expand(B, -1)  # [B, T_max]
    arange_tokens = torch.arange(N_max).unsqueeze(0).expand(B, -1)  # [B, N_max]

    frames_mask     = arange_frames >= frame_lengths.unsqueeze(1)  # True = padding
    embeddings_mask = arange_tokens >= token_lengths.unsqueeze(1)

    out = (
        keypoints_padded.to(torch.float32),
        frames_mask.to(torch.bool),
        embeddings_padded.to(torch.float32),
        embeddings_mask.to(torch.bool)
    )

    # Experimento CE-vocab: el tercer slot del dataset lleva los token IDs.
    if len(batch[0]) > 2 and torch.is_tensor(batch[0][2]):
        ids_list = [item[2].long() for item in batch]
        ids_padded = pad_sequence(ids_list, batch_first=True, padding_value=IGNORE_INDEX)
        if ids_padded.size(1) < N_max:
            pad = ids_padded.new_full((B, N_max - ids_padded.size(1)), IGNORE_INDEX)
            ids_padded = torch.cat([ids_padded, pad], dim=1)
        out = out + (ids_padded,)

    return out
