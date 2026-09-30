"""Synthetic temporal pretraining samples for v126.

The dataset builds continuous signing clips by concatenating isolated dataset1
clips, inserting neutral frames between signs, and exposing the known sign
boundaries so CIF can be supervised before moving to real continuous data.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .data_augmentation import normalize_augment_data, remove_keypoints


@dataclass(frozen=True)
class SyntheticTemporalSample:
    keypoints: torch.Tensor
    token_ids: torch.Tensor
    boundaries: torch.Tensor
    glosses: tuple[str, ...]
    clip_ids: tuple[str, ...]
    token_spans: torch.Tensor
    target_embeddings: torch.Tensor | None = None


class SyntheticTemporalSignDataset(Dataset):
    """Concatenate 2-8 isolated clips into one synthetic continuous sequence.

    Args:
        h5_path: HDF5 containing ``dataset_name/keypoints``.
        clip_ids: candidate isolated clip ids.
        label_by_clip: map clip id -> gloss text.
        token_ids_by_label: map gloss text -> token id sequence.
        embedding_table: optional ``[vocab, dim]`` tensor used to materialize
            target embeddings for the concatenated token ids.
        apply_remove_keypoints: set True for the real dataset1 133->111 keypoint
            preprocessing. Tests can keep it False for small fixtures.
    """

    def __init__(
        self,
        h5_path,
        clip_ids: Sequence[str],
        label_by_clip: Mapping[str, str],
        token_ids_by_label: Mapping[str, Sequence[int]],
        *,
        dataset_name: str = "dataset1",
        min_clips: int = 2,
        max_clips: int = 8,
        min_neutral_frames: int = 0,
        max_neutral_frames: int = 8,
        samples_per_epoch: int | None = None,
        seed: int = 23,
        embedding_table: torch.Tensor | None = None,
        apply_remove_keypoints: bool = False,
        normalize: bool = False,
        n_keypoints: int = 111,
        rescue_augmentation: bool = False,
        jitter_std: float = 0.01,
        temporal_drop_rate: float = 0.15,
        augmentation_probability: float = 0.5,
    ):
        if min_clips < 1 or max_clips < min_clips:
            raise ValueError("clip count range must satisfy 1 <= min <= max")
        if min_neutral_frames < 0 or max_neutral_frames < min_neutral_frames:
            raise ValueError("neutral frame range must satisfy 0 <= min <= max")
        normalized_clip_ids = tuple(str(cid) for cid in clip_ids)
        normalized_label_by_clip = {str(k): v for k, v in label_by_clip.items()}
        missing = [
            cid
            for cid in normalized_clip_ids
            if normalized_label_by_clip[cid] not in token_ids_by_label
        ]
        if missing:
            raise KeyError(f"missing token ids for labels of clips: {missing[:3]}")

        self.h5_path = h5_path
        self.dataset_name = dataset_name
        self.clip_ids = normalized_clip_ids
        self.label_by_clip = normalized_label_by_clip
        self.token_ids_by_label = {
            k: tuple(int(t) for t in v) for k, v in token_ids_by_label.items()
        }
        self.min_clips = int(min_clips)
        self.max_clips = int(max_clips)
        self.min_neutral_frames = int(min_neutral_frames)
        self.max_neutral_frames = int(max_neutral_frames)
        self.samples_per_epoch = int(samples_per_epoch or len(self.clip_ids))
        self.seed = int(seed)
        self.embedding_table = embedding_table
        self.apply_remove_keypoints = bool(apply_remove_keypoints)
        self.normalize = bool(normalize)
        self.n_keypoints = int(n_keypoints)
        self.rescue_augmentation = bool(rescue_augmentation)
        self.jitter_std = float(jitter_std)
        self.temporal_drop_rate = float(temporal_drop_rate)
        self.augmentation_probability = float(augmentation_probability)

    def _augment_clip(self, keypoints: torch.Tensor, rng: random.Random) -> torch.Tensor:
        """Apply the single predefined AR rescue augmentation to one clip."""
        if not self.rescue_augmentation:
            return keypoints
        if rng.random() < self.augmentation_probability:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(rng.randrange(2**31))
            noise = torch.randn(
                keypoints.shape, generator=generator, dtype=keypoints.dtype
            ).to(keypoints.device)
            keypoints = keypoints + self.jitter_std * noise
        if rng.random() < self.augmentation_probability and keypoints.size(0) > 1:
            keep_count = max(1, round(keypoints.size(0) * (1.0 - self.temporal_drop_rate)))
            kept = sorted(rng.sample(range(keypoints.size(0)), keep_count))
            keypoints = keypoints[torch.tensor(kept, device=keypoints.device)]
        return keypoints

    def __len__(self):
        return self.samples_per_epoch

    def _rng(self, idx: int) -> random.Random:
        return random.Random(self.seed + int(idx))

    def _read_keypoints(self, h5_file, clip_id: str) -> torch.Tensor:
        array = h5_file[self.dataset_name]["keypoints"][clip_id][:]
        if self.apply_remove_keypoints:
            array = remove_keypoints(array)
        if self.normalize:
            array = normalize_augment_data(array, "Original", self.n_keypoints)
        return torch.as_tensor(array, dtype=torch.float32)

    def __getitem__(self, idx: int) -> SyntheticTemporalSample:
        rng = self._rng(idx)
        n_clips = rng.randint(self.min_clips, self.max_clips)
        chosen = tuple(rng.choice(self.clip_ids) for _ in range(n_clips))

        parts: list[torch.Tensor] = []
        boundaries: list[tuple[int, int]] = []
        token_ids: list[int] = []
        token_spans: list[tuple[int, int]] = []
        glosses: list[str] = []
        frame_cursor = 0
        token_cursor = 0

        with h5py.File(self.h5_path, "r") as f:
            for pos, clip_id in enumerate(chosen):
                kp = self._read_keypoints(f, clip_id)
                kp = self._augment_clip(kp, rng)
                start = frame_cursor
                end = start + kp.size(0)
                parts.append(kp)
                boundaries.append((start, end))
                frame_cursor = end

                gloss = self.label_by_clip[clip_id]
                ids = self.token_ids_by_label[gloss]
                token_ids.extend(ids)
                token_spans.append((token_cursor, token_cursor + len(ids)))
                token_cursor += len(ids)
                glosses.append(gloss)

                if pos < n_clips - 1:
                    gap = rng.randint(self.min_neutral_frames, self.max_neutral_frames)
                    if gap:
                        parts.append(kp.new_zeros((gap, *kp.shape[1:])))
                        frame_cursor += gap

        keypoints = torch.cat(parts, dim=0)
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        embeddings = None
        if self.embedding_table is not None:
            embeddings = self.embedding_table[token_tensor]

        return SyntheticTemporalSample(
            keypoints=keypoints,
            token_ids=token_tensor,
            boundaries=torch.tensor(boundaries, dtype=torch.long),
            glosses=tuple(glosses),
            clip_ids=chosen,
            token_spans=torch.tensor(token_spans, dtype=torch.long),
            target_embeddings=embeddings,
        )


def permute_video_segments(
    keypoints: torch.Tensor,
    boundaries: torch.Tensor,
    length: int,
    rng: random.Random,
    *,
    require_change: bool = False,
) -> torch.Tensor:
    """Shuffle whole gloss segments (each with its leading neutral gap) in time.

    Used only for the v126b diagnostic eval: targets stay in their original
    order while the visual segments are permuted, so a drop in token accuracy
    shows the model actually relies on temporal order rather than a bag of
    signs.

    Frames before the first boundary start and frames at/after ``length``
    (padding) are left untouched; only the ``[prev_end, end)`` chunks for each
    real boundary are reordered among themselves.
    """
    valid = [(s, e) for s, e in boundaries.tolist() if s >= 0 and e > s]
    if len(valid) < 2:
        return keypoints.clone()

    prefix_end = valid[0][0]
    chunks = []
    prev_end = prefix_end
    for start, end in valid:
        chunks.append((prev_end, end))
        prev_end = end

    order = list(range(len(chunks)))
    rng.shuffle(order)
    if require_change and order == list(range(len(chunks))):
        # Deterministically avoid a no-op intervention.  This changes only an
        # identity draw; all sampled non-identity permutations are preserved.
        order = order[1:] + order[:1]

    pieces = [keypoints[:prefix_end]] if prefix_end > 0 else []
    for idx in order:
        start, end = chunks[idx]
        pieces.append(keypoints[start:end])
    if prev_end < length:
        pieces.append(keypoints[prev_end:length])
    if length < keypoints.size(0):
        pieces.append(keypoints[length:])

    return torch.cat(pieces, dim=0)


def synthetic_temporal_collate(batch: Sequence[SyntheticTemporalSample]):
    keypoints = pad_sequence([item.keypoints for item in batch], batch_first=True)
    token_ids = pad_sequence(
        [item.token_ids for item in batch],
        batch_first=True,
        padding_value=-100,
    )
    frame_lengths = torch.tensor([item.keypoints.size(0) for item in batch], dtype=torch.long)
    token_lengths = torch.tensor([item.token_ids.numel() for item in batch], dtype=torch.long)
    max_signs = max(item.boundaries.size(0) for item in batch)
    boundaries = torch.full((len(batch), max_signs, 2), -1, dtype=torch.long)
    token_spans = torch.full((len(batch), max_signs, 2), -1, dtype=torch.long)
    sign_counts = torch.tensor([item.boundaries.size(0) for item in batch], dtype=torch.long)
    for i, item in enumerate(batch):
        boundaries[i, : item.boundaries.size(0)] = item.boundaries
        token_spans[i, : item.token_spans.size(0)] = item.token_spans

    target_embeddings = None
    if all(item.target_embeddings is not None for item in batch):
        target_embeddings = pad_sequence(
            [item.target_embeddings for item in batch],
            batch_first=True,
            padding_value=0.0,
        )

    return {
        "keypoints": keypoints,
        "frame_lengths": frame_lengths,
        "token_ids": token_ids,
        "token_lengths": token_lengths,
        "boundaries": boundaries,
        "sign_counts": sign_counts,
        "token_spans": token_spans,
        "target_embeddings": target_embeddings,
        "glosses": [item.glosses for item in batch],
        "clip_ids": [item.clip_ids for item in batch],
    }
