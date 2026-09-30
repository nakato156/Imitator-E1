"""IsolatedKeypointDataset — dataset slim para clasificación de señas aisladas
(v120, dataset1: 64 glosas x 50 ejemplos c/u).

No reusa `KeypointDataset` porque esa clase exige un grupo `embeddings` en el
h5 (`processData` enumera clips desde `f[dataset]["embeddings"].keys()`) que
dataset1 (clasificación, sin embeddings de Gemma) no tiene. Sí reusa
`remove_keypoints` y `normalize_augment_data` (incluido `temporal_drop`,
agregado en v119) para no duplicar esa lógica.
"""
import random

import h5py
import torch
from torch.utils.data import Dataset

from .data_augmentation import (
    augment_isolated_keypoints,
    filter_unstable_keypoints_to_num,
    normalize_augment_data,
    normalize_keypoints_torso,
    remove_keypoints,
    resample_temporal,
    trim_active_interval,
)

AUGMENTATIONS = ["Gaussian_jitter", "Rotation_2D", "Scaling", "Temporal_drop", "Length_variance"]


def list_clips(h5_path, dataset_name="dataset1"):
    """Enumera (clip_id, label) de un grupo del h5 que solo tiene
    keypoints/labels (sin embeddings), ordenado por clip_id numérico."""
    with h5py.File(h5_path, "r") as f:
        g = f[dataset_name]
        clip_ids = sorted(g["keypoints"].keys(), key=int)
        labels = [g["labels"][cid][:][0].decode() for cid in clip_ids]
    return clip_ids, labels


def list_clip_records(h5_path, dataset_name="dataset1"):
    """Devuelve metadata para splits por signer, con fallback desde video_id."""
    with h5py.File(h5_path, "r") as f:
        g = f[dataset_name]
        clip_ids = sorted(g["keypoints"].keys(), key=int)
        records = []
        for cid in clip_ids:
            label = g["labels"][cid][0].decode()
            video_id = g["video_id"][cid][0].decode() if "video_id" in g else cid
            signer = int(g["signer_id"][cid][0]) if "signer_id" in g else None
            repetition = int(g["repetition"][cid][0]) if "repetition" in g else None
            records.append(
                {
                    "clip_id": cid,
                    "label": label,
                    "video_id": video_id,
                    "signer_id": signer,
                    "repetition": repetition,
                }
            )
    return records


class IsolatedKeypointDataset(Dataset):
    def __init__(
        self,
        h5_path,
        clip_ids,
        labels,
        dataset_name="dataset1",
        n_keypoints=111,
        augment=False,
        normalization="minmax",
        augmentation_profile="legacy",
        trim_active=False,
        target_frames=None,
        downsample=1,
    ):
        self.h5_path = h5_path
        self.dataset_name = dataset_name
        self.clip_ids = clip_ids
        self.labels = labels
        self.n_keypoints = n_keypoints
        self.augment = augment
        self.normalization = normalization
        self.augmentation_profile = augmentation_profile
        self.trim_active = trim_active
        self.target_frames = target_frames
        self.downsample = max(1, int(downsample))

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        with h5py.File(self.h5_path, "r") as f:
            keypoint = f[self.dataset_name]["keypoints"][clip_id][:]

        keypoint = remove_keypoints(keypoint)
        if self.trim_active:
            keypoint = trim_active_interval(keypoint)
        if self.downsample > 1:
            keypoint = keypoint[:: self.downsample]
        if self.target_frames:
            keypoint = resample_temporal(keypoint, self.target_frames)

        if self.normalization == "torso":
            keypoint, _ = filter_unstable_keypoints_to_num(
                torch.as_tensor(keypoint), self.n_keypoints
            )
            keypoint = normalize_keypoints_torso(keypoint)
            if self.augment and self.augmentation_profile == "composed":
                keypoint = augment_isolated_keypoints(keypoint)
        elif self.normalization == "minmax":
            transform = (
                random.choice(AUGMENTATIONS)
                if self.augment and self.augmentation_profile == "legacy"
                else "Original"
            )
            keypoint = normalize_augment_data(keypoint, transform, self.n_keypoints)
            if self.augment and self.augmentation_profile == "composed":
                keypoint = augment_isolated_keypoints(keypoint)
        else:
            raise ValueError(f"normalization desconocida: {self.normalization!r}")
        if not isinstance(keypoint, torch.Tensor):
            keypoint = torch.as_tensor(keypoint)
        return keypoint.float(), self.labels[idx]


def isolated_collate_fn(batch, label_to_idx):
    """NO pad-ea: cada clip dura distinto y un STGCN multi-capa con padding+
    máscara filtra entre samples (ver docstring de
    IsolatedSignClassifier.forward). Devuelve la lista de keypoints a su
    longitud real; el training loop llama al modelo una vez por clip (B=1)."""
    keypoints = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    target = torch.tensor([label_to_idx[lbl] for lbl in labels], dtype=torch.long)
    return keypoints, target
