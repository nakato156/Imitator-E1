#!/usr/bin/env python
"""Export one clip's 111 model-input keypoints to a small .npz for the README GIF.

Run once, with an interpreter that has h5py; the result is committed so that
make_imitator_animation.py needs neither the dataset nor h5py.

Mirrors remove_keypoints + keypoint_normalization from
src/mslm/dataloader/data_augmentation.py (reimplemented in numpy so this script
does not need torch either).

    python scripts/docs/export_clip_keypoints.py --clip 1660
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

H5 = Path("/shared/Code/Sign-AI/data/processed/dataset1_isolated.hdf5")


def remove_keypoints(kp: np.ndarray) -> np.ndarray:
    """(T, 134, 2) -> (T, 111, 2); same slices as data_augmentation.py:142-151."""
    pose = kp[:, :7, :]           # 7 pose
    face = kp[:, 28:92, :]        # 64 face
    left = kp[:, 93:113, :]       # 20 left hand
    right = kp[:, 113:133, :]     # 20 right hand
    return np.concatenate([pose, face, left, right], axis=1)


def normalize(kp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min-max to [0, 1] over valid points; same as data_augmentation.py:76-96."""
    kp = np.abs(kp)
    valid = ~((kp[..., 0] < 5) & (kp[..., 1] < 5))
    pts = kp[valid].reshape(-1, 2)
    lo = pts.min(axis=0)
    rng = pts.max(axis=0) - lo
    rng[rng == 0] = 1.0
    out = (kp - lo) / rng
    out[~valid] = 0.0
    return out.astype(np.float32), valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="1660")
    ap.add_argument("--h5", type=Path, default=H5)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[2] / "docs" / "clip_keypoints.npz")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        g = f["dataset1"]
        raw = g["keypoints"][args.clip][:]
        gloss = g["labels"][args.clip][0].decode()

    kp = remove_keypoints(raw)
    norm, valid = normalize(kp)
    assert norm.shape[1:] == (111, 2), norm.shape

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        keypoints=norm.astype(np.float16),
        valid=valid,
        clip_id=args.clip,
        gloss=gloss,
    )
    kb = args.out.stat().st_size / 1024
    print(f"{args.out}: clip {args.clip} '{gloss}' {norm.shape} -> {kb:.0f} KB")


if __name__ == "__main__":
    main()
