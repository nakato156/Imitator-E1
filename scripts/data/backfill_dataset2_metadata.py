"""Backfill de metadata en el HDF5 ya construido de dataset2 (v125).

No re-extrae keypoints ni recalcula embeddings. Reusa
`build_dataset2_h5.select_clips(n, seed)` para recuperar, en orden, el
(fname, label) original de cada índice — se verificó que esa función es
determinista y reproduce exactamente el orden usado para construir
`dataset_v6_unsloth.hdf5` (label del clip "0" coincide carácter a carácter).

Uso:
    PYTHONPATH=. python scripts/data/backfill_dataset2_metadata.py \
        --h5 data/processed/dataset_v6_unsloth.hdf5 --n 5600 --seed 23
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset2_h5 import select_clips, RAW  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mslm.dataloader.vocab import tokenize  # noqa: E402


def compute_metadata_fields(fname: str, video_column: str, label: str, frame_count: int, token_count: int) -> dict:
    return {
        "video_id": Path(fname).stem,
        "source_group": video_column,
        "frame_count": int(frame_count),
        "token_count": int(token_count),
        "word_count": len(tokenize(label)),
    }


def main(h5_path: Path, n: int, seed: int) -> None:
    meta = pd.read_csv(RAW / "meta.csv")
    video_by_id = dict(zip(meta["id"], meta["video"]))

    clips = select_clips(n, seed)  # [(fname, label_lower), ...] mismo orden que el build original

    with h5py.File(h5_path, "a") as f:
        g = f["dataset2"]
        g_kp, g_ids = g["keypoints"], g.get("token_ids")
        g_video = g.require_group("video_id")
        g_group = g.require_group("source_group")
        g_frames = g.require_group("frame_count")
        g_tokens = g.require_group("token_count")
        g_words = g.require_group("word_count")
        dt = h5py.string_dtype(encoding="utf-8")

        done = skipped = 0
        for idx, (fname, label) in enumerate(clips):
            key = str(idx)
            if key not in g_kp:
                continue
            stored_label = g["labels"][key][0].decode()
            if stored_label != label:
                raise ValueError(
                    f"clip {key}: el orden reconstruido no coincide con el H5 "
                    f"({label!r} != {stored_label!r})"
                )
            clip_id_csv = Path(fname).stem
            video_column = video_by_id.get(clip_id_csv)
            if video_column is None:
                print(f"[backfill] {key} SKIP: {clip_id_csv!r} no está en meta.csv")
                skipped += 1
                continue
            frame_count = g_kp[key].shape[0]
            token_count = g_ids[key].shape[0] if g_ids is not None and key in g_ids else 0
            fields = compute_metadata_fields(fname, video_column, label, frame_count, token_count)

            if key not in g_video:
                g_video.create_dataset(key, data=[fields["video_id"]], dtype=dt)
            if key not in g_group:
                g_group.create_dataset(key, data=[fields["source_group"]], dtype=dt)
            if key not in g_frames:
                g_frames.create_dataset(
                    key, data=np.array([fields["frame_count"]], dtype=np.int32)
                )
            else:
                g_frames[key][0] = fields["frame_count"]
            if key not in g_tokens:
                g_tokens.create_dataset(
                    key, data=np.array([fields["token_count"]], dtype=np.int32)
                )
            else:
                g_tokens[key][0] = fields["token_count"]
            if key not in g_words:
                g_words.create_dataset(
                    key, data=np.array([fields["word_count"]], dtype=np.int32)
                )
            else:
                g_words[key][0] = fields["word_count"]
            done += 1
            if done % 500 == 0:
                f.flush()
                print(f"[backfill] {done} hechos")
        f.flush()
        print(f"[backfill] listo: {done} hechos, {skipped} saltados (sin match en meta.csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=Path("data/processed/dataset_v6_unsloth.hdf5"))
    ap.add_argument("--n", type=int, default=5600)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    h5 = args.h5 if args.h5.is_absolute() else Path(__file__).resolve().parents[2] / args.h5
    main(h5, args.n, args.seed)
