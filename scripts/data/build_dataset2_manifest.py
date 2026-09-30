"""Construye el manifiesto de integridad de dataset2 sobre el HDF5 real (v125).

Requiere que `backfill_dataset2_metadata.py` ya corrió (lee video_id,
source_group, frame_count, token_count, que no vienen del build original).
El truncamiento se detecta comparando frame_count (h5) contra el frame count
real del video fuente (`cv2.VideoCapture(...).get(cv2.CAP_PROP_FRAME_COUNT)`,
lectura de metadata, no decodifica frames): un clip está truncado si
frame_count_h5 / frame_count_video < TRUNCATION_RATIO_THRESHOLD. El viejo
heurístico (frame_count >= 250, el cap de `build_dataset2_h5.py
--max-frames` de un build anterior de 600 clips) ya no aplica: el build
actual de 5600 clips no usó ese cap, y marcaba como truncados ~42% de los
clips que en realidad solo eran videos largos.
"""
import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mslm.dataloader.integrity_filter import ClipRecord, filter_clip_records  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset2_h5 import RAW  # noqa: E402

TRUNCATION_RATIO_THRESHOLD = 0.9


def _is_truncated(video_id: str, frame_count_h5: int) -> bool:
    """True si frame_count_h5 cubre menos del THRESHOLD del video fuente.

    Si el video no se puede abrir o cv2 reporta 0 frames, no se puede
    verificar completitud -> se marca truncado (fail-safe) y se loggea.
    """
    video_path = RAW / "videos" / f"{video_id}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warnings.warn(f"[truncation-check] no se pudo abrir video para clip video_id={video_id} ({video_path})")
        cap.release()
        return True
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        warnings.warn(f"[truncation-check] cv2 reportó 0 frames para clip video_id={video_id} ({video_path})")
        return True
    return (frame_count_h5 / total_frames) < TRUNCATION_RATIO_THRESHOLD


def build_records(h5_path: Path) -> list[ClipRecord]:
    records = []
    with h5py.File(h5_path, "r") as f:
        g = f["dataset2"]
        groups_to_cover = (
            "keypoints",
            "labels",
            "embeddings",
            "token_ids",
            "video_id",
            "source_group",
            "frame_count",
            "token_count",
            "word_count",
        )
        clip_ids = sorted(
            {
                key
                for group_name in groups_to_cover
                if group_name in g
                for key in g[group_name].keys()
            },
            key=int,
        )
        for key in clip_ids:
            has_kp = key in g["keypoints"]
            has_label = key in g["labels"]
            has_emb = key in g["embeddings"]
            has_ids = "token_ids" in g and key in g["token_ids"]
            frame_count = int(g["frame_count"][key][0]) if "frame_count" in g and key in g["frame_count"] else 0
            token_count = int(g["token_count"][key][0]) if "token_count" in g and key in g["token_count"] else 0
            embedding_rows = g["embeddings"][key].shape[0] if has_emb else 0
            kp_arr = g["keypoints"][key][:] if has_kp else np.array([])
            emb_arr = g["embeddings"][key][:] if has_emb else np.array([])
            video_id = g["video_id"][key][0].decode() if "video_id" in g and key in g["video_id"] else None
            if video_id:
                truncated = _is_truncated(video_id, frame_count)
            else:
                warnings.warn(f"[truncation-check] clip {key} no tiene video_id (backfill no corrió?) -> truncated=True")
                truncated = True
            records.append(
                ClipRecord(
                    clip_id=key,
                    has_keypoints=has_kp,
                    has_label=has_label,
                    has_embeddings=has_emb,
                    has_token_ids=has_ids,
                    frame_count=frame_count,
                    token_count=token_count,
                    embedding_rows=embedding_rows,
                    truncated=truncated,
                    keypoints_have_nan_inf=bool(kp_arr.size and not np.isfinite(kp_arr).all()),
                    embeddings_have_nan_inf=bool(emb_arr.size and not np.isfinite(emb_arr).all()),
                )
            )
    return records


def main(h5_path: Path, out_path: Path) -> None:
    records = build_records(h5_path)
    kept, manifest = filter_clip_records(records)
    causes = Counter(m["cause"] for m in manifest if not m["kept"])
    summary = {"total": len(manifest), "kept": len(kept), "excluded": len(manifest) - len(kept), "by_cause": dict(causes)}
    out_path.write_text(json.dumps({"summary": summary, "manifest": manifest}, indent=2, ensure_ascii=False))
    print(f"[manifest] {summary}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=Path("data/processed/dataset_v6_unsloth.hdf5"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/dataset2_manifest_v125.json"))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    h5 = args.h5 if args.h5.is_absolute() else root / args.h5
    out = args.out if args.out.is_absolute() else root / args.out
    main(h5, out)
