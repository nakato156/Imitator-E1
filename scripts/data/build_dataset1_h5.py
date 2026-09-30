"""v120 — construye el HDF5 de keypoints para dataset1 (reconocimiento de señas
aislado, 64 glosas x 50 ejemplos c/u).

A diferencia de dataset2 (frases en español, sin glosas, usado por v113-v119
para una tarea de traducción que la literatura cargada no resuelve con estos
datos), dataset1 es clasificación: 1 palabra (glosa) por label. No hay fase de
embeddings de LLM -- la clasificación usa class-id, no vectores de Gemma.

Reusa literal `extract_keypoints` de build_dataset2_h5.py (RTMPose + selección
de signer por movimiento de muñecas, formato (T,137,2) que espera
`remove_keypoints` del dataloader).

Escribe en data/processed/<out> bajo el grupo `dataset1/{keypoints,labels}/<idx>`.
Resumible (salta claves ya presentes).

Uso (lanzar en background, GPU, ~3200 videos):
    PYTHONPATH=. python scripts/data/build_dataset1_h5.py --max-frames 150
"""
import argparse
import os
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

RAW = Path("data/raw/dataset1")
N_OPENPOSE = 134  # rtmlib to_openpose -> 134 puntos (layout que asume remove_keypoints del dataloader)
LH_WRIST, RH_WRIST = 93, 113  # muñecas: hand_l empieza en 93, hand_r en 113 (ver remove_keypoints)


def select_clips(n, seed):
    meta = pd.read_csv(RAW / "meta.csv")
    meta = meta[meta["label"].notna() & (meta["label"].astype(str).str.strip() != "")]
    videos = set(os.listdir(RAW / "videos"))
    rows = []
    for _, r in meta.iterrows():
        fname = f"{r['id']}.mp4"
        if fname in videos:
            rows.append((fname, str(r["label"]).strip()))
    random.Random(seed).shuffle(rows)
    return rows if n is None else rows[:n]


# ----------------------------- Keypoints (RTMPose) -----------------------------
# Idéntico a build_dataset2_h5.py::extract_keypoints (mismo formato de entrada
# para remove_keypoints, mismo criterio de selección de signer).
def extract_keypoints(video_path, model, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    per_person = {}  # person_idx -> list[(frame_idx, kpts(137,2))]
    fi = 0
    while fi < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        keypoints, _ = model(frame_rgb)          # (P, 137, 2), (P, 137)
        kp = np.asarray(keypoints, dtype=np.float32)
        if kp.ndim == 3 and kp.shape[1] == N_OPENPOSE:
            for p in range(kp.shape[0]):
                per_person.setdefault(p, []).append((fi, kp[p]))
        fi += 1
    cap.release()

    if not per_person:
        return None

    # signer = persona con mayor movimiento total de muñecas
    def movement(seq):
        tot = 0.0
        for (_, a), (_, b) in zip(seq[:-1], seq[1:]):
            tot += np.linalg.norm(a[LH_WRIST] - b[LH_WRIST])
            tot += np.linalg.norm(a[RH_WRIST] - b[RH_WRIST])
        return tot

    best = max(per_person.values(), key=movement)
    frames = np.stack([k for _, k in sorted(best, key=lambda t: t[0])])  # (T, 137, 2)
    return frames


def phase_keypoints(f, clips, max_frames, include_metadata=False):
    from rtmlib import Custom
    model = Custom(
        to_openpose=True,
        det_class="RTMDet",
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
        det_input_size=(640, 640),
        pose_class="RTMPose",
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip",
        pose_input_size=(288, 384),
        backend="onnxruntime",
        device="cuda",
    )
    g_kp = f["dataset1"].require_group("keypoints")
    g_lb = f["dataset1"].require_group("labels")
    g_video = f["dataset1"].require_group("video_id") if include_metadata else None
    g_signer = f["dataset1"].require_group("signer_id") if include_metadata else None
    g_rep = f["dataset1"].require_group("repetition") if include_metadata else None
    dt = h5py.string_dtype(encoding="utf-8")

    done = skipped = 0
    for idx, (fname, label) in enumerate(clips):
        key = str(idx)
        if key not in g_lb:
            g_lb.create_dataset(key, data=[label], dtype=dt, compression="gzip")
        if include_metadata:
            stem = Path(fname).stem
            parts = stem.split("_")
            if key not in g_video:
                g_video.create_dataset(key, data=[fname], dtype=dt, compression="gzip")
            if key not in g_signer:
                g_signer.create_dataset(key, data=[int(parts[1])], compression="gzip")
            if key not in g_rep:
                g_rep.create_dataset(key, data=[int(parts[2])], compression="gzip")
        if key in g_kp:
            continue
        kp = extract_keypoints(RAW / "videos" / fname, model, max_frames)
        if kp is None or kp.shape[0] < 2:
            skipped += 1
            print(f"[kp] {idx} SKIP ({fname})")
            continue
        g_kp.create_dataset(key, data=kp, compression="gzip", compression_opts=4)
        done += 1
        if done % 100 == 0:
            f.flush()
            print(f"[kp] {done} hechos | último {kp.shape} ({fname})")
    f.flush()
    print(f"[kp] FASE keypoints lista: {done} hechos, {skipped} saltados")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="nº de clips a procesar (default: todos)")
    ap.add_argument("--max-frames", type=int, default=150, help="frames máx por video")
    ap.add_argument("--out", type=Path, default=Path("data/processed/dataset1_isolated.hdf5"))
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument(
        "--include-metadata",
        action="store_true",
        help="guarda video_id, signer_id y repetition por clip",
    )
    args = ap.parse_args()

    out = args.out
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[2] / out
    out.parent.mkdir(parents=True, exist_ok=True)

    clips = select_clips(args.n, args.seed)
    print(f"Subconjunto: {len(clips)} clips | salida: {out}")

    with h5py.File(out, "a") as f:
        f.require_group("dataset1")
        phase_keypoints(f, clips, args.max_frames, include_metadata=args.include_metadata)
        nk = len(f["dataset1"]["keypoints"])
        print(f"HDF5 listo: dataset1 con {nk} keypoints")
