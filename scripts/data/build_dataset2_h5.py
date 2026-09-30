"""Fase 0 (subconjunto): construye el HDF5 de entrenamiento para dataset2.

Para cada clip seleccionado:
  - keypoints: extraídos con RTMPose (rtmlib, to_openpose -> 137 puntos), decodificando con cv2.
    Se elige el "signer" por mayor movimiento de muñecas. Formato (T, 137, 2), el que espera
    `remove_keypoints` del dataloader (reduce a 111).
  - embeddings objetivo: embeddings de token de entrada de gemma-3n (E2B, hidden=2048 =
    output_size del Imitator). Requiere transformers >= 4.53 (aquí 5.x).
  - labels: la transcripción (col `label` de meta.csv).

Escribe en data/processed/<out> bajo el grupo `dataset2/{keypoints,embeddings,labels}/<idx>`,
con claves enteras alineadas (formato que lee KeypointDataset). Es resumible (salta claves ya
presentes) y procesa por fases (keypoints -> libera RTMPose -> embeddings) para no saturar GPU.

Uso (lanzar en tmux, ver tiempos largos):
    PYTHONPATH=. python scripts/data/build_dataset2_h5.py --n 600 --max-frames 250
"""
import argparse
import os
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

RAW = Path("data/raw/dataset2")
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
            rows.append((fname, str(r["label"]).lower()))
    random.Random(seed).shuffle(rows)
    return rows[:n]


# ----------------------------- Keypoints (RTMPose) -----------------------------
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


def phase_keypoints(f, clips, max_frames):
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
    g_kp = f["dataset2"].require_group("keypoints")
    g_lb = f["dataset2"].require_group("labels")
    dt = h5py.string_dtype(encoding="utf-8")

    done = skipped = 0
    for idx, (fname, label) in enumerate(clips):
        key = str(idx)
        if key in g_kp:
            continue
        kp = extract_keypoints(RAW / "videos" / fname, model, max_frames)
        if kp is None or kp.shape[0] < 2:
            skipped += 1
            print(f"[kp] {idx} SKIP ({fname})")
            continue
        g_kp.create_dataset(key, data=kp, compression="gzip", compression_opts=4)
        if key not in g_lb:
            g_lb.create_dataset(key, data=[label], dtype=dt, compression="gzip")
        done += 1
        if done % 25 == 0:
            f.flush()
            print(f"[kp] {done} hechos | último {kp.shape} ({fname})")
    f.flush()
    print(f"[kp] FASE keypoints lista: {done} hechos, {skipped} saltados")


# ----------------------------- Embeddings (LLM) -----------------------------
def _load_lm(llm_model):
    """Carga el LLM para extraer la tabla de embeddings de entrada.

    gemma-3n es multimodal y viene pre-cuantizado 4bit (unsloth-bnb); from_pretrained lee la
    quantization_config embebida. Se prueban varias clases auto según la arquitectura.
    """
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(llm_model, device_map="cuda")
    except Exception as e1:
        print(f"[emb] AutoModelForCausalLM no aplica ({type(e1).__name__}); probando ImageTextToText")
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(llm_model, device_map="cuda")


def phase_embeddings(f, clips, llm_model):
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(llm_model)
    model = _load_lm(llm_model)
    emb = model.get_input_embeddings()
    dim = emb.weight.shape[1]
    print(f"[emb] {llm_model} | dim embeddings = {dim}  (output_size del Imitator debe ser {dim})")

    g_emb = f["dataset2"].require_group("embeddings")
    g_kp = f["dataset2"]["keypoints"]

    done = 0
    for idx, (fname, label) in enumerate(clips):
        key = str(idx)
        if key not in g_kp:       # solo clips con keypoints válidos
            continue
        if key in g_emb:
            continue
        with torch.no_grad():
            ids = tok(label, return_tensors="pt").input_ids.to(emb.weight.device)
            vecs = emb(ids[0]).detach().cpu().float().numpy()   # (n_tokens, dim)
        g_emb.create_dataset(key, data=vecs, compression="gzip", compression_opts=4)
        done += 1
        if done % 50 == 0:
            f.flush()
            print(f"[emb] {done} hechos")
    f.flush()
    print(f"[emb] FASE embeddings lista: {done} hechos | dim={dim}")
    return dim


# ----------------------------- Text contextual (v118e) -----------------------------
TEXT_CTX_MODEL = "unsloth/gemma-3n-E2B-it"  # variante SIN cuantizar (bf16), ver nota abajo


def _load_lm_bf16(llm_model):
    """Igual que _load_lm pero sin bitsandbytes 4bit.

    phase_text_ctx necesita un forward real (no solo `get_input_embeddings()`).
    La variante -bnb-4bit usada en phase_embeddings revienta con un
    AssertionError dentro de bitsandbytes (`fix_4bit_weight_quant_state_from_module`,
    `assert module.weight.shape[1] == 1`) al pasar por las capas `altup_projections`
    propias de la arquitectura AltUp de Gemma-3n -- bug de compatibilidad
    bitsandbytes/transformers para esa capa concreta, no de este código. La
    variante sin cuantizar (bf16) no tiene ese problema. Requiere ~5-7GB de
    descarga la primera vez; usar HF_HOME apuntando a un disco con espacio
    (en esta máquina /home estaba al 99%, se usó .hf_cache).
    """
    from transformers import AutoModelForCausalLM
    import torch
    try:
        return AutoModelForCausalLM.from_pretrained(llm_model, device_map="cuda", dtype=torch.bfloat16)
    except Exception as e1:
        print(f"[ctx] AutoModelForCausalLM no aplica ({type(e1).__name__}); probando ImageTextToText")
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(llm_model, device_map="cuda", dtype=torch.bfloat16)


def phase_text_ctx(f, clips, llm_model=None):
    """Último hidden state contextual de Gemma por clip (1 vector, NO per-token).

    v118 (contrastivo) promediaba la TABLA DE EMBEDDINGS DE ENTRADA por token
    (`phase_embeddings`) como representación de frase. El diagnóstico de
    discriminabilidad mostró que ese espacio es casi isotrópico entre frases
    distintas (coseno medio entre pares = 0.80, vecino más cercano = 0.90 de
    mediana) -> techo de retrieval bajo por construcción, no por el encoder de
    vídeo. Esta fase corre un forward real (no solo lookup) y toma el ÚLTIMO
    token de la representación final de la última capa: en un decoder causal
    es el único token que ya vio toda la frase (mean-pool mezclaría contexto
    parcial de posiciones tempranas). Se guarda como (1, dim) para no tocar
    collate_fn/masked_mean (tratado como secuencia de 1 "token" sin padding).

    IMPORTANTE (AltUp): `output_hidden_states=True` + `outputs.hidden_states[-1]`
    en Gemma3n NO da la representación final -- da el tensor crudo por capa con
    una dimensión extra de 4 "streams" (mecanismo AltUp, ver
    `Gemma3nTextModel.forward` en transformers). La representación correcta
    (la que de hecho alimenta al LM head) es `outputs.last_hidden_state`,
    calculada internamente como la media de los 4 streams re-escalados +
    RMSNorm final. Por eso se llama a `model.model(...)` (el submódulo base)
    en vez de pasar por el wrapper ForCausalLM/ConditionalGeneration completo.
    """
    import torch

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEXT_CTX_MODEL)
    model = _load_lm_bf16(TEXT_CTX_MODEL)
    model.eval()
    device = next(model.parameters()).device
    dim = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size

    g_ctx = f["dataset2"].require_group("text_ctx")
    g_kp = f["dataset2"]["keypoints"]

    done = 0
    for idx, (fname, label) in enumerate(clips):
        key = str(idx)
        if key not in g_kp:        # solo clips con keypoints válidos
            continue
        if key in g_ctx:
            continue
        with torch.no_grad():
            inputs = tok(label, return_tensors="pt").to(device)
            out = model.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            vec = out.last_hidden_state[0, -1].detach().cpu().float().numpy()  # último token, ya reducido (no AltUp)
        g_ctx.create_dataset(key, data=vec[None, :], compression="gzip", compression_opts=4)
        done += 1
        if done % 50 == 0:
            f.flush()
            print(f"[ctx] {done} hechos")
    f.flush()
    print(f"[ctx] FASE text_ctx lista: {done} hechos | dim={dim}")
    return dim


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="nº de clips del subconjunto")
    ap.add_argument("--max-frames", type=int, default=250, help="frames máx por video")
    ap.add_argument("--out", type=Path, default=Path("data/processed/dataset_v6_unsloth.hdf5"))
    ap.add_argument("--llm-model", type=str, default="unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--phase", choices=["all", "keypoints", "embeddings", "text_ctx"], default="all")
    args = ap.parse_args()

    out = args.out
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[2] / out
    out.parent.mkdir(parents=True, exist_ok=True)

    clips = select_clips(args.n, args.seed)
    print(f"Subconjunto: {len(clips)} clips | salida: {out}")

    with h5py.File(out, "a") as f:
        f.require_group("dataset2")
        if args.phase in ("all", "keypoints"):
            phase_keypoints(f, clips, args.max_frames)
        if args.phase in ("all", "embeddings"):
            phase_embeddings(f, clips, args.llm_model)
        if args.phase == "text_ctx":
            phase_text_ctx(f, clips, args.llm_model)
        nk = len(f["dataset2"]["keypoints"]) if "keypoints" in f["dataset2"] else 0
        ne = len(f["dataset2"]["embeddings"]) if "embeddings" in f["dataset2"] else 0
        nc = len(f["dataset2"]["text_ctx"]) if "text_ctx" in f["dataset2"] else 0
        print(f"HDF5 listo: dataset2 con {nk} keypoints, {ne} embeddings, {nc} text_ctx")
