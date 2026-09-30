"""Diagnóstico de interpretabilidad para CTCEncoder (v119) — "qué pasa, cómo,
dónde" en el plateau de val WER ~99-100% que se repite en cada formulación de
v119, independientemente de model-oriented (arquitectura) o data-oriented
(filtro+augmentation) palancas.

Técnica: visualización de las posteriors de CTC por frame (estándar en
debugging de ASR/CTC) -- para una muestra de clips de val, grafica P(blank) y
P(top no-blank) por paso temporal post-downsampling (T'), con marcas en los
picos donde el greedy decode emite una palabra. Esto muestra directamente SI
el modelo colapsa a blank uniforme (sin señal en ningún frame) o tiene picos
de señal localizados que simplemente no alcanzan para decodificar la frase
completa -- son causas raíz distintas y piden soluciones distintas.

Complementa con un resumen agregado sobre el split de val completo (o un
subconjunto): fracción de frames decodificados (argmax) como blank, % de
clips con colapso total (0 palabras decodificadas), distribución de WER.

Uso (mismo patrón que train_ctc_v119.py, debe correr desde la raíz del repo):
    MSLM_EXPERIMENT_CONFIG=experiments/v119_ctc/ctc_v119.toml \
        PYTHONPATH=. python scripts/diagnostics/diagnose_ctc_posteriors.py --tag best_wer
"""
import argparse
import functools
import os
import random

os.environ.setdefault("MSLM_EXPERIMENT_CONFIG", "experiments/v119_ctc/ctc_v119.toml")

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train_ctc_v119 as tv
from src.mslm.utils.paths import path_vars


def _load_state_dict_skip_mismatched(model, state_dict):
    """Como load_state_dict(strict=False) pero también tolera mismatch de
    forma (no solo de claves) -- strict=False de PyTorch sigue lanzando
    RuntimeError si una clave existe en ambos lados con shape distinta.
    Necesario para checkpoints donde vocab.json fue sobrescrito después del
    entrenamiento original (el classifier quedó dimensionado para un vocab
    distinto al guardado en disco). Solo seguro de usar cuando las capas con
    mismatch no se usan (p.ej. probing de activaciones que no llega a
    `classifier`)."""
    own = model.state_dict()
    filtered, skipped = {}, []
    for k, v in state_dict.items():
        if k in own and own[k].shape != v.shape:
            skipped.append(k)
            continue
        filtered[k] = v
    if skipped:
        print(f"[diag] state_dict: claves omitidas por mismatch de forma: {skipped}")
    model.load_state_dict(filtered, strict=False)


def build_val(tag, run_dir: str | None = None, strict: bool = True):
    """run_dir: si se pasa, reemplaza str(run_id) en la ruta de checkpoint, de
    vocab y de salida de diagnósticos -- permite apuntar a checkpoints fuera
    del esquema {version}/{run_id} (p.ej. 'run_id_baseline_no_data_levers').
    Sin el parámetro, comportamiento idéntico al original.
    strict=False usa `_load_state_dict_skip_mismatched` -- solo seguro si el
    caller no usa las capas con mismatch (ver su docstring)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, h5_file = tv.setup_paths()
    A = np.load(path_vars.data_path / "processed" / "adjacency_matrix.npy", allow_pickle=True)

    model_cfg = dict(tv.cfg.model)
    use_motion_stream = bool(model_cfg.get("use_motion_stream", False))
    include = tv._cfg("data", "primary_datasets", ["dataset2"])
    train_ratio = float(tv._cfg("data", "train_ratio", 0.8))
    text_group = tv._cfg("data", "text_group", "text_ctx")
    n_keypoints = int(tv._cfg("data", "n_keypoints", model_cfg.get("input_size", 111)))
    min_frames = int(tv._cfg("data", "min_frames", 0))
    filter_invalid_labels = bool(tv._cfg("data", "filter_invalid_labels", False))
    max_samples = tv._cfg("data", "max_samples", None)
    version = int(tv._cfg("training", "model_version", 119))
    run_id = int(tv._cfg("training", "run_id", 1))
    dir_name = run_dir if run_dir is not None else str(run_id)

    ds = tv.KeypointDataset(
        h5Path=h5_file, n_keypoints=n_keypoints, return_label=True,
        text_group=text_group, include_datasets=include,
        data_augmentation=False, max_length=4000,
        min_frames=min_frames, filter_invalid_labels=filter_invalid_labels,
    )
    if max_samples and int(max_samples) < len(ds.valid_index):
        rng = random.Random(23)
        idx = list(range(len(ds.valid_index)))
        rng.shuffle(idx)
        keep = sorted(idx[: int(max_samples)])
        ds.valid_index = [ds.valid_index[i] for i in keep]
        ds.video_lengths = [ds.video_lengths[i] for i in keep]
        ds.dataset_length = len(ds.valid_index)

    # Mismo seed que train_ctc_v119.py (torch.manual_seed(23) a nivel de módulo,
    # ya ejecutado al importar `tv`) -- random_split() en split_dataset() usa su
    # propio generator(seed=42) fijo, así que el split es reproducible sin
    # depender de este seed global; se mantiene por paridad con el script real.
    _, val_subset, _, _ = ds.split_dataset(train_ratio)

    vocab_path = f"../outputs/checkpoints/{version}/{dir_name}/vocab.json"
    vocab = tv.Vocab.load(vocab_path)

    collate = functools.partial(tv.ctc_collate_fn, vocab=vocab)
    val_dl = torch.utils.data.DataLoader(val_subset, batch_size=1, shuffle=False, collate_fn=collate)

    model = tv.CTCEncoder(
        A=A, input_size=model_cfg.get("input_size", n_keypoints),
        gcn_channels=tuple(model_cfg.get("gcn_channels", [32, 64, 128])),
        hidden_size=model_cfg.get("hidden_size", 256),
        lstm_layers=model_cfg.get("lstm_layers", 2),
        vocab_size=len(vocab) - 1,
        use_motion_stream=use_motion_stream,
    ).to(device)

    ckpt_path = f"../outputs/checkpoints/{version}/{dir_name}/{tag}/checkpoint.pth"
    state = torch.load(ckpt_path, map_location=device)
    if strict:
        model.load_state_dict(state["model_state"])
    else:
        _load_state_dict_skip_mismatched(model, state["model_state"])
    model.eval()
    print(f"[diag] Checkpoint cargado: {ckpt_path} (epoch={state.get('epoch')})")

    out_dir = f"../outputs/diagnostics/{version}/{dir_name}"
    os.makedirs(out_dir, exist_ok=True)
    return model, val_dl, vocab, device, out_dir


def run_clip(model, keypoint, vocab, device):
    """Un forward pass por clip (sin padding, igual que _encode_batch en
    train_ctc_v119.py). Devuelve log_probs_long [T',V+1], decoded_ids,
    fracción de frames cuyo argmax es blank (definición única, reusada tanto
    para graficar como para el resumen agregado -- antes este script tenía
    dos criterios distintos según si el clip se graficaba o no)."""
    keypoint = keypoint.to(device)
    with torch.no_grad():
        fm = torch.zeros(1, keypoint.size(1), dtype=torch.bool, device=device)
        _, log_probs_long, seq_lengths, _ = model(keypoint, fm)

    T = int(seq_lengths[0].item())
    log_probs = log_probs_long[0, :T].cpu()
    decoded_ids = tv.greedy_ctc_decode(log_probs_long.cpu(), seq_lengths.cpu(), blank=vocab.blank_id)[0]
    argmax_ids = log_probs.argmax(dim=-1)
    blank_frac = float((argmax_ids == vocab.blank_id).float().mean())
    return log_probs, decoded_ids, blank_frac


def plot_clip(log_probs, vocab, decoded_ids, clip_idx, out_dir):
    probs = log_probs.exp()
    blank_p = probs[:, vocab.blank_id].numpy()
    nonblank = probs.clone()
    nonblank[:, vocab.blank_id] = 0.0
    top_p, top_id = nonblank.max(dim=-1)
    top_p, top_id = top_p.numpy(), top_id.numpy()
    T = len(blank_p)

    fig, ax = plt.subplots(figsize=(max(6, T * 0.15), 3))
    ax.plot(blank_p, label="P(blank)", color="tab:gray")
    ax.plot(top_p, label="P(top no-blank)", color="tab:red")
    prev = None
    for t, (tid, p) in enumerate(zip(top_id, top_p)):
        if tid != prev and p > blank_p[t]:
            ax.annotate(vocab.id_to_token.get(int(tid), "?"), (t, p), fontsize=7, rotation=60)
        prev = tid
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("frame post-downsampling (T')")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"clip {clip_idx} | decoded: {' '.join(vocab.decode(decoded_ids)) or '(vacío)'}")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/clip_{clip_idx:04d}.png", dpi=120)
    plt.close(fig)


def main(args):
    model, val_dl, vocab, device, out_dir = build_val(args.tag)

    wers, blank_fracs, n_empty = [], [], 0
    for i, (keypoint, frames_mask, targets, target_lengths, labels) in enumerate(val_dl):
        if i >= args.n_eval:
            break
        log_probs, decoded_ids, blank_frac = run_clip(model, keypoint, vocab, device)
        if i < args.n_plot:
            plot_clip(log_probs, vocab, decoded_ids, i, out_dir)

        hyp = vocab.decode(decoded_ids)
        ref = tv.tokenize(labels[0])
        if ref:
            wers.append(tv.word_error_rate(hyp, ref))
        blank_fracs.append(blank_frac)
        if len(decoded_ids) == 0:
            n_empty += 1

    n = len(wers)
    report = [
        f"Clips evaluados: {n}",
        f"WER promedio: {sum(wers) / max(1, n):.1%}",
        f"% clips con colapso total (0 palabras decodificadas): {n_empty / max(1, n):.1%}",
        f"% frames con argmax==blank, promedio por clip: {sum(blank_fracs) / max(1, len(blank_fracs)):.1%}",
        f"% clips con WER==100%: {sum(1 for w in wers if w >= 1.0) / max(1, n):.1%}",
        f"% clips con WER==0%: {sum(1 for w in wers if w == 0.0) / max(1, n):.1%}",
    ]
    text = "\n".join(report)
    print("\n[diag] Resumen agregado:\n" + text)
    with open(f"{out_dir}/summary.txt", "w") as f:
        f.write(text + "\n")
    print(f"\n[diag] Plots de {args.n_plot} clips + resumen en {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="best_wer", help="Subdirectorio de checkpoint (best_wer, o número de época).")
    parser.add_argument("--n-plot", type=int, default=6, help="Clips a graficar (espectro de posteriors).")
    parser.add_argument("--n-eval", type=int, default=200, help="Clips a evaluar para el resumen agregado.")
    main(parser.parse_args())
