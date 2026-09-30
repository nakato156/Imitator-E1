"""Fase 0 del plan post-v118c: diagnóstico train-vs-val retrieval sobre best_r1.

Decide si el techo de R@1 (~2.9%) es sobreajuste (capacidad sobra -> recortar
ayuda, v118d) o un muro de datos/dificultad de tarea (recortar no ayuda -> pivotar
a v119 / más datos). Carga el checkpoint 118/4/best_r1, encodea TODO train y TODO
val con el mismo encoder, y compara retrieval@1/5/10 + median_rank en ambos splits
contra su propio azar (1/N_train vs 1/N_val).

Uso:
    MSLM_EXPERIMENT_CONFIG=experiments/v118_contrastive/contrastive_v118c.toml \
        PYTHONPATH=. python scripts/diagnostics/diag_v118_train_val_gap.py
"""
import os
os.environ.setdefault("MSLM_EXPERIMENT_CONFIG", "experiments/v118_contrastive/contrastive_v118c.toml")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from settings import initialize
initialize()

import json
import torch
import torch.nn.functional as F

from src.mslm.utils.setup_train import setup_paths, build_model, prepare_datasets, create_dataloaders
from src.mslm.utils.config_loader import cfg
from src.mslm.models.imitator import PrefixImitator
from src.mslm.models.contrastive import ContrastiveAligner
from src.mslm.training.loss_contrastive import retrieval_metrics

CKPT_PATH = "../outputs/checkpoints/118/4/best_r1/checkpoint.pth"
MAX_FRAMES = 1024


def _cfg(section, key, default):
    s = getattr(cfg, section, {}) or {}
    return s.get(key, default)


@torch.no_grad()
def encode_split(model, dl, device, max_frames):
    V, T = [], []
    for batch in dl:
        keypoint, frames_mask = batch[0].to(device), batch[1].to(device)
        text_emb, text_mask = batch[2].to(device), batch[3].to(device)
        if keypoint.size(1) > max_frames:
            keypoint = keypoint[:, :max_frames]
            frames_mask = frames_mask[:, :max_frames]
        B = keypoint.size(0)
        for i in range(B):
            v = model.encode_video(keypoint[i:i + 1], frames_mask[i:i + 1], normalize=False)
            V.append(v)
        t = model.encode_text(text_emb, text_mask, normalize=False)
        T.append(t)
    V = torch.cat(V, dim=0)
    T = torch.cat(T, dim=0)
    # retrieval_metrics espera vectores L2-normalizados (similitud coseno)
    return F.normalize(V, dim=-1), F.normalize(T, dim=-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, h5_file = setup_paths()

    model_cfg = dict(cfg.model)
    include = _cfg("data", "primary_datasets", ["dataset2"])
    train_ratio = float(_cfg("data", "train_ratio", 0.8))

    tr_ds, val_ds, _, _ = prepare_datasets(
        h5_file, train_ratio, model_cfg.get("input_size", 111),
        include_datasets=include, return_token_ids=False)
    # batch_size grande solo para empaquetar; el encode es muestra-a-muestra adentro
    tr_dl, val_dl = create_dataloaders(tr_ds, val_ds, batch_size=32, num_workers=8)

    aligner = ContrastiveAligner(
        PrefixImitator(build_model(**model_cfg)),
        hidden=model_cfg.get("output_size", 2048),
        proj_dim=int(_cfg("loss", "proj_dim", 256)),
    ).to(device)

    state = torch.load(CKPT_PATH, map_location=device)
    aligner.load_state_dict(state["model_state"])
    aligner.eval()
    print(f"[diag] Checkpoint cargado: epoch={state.get('epoch')} de {CKPT_PATH}")
    print(f"[diag] Train N={len(tr_ds)} (chance R@1={1/len(tr_ds):.2%}) | "
          f"Val N={len(val_ds)} (chance R@1={1/len(val_ds):.2%})")

    results = {}
    for name, dl, ds in [("train", tr_dl, tr_ds), ("val", val_dl, val_ds)]:
        V, T = encode_split(aligner, dl, device, MAX_FRAMES)
        m = retrieval_metrics(V, T)
        results[name] = m
        print(f"\n=== {name} (N={len(ds)}) ===")
        print(f"  R@1={m['R@1']:.2%}  R@5={m['R@5']:.2%}  R@10={m['R@10']:.2%}  "
              f"median_rank={m['median_rank']:.0f}  chance={m['chance']:.2%}")

    gap_factor = (results["train"]["R@1"] / max(results["train"]["chance"], 1e-9)) / \
                 (results["val"]["R@1"] / max(results["val"]["chance"], 1e-9))
    print(f"\n=== VEREDICTO ===")
    print(f"train R@1/chance = {results['train']['R@1']/results['train']['chance']:.1f}x")
    print(f"val   R@1/chance = {results['val']['R@1']/results['val']['chance']:.1f}x")
    print(f"ratio (train_over_chance / val_over_chance) = {gap_factor:.1f}x")
    if gap_factor > 3:
        print("-> GAP GRANDE: el modelo memoriza train mejor que generaliza a val.")
        print("   Sugiere SOBREAJUSTE -> recortar capacidad (v118d) es razonable.")
    else:
        print("-> GAP PEQUEÑO: train y val están igual de mal.")
        print("   El modelo NI SIQUIERA ajusta train -> el muro es datos/dificultad")
        print("   de tarea, no exceso de capacidad. Recortar params no ayudará por sí solo.")

    out_path = "outputs/diag_v118_train_val_gap.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results, "gap_factor": gap_factor}, f, indent=2)
    print(f"\n[diag] Resultados guardados en {out_path}")


if __name__ == "__main__":
    main()
