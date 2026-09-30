"""Experimento CE-vocab (v115): materializa los token IDs y la tabla de embeddings.

El HDF5 de fase 0 guarda por clip los *embeddings* de los tokens de la transcripción,
pero la pérdida CE necesita los *IDs* discretos y la tabla de embeddings de entrada del
LLM (para la cabeza de salida atada: logits = pred @ E^T).

Dos fases independientes:
  - token_ids: re-tokeniza las labels con el mismo tokenizer (CPU, segundos) y guarda
    `dataset2/token_ids/<idx>` (int32). Verifica que el nº de tokens coincida con el nº
    de embeddings almacenados para ese clip (misma llamada al tokenizer que en fase 0).
  - table: carga el LLM una vez (GPU, igual que build_dataset2_h5) y vuelca la tabla de
    embeddings de entrada a `data/processed/gemma3n_embed_table.pt` (fp16). Verifica
    contra los embeddings almacenados de un clip de muestra.

Uso:
    PYTHONPATH=. python scripts/data/add_token_ids_h5.py --phase token_ids
    PYTHONPATH=. python scripts/data/add_token_ids_h5.py --phase table   # en la máquina con GPU
"""
import argparse
from pathlib import Path

import h5py
import numpy as np


def phase_token_ids(f, llm_model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(llm_model)
    g = f["dataset2"]
    g_ids = g.require_group("token_ids")
    g_lb = g["labels"]
    g_emb = g["embeddings"]

    done = mismatched = 0
    for key in g_emb.keys():
        if key in g_ids:
            continue
        label = g_lb[key][:][0].decode()
        ids = tok(label, return_tensors="np").input_ids[0].astype(np.int32)
        n_emb = g_emb[key].shape[0]
        if len(ids) != n_emb:
            mismatched += 1
            print(f"[ids] {key} MISMATCH: {len(ids)} tokens vs {n_emb} embeddings — SKIP")
            continue
        g_ids.create_dataset(key, data=ids, compression="gzip")
        done += 1
    f.flush()
    print(f"[ids] FASE token_ids lista: {done} hechos, {mismatched} con mismatch")


def phase_table(f, llm_model, out_path):
    import torch
    from build_dataset2_h5 import _load_lm

    model = _load_lm(llm_model)
    emb = model.get_input_embeddings()
    # Se materializa pasando los IDs por el módulo (no leyendo emb.weight): gemma-3n usa
    # un scaled word embedding (multiplica por ~sqrt(hidden) en el forward) y la fase 0
    # guardó los embeddings YA escalados; la tabla debe vivir en el mismo espacio.
    vocab = emb.weight.shape[0]
    chunks = []
    with torch.no_grad():
        for s in range(0, vocab, 8192):
            ids = torch.arange(s, min(s + 8192, vocab), device=emb.weight.device)
            chunks.append(emb(ids).detach().cpu().to(torch.float16))
    table = torch.cat(chunks, dim=0)
    print(f"[table] {llm_model} | tabla {tuple(table.shape)} fp16 (post-escala del módulo)")

    # Verificación: los embeddings almacenados deben ser filas exactas de la tabla.
    g = f["dataset2"]
    if "token_ids" in g and len(g["token_ids"]):
        key = next(iter(g["token_ids"].keys()))
        ids = g["token_ids"][key][:]
        stored = torch.from_numpy(g["embeddings"][key][:])
        rows = table[torch.from_numpy(ids.astype(np.int64))].float()
        err = (stored - rows).abs().max().item()
        print(f"[table] verificación clip {key}: max|err| = {err:.3e}")
        assert err < 1e-2, "los embeddings almacenados no coinciden con la tabla"

    torch.save(table, out_path)
    print(f"[table] guardada en {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=Path("data/processed/dataset_v6_unsloth.hdf5"))
    ap.add_argument("--table-out", type=Path, default=Path("data/processed/gemma3n_embed_table.pt"))
    ap.add_argument("--llm-model", type=str, default="unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit")
    ap.add_argument("--phase", choices=["all", "token_ids", "table"], default="all")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    h5 = args.h5 if args.h5.is_absolute() else root / args.h5
    table_out = args.table_out if args.table_out.is_absolute() else root / args.table_out

    with h5py.File(h5, "a") as f:
        if args.phase in ("all", "token_ids"):
            phase_token_ids(f, args.llm_model)
        if args.phase in ("all", "table"):
            phase_table(f, args.llm_model, table_out)
