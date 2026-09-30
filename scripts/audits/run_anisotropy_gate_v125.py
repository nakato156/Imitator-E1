"""Gate de fidelidad de PCA-512+whitening sobre la tabla real de embeddings
de Gemma (v125). Ajusta SOLO con token IDs vistos en clips de train; evalúa
coseno de reconstrucción sobre token IDs de val (held out)."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mslm.utils.embedding_space import (  # noqa: E402
    fit_embedding_transform,
    fit_standardized_embedding_transform,
    reconstruction_cosine,
)


def collect_token_ids(h5_path: Path, clip_ids: list[str]) -> set[int]:
    ids: set[int] = set()
    with h5py.File(h5_path, "r") as f:
        g = f["dataset2"]["token_ids"]
        for cid in clip_ids:
            if cid in g:
                ids.update(int(t) for t in g[cid][:])
    return ids


def main(h5_path: Path, table_path: Path, split_path: Path, out_path: Path, n_components: int) -> None:
    split = json.loads(split_path.read_text())
    table = torch.load(table_path, map_location="cpu").float().numpy()  # [vocab, 2048]

    train_ids = sorted(collect_token_ids(h5_path, split["train"]))
    val_ids = sorted(collect_token_ids(h5_path, split["val"]) - set(train_ids))

    train_emb = table[train_ids]
    val_emb = table[val_ids] if val_ids else table[train_ids[: max(1, len(train_ids) // 10)]]

    transform = fit_embedding_transform(train_emb, n_components=n_components, epsilon=1e-5)
    recon = transform.inverse_transform(transform.transform(val_emb))
    cos = reconstruction_cosine(val_emb, recon)

    result = {
        "n_components": n_components,
        "n_train_tokens": len(train_ids),
        "n_val_tokens": len(val_ids),
        "cosine_mean": float(cos.mean()),
        "cosine_p5": float(np.percentile(cos, 5)),
        "gate_cosine_mean_ge_0.99": bool(cos.mean() >= 0.99),
        "gate_cosine_p5_ge_0.97": bool(np.percentile(cos, 5) >= 0.97),
    }
    pca_accepted = result["gate_cosine_mean_ge_0.99"] and result["gate_cosine_p5_ge_0.97"]
    result["selected_mode"] = "pca_whitening" if pca_accepted else "full_standardized"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    if pca_accepted:
        transform_blob = {
            "mode": "pca_whitening",
            "mean": torch.from_numpy(transform.mean),
            "components": torch.from_numpy(transform.components),
            "singular_values": torch.from_numpy(transform.singular_values),
            "n_samples": transform.n_samples,
            "epsilon": transform.epsilon,
        }
    else:
        fallback = fit_standardized_embedding_transform(train_emb, epsilon=1e-5)
        transform_blob = {
            "mode": "full_standardized",
            "mean": torch.from_numpy(fallback.mean),
            "scale": torch.from_numpy(fallback.scale),
            "epsilon": fallback.epsilon,
        }
    torch.save(transform_blob, out_path.with_suffix(".transform.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=Path("data/processed/dataset_v6_unsloth.hdf5"))
    ap.add_argument("--table", type=Path, default=Path("data/processed/gemma3n_embed_table.pt"))
    ap.add_argument("--split", type=Path, default=Path("data/processed/dataset2_split_v125.json"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/v125_anisotropy_gate.json"))
    ap.add_argument("--n-components", type=int, default=512)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    main(
        root / args.h5 if not args.h5.is_absolute() else args.h5,
        root / args.table if not args.table.is_absolute() else args.table,
        root / args.split if not args.split.is_absolute() else args.split,
        root / args.out if not args.out.is_absolute() else args.out,
        args.n_components,
    )
