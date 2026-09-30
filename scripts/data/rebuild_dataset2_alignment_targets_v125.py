"""Regenera token_ids/embeddings de dataset2 con las reglas cerradas de v125.

La etiqueta generativa original se conserva. Las unidades visuales usan texto
NFC, minúsculas, espacios normalizados, sin puntuación y sin special tokens.
Los embeddings se reconstruyen por lookup en la tabla versionada de Gemma.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mslm.dataloader.alignment_text import normalize_alignment_text  # noqa: E402

MODEL_ID = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"


def rebuild_targets(h5_path: Path, table_path: Path, model_id: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    table = torch.load(table_path, map_location="cpu").float().numpy()

    with h5py.File(h5_path, "a") as f:
        dataset = f["dataset2"]
        labels = dataset["labels"]
        old_ids = dataset["token_ids"]
        old_embeddings = dataset["embeddings"]
        old_counts = dataset["token_count"]

        for name in ("token_ids_v125_tmp", "embeddings_v125_tmp", "token_count_v125_tmp"):
            if name in dataset:
                del dataset[name]
        new_ids = dataset.create_group("token_ids_v125_tmp")
        new_embeddings = dataset.create_group("embeddings_v125_tmp")
        new_counts = dataset.create_group("token_count_v125_tmp")

        done = 0
        for clip_id in sorted(labels.keys(), key=int):
            label = labels[clip_id][0].decode()
            normalized = normalize_alignment_text(label)
            ids = tokenizer(
                normalized,
                add_special_tokens=False,
                return_attention_mask=False,
            ).input_ids
            ids_array = np.asarray(ids, dtype=np.int32)
            embeddings = table[ids_array.astype(np.int64)]
            new_ids.create_dataset(clip_id, data=ids_array, compression="gzip")
            new_embeddings.create_dataset(
                clip_id, data=embeddings, compression="gzip", compression_opts=4
            )
            new_counts.create_dataset(
                clip_id, data=np.asarray([len(ids)], dtype=np.int32)
            )
            done += 1
            if done % 500 == 0:
                f.flush()
                print(f"[alignment-targets] {done} clips")

        del dataset["token_ids"]
        del dataset["embeddings"]
        del dataset["token_count"]
        dataset.move("token_ids_v125_tmp", "token_ids")
        dataset.move("embeddings_v125_tmp", "embeddings")
        dataset.move("token_count_v125_tmp", "token_count")
        f.flush()
        print(f"[alignment-targets] listos: {done} clips")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5", type=Path, default=Path("data/processed/dataset_v6_unsloth.hdf5")
    )
    parser.add_argument(
        "--table", type=Path, default=Path("data/processed/gemma3n_embed_table.pt")
    )
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    h5_path = args.h5 if args.h5.is_absolute() else root / args.h5
    table_path = args.table if args.table.is_absolute() else root / args.table
    rebuild_targets(h5_path, table_path, args.model)
