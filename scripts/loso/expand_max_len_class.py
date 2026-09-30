"""Expand a clean-LOSO checkpoint's max_len_class (16 -> 32) for Level B.

Any saved tensor whose shape no longer matches a freshly constructed model at
the new ``max_len_class`` gets its old rows copied into the matching prefix
of the new (randomly initialized) tensor; everything else copies through
unchanged. This is shape-driven rather than hardcoded to specific module
names, so it transfers both ``token_head.position_embedding`` (contextual
variant) and ``length_head.classifier.1`` (both length_head variants)
without the script needing to know which architecture variant is in play.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.mslm.models.temporal_sign_prompt import STGCNTemporalFrameEncoder, TemporalSignPromptModel

DEFAULT_ADJACENCY = Path("data/processed/adjacency_matrix.npy")


def infer_model_kwargs(state_dict: dict, arch_config: dict) -> dict:
    token_head_variant = arch_config.get("token_head", "contextual")
    length_head_variant = arch_config.get("length_head", "attention")
    embedding_dim, hidden_size = state_dict["embedding_head.weight"].shape
    vocab_size = state_dict["token_head.classifier.weight"].shape[0]
    old_max_len_class = state_dict["length_head.classifier.1.weight"].shape[0] - 1
    return {
        "hidden_size": int(hidden_size),
        "vocab_size": int(vocab_size),
        "embedding_dim": int(embedding_dim),
        "token_head_variant": token_head_variant,
        "length_head_variant": length_head_variant,
        "old_max_len_class": int(old_max_len_class),
    }


def expand_state_dict(old_state: dict, new_scaffold: dict) -> tuple[dict, list[dict]]:
    merged = dict(new_scaffold)
    expanded_params = []
    for name, new_tensor in new_scaffold.items():
        if name not in old_state:
            continue  # genuinely new param (shouldn't happen for this expansion); keep scaffold init
        old_tensor = old_state[name]
        if old_tensor.shape == new_tensor.shape:
            merged[name] = old_tensor
            continue
        slices = tuple(slice(0, min(o, n)) for o, n in zip(old_tensor.shape, new_tensor.shape))
        merged_tensor = new_tensor.clone()
        merged_tensor[slices] = old_tensor[slices]
        merged[name] = merged_tensor
        expanded_params.append(
            {"name": name, "old_shape": list(old_tensor.shape), "new_shape": list(new_tensor.shape)}
        )
    return merged, expanded_params


def build_scaffold_model(model_kwargs: dict, new_max_len_class: int, adjacency_path: Path) -> TemporalSignPromptModel:
    A = np.load(adjacency_path, allow_pickle=True)
    encoder = STGCNTemporalFrameEncoder(A, hidden_size=model_kwargs["hidden_size"])
    return TemporalSignPromptModel(
        encoder,
        hidden_size=model_kwargs["hidden_size"],
        vocab_size=model_kwargs["vocab_size"],
        embedding_dim=model_kwargs["embedding_dim"],
        max_len_class=new_max_len_class,
        token_head_variant=model_kwargs["token_head_variant"],
        length_head_variant=model_kwargs["length_head_variant"],
    )


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand_checkpoint(
    checkpoint_path: Path,
    new_max_len_class: int = 32,
    adjacency_path: Path = DEFAULT_ADJACENCY,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    arch_config = checkpoint.get("arch_config", {})
    model_kwargs = infer_model_kwargs(checkpoint["model"], arch_config)
    if model_kwargs["old_max_len_class"] >= new_max_len_class:
        raise ValueError(
            f"new_max_len_class={new_max_len_class} must exceed "
            f"old_max_len_class={model_kwargs['old_max_len_class']}"
        )
    scaffold = build_scaffold_model(model_kwargs, new_max_len_class, adjacency_path)
    merged_state, expanded_params = expand_state_dict(checkpoint["model"], scaffold.state_dict())

    new_checkpoint = dict(checkpoint)
    new_checkpoint["model"] = merged_state
    new_checkpoint["arch_config"] = {**arch_config, "max_len_class": new_max_len_class}
    new_checkpoint["max_len_class_expansion"] = {
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_of(checkpoint_path),
        "old_max_len_class": model_kwargs["old_max_len_class"],
        "new_max_len_class": new_max_len_class,
        "expanded_params": expanded_params,
    }
    # optimizer state shapes no longer match the expanded model; drop it so a
    # fresh AdamW state is built when this checkpoint is resumed.
    new_checkpoint.pop("optimizer", None)
    return new_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-max-len-class", type=int, default=32)
    parser.add_argument("--adjacency", type=Path, default=DEFAULT_ADJACENCY)
    return parser.parse_args()


def main():
    args = parse_args()
    new_checkpoint = expand_checkpoint(args.checkpoint, args.new_max_len_class, args.adjacency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_checkpoint, args.output)
    print(
        f"[expand_max_len_class] wrote {args.output} "
        f"expanded_params={[p['name'] for p in new_checkpoint['max_len_class_expansion']['expanded_params']]}"
    )


if __name__ == "__main__":
    main()
