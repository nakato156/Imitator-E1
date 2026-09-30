"""A2 CIF length/count calibration analysis for the dataset1 token imitator."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", ".hf_cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from settings import initialize
from scripts.train.train_temporal_v126 import (
    IsolatedTokenEvalDataset,
    align_time_to_targets,
    evaluate_alpha_mode,
    forward_from_cif,
    isolated_eval_collate,
    load_embedding_rows,
    make_label_tokens,
    move_batch,
    split_records,
)
from src.mslm.dataloader.isolated_keypoint_dataset import list_clip_records
from src.mslm.models.temporal_sign_prompt import (
    CIFAggregator,
    STGCNTemporalFrameEncoder,
    TemporalSignPromptModel,
    length_mask_from_lengths,
    load_visual_low_level_weights,
    rescale_alphas_to_predicted_lengths,
    rescale_alphas_to_rounded_count,
    rescale_alphas_to_target_lengths,
)


DEFAULT_TOKENIZER = (
    ".hf_cache/hub/"
    "models--unsloth--gemma-3n-E2B-it-unsloth-bnb-4bit/"
    "snapshots/3d26ffdd2276698f582562bf01511aa625bc6f30"
)
DEFAULT_H5 = "data/processed/dataset1_isolated_v122.hdf5"
DEFAULT_EMBEDDING_TABLE = "data/processed/gemma3n_embed_table.pt"
DEFAULT_V121 = "../outputs/checkpoints/121/23/best_top1/checkpoint.pth"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h5", type=Path, default=Path(DEFAULT_H5))
    parser.add_argument("--tokenizer", type=Path, default=Path(DEFAULT_TOKENIZER))
    parser.add_argument("--embedding-table", type=Path, default=Path(DEFAULT_EMBEDDING_TABLE))
    parser.add_argument("--checkpoint-v121", type=Path, default=Path(DEFAULT_V121))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--heldout-signer",
        type=int,
        help="Score only this signer_id (must match the value the checkpoint was trained with via --heldout-signer).",
    )
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--token-head", choices=["linear", "contextual"], default=None)
    parser.add_argument("--length-head", choices=["mean", "attention"], default=None)
    parser.add_argument(
        "--rounded-count-bias",
        type=float,
        default=0.0,
        help=(
            "Additive bias for the pred_rescaled_to_rounded_count mode "
            "(count = round(alpha.sum() + bias)). Fit on train signers only."
        ),
    )
    return parser.parse_args()


def bucket_len(length: int) -> str:
    if length <= 1:
        return "1"
    if length == 2:
        return "2"
    return "3+"


LEGACY_ARCH_CONFIG = {"token_head": "contextual", "length_head": "attention"}


def resolve_arch_config(checkpoint_state: dict, cli_token_head, cli_length_head) -> dict:
    """Pick token_head/length_head: explicit CLI > checkpoint metadata > legacy default."""
    base = dict(checkpoint_state.get("arch_config", LEGACY_ARCH_CONFIG))
    if cli_token_head is not None:
        base["token_head"] = cli_token_head
    if cli_length_head is not None:
        base["length_head"] = cli_length_head
    return base


def effective_split_seed(seed: int, split_seed) -> int:
    return split_seed if split_seed is not None else seed


def add_bucket(row, key, value):
    row[key] += float(value)
    row[f"{key}_n"] += 1


def finalize_bucket(row):
    out = {"samples": int(row["samples"])}
    for key, value in sorted(row.items()):
        if key == "samples" or key.endswith("_n"):
            continue
        n = max(1.0, row[f"{key}_n"])
        out[key] = value / n
    return out


def collect_mode_rows(model, loader, device, mode, rounded_count_bias=0.0):
    rows = []
    hist_quantity = Counter()
    hist_count = Counter()
    by_len = defaultdict(lambda: defaultdict(float))
    by_gloss = defaultdict(lambda: defaultdict(float))
    quantity_close_count_wrong = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            alpha_target = CIFAggregator.boundary_targets(
                batch["boundaries"],
                batch["keypoints"].size(1),
                token_spans=batch["token_spans"],
            )
            frame_features = model.frame_encoder(batch["keypoints"], batch["frame_lengths"])
            alpha_logits = model.cif.predict_alpha_logits(frame_features, batch["frame_lengths"])
            alpha_pred = torch.sigmoid(alpha_logits).masked_fill(
                ~length_mask_from_lengths(batch["frame_lengths"], alpha_logits.size(1)),
                0.0,
            )
            if mode == "teacher_alpha":
                alphas = alpha_target
            elif mode == "pred_rescaled_to_target_len":
                alphas = rescale_alphas_to_target_lengths(alpha_pred, batch["token_lengths"])
            elif mode == "pred_rescaled_to_pred_len":
                pred_lengths = model.predict_length_logits(
                    frame_features, batch["frame_lengths"]
                ).argmax(dim=-1)
                alphas, _ = rescale_alphas_to_predicted_lengths(
                    alpha_pred, pred_lengths, max_len=model.max_len_class
                )
            elif mode == "pred_rescaled_to_rounded_count":
                alphas, _ = rescale_alphas_to_rounded_count(
                    alpha_pred,
                    max_len=model.max_len_class,
                    bias=rounded_count_bias,
                )
            elif mode == "pred_raw":
                alphas = alpha_pred
            else:
                raise ValueError(mode)

            out = forward_from_cif(model, frame_features, batch["frame_lengths"], alphas=alphas)
            logits = align_time_to_targets(out["token_logits"], batch["token_ids"].size(1))
            pred = logits.argmax(dim=-1)
            for i, target_len in enumerate(batch["token_lengths"].detach().cpu().tolist()):
                target = batch["token_ids"][i, :target_len]
                pred_ids = pred[i, :target_len]
                token_acc = pred_ids.eq(target).float().mean().item()
                exact = int(torch.equal(pred_ids.cpu(), target.cpu()))
                count = int(out["cif"].counts[i].item())
                quantity_delta = float(out["cif"].quantity[i].item() - target_len)
                count_delta = count - target_len
                hist_quantity[round(quantity_delta, 1)] += 1
                hist_count[count_delta] += 1

                label = bucket_len(target_len)
                by_len[label]["samples"] += 1
                add_bucket(by_len[label], "token_top1", token_acc)
                add_bucket(by_len[label], "exact", exact)
                add_bucket(by_len[label], "quantity_abs_error", abs(quantity_delta))
                add_bucket(by_len[label], "count_abs_error", abs(count_delta))
                add_bucket(by_len[label], "count_match", int(count_delta == 0))
                if count_delta == 0:
                    add_bucket(by_len[label], "token_accuracy_when_count_correct", token_acc)
                else:
                    add_bucket(by_len[label], "token_accuracy_when_count_wrong", token_acc)

                gloss = batch["glosses"][i][0]
                by_gloss[gloss]["samples"] += 1
                add_bucket(by_gloss[gloss], "exact", exact)
                add_bucket(by_gloss[gloss], "token_accuracy", token_acc)
                add_bucket(by_gloss[gloss], "count_match", int(count_delta == 0))

                row = {
                    "clip_id": batch["clip_ids"][i][0],
                    "gloss": batch["glosses"][i][0],
                    "target_len": target_len,
                    "quantity": out["cif"].quantity[i].item(),
                    "pred_count": count,
                    "quantity_delta": quantity_delta,
                    "count_delta": count_delta,
                    "token_accuracy": token_acc,
                    "exact": exact,
                }
                rows.append(row)
                if abs(quantity_delta) <= 0.25 and count_delta != 0:
                    quantity_close_count_wrong.append(row)

    return {
        "by_target_length": {
            key: finalize_bucket(value) for key, value in sorted(by_len.items())
        },
        "by_gloss": {
            key: finalize_bucket(value) for key, value in sorted(by_gloss.items())
        },
        "hist_quantity_minus_target_len": dict(sorted(hist_quantity.items())),
        "hist_pred_count_minus_target_len": dict(sorted(hist_count.items())),
        "quantity_close_but_count_wrong_examples": quantity_close_count_wrong[:25],
        "worst_examples": sorted(
            rows,
            key=lambda row: (
                -abs(row["count_delta"]),
                row["token_accuracy"],
                row["gloss"],
            ),
        )[:25],
    }


def main():
    args = parse_args()
    initialize(seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = list_clip_records(args.h5, "dataset1")
    split_seed = effective_split_seed(args.seed, args.split_seed)
    _, val_records = split_records(records, split_seed, args.heldout_signer)
    if args.max_samples:
        val_records = val_records[: args.max_samples]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    token_ids_by_label = make_label_tokens(records, tokenizer)
    all_token_ids = {tid for ids in token_ids_by_label.values() for tid in ids}
    embedding_rows = load_embedding_rows(args.embedding_table, all_token_ids, 2048)
    loader = DataLoader(
        IsolatedTokenEvalDataset(args.h5, val_records, token_ids_by_label, embedding_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=isolated_eval_collate,
        pin_memory=torch.cuda.is_available(),
    )

    state = torch.load(args.checkpoint, map_location=device)
    arch_config = resolve_arch_config(state, args.token_head, args.length_head)
    A = np.load("data/processed/adjacency_matrix.npy", allow_pickle=True)
    encoder = STGCNTemporalFrameEncoder(A, hidden_size=128)
    load_info = load_visual_low_level_weights(encoder, args.checkpoint_v121)
    model = TemporalSignPromptModel(
        encoder,
        hidden_size=128,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=2048,
        token_head_variant=arch_config["token_head"],
        length_head_variant=arch_config["length_head"],
    ).to(device)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)

    summaries = {}
    cuts = {}
    for mode in (
        "teacher_alpha",
        "pred_rescaled_to_target_len",
        "pred_rescaled_to_pred_len",
        "pred_rescaled_to_rounded_count",
        "pred_raw",
    ):
        summaries[mode] = evaluate_alpha_mode(
            model, loader, device, mode, rounded_count_bias=args.rounded_count_bias
        )
        cuts[mode] = collect_mode_rows(
            model, loader, device, mode, rounded_count_bias=args.rounded_count_bias
        )

    result = {
        "checkpoint": str(args.checkpoint),
        "heldout_signer": args.heldout_signer,
        "samples": len(val_records),
        "device": device,
        "load_info": load_info,
        "arch_config": arch_config,
        "state_load": {"missing": missing, "unexpected": unexpected},
        "rounded_count_bias": args.rounded_count_bias,
        "mode_summaries": summaries,
        "mode_cuts": cuts,
        "interpretation": (
            "If pred_rescaled_to_target_len stays near teacher_alpha while pred_raw drops, "
            "the deployable bottleneck is length/count calibration or CIF discretization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md = args.output.with_suffix(".md")
    lines = ["# A2 Imitator Length/Count Analysis", ""]
    for mode, summary in summaries.items():
        lines.append(f"## {mode}")
        keys = [
            "top1",
            "top5",
            "exact",
            "quantity_mae",
            "count_mae",
            "count_match_rate",
            "token_accuracy_when_count_correct",
            "token_accuracy_when_count_wrong",
            "boundary_mae_when_count_correct",
        ]
        for key in keys:
            if key in summary:
                lines.append(f"- {key}: {summary[key]:.4f}")
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"wrote": str(args.output), "markdown": str(md)}, indent=2))


if __name__ == "__main__":
    main()
