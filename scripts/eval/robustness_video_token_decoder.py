#!/usr/bin/env python3
"""Pre-registered outer-test robustness battery for Imitator E1.

The script evaluates one closed checkpoint on the same deterministic 896
synthetic sequences under clean input, a gold-boundary segment permutation,
Gaussian keypoint jitter, frame deletion, and temporal speed changes.  Every
stochastic operation is seeded per sample so results do not depend on batch
size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F

from scripts.train import train_video_token_decoder as protocol
from src.mslm.dataloader.isolated_keypoint_dataset import list_clip_records
from src.mslm.dataloader.synthetic_temporal import permute_video_segments
from src.mslm.utils.sequence_metrics import gloss_sequence_diagnostics


PERMUTATION_SEED = 7_800_023
JITTER_SEED = 7_800_024
DROPOUT_SEED = 7_800_025
BOOTSTRAP_SEED = 7_800_026
BOOTSTRAP_REPLICATES = 10_000
JITTER_SIGMAS = (0.01, 0.02, 0.05)
FRAME_DROPOUT_RATES = (0.1, 0.2)
TEMPORAL_SPEEDS = (0.75, 1.25)

BatchTransform = Callable[[dict, int], tuple[torch.Tensor, torch.Tensor]]


def gaussian_jitter(sequence: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """Add normalized-coordinate Gaussian noise with a sample-local seed."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0:
        return sequence.clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(sequence.shape, generator=generator, dtype=sequence.dtype)
    return sequence + float(sigma) * noise.to(sequence.device)


def drop_frames(sequence: torch.Tensor, rate: float, seed: int) -> torch.Tensor:
    """Delete frames independently, retaining at least one frame."""
    if not 0 <= rate < 1:
        raise ValueError("frame dropout rate must satisfy 0 <= rate < 1")
    if rate == 0 or sequence.size(0) <= 1:
        return sequence.clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    keep = torch.rand(sequence.size(0), generator=generator) >= float(rate)
    if not bool(keep.any()):
        keep[torch.randint(sequence.size(0), (1,), generator=generator)] = True
    return sequence[keep.to(sequence.device)]


def temporal_resample(sequence: torch.Tensor, speed: float) -> torch.Tensor:
    """Resample at playback ``speed``; 0.75x lengthens and 1.25x shortens."""
    if speed <= 0:
        raise ValueError("temporal speed must be positive")
    if speed == 1 or sequence.size(0) <= 1:
        return sequence.clone()
    output_length = max(1, round(sequence.size(0) / float(speed)))
    flat = sequence.reshape(sequence.size(0), -1).transpose(0, 1).unsqueeze(0)
    resized = F.interpolate(flat, size=output_length, mode="linear", align_corners=False)
    return resized.squeeze(0).transpose(0, 1).reshape(output_length, *sequence.shape[1:])


def repad(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([sequence.size(0) for sequence in sequences], dtype=torch.long)
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True), lengths


def identity_transform(batch: dict, _: int) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["keypoints"].clone(), batch["frame_lengths"].clone()


def permutation_transform(batch: dict, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    transformed = batch["keypoints"].clone()
    for index, length in enumerate(batch["frame_lengths"].tolist()):
        transformed[index] = permute_video_segments(
            transformed[index],
            batch["boundaries"][index],
            length,
            random.Random(PERMUTATION_SEED + offset + index),
            require_change=True,
        )
    return transformed, batch["frame_lengths"].clone()


def jitter_transform(sigma: float) -> BatchTransform:
    def apply(batch: dict, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = [
            gaussian_jitter(
                batch["keypoints"][index, :length], sigma, JITTER_SEED + offset + index
            )
            for index, length in enumerate(batch["frame_lengths"].tolist())
        ]
        return repad(sequences)

    return apply


def dropout_transform(rate: float) -> BatchTransform:
    def apply(batch: dict, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = [
            drop_frames(
                batch["keypoints"][index, :length], rate, DROPOUT_SEED + offset + index
            )
            for index, length in enumerate(batch["frame_lengths"].tolist())
        ]
        return repad(sequences)

    return apply


def speed_transform(speed: float) -> BatchTransform:
    def apply(batch: dict, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = [
            temporal_resample(batch["keypoints"][index, :length], speed)
            for index, length in enumerate(batch["frame_lengths"].tolist())
        ]
        return repad(sequences)

    return apply


@torch.no_grad()
def evaluate_condition(model, loader, device, tokenizer, transform: BatchTransform):
    model.eval()
    predictions: list[list[int]] = []
    eos_flags: list[bool] = []
    targets: list[list[int]] = []
    metadata: list[dict] = []
    offset = 0
    for batch in loader:
        keypoints, transformed_lengths = transform(batch, offset)
        decoded = model.greedy_decode(keypoints.to(device), transformed_lengths.to(device))
        predictions.extend(decoded.token_ids)
        eos_flags.extend(decoded.emitted_eos.cpu().tolist())
        target_rows = [row[row.ne(-100)].tolist() for row in batch["token_ids"]]
        targets.extend(target_rows)
        for index, target in enumerate(target_rows):
            metadata.append(
                {
                    "sample_index": offset + index,
                    "sign_count": int(batch["sign_counts"][index]),
                    "frame_length": int(batch["frame_lengths"][index]),
                    "transformed_frame_length": int(transformed_lengths[index]),
                    "target_token_length": len(target),
                }
            )
        offset += len(target_rows)

    metrics, rows = protocol.sequence_metrics(predictions, eos_flags, targets, tokenizer)
    for row, fields in zip(rows, metadata):
        row.update(fields)
    return metrics, rows


def bootstrap_mean_ci(
    values: list[float] | np.ndarray,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("bootstrap requires at least one value")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    cursor = 0
    while cursor < replicates:
        count = min(512, replicates - cursor)
        indices = rng.integers(0, array.size, size=(count, array.size))
        means[cursor : cursor + count] = array[indices].mean(axis=1)
        cursor += count
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "estimate": float(array.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": 0.95,
        "method": "sample-level nonparametric percentile bootstrap",
        "replicates": int(replicates),
        "seed": int(seed),
    }


def metric_cis(rows: list[dict], seed: int) -> dict:
    return {
        metric: bootstrap_mean_ci([row[metric] for row in rows], seed=seed + index)
        for index, metric in enumerate(("strict_exact", "token_edit_similarity"))
    }


def paired_delta_cis(clean_rows: list[dict], perturbed_rows: list[dict], seed: int) -> dict:
    if len(clean_rows) != len(perturbed_rows):
        raise ValueError("paired conditions contain different sample counts")
    result = {}
    for index, metric in enumerate(("strict_exact", "token_edit_similarity")):
        deltas = [
            float(clean[metric]) - float(perturbed[metric])
            for clean, perturbed in zip(clean_rows, perturbed_rows)
        ]
        result[metric] = bootstrap_mean_ci(deltas, seed=seed + index)
    return result


def summarize_indices(rows: list[dict], indices: list[int]) -> dict:
    if not indices:
        return {"n": 0, "strict_exact": None, "token_edit_similarity": None}
    return {
        "n": len(indices),
        "strict_exact": float(np.mean([rows[i]["strict_exact"] for i in indices])),
        "token_edit_similarity": float(
            np.mean([rows[i]["token_edit_similarity"] for i in indices])
        ),
    }


def quantile_strata(rows: list[dict], field: str) -> dict:
    values = np.asarray([row[field] for row in rows])
    edges = np.quantile(values, [0.25, 0.5, 0.75], method="nearest")
    groups = np.searchsorted(edges, values, side="right")
    return {
        "field": field,
        "edge_values": [int(value) for value in edges],
        "groups": {
            f"Q{quartile + 1}": summarize_indices(
                rows, np.flatnonzero(groups == quartile).tolist()
            )
            for quartile in range(4)
        },
    }


def stratified_metrics(rows: list[dict]) -> dict:
    return {
        "by_sign_count": {
            str(sign_count): summarize_indices(
                rows,
                [i for i, row in enumerate(rows) if row["sign_count"] == sign_count],
            )
            for sign_count in range(2, 9)
        },
        "by_frame_length_quartile": quantile_strata(rows, "frame_length"),
        "by_target_token_length_quartile": quantile_strata(rows, "target_token_length"),
    }


def two_sided_sign_test(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * math.ldexp(float(tail), -discordant))


def paired_cif_result(clean_rows: list[dict], comparator: dict, fold: int) -> dict:
    if int(comparator["fold"]) != fold:
        raise RuntimeError("CIF comparator fold mismatch")
    cif_rows = comparator["outer_test"]["predictions"]
    if len(cif_rows) != len(clean_rows):
        raise RuntimeError("CIF and E1 sample counts differ")
    differences = []
    for e1, cif in zip(clean_rows, cif_rows):
        if list(e1["target"]) != list(cif["target"]):
            raise RuntimeError("CIF and E1 targets are not row-aligned")
        differences.append(int(e1["strict_exact"]) - int(cif["strict_exact"]))
    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    ties = sum(value == 0 for value in differences)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "discordant": wins + losses,
        "strict_exact_delta": sum(differences) / len(differences),
        "two_sided_sign_test_p": two_sided_sign_test(wins, losses),
    }


def validate_e1_checkpoint(state: dict, fold: int, seed: int) -> None:
    config = state.get("config", {})
    expected = {
        "epochs": 30,
        "seed": seed,
        "encoder_pe": True,
        "ctc_weight": 0.0,
        "label_smoothing": 0.0,
        "unfreeze_stgcn_epoch": None,
        "select": "edit",
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if not config.get("vocab_map"):
        mismatches["vocab_map"] = {"expected": "restricted", "observed": None}
    if int(state.get("provenance", {}).get("fold", -1)) != fold:
        mismatches["fold"] = {
            "expected": fold,
            "observed": state.get("provenance", {}).get("fold"),
        }
    if mismatches:
        raise RuntimeError(f"checkpoint is not frozen E1: {mismatches}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def condition_transforms(selected: list[str] | None = None) -> dict[str, BatchTransform]:
    transforms: dict[str, BatchTransform] = {
        "clean": identity_transform,
        "segment_permutation": permutation_transform,
    }
    transforms.update({f"jitter_sigma_{sigma:.2f}": jitter_transform(sigma) for sigma in JITTER_SIGMAS})
    transforms.update(
        {f"frame_dropout_p_{rate:.1f}": dropout_transform(rate) for rate in FRAME_DROPOUT_RATES}
    )
    transforms.update(
        {f"temporal_speed_{speed:.2f}x": speed_transform(speed) for speed in TEMPORAL_SPEEDS}
    )
    if selected is None:
        return transforms
    unknown = [name for name in selected if name not in transforms]
    if unknown:
        raise ValueError(f"unknown robustness condition(s): {unknown}")
    required = {"clean", "segment_permutation"}
    if not required.issubset(selected):
        raise ValueError(
            "H1-H3 evaluation requires both clean and segment_permutation conditions"
        )
    return {name: transforms[name] for name in selected}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    protocol.shared_parser(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cif-comparator", type=Path, required=True)
    parser.add_argument("--eval-samples", type=int, default=896)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="condition names to run; default runs the full secondary battery",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = protocol.load_manifest(args.manifest)
    spec = protocol.fold_spec(manifest, args.fold)
    model, state = protocol.load_closed_decoder(args, spec, device)
    validate_e1_checkpoint(state, args.fold, args.seed)

    records = list_clip_records(args.h5, "dataset1")
    tokenizer = protocol.load_tokenizer(args.tokenizer)
    tokens = protocol.label_tokens(records, tokenizer)
    outer_rows = protocol.rows_for_signers(records, [spec["outer_test_signer"]])
    dataset = protocol.make_dataset(
        args,
        outer_rows,
        tokens,
        samples=args.eval_samples,
        seed=args.seed + 200_000,
    )

    conditions = {}
    clean_rows = None
    transforms = condition_transforms(args.conditions)
    for condition_index, (name, transform) in enumerate(transforms.items()):
        loader = protocol.make_loader(dataset, args.eval_batch_size, False)
        metrics, rows = evaluate_condition(model, loader, device, tokenizer, transform)
        if clean_rows is None:
            clean_rows = rows
        conditions[name] = {
            "metrics": metrics,
            "bootstrap_95_ci": metric_cis(rows, BOOTSTRAP_SEED + condition_index * 100),
            "gloss_diagnostics": gloss_sequence_diagnostics(
                [row["pred"] for row in rows], [row["target"] for row in rows], tokens
            ),
            "stratified": stratified_metrics(rows),
            "predictions": rows,
        }
        if name != "clean":
            conditions[name]["paired_clean_minus_condition_bootstrap_95_ci"] = paired_delta_cis(
                clean_rows, rows, BOOTSTRAP_SEED + condition_index * 100 + 20
            )

    assert clean_rows is not None
    comparator = json.loads(args.cif_comparator.read_text(encoding="utf-8"))
    paired_cif = paired_cif_result(clean_rows, comparator, args.fold)
    clean_exact = conditions["clean"]["metrics"]["strict_exact"]
    clean_order = conditions["clean"]["gloss_diagnostics"]["pairwise_order_accuracy"]
    permutation_delta = conditions["segment_permutation"][
        "paired_clean_minus_condition_bootstrap_95_ci"
    ]["token_edit_similarity"]["estimate"]
    confirmatory = args.fold in (7, 8)
    decisions = {
        "scope": "confirmatory" if confirmatory else "development validation only",
        "H1": {
            "pass": bool(
                paired_cif["wins"] > paired_cif["losses"]
                and paired_cif["two_sided_sign_test_p"] < 0.01
            ),
            "criterion": "wins > losses and two-sided sign-test p < 0.01",
        },
        "H2": {
            "pass": bool(clean_exact >= 0.119),
            "criterion": "clean strict_exact >= 0.119 in this fold",
        },
        "H3": {
            "pass": bool(clean_order is not None and clean_order >= 0.9 and permutation_delta >= 0.15),
            "criterion": "pairwise gloss order >= 0.9 and paired absolute edit_sim drop >= 0.15",
        },
    }
    result = {
        "schema_version": 1,
        "fold": args.fold,
        "outer_test_signer": int(spec["outer_test_signer"]),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "cif_comparator": str(args.cif_comparator),
        "cif_comparator_sha256": sha256_file(args.cif_comparator),
        "protocol": {
            "eval_samples": args.eval_samples,
            "dataset_seed": args.seed + 200_000,
            "permutation_seed": PERMUTATION_SEED,
            "jitter_seed": JITTER_SEED,
            "dropout_seed": DROPOUT_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "permutation_requires_non_identity": True,
            "temporal_factor_semantics": "playback speed; output frames = round(input frames / speed)",
            "evaluated_conditions": list(transforms),
            "secondary_battery_complete": args.conditions is None,
        },
        "paired_vs_affine_cif": paired_cif,
        "conditions": conditions,
        "pre_registered_decisions_for_this_fold": decisions,
        "checkpoint_config": state.get("config", {}),
        "checkpoint_provenance": state.get("provenance", {}),
    }
    output = args.output or args.checkpoint.with_name("outer_test_robustness.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
