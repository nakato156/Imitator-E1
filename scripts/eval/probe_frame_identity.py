#!/usr/bin/env python3
"""Clip-balanced linear probe of per-frame gloss identity for frozen E1 features.

This is a post-hoc diagnostic, not an E1 training stage and not a model ceiling.
It measures a lower bound on gloss identity linearly decodable from the final
frame encoder.  The encoder is always in eval mode and never receives gradients.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from scripts.eval.robustness_video_token_decoder import sha256_file, validate_e1_checkpoint
from scripts.train import train_video_token_decoder as protocol
from src.mslm.dataloader.data_augmentation import normalize_augment_data, remove_keypoints
from src.mslm.dataloader.isolated_keypoint_dataset import list_clip_records


PROBE_SEED = 7_800_027
PROBE_EPOCHS = 100
PROBE_BATCH_CLIPS = 32
PROBE_LR = 1e-2
PROBE_WEIGHT_DECAY = 1e-4


def clip_balanced_moments(clips: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Feature mean/std giving every clip total weight one."""
    if not clips or any(clip.ndim != 2 or clip.size(0) == 0 for clip in clips):
        raise ValueError("expected a non-empty list of non-empty [frames, features] clips")
    clip_means = torch.stack([clip.float().mean(0) for clip in clips])
    clip_second = torch.stack([(clip.float() ** 2).mean(0) for clip in clips])
    mean = clip_means.mean(0)
    variance = (clip_second.mean(0) - mean.square()).clamp_min(0.0)
    return mean, variance.sqrt().clamp_min(1e-6)


def clip_balanced_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, clip_lengths: list[int]
) -> torch.Tensor:
    """Mean of within-clip mean losses, independent of clip duration."""
    if sum(clip_lengths) != logits.size(0) or labels.numel() != logits.size(0):
        raise ValueError("clip lengths do not match frame logits")
    losses = F.cross_entropy(logits, labels, reduction="none")
    return torch.stack(
        [chunk.mean() for chunk in torch.split(losses, clip_lengths)]
    ).mean()


def classification_metrics(
    probe: torch.nn.Module,
    clips: list[torch.Tensor],
    labels: list[int],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> dict:
    """Report frame micro/macro accuracy and clip accuracy from mean logits."""
    frame_correct = frame_total = clip_correct = 0
    per_clip_frame_accuracy = []
    with torch.no_grad():
        for clip, label in zip(clips, labels):
            logits = probe(((clip - mean) / std).to(device))
            predictions = logits.argmax(-1)
            correct = int(predictions.eq(label).sum())
            frame_correct += correct
            frame_total += clip.size(0)
            per_clip_frame_accuracy.append(correct / clip.size(0))
            clip_correct += int(logits.mean(0).argmax().item() == label)
    return {
        "clip_count": len(clips),
        "frame_count": frame_total,
        "frame_accuracy_micro": frame_correct / frame_total,
        "frame_accuracy_macro_by_clip": float(np.mean(per_clip_frame_accuracy)),
        "clip_accuracy_mean_logits": clip_correct / len(clips),
    }


def bootstrap_clip_metrics(
    probe: torch.nn.Module,
    clips: list[torch.Tensor],
    labels: list[int],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    replicates: int = 10_000,
) -> dict:
    """Percentile CIs resampling clips, the independent observational unit."""
    clip_frame_accuracy = []
    clip_accuracy = []
    with torch.no_grad():
        for clip, label in zip(clips, labels):
            logits = probe(((clip - mean) / std).to(device))
            clip_frame_accuracy.append(float(logits.argmax(-1).eq(label).float().mean()))
            clip_accuracy.append(float(logits.mean(0).argmax().item() == label))
    values = {
        "frame_accuracy_macro_by_clip": np.asarray(clip_frame_accuracy),
        "clip_accuracy_mean_logits": np.asarray(clip_accuracy),
    }
    rng = np.random.default_rng(PROBE_SEED)
    indices = rng.integers(0, len(clips), size=(replicates, len(clips)))
    result = {}
    for name, array in values.items():
        estimates = array[indices].mean(axis=1)
        lower, upper = np.quantile(estimates, [0.025, 0.975])
        result[name] = {
            "estimate": float(array.mean()),
            "lower": float(lower),
            "upper": float(upper),
            "confidence": 0.95,
            "unit": "isolated clip",
            "replicates": replicates,
            "seed": PROBE_SEED,
        }
    return result


def preprocess_clip(h5_file, clip_id: str) -> torch.Tensor:
    array = h5_file["dataset1"]["keypoints"][str(clip_id)][:]
    array = remove_keypoints(array)
    array = normalize_augment_data(array, "Original", 111)
    return torch.as_tensor(array, dtype=torch.float32)


@torch.no_grad()
def encode_records(model, records: list[dict], h5_path: Path, device, batch_size: int = 16):
    features: list[torch.Tensor] = []
    model.eval()
    with h5py.File(h5_path, "r") as h5_file:
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            keypoints = [preprocess_clip(h5_file, row["clip_id"]) for row in chunk]
            lengths = torch.tensor([clip.size(0) for clip in keypoints], dtype=torch.long)
            padded = torch.nn.utils.rnn.pad_sequence(keypoints, batch_first=True).to(device)
            encoded = model.frame_encoder(padded, lengths.to(device))
            features.extend(
                encoded[index, :length].cpu()
                for index, length in enumerate(lengths.tolist())
            )
    return features


def fit_probe(
    clips: list[torch.Tensor],
    labels: list[int],
    feature_dim: int,
    class_count: int,
    device: torch.device,
):
    torch.manual_seed(PROBE_SEED)
    random.seed(PROBE_SEED)
    np.random.seed(PROBE_SEED)
    probe = torch.nn.Linear(feature_dim, class_count).to(device)
    optimizer = torch.optim.Adam(
        probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY
    )
    mean, std = clip_balanced_moments(clips)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PROBE_SEED)
    final_loss = None
    for epoch in range(PROBE_EPOCHS):
        permutation = torch.randperm(len(clips), generator=generator).tolist()
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, len(permutation), PROBE_BATCH_CLIPS):
            indices = permutation[start : start + PROBE_BATCH_CLIPS]
            batch_clips = [((clips[index] - mean) / std).to(device) for index in indices]
            lengths = [clip.size(0) for clip in batch_clips]
            frame_labels = torch.cat(
                [
                    torch.full((length,), labels[index], dtype=torch.long, device=device)
                    for index, length in zip(indices, lengths)
                ]
            )
            optimizer.zero_grad(set_to_none=True)
            logits = probe(torch.cat(batch_clips))
            loss = clip_balanced_cross_entropy(logits, frame_labels, lengths)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batch_count += 1
        final_loss = epoch_loss / batch_count
        if epoch in {0, 24, 49, 74, 99}:
            print(f"probe epoch={epoch:03d} clip_balanced_loss={final_loss:.6f}", flush=True)
    return probe, mean, std, final_loss


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    protocol.shared_parser(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-checkpoint", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = protocol.load_manifest(args.manifest)
    spec = protocol.fold_spec(manifest, args.fold)
    model, state = protocol.load_closed_decoder(args, spec, device)
    validate_e1_checkpoint(state, args.fold, args.seed)

    records = list_clip_records(args.h5, "dataset1")
    labels = sorted({row["label"] for row in records})
    label_to_index = {label: index for index, label in enumerate(labels)}
    train_signers = set(map(int, spec["train_signers"]))
    outer_signer = int(spec["outer_test_signer"])
    train_records = [row for row in records if int(row["signer_id"]) in train_signers]
    outer_records = [row for row in records if int(row["signer_id"]) == outer_signer]
    if {int(row["signer_id"]) for row in train_records} & {outer_signer}:
        raise RuntimeError("outer signer leaked into probe training records")
    if {row["label"] for row in train_records} != set(labels):
        raise RuntimeError("probe training records do not contain every gloss")

    train_clips = encode_records(model, train_records, args.h5, device)
    outer_clips = encode_records(model, outer_records, args.h5, device)
    train_labels = [label_to_index[row["label"]] for row in train_records]
    outer_labels = [label_to_index[row["label"]] for row in outer_records]
    probe, mean, std, final_loss = fit_probe(
        train_clips, train_labels, train_clips[0].size(1), len(labels), device
    )

    train_metrics = classification_metrics(
        probe, train_clips, train_labels, mean, std, device
    )
    outer_metrics = classification_metrics(
        probe, outer_clips, outer_labels, mean, std, device
    )
    result = {
        "schema_version": 1,
        "interpretation": "lower bound on linearly decodable frame-level gloss identity; not a ceiling",
        "fold": args.fold,
        "outer_test_signer": outer_signer,
        "inner_val_signer_excluded": int(spec["inner_val_signer"]),
        "train_signers": sorted(train_signers),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "feature_source": "final E1 STGCNTemporalFrameEncoder output; encoder eval mode, no gradients",
        "protocol": {
            "seed": PROBE_SEED,
            "epochs_fixed_without_selection": PROBE_EPOCHS,
            "batch_clips": PROBE_BATCH_CLIPS,
            "learning_rate": PROBE_LR,
            "weight_decay": PROBE_WEIGHT_DECAY,
            "training_objective": "mean across clips of mean frame cross-entropy",
            "standardization": "train-only moments with equal total weight per clip",
            "clip_prediction": "argmax of mean frame logits",
        },
        "final_train_objective": final_loss,
        "train": train_metrics,
        "outer": outer_metrics,
        "outer_bootstrap_95_ci": bootstrap_clip_metrics(
            probe, outer_clips, outer_labels, mean, std, device
        ),
    }
    output = args.output or args.checkpoint.with_name("frame_identity_probe.json")
    probe_checkpoint = args.probe_checkpoint or args.checkpoint.with_name(
        "frame_identity_probe.pt"
    )
    torch.save(
        {
            "probe": probe.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "label_to_index": label_to_index,
            "protocol": result["protocol"],
            "source_checkpoint_sha256": result["checkpoint_sha256"],
        },
        probe_checkpoint,
    )
    result["probe_checkpoint"] = str(probe_checkpoint)
    result["probe_checkpoint_sha256"] = sha256_file(probe_checkpoint)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "outer": outer_metrics}, indent=2))


if __name__ == "__main__":
    main()
