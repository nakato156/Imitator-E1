#!/usr/bin/env python3
"""Keypoint saliency and region-ablation analysis for closed AR decoders."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import train_video_token_decoder as protocol
from src.mslm.dataloader.data_augmentation import normalize_augment_data, remove_keypoints
from src.mslm.models.video_token_decoder import build_teacher_forcing


REGIONS = (
    ("pose", 0, 7),
    ("face", 7, 71),
    ("left_hand", 71, 91),
    ("right_hand", 91, 111),
)


@dataclass
class SequenceInput:
    sequence_id: str
    clip_ids: tuple[str, ...]
    keypoints: torch.Tensor
    token_ids: torch.Tensor
    glosses: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be ALIAS=PATH")
    alias, raw_path = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", alias):
        raise argparse.ArgumentTypeError(f"invalid checkpoint alias: {alias!r}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return alias, path


def load_sequence_spec(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("sequences") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("sequences JSON must contain a non-empty 'sequences' list")
    result, seen = [], set()
    for row in rows:
        sequence_id = str(row.get("id", "")).strip()
        clip_ids = tuple(str(value) for value in row.get("clip_ids", []))
        if not sequence_id or sequence_id in seen:
            raise ValueError(f"sequence ids must be non-empty and unique: {sequence_id!r}")
        if not clip_ids:
            raise ValueError(f"sequence {sequence_id!r} has no clip_ids")
        seen.add(sequence_id)
        result.append((sequence_id, clip_ids))
    return result


def build_sequences(h5_path: Path, specs, records, tokenizer) -> list[SequenceInput]:
    record_by_id = {str(row["clip_id"]): row for row in records}
    tokens = protocol.label_tokens(records, tokenizer)
    sequences = []
    with h5py.File(h5_path, "r") as h5:
        group = h5["dataset1"]["keypoints"]
        for sequence_id, clip_ids in specs:
            parts, target, glosses = [], [], []
            for clip_id in clip_ids:
                if clip_id not in record_by_id or clip_id not in group:
                    raise KeyError(f"unknown clip_id {clip_id!r} in sequence {sequence_id!r}")
                label = record_by_id[clip_id]["label"]
                keypoints = remove_keypoints(group[clip_id][:])
                keypoints = normalize_augment_data(keypoints, "Original", 111)
                parts.append(torch.as_tensor(keypoints, dtype=torch.float32))
                target.extend(tokens[label])
                glosses.append(label)
            sequences.append(SequenceInput(
                sequence_id=sequence_id,
                clip_ids=clip_ids,
                keypoints=torch.cat(parts),
                token_ids=torch.tensor(target, dtype=torch.long),
                glosses=tuple(glosses),
            ))
    return sequences


def normalize_saliency(value: torch.Tensor) -> torch.Tensor:
    denominator = value.sum()
    return value / denominator if float(denominator) > 0 else torch.zeros_like(value)


def selected_log_probs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits[0], dim=-1).gather(1, ids[:, None]).squeeze(1)


def analyze_sequence(model, sample: SequenceInput, device: torch.device) -> dict:
    keypoints = sample.keypoints.unsqueeze(0).to(device).requires_grad_(True)
    frame_lengths = torch.tensor([sample.keypoints.size(0)], device=device)
    decoder_input, _ = build_teacher_forcing(sample.token_ids.unsqueeze(0))
    decoder_input = model.to_dense(decoder_input.to(device))
    reference_ids = model.to_dense(sample.token_ids.to(device))
    if bool((reference_ids < 0).any()):
        raise ValueError(f"target token absent from checkpoint vocabulary: {sample.sequence_id}")

    logits = model(keypoints, frame_lengths, decoder_input)[:, : reference_ids.numel()]
    log_probs = F.log_softmax(logits[0], dim=-1)
    predicted_ids = logits[0].argmax(dim=-1)
    reference_logp = log_probs.gather(1, reference_ids[:, None]).squeeze(1)
    predicted_logp = log_probs.gather(1, predicted_ids[:, None]).squeeze(1)
    reference_maps, predicted_maps = [], []
    for position in range(reference_ids.numel()):
        reference_grad = torch.autograd.grad(
            reference_logp[position], keypoints, retain_graph=True
        )[0][0]
        predicted_grad = torch.autograd.grad(
            predicted_logp[position], keypoints, retain_graph=True
        )[0][0]
        reference_maps.append((keypoints[0] * reference_grad).abs().sum(dim=-1).detach().cpu())
        predicted_maps.append((keypoints[0] * predicted_grad).abs().sum(dim=-1).detach().cpu())

    reference_raw = torch.stack(reference_maps)
    predicted_raw = torch.stack(predicted_maps)
    reference_norm = torch.stack([normalize_saliency(row) for row in reference_raw])
    predicted_norm = torch.stack([normalize_saliency(row) for row in predicted_raw])

    baseline_ref = reference_logp.detach()
    baseline_pred = predicted_logp.detach()
    ref_ablation, pred_ablation = [], []
    with torch.no_grad():
        for _, start, end in REGIONS:
            ablated = keypoints.detach().clone()
            ablated[:, :, start:end] = 0
            changed = model(ablated, frame_lengths, decoder_input)[:, : reference_ids.numel()]
            ref_changed = selected_log_probs(changed, reference_ids)
            pred_changed = selected_log_probs(changed, predicted_ids)
            ref_ablation.append((baseline_ref - ref_changed).cpu())
            pred_ablation.append((baseline_pred - pred_changed).cpu())

    if model.vocab_map is None:
        predicted_gemma = predicted_ids.detach().cpu().numpy()
    else:
        predicted_gemma = model.dense_to_gemma[predicted_ids].detach().cpu().numpy()
    return {
        "reference_token_ids": sample.token_ids.numpy(),
        "predicted_token_ids": predicted_gemma,
        "reference_logp": reference_logp.detach().cpu().numpy(),
        "predicted_logp": predicted_logp.detach().cpu().numpy(),
        "reference_raw": reference_raw.numpy(),
        "reference_normalized": reference_norm.numpy(),
        "predicted_raw": predicted_raw.numpy(),
        "predicted_normalized": predicted_norm.numpy(),
        "reference_ablation": torch.stack(ref_ablation, dim=1).numpy(),
        "predicted_ablation": torch.stack(pred_ablation, dim=1).numpy(),
    }


def pack_results(samples: list[SequenceInput], results: list[dict]) -> dict[str, np.ndarray]:
    offsets = [0]
    for sample, result in zip(samples, results):
        offsets.append(offsets[-1] + len(result["reference_token_ids"]) * sample.keypoints.size(0))
    packed = {
        "schema_version": np.array(1, dtype=np.int64),
        "sequence_ids": np.asarray([s.sequence_id for s in samples]),
        "clip_ids_json": np.asarray([json.dumps(s.clip_ids) for s in samples]),
        "glosses_json": np.asarray([json.dumps(s.glosses, ensure_ascii=False) for s in samples]),
        "frame_lengths": np.asarray([s.keypoints.size(0) for s in samples], dtype=np.int64),
        "token_lengths": np.asarray([len(r["reference_token_ids"]) for r in results], dtype=np.int64),
        "sequence_offsets": np.asarray(offsets, dtype=np.int64),
        "region_names": np.asarray([r[0] for r in REGIONS]),
        "region_bounds": np.asarray([[r[1], r[2]] for r in REGIONS], dtype=np.int64),
    }
    vector_keys = ("reference_token_ids", "predicted_token_ids", "reference_logp", "predicted_logp")
    matrix_keys = ("reference_ablation", "predicted_ablation")
    map_keys = ("reference_raw", "reference_normalized", "predicted_raw", "predicted_normalized")
    for key in vector_keys + matrix_keys:
        packed[key] = np.concatenate([result[key] for result in results], axis=0)
    for key in map_keys:
        packed[key] = np.concatenate([result[key].reshape(-1, 111) for result in results], axis=0)
    return packed


def unpack_map(packed: dict, sequence_index: int, key: str) -> np.ndarray:
    frames = int(packed["frame_lengths"][sequence_index])
    tokens = int(packed["token_lengths"][sequence_index])
    start, end = packed["sequence_offsets"][sequence_index:sequence_index + 2]
    return packed[key][start:end].reshape(tokens, frames, 111)


def plot_heatmap(data, title, output: Path, xlabel, ylabel, *, center=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = min(20, max(7, data.shape[1] / 12))
    height = min(14, max(4, data.shape[0] / 2))
    fig, ax = plt.subplots(figsize=(width, height))
    limit = float(np.max(np.abs(data))) if center and data.size else None
    image = ax.imshow(
        data, aspect="auto", interpolation="nearest",
        cmap="coolwarm" if center else "magma",
        vmin=-limit if center and limit else None,
        vmax=limit if center and limit else None,
    )
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_checkpoint_heatmaps(alias: str, samples, packed, output_dir: Path):
    heatmaps = output_dir / alias / "heatmaps"
    heatmaps.mkdir(parents=True, exist_ok=True)
    token_cursor = 0
    for index, sample in enumerate(samples):
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample.sequence_id)
        token_count = int(packed["token_lengths"][index])
        for target, prefix in (("reference_normalized", "reference"), ("predicted_normalized", "predicted")):
            saliency = unpack_map(packed, index, target)
            plot_heatmap(saliency.sum(2), f"{alias} · {sample.sequence_id} · {prefix}", heatmaps / f"{safe_id}_{prefix}_token_frame.png", "frame", "token position")
            plot_heatmap(saliency.sum(1), f"{alias} · {sample.sequence_id} · {prefix}", heatmaps / f"{safe_id}_{prefix}_token_keypoint.png", "keypoint", "token position")
            ablation = packed[f"{prefix}_ablation"][token_cursor:token_cursor + token_count]
            plot_heatmap(ablation, f"{alias} · {sample.sequence_id} · {prefix} ablation", heatmaps / f"{safe_id}_{prefix}_region_ablation.png", "body region", "token position", center=True)
        token_cursor += token_count


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def similarity(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    cosine = float(np.dot(left, right) / denominator) if denominator else 0.0
    lr, rr = rankdata(left), rankdata(right)
    spearman = float(np.corrcoef(lr, rr)[0, 1]) if left.size > 1 else 1.0
    return cosine, spearman


def compare_checkpoints(baseline_alias, baseline, alias, current, samples, output_dir):
    for key in ("sequence_ids", "frame_lengths", "token_lengths", "reference_token_ids"):
        if not np.array_equal(baseline[key], current[key]):
            raise RuntimeError(f"checkpoint comparison alignment failed for {key}")
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    token_cursor = 0
    for index, sample in enumerate(samples):
        token_count = int(baseline["token_lengths"][index])
        base_map = unpack_map(baseline, index, "reference_normalized")
        current_map = unpack_map(current, index, "reference_normalized")
        delta = current_map - base_map
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample.sequence_id)
        plot_heatmap(delta.sum(2), f"{alias} - {baseline_alias} · {sample.sequence_id}", comparison_dir / f"{safe_id}_{alias}_minus_{baseline_alias}_token_frame.png", "frame", "token position", center=True)
        plot_heatmap(delta.sum(1), f"{alias} - {baseline_alias} · {sample.sequence_id}", comparison_dir / f"{safe_id}_{alias}_minus_{baseline_alias}_token_keypoint.png", "keypoint", "token position", center=True)
        for position in range(token_count):
            cosine, spearman = similarity(base_map[position].ravel(), current_map[position].ravel())
            rows.append({
                "baseline": baseline_alias, "checkpoint": alias,
                "sequence_id": sample.sequence_id, "token_position": position,
                "reference_token_id": int(baseline["reference_token_ids"][token_cursor + position]),
                "cosine": cosine, "spearman": spearman,
                **{
                    f"ablation_delta_{name}": float(current["reference_ablation"][token_cursor + position, region] - baseline["reference_ablation"][token_cursor + position, region])
                    for region, (name, _, _) in enumerate(REGIONS)
                },
            })
        token_cursor += token_count
    return rows


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    protocol.shared_parser(parser)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True, metavar="ALIAS=PATH")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    aliases = [alias for alias, _ in args.checkpoint]
    if len(set(aliases)) != len(aliases):
        raise ValueError("checkpoint aliases must be unique")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    manifest = protocol.load_manifest(args.manifest)
    spec = protocol.fold_spec(manifest, args.fold)
    records = protocol.list_clip_records(args.h5, "dataset1")
    tokenizer = protocol.load_tokenizer(args.tokenizer)
    sequence_specs = load_sequence_spec(args.sequences)
    sequences = build_sequences(args.h5, sequence_specs, records, tokenizer)
    output_dir = (args.output_dir or ROOT / "interpretability" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    packed_by_alias, checkpoint_manifest = {}, []
    for alias, path in args.checkpoint:
        model, state = protocol.load_closed_decoder(argparse.Namespace(**{**vars(args), "checkpoint": path}), spec, device)
        model.eval()
        results = [analyze_sequence(model, sample, device) for sample in sequences]
        packed = pack_results(sequences, results)
        checkpoint_dir = output_dir / alias
        checkpoint_dir.mkdir()
        np.savez_compressed(checkpoint_dir / "saliency.npz", **packed)
        write_checkpoint_heatmaps(alias, sequences, packed, output_dir)
        packed_by_alias[alias] = packed
        checkpoint_manifest.append({"alias": alias, "path": str(path), "sha256": sha256_file(path), "config": state.get("config", {})})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison_rows = []
    if len(aliases) > 1:
        baseline = packed_by_alias[aliases[0]]
        for alias in aliases[1:]:
            comparison_rows.extend(compare_checkpoints(aliases[0], baseline, alias, packed_by_alias[alias], sequences, output_dir))
        comparison_path = output_dir / "comparison" / "summary.csv"
        with comparison_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
            writer.writeheader()
            writer.writerows(comparison_rows)
        (output_dir / "comparison" / "summary.json").write_text(json.dumps(comparison_rows, indent=2), encoding="utf-8")

    run_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device), "seed": args.seed, "fold": args.fold,
        "outer_test_signer": int(spec["outer_test_signer"]),
        "sequences_file": str(args.sequences.resolve()),
        "sequences": [{"id": row.sequence_id, "clip_ids": list(row.clip_ids)} for row in sequences],
        "checkpoints": checkpoint_manifest,
    }
    (output_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "checkpoints": aliases, "sequences": len(sequences)}, indent=2))


if __name__ == "__main__":
    main()
