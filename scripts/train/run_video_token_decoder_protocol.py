#!/usr/bin/env python3
"""Execute the fixed 24-hour video-token-decoder protocol for folds 1-6."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/train/train_video_token_decoder.py"
OUTPUT_ROOT = ROOT.parent / "outputs/video_token_decoder"
FREEZE_FILES = (
    ROOT / "src/mslm/models/video_token_decoder.py",
    ROOT / "src/mslm/dataloader/synthetic_temporal.py",
    RUNNER,
)


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in FREEZE_FILES:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def call(*args: str) -> None:
    subprocess.run([sys.executable, str(RUNNER), *args], cwd=ROOT, check=True)


def paths(output_root: Path, fold: int, seed: int, rescue: bool):
    variant = "rescue" if rescue else "base"
    directory = output_root / f"fold{fold}_seed{seed}" / variant
    return directory / "checkpoint_closed.pt", directory / "outer_test.json"


def ensure_cif(output_root: Path, fold: int, common: list[str]) -> Path:
    output = output_root / f"fold{fold}_cif_comparator.json"
    if not output.exists():
        call("cif-comparator", "--fold", str(fold), "--output", str(output), *common)
    return output


def ensure_decoder(
    output_root: Path, fold: int, seed: int, rescue: bool, common: list[str]
) -> dict:
    checkpoint, evaluation = paths(output_root, fold, seed, rescue)
    train_args = ["train", "--fold", str(fold), "--seed", str(seed), *common]
    if rescue:
        train_args.append("--rescue-augmentation")
    if not checkpoint.exists():
        call(*train_args)
    cif = ensure_cif(output_root, fold, common)
    if not evaluation.exists():
        call(
            "evaluate", "--fold", str(fold), "--seed", str(seed),
            "--checkpoint", str(checkpoint), "--cif-comparator", str(cif),
            "--output", str(evaluation), *common,
        )
    return json.loads(evaluation.read_text(encoding="utf-8"))


def exact_scores(results: list[dict]) -> list[float]:
    return [float(row["metrics"]["strict_exact"]) for row in results]


def classify_confirmation(results: list[dict]) -> dict:
    scores = exact_scores(results)
    deltas = [float(row["paired_vs_affine_cif"]["strict_exact_delta"]) for row in results]
    mean_exact = sum(scores) / len(scores)
    improved_folds = sum(delta >= 0.05 for delta in deltas)
    if mean_exact >= 0.40 and improved_folds >= 2:
        outcome = "SUCCESS"
    elif 0.25 <= mean_exact < 0.40 or any(delta > 0 for delta in deltas):
        outcome = "PARTIAL"
    else:
        outcome = "FAILURE"
    return {
        "outcome": outcome,
        "mean_strict_exact": mean_exact,
        "strict_exact_by_fold": scores,
        "affine_cif_delta_by_fold": deltas,
        "folds_improved_by_5pp": improved_folds,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--remaining-hours", type=float, default=0.0)
    parser.add_argument("--h5", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--adjacency", type=Path)
    parser.add_argument("--source-root", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    common = ["--output-root", str(args.output_root)]
    for name in ("h5", "tokenizer", "manifest", "adjacency", "source_root"):
        value = getattr(args, name)
        if value is not None:
            common.extend([f"--{name.replace('_', '-')}", str(value)])

    # The clean CIF baseline is computed for exactly the six allowed folds.
    for fold in range(1, 7):
        ensure_cif(args.output_root, fold, common)

    base = [ensure_decoder(args.output_root, fold, 23, False, common) for fold in range(1, 4)]
    base_scores = exact_scores(base)
    base_mean = sum(base_scores) / 3
    if base_mean < 0.20 or min(base_scores) < 0.20:
        report = {"outcome": "FAILURE", "stage": "development", "scores": base_scores}
        (args.output_root / "protocol_result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 1

    adopted_rescue = False
    rescue_scores = None
    if base_mean < 0.35:
        rescue = [ensure_decoder(args.output_root, fold, 23, True, common) for fold in range(1, 4)]
        rescue_scores = exact_scores(rescue)
        adopted_rescue = (
            sum(rescue_scores) / 3 >= base_mean + 0.03
            and all(new >= old - 0.05 for new, old in zip(rescue_scores, base_scores))
        )

    freeze = {
        "source_sha256": source_hash(),
        "seed": 23,
        "rescue_augmentation": adopted_rescue,
        "development_base": base_scores,
        "development_rescue": rescue_scores,
    }
    freeze_path = args.output_root / "frozen_protocol.json"
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    # Confirmation may start only if the implementation still matches the freeze.
    if source_hash() != freeze["source_sha256"]:
        raise RuntimeError("implementation changed after protocol freeze")
    confirmation = [
        ensure_decoder(args.output_root, fold, 23, adopted_rescue, common)
        for fold in range(4, 7)
    ]
    report = {
        "development": freeze,
        "confirmation": classify_confirmation(confirmation),
        "stability_seed42": None,
    }
    if args.remaining_hours >= 5 and not adopted_rescue:
        stability = [
            ensure_decoder(args.output_root, fold, 42, False, common)
            for fold in range(4, 7)
        ]
        report["stability_seed42"] = exact_scores(stability)
    (args.output_root / "protocol_result.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
