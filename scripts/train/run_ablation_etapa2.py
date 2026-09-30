"""Resumable 2x2x2x3-seed Etapa 2 ablation orchestrator.

Run inside tmux session 0:
  tmux send-keys -t 0 \
    "PYTHONPATH=. python scripts/train/run_ablation_etapa2.py --run-tag $(date +%Y%m%d_%H%M%S)" Enter
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

TOKEN_HEADS = ("linear", "contextual")
LENGTH_HEADS = ("mean", "attention")
LABEL_SMOOTHINGS = (0.0, 0.1)
SEEDS = (23, 42, 101)
SPLIT_SEED = 23
EPOCHS = 15

DEFAULT_RESUME_CKPT = Path(
    "../outputs/v126_temporal/diag_A3_length_head_rerun_20260627_021153/checkpoint_best.pt"
)
DEFAULT_OUT_ROOT = Path("../outputs/v126_temporal")
DEFAULT_REGISTRY_DIR = Path("artifacts/v126_closeout/ablation_etapa2/registry")
DEFAULT_PYTHON_BIN = "python"

FIXED_TRAIN_FLAGS = [
    "--phase", "learned_cif",
    "--resume-weights-only",
    "--epochs", str(EPOCHS),
    "--min-clips", "1", "--max-clips", "1",
    "--min-neutral-frames", "0", "--max-neutral-frames", "8",
    "--alpha-schedule", "target_only",
    "--diag-alpha-loss", "logit_l1",
    "--diag-freeze", "target_only_stage1",
    "--stgcn-lr-scale", "0.1",
    "--prediction-alpha-mode", "pred_rescaled_to_pred_len",
]


def generate_variants() -> list[dict]:
    variants = []
    for token_head, length_head, smoothing, seed in itertools.product(
        TOKEN_HEADS, LENGTH_HEADS, LABEL_SMOOTHINGS, SEEDS
    ):
        variants.append(
            {
                "token_head": token_head,
                "length_head": length_head,
                "token_label_smoothing": smoothing,
                "seed": seed,
                "split_seed": SPLIT_SEED,
            }
        )
    return variants


def run_name_for(variant: dict, run_tag: str) -> str:
    smoothing = variant["token_label_smoothing"]
    return (
        f"A3_etapa2_ablation_{variant['token_head']}_{variant['length_head']}"
        f"_ls{smoothing}_seed{variant['seed']}_{run_tag}"
    )


def train_command_for(
    variant: dict,
    run_name: str,
    resume_ckpt: Path,
    python_bin: str = DEFAULT_PYTHON_BIN,
    out_root: Path = DEFAULT_OUT_ROOT,
) -> list[str]:
    return [
        python_bin, "scripts/train/train_temporal_v126.py",
        *FIXED_TRAIN_FLAGS,
        "--resume", str(resume_ckpt),
        "--token-head", variant["token_head"],
        "--length-head", variant["length_head"],
        "--token-label-smoothing", str(variant["token_label_smoothing"]),
        "--split-seed", str(variant["split_seed"]),
        "--seed", str(variant["seed"]),
        "--run-name", run_name,
        "--output-root", str(out_root),
    ]


def audit_command_for(variant: dict, run_name: str, out_root: Path, python_bin: str = DEFAULT_PYTHON_BIN) -> list[str]:
    checkpoint = out_root / f"diag_{run_name}" / "checkpoint_best.pt"
    output = out_root / f"diag_{run_name}_audit.json"
    return [
        python_bin, "scripts/diagnostics/analyze_imitator_a2.py",
        "--checkpoint", str(checkpoint),
        "--output", str(output),
        "--split-seed", str(variant["split_seed"]),
        "--seed", str(variant["seed"]),
    ]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_reuse(registry_entry: dict, variant: dict, checkpoint_path: Path) -> bool:
    if not checkpoint_path.is_file():
        return False
    if registry_entry.get("variant") != variant:
        return False
    return registry_entry.get("checkpoint_sha256") == sha256_of(checkpoint_path)


def main(
    *,
    variants: list[dict],
    run_tag: str,
    out_root: Path = DEFAULT_OUT_ROOT,
    registry_dir: Path = DEFAULT_REGISTRY_DIR,
    resume_ckpt: Path = DEFAULT_RESUME_CKPT,
    python_bin: str = DEFAULT_PYTHON_BIN,
    runner=subprocess.run,
) -> None:
    registry_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        run_name = run_name_for(variant, run_tag)
        registry_path = registry_dir / f"{run_name}.json"
        checkpoint_path = out_root / f"diag_{run_name}" / "checkpoint_best.pt"

        if registry_path.is_file():
            entry = json.loads(registry_path.read_text(encoding="utf-8"))
            if is_valid_reuse(entry, variant, checkpoint_path):
                print(f"[ablation] skip valid reuse run={run_name}")
                continue
            print(f"[ablation] registry stale/mismatched for run={run_name}, re-running")

        print(f"[ablation] run start {run_name}")
        runner(train_command_for(variant, run_name, resume_ckpt, python_bin, out_root), check=True)
        runner(audit_command_for(variant, run_name, out_root, python_bin), check=True)

        if not checkpoint_path.is_file():
            raise RuntimeError(f"training did not produce a checkpoint for {run_name}: {checkpoint_path}")
        registry_path.write_text(
            json.dumps(
                {
                    "variant": variant,
                    "run_name": run_name,
                    "checkpoint_sha256": sha256_of(checkpoint_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[ablation] run done {run_name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--resume-ckpt", type=Path, default=DEFAULT_RESUME_CKPT)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        variants=generate_variants(),
        run_tag=args.run_tag,
        out_root=args.out_root,
        registry_dir=args.registry_dir,
        resume_ckpt=args.resume_ckpt,
        python_bin=args.python_bin,
    )
    print("ALL_ABLATION_RUNS_DONE", file=sys.stderr)
