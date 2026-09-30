"""Sequential, resumable orchestrator for the clean 10-fold LOSO reconstruction.

Per fold, rebuilds the full chain from scratch without reusing any weights
trained on the outer test signer:

    v121 (80ep) -> A1 (30ep, linear/mean) -> A2 (40ep, linear/mean, target_only)
    -> A3_pre_decoder (30ep, linear/mean) -> decoder_final (15ep, contextual/attention,
    smoothing 0.1) -> promotion (1ep, target_only)

Each stage is skipped only if its recorded manifest hash, stage config hash,
and parent-checkpoint hash all still match what's on disk; any mismatch
reruns the stage rather than trusting stale state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
OUT_ROOT = ROOT.parent / "outputs" / "loso_clean"
V121_CHECKPOINTS_ROOT = ROOT.parent / "outputs" / "checkpoints" / "121"
V121_CONFIG = "experiments/v121_v124_isolated_staged/cls_v121.toml"
# cls_v121.toml's original h5 (dataset1_isolated.hdf5) has no signer_id metadata at all,
# so LOSO splitting on it is impossible. v122's re-extraction does carry signer_id, and
# it's the same file train_temporal_v126.py uses by default, so the whole chain stays on
# one canonical dataset source.
V121_H5_FILENAME = "dataset1_isolated_v122.hdf5"

STAGE_SPECS = [
    {
        "name": "A1",
        "epochs": 30,
        "extra": ["--phase", "teacher_forced", "--token-head", "linear", "--length-head", "mean"],
    },
    {
        "name": "A2",
        "epochs": 40,
        "extra": [
            "--phase", "learned_cif",
            "--token-head", "linear", "--length-head", "mean",
            "--alpha-schedule", "target_only",
            "--diag-alpha-loss", "logit_l1",
            "--diag-freeze", "target_only_stage1",
        ],
    },
    {
        "name": "A3_pre_decoder",
        "epochs": 30,
        "extra": [
            "--phase", "learned_cif",
            "--token-head", "linear", "--length-head", "mean",
            "--alpha-schedule", "current",
        ],
    },
    {
        "name": "decoder_final",
        "epochs": 15,
        "extra": [
            "--phase", "learned_cif",
            "--token-head", "contextual", "--length-head", "attention",
            "--token-label-smoothing", "0.1",
            "--alpha-schedule", "current",
        ],
    },
    {
        "name": "promotion",
        "epochs": 1,
        "extra": [
            "--phase", "learned_cif",
            "--token-head", "contextual", "--length-head", "attention",
            "--alpha-schedule", "target_only",
            "--diag-alpha-loss", "logit_l1",
            "--diag-freeze", "target_only_stage1",
        ],
    },
]


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_dict(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in data.items() if k != "manifest_sha256"}
    if sha256_of_dict(payload) != data["manifest_sha256"]:
        raise RuntimeError(f"manifest hash mismatch: {path} was modified after generation")
    return data


def _arg_value(args_list: list[str], flag: str, default: str) -> str:
    if flag in args_list:
        return args_list[args_list.index(flag) + 1]
    return default


def expected_run_name(run_name: str, extra: list[str]) -> str:
    """Mirror train_temporal_v126.py's auto diag_ prefix so we find the right out_dir."""
    diag_alpha_loss = _arg_value(extra, "--diag-alpha-loss", "current")
    diag_freeze = _arg_value(extra, "--diag-freeze", "full_current")
    diagnostic_mode = diag_alpha_loss != "current" or diag_freeze != "full_current"
    if diagnostic_mode and not run_name.startswith("diag_"):
        return f"diag_{run_name}"
    return run_name


def run_subprocess(argv: list[str]) -> None:
    print(f"[orchestrator] $ {' '.join(argv)}", flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(argv, cwd=ROOT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"stage failed (exit={result.returncode}): {' '.join(argv)}")


def _try_skip(done_path: Path, stage_hash: str) -> Path | None:
    if not done_path.exists():
        return None
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("stage_hash") != stage_hash:
        print(f"[orchestrator] stale state at {done_path}; rerunning", flush=True)
        return None
    ckpt = Path(done.get("checkpoint", ""))
    if done.get("pruned"):
        # Deliberately deleted after a child stage consumed it (see prune_checkpoint_files).
        # Trust the attested hash rather than requiring the file to still exist.
        return ckpt
    if ckpt.exists() and sha256_of(ckpt) == done.get("checkpoint_sha256"):
        return ckpt
    print(f"[orchestrator] stale state at {done_path}; rerunning", flush=True)
    return None


def _recorded_checkpoint_sha256(parent_checkpoint: Path) -> tuple[str | None, bool]:
    for candidate_dir in (parent_checkpoint.parent, parent_checkpoint.parent.parent):
        done_path = candidate_dir / "stage_done.json"
        if not done_path.exists():
            continue
        recorded = json.loads(done_path.read_text(encoding="utf-8"))
        if Path(recorded.get("checkpoint", "")) == parent_checkpoint and "checkpoint_sha256" in recorded:
            return recorded["checkpoint_sha256"], bool(recorded.get("pruned", False))
    return None, False


def resolved_parent_sha256(parent_checkpoint: Path) -> str:
    """Hash of the parent checkpoint, preferring the attested hash once pruned.

    Disk retention deletes a stage's checkpoint once its child stage has
    successfully consumed it (see prune_checkpoint_files), which marks the
    stage "pruned" in its own stage_done.json. That marker is authoritative:
    if a *new* file later appears at the same path (e.g. an interrupted rerun
    that got partway through before being killed), it is NOT the checkpoint
    this hash chain was built on, so the recorded hash wins over re-reading
    whatever bytes are currently on disk.
    """
    recorded_hash, pruned = _recorded_checkpoint_sha256(parent_checkpoint)
    if pruned and recorded_hash is not None:
        return recorded_hash
    if parent_checkpoint.exists():
        return sha256_of(parent_checkpoint)
    if recorded_hash is not None:
        return recorded_hash
    raise FileNotFoundError(
        f"parent checkpoint missing and no recorded hash found for {parent_checkpoint}"
    )


def mark_pruned(done_path: Path) -> None:
    """Flag a stage as deliberately pruned so _try_skip trusts it without the file."""
    if not done_path.exists():
        return
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["pruned"] = True
    done_path.write_text(json.dumps(done, indent=2), encoding="utf-8")


def prune_checkpoint_files(out_dir: Path) -> None:
    """Delete a v126 stage's checkpoint files once its child stage has consumed them."""
    for name in ("checkpoint_best.pt", "checkpoint_latest.pt"):
        path = out_dir / name
        if path.exists():
            path.unlink()
            print(f"[orchestrator] pruned {path}", flush=True)
    mark_pruned(out_dir / "stage_done.json")


def prune_v121_checkpoint(run_id: int) -> None:
    """Delete v121's checkpoint.pth files (best_top1/ and periodic epoch dirs)."""
    run_dir = V121_CHECKPOINTS_ROOT / str(run_id)
    if not run_dir.exists():
        return
    for ckpt_path in run_dir.glob("*/checkpoint.pth"):
        ckpt_path.unlink()
        print(f"[orchestrator] pruned {ckpt_path}", flush=True)
    mark_pruned(run_dir / "stage_done.json")


def _write_done(done_path: Path, stage_hash: str, checkpoint: Path) -> None:
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text(
        json.dumps(
            {
                "stage_hash": stage_hash,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_of(checkpoint),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_commit(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_v121_stage(fold: dict, manifest_sha256: str) -> Path:
    run_id = 9000 + fold["fold"]
    out_dir = V121_CHECKPOINTS_ROOT / str(run_id)
    checkpoint = out_dir / "best_top1" / "checkpoint.pth"
    stage_config = {
        "stage": "v121",
        "fold": fold["fold"],
        "config": V121_CONFIG,
        "h5_filename": V121_H5_FILENAME,
        "manifest_sha256": manifest_sha256,
    }
    stage_hash = sha256_of_dict(stage_config)
    done_path = out_dir / "stage_done.json"

    cached = _try_skip(done_path, stage_hash)
    if cached is not None:
        print(f"[orchestrator] skip fold={fold['fold']} stage=v121 (hash-verified)", flush=True)
        return cached

    run_subprocess(
        [
            PYTHON, "scripts/train/train_isolated_staged.py",
            "--config", V121_CONFIG,
            "--h5-filename", V121_H5_FILENAME,
            "--heldout-signer", str(fold["inner_val_signer"]),
            "--exclude-signer", str(fold["outer_test_signer"]),
            "--run-id", str(run_id),
        ]
    )
    if not checkpoint.exists():
        raise RuntimeError(f"expected v121 checkpoint missing: {checkpoint}")
    _write_done(done_path, stage_hash, checkpoint)
    return checkpoint


def run_v126_stage(
    fold: dict,
    stage_spec: dict,
    parent_checkpoint: Path,
    checkpoint_v121: Path,
    manifest_sha256: str,
) -> Path:
    base_run_name = f"fold{fold['outer_test_signer']}_{stage_spec['name']}"
    run_name = expected_run_name(base_run_name, stage_spec["extra"])
    out_dir = OUT_ROOT / run_name
    checkpoint = out_dir / "checkpoint_best.pt"
    stage_config = {
        "stage": stage_spec["name"],
        "fold": fold["fold"],
        "extra": stage_spec["extra"],
        "epochs": stage_spec["epochs"],
        "manifest_sha256": manifest_sha256,
        "parent_checkpoint_sha256": resolved_parent_sha256(parent_checkpoint),
    }
    stage_hash = sha256_of_dict(stage_config)
    done_path = out_dir / "stage_done.json"

    cached = _try_skip(done_path, stage_hash)
    if cached is not None:
        print(
            f"[orchestrator] skip fold={fold['fold']} stage={stage_spec['name']} (hash-verified)",
            flush=True,
        )
        return cached

    if not parent_checkpoint.exists():
        raise RuntimeError(
            f"cannot run stage={stage_spec['name']}: parent checkpoint was pruned "
            f"({parent_checkpoint}) and this stage isn't already done. Aggressive "
            f"retention only keeps the chain valid for forward progress; rebuilding "
            f"from a pruned ancestor isn't supported."
        )

    argv = [
        PYTHON, "scripts/train/train_temporal_v126.py",
        "--heldout-signer", str(fold["inner_val_signer"]),
        "--exclude-signer", str(fold["outer_test_signer"]),
        "--seed", "23",
        "--epochs", str(stage_spec["epochs"]),
        "--output-root", str(OUT_ROOT),
        "--run-name", base_run_name,
        *stage_spec["extra"],
    ]
    if stage_spec["name"] == "A1":
        argv += ["--checkpoint-v121", str(checkpoint_v121)]
    else:
        argv += ["--resume", str(parent_checkpoint), "--resume-weights-only"]

    run_subprocess(argv)
    if not checkpoint.exists():
        raise RuntimeError(f"expected checkpoint missing: {checkpoint}")
    _write_done(done_path, stage_hash, checkpoint)
    return checkpoint


def run_test_signer_eval(fold: dict, final_checkpoint: Path) -> Path:
    """Load the outer test signer for the first time, against the frozen checkpoint."""
    eval_path = OUT_ROOT / f"fold{fold['outer_test_signer']}_test_eval.json"
    if eval_path.exists():
        print(f"[orchestrator] skip fold={fold['fold']} test-eval (already written)", flush=True)
        return eval_path
    run_subprocess(
        [
            PYTHON, "scripts/diagnostics/analyze_imitator_a2.py",
            "--checkpoint", str(final_checkpoint),
            "--heldout-signer", str(fold["outer_test_signer"]),
            "--output", str(eval_path),
            "--seed", "23",
        ]
    )
    if not eval_path.exists():
        raise RuntimeError(f"expected test-signer eval output missing: {eval_path}")
    return eval_path


# Disk retention: /shared has very little headroom for 10 folds x 6 stages x
# ~800MB. Once a stage's checkpoint has been consumed by its child (the child
# stage is confirmed done), delete it -- keep only decoder_final and
# promotion, the two checkpoints that matter for Level B and the final report.
PRUNABLE_AFTER = {"A1": "v121", "A2": "A1", "A3_pre_decoder": "A2", "decoder_final": "A3_pre_decoder"}


def run_fold(fold: dict, manifest_sha256: str, *, evaluate_outer: bool = True) -> Path:
    v121_checkpoint = run_v121_stage(fold, manifest_sha256)
    parent = v121_checkpoint
    stage_out_dirs: dict[str, Path] = {}
    for stage_spec in STAGE_SPECS:
        parent = run_v126_stage(fold, stage_spec, parent, v121_checkpoint, manifest_sha256)
        stage_out_dirs[stage_spec["name"]] = parent.parent

        prune_target = PRUNABLE_AFTER.get(stage_spec["name"])
        if prune_target == "v121":
            prune_v121_checkpoint(9000 + fold["fold"])
        elif prune_target in stage_out_dirs:
            prune_checkpoint_files(stage_out_dirs[prune_target])

    if evaluate_outer:
        run_test_signer_eval(fold, parent)
    else:
        print(
            f"[orchestrator] fold={fold['fold']} preparation complete; "
            "outer-test evaluation deliberately skipped",
            flush=True,
        )
    return parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("experiments/a3_etapa4_clean_loso/manifest.json")
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--fold", type=int, default=None, help="Run only this manifest fold."
    )
    selection.add_argument(
        "--folds", type=int, nargs="+", default=None, help="Run only these manifest folds."
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build final checkpoints but never load or evaluate the outer-test signer.",
    )
    return parser.parse_args(argv)


def selected_folds(manifest: dict, *, fold: int | None, folds: list[int] | None) -> list[dict]:
    requested = [fold] if fold is not None else folds
    available = manifest["folds"]
    if requested is None:
        return available
    if len(set(requested)) != len(requested):
        raise ValueError("fold selection contains duplicates")
    by_number = {int(row["fold"]): row for row in available}
    missing = [number for number in requested if number not in by_number]
    if missing:
        raise ValueError(f"fold(s) not found in manifest: {missing}")
    return [by_number[number] for number in requested]


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest)
    folds = selected_folds(manifest, fold=args.fold, folds=args.folds)

    for fold in folds:
        print(
            f"=== fold {fold['fold']} start test={fold['outer_test_signer']} "
            f"val={fold['inner_val_signer']} train={fold['train_signers']} "
            f"{datetime.now(timezone.utc).isoformat()} ===",
            flush=True,
        )
        final_checkpoint = run_fold(
            fold, manifest["manifest_sha256"], evaluate_outer=not args.prepare_only
        )
        print(
            f"=== fold {fold['fold']} done final_checkpoint={final_checkpoint} "
            f"{datetime.now(timezone.utc).isoformat()} ===",
            flush=True,
        )
    print("ALL_FOLDS_DONE", flush=True)


if __name__ == "__main__":
    main()
