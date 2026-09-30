"""Optuna study for v126 learned-CIF calibration.

The search is intentionally centered on the 2026-06-24 diagnosis:
keep ST-GCN frozen, start from the calibrated teacher-only checkpoint, and
explore only target-only warmup or very gradual predicted-alpha mixing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna
from optuna.exceptions import TrialPruned


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "scripts" / "train" / "train_temporal_v126.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default="v126_cif_optuna_20260624")
    parser.add_argument(
        "--storage",
        default="sqlite:///../outputs/v126_temporal/optuna_v126_cif.db",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples-per-epoch", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument(
        "--resume",
        type=Path,
        default=Path("../outputs/v126_temporal/diag_target_only_stage1_long/checkpoint_best.pt"),
        help="Calibrated warmup checkpoint used as model-weight initialization.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../outputs/v126_temporal/optuna_trials"),
    )
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric(row: dict, dotted_key: str, default: float = 0.0) -> float:
    value = row
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return float(value)


def collapse_penalty(row: dict) -> float:
    teacher_top1 = metric(row, "val_teacher.top1")
    target_len = metric(row, "val_pred_raw.target_len_mean", 1.0)
    quantity = metric(row, "val_pred_raw.quantity_mean")
    pred_count = metric(row, "val_pred_raw.pred_count_mean")
    alpha_p50 = metric(row, "val_pred_raw.alpha_logit_p50")

    penalty = 0.0
    penalty += max(0.0, 0.70 - teacher_top1) * 6.0
    penalty += max(0.0, 0.55 * target_len - quantity) * 0.15
    penalty += max(0.0, 0.55 * target_len - pred_count) * 0.15
    if alpha_p50 < -12.0:
        penalty += 2.0
    return penalty


def objective_score(row: dict) -> float:
    """Single scalar for Optuna; gates remain visible in trial attrs."""
    raw_top1 = metric(row, "val_pred_raw.top1")
    rescaled_top1 = metric(row, "val_pred_rescaled_to_target_len.top1")
    raw_top5 = metric(row, "val_pred_raw.top5")
    boundary_mae = metric(row, "val_pred_rescaled_to_target_len.boundary_mae", 1000.0)
    raw_mae_len = metric(row, "val_pred_raw.mae_len", 1000.0)
    permuted_drop = metric(row, "permuted_top1_drop")

    return (
        2.00 * rescaled_top1
        + 1.25 * raw_top1
        + 0.25 * raw_top5
        + 0.75 * permuted_drop
        - 0.020 * boundary_mae
        - 0.080 * raw_mae_len
        - collapse_penalty(row)
    )


def suggest_trial(trial: optuna.Trial) -> dict:
    mode = trial.suggest_categorical("mode", ["target_only", "linear_pred_mix"])
    params = {
        "mode": mode,
        "lr": trial.suggest_float("lr", 5e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True),
        "alpha_loss_weight": trial.suggest_float("alpha_loss_weight", 2.0, 12.0),
        "qty_loss_weight": trial.suggest_float("qty_loss_weight", 0.5, 2.5),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 2.0),
        "stgcn_lr_scale": trial.suggest_float("stgcn_lr_scale", 0.01, 0.10, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [2, 4]),
        "max_clips": trial.suggest_categorical("max_clips", [6, 8]),
        "diag_alpha_loss": trial.suggest_categorical("diag_alpha_loss", ["current", "logit_l1"]),
    }
    if mode == "linear_pred_mix":
        start = trial.suggest_float("mix_w_pred_start", 0.02, 0.10)
        params["mix_w_pred_start"] = start
        params["mix_w_pred_end"] = trial.suggest_float("mix_w_pred_end", start, 0.25)
        params["mix_ramp_epochs"] = trial.suggest_int("mix_ramp_epochs", 6, 20)
        params["mix_start_epoch"] = trial.suggest_int("mix_start_epoch", 0, 3)
    else:
        params["mix_w_pred_start"] = 0.0
        params["mix_w_pred_end"] = 0.0
        params["mix_ramp_epochs"] = 1
        params["mix_start_epoch"] = 0
    return params


def build_command(args, trial: optuna.Trial, params: dict, run_name: str) -> list[str]:
    alpha_schedule = "target_only" if params["mode"] == "target_only" else "linear_pred_mix"
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--phase",
        "learned_cif",
        "--resume",
        str(args.resume),
        "--resume-weights-only",
        "--epochs",
        str(args.epochs),
        "--samples-per-epoch",
        str(args.samples_per_epoch),
        "--val-samples",
        str(args.val_samples),
        "--seed",
        str(args.seed + trial.number),
        "--run-name",
        run_name,
        "--output-root",
        str(args.output_root),
        "--diag-freeze",
        "target_only_stage1",
        "--diag-alpha-loss",
        params["diag_alpha_loss"],
        "--alpha-schedule",
        alpha_schedule,
        "--mix-start-epoch",
        str(params["mix_start_epoch"]),
        "--mix-ramp-epochs",
        str(params["mix_ramp_epochs"]),
        "--mix-w-pred-start",
        str(params["mix_w_pred_start"]),
        "--mix-w-pred-end",
        str(params["mix_w_pred_end"]),
        "--lr",
        str(params["lr"]),
        "--weight-decay",
        str(params["weight_decay"]),
        "--alpha-loss-weight",
        str(params["alpha_loss_weight"]),
        "--qty-loss-weight",
        str(params["qty_loss_weight"]),
        "--grad-clip",
        str(params["grad_clip"]),
        "--stgcn-lr-scale",
        str(params["stgcn_lr_scale"]),
        "--batch-size",
        str(params["batch_size"]),
        "--max-clips",
        str(params["max_clips"]),
    ]


def run_trial(args, trial: optuna.Trial) -> float:
    params = suggest_trial(trial)
    run_name = f"diag_trial_{trial.number:04d}_{params['mode']}"
    out_dir = args.output_root / run_name
    metrics_path = out_dir / "metrics.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args, trial, params, run_name)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    reported = 0
    last_rows: list[dict] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if not line.startswith("ep"):
            continue
        time.sleep(0.1)
        rows = read_metrics(metrics_path)
        for row in rows[reported:]:
            score = objective_score(row)
            trial.report(score, step=int(row["epoch"]))
            reported += 1
            last_rows = rows
            if trial.should_prune():
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise TrialPruned(f"pruned at epoch {row['epoch']} score={score:.4f}")

    rc = process.wait()
    rows = read_metrics(metrics_path)
    if rc != 0:
        raise RuntimeError(f"training command failed with exit code {rc}: {' '.join(cmd)}")
    if not rows:
        raise RuntimeError(f"training finished without metrics: {metrics_path}")
    last_rows = rows

    best_row = max(last_rows, key=objective_score)
    trial.set_user_attr("run_dir", str(out_dir))
    trial.set_user_attr("best_epoch", int(best_row["epoch"]))
    for key in (
        "val_teacher.top1",
        "val_pred_raw.top1",
        "val_pred_raw.top5",
        "val_pred_raw.mae_len",
        "val_pred_raw.quantity_mean",
        "val_pred_raw.pred_count_mean",
        "val_pred_rescaled_to_target_len.top1",
        "val_pred_rescaled_to_target_len.boundary_mae",
        "permuted_top1_drop",
    ):
        trial.set_user_attr(key, metric(best_row, key))
    return objective_score(best_row)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(lambda trial: run_trial(args, trial), n_trials=args.n_trials, timeout=args.timeout)
    print("Best value:", study.best_value)
    print("Best params:", study.best_trial.params)
    print("Best attrs:", study.best_trial.user_attrs)


if __name__ == "__main__":
    main()
