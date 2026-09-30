"""Runner configurable para los experimentos aislados v121-v124."""
import argparse
import json
import os
import random
import subprocess
import sys
import tomllib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="experiments/v121_v124_isolated_staged/cls_v121.toml", help="TOML del experimento"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--heldout-signer", type=int)
    parser.add_argument(
        "--exclude-signer",
        type=int,
        default=None,
        help="Outer-test signer for clean LOSO: excluded from both train and val/--heldout-signer.",
    )
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument(
        "--h5-filename",
        default=None,
        help="Override data.h5_filename from the TOML. Needed for clean LOSO: cls_v121.toml's "
        "original h5 has no signer_id metadata, so the orchestrator points this at "
        "dataset1_isolated_v122.hdf5 (the same file train_temporal_v126.py uses by default).",
    )
    return parser.parse_args()


ARGS = _parse_args()
os.environ["MSLM_EXPERIMENT_CONFIG"] = ARGS.config
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

with open(ARGS.config, "rb") as _f:
    _raw_cfg = tomllib.load(_f)
_seed = ARGS.seed if ARGS.seed is not None else int(_raw_cfg.get("experiment", {}).get("seed", 23))

from settings import initialize

initialize(seed=_seed)

import functools

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.mslm.checkpoint.manager import CheckpointManager
from src.mslm.dataloader.isolated_keypoint_dataset import (
    IsolatedKeypointDataset,
    isolated_collate_fn,
    list_clip_records,
)
from src.mslm.models.isolated_classifier import IsolatedSignClassifier
from src.mslm.utils.config_loader import cfg
from src.mslm.utils.early_stopping import EarlyStopping
from src.mslm.utils.metrics import top_k_accuracy


def _cfg(section, key, default):
    value = getattr(cfg, section, {}) or {}
    return value.get(key, default)


def _forward_batch(model, keypoints, device):
    return torch.cat([model(kp.unsqueeze(0).to(device)) for kp in keypoints], dim=0)


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def stratified_split(labels, n_val_per_class, seed):
    by_label = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for idxs in by_label.values():
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        val_idx.extend(shuffled[:n_val_per_class])
        train_idx.extend(shuffled[n_val_per_class:])
    return train_idx, val_idx


def split_records(records, seed, n_val_per_class, heldout_signer=None, exclude_signer=None):
    if exclude_signer is not None and heldout_signer is None:
        raise ValueError("exclude_signer requiere heldout_signer")
    if exclude_signer is not None and exclude_signer == heldout_signer:
        raise ValueError("exclude_signer debe ser distinto de heldout_signer")
    if heldout_signer is None:
        return stratified_split([r["label"] for r in records], n_val_per_class, seed)
    excluded = {heldout_signer, exclude_signer} - {None}
    train_idx = [i for i, r in enumerate(records) if r["signer_id"] not in excluded]
    val_idx = [i for i, r in enumerate(records) if r["signer_id"] == heldout_signer]
    if not train_idx or not val_idx:
        raise ValueError(f"heldout_signer={heldout_signer} no produce ambos splits")
    return train_idx, val_idx


def _dataset_kwargs(augment):
    return {
        "dataset_name": _cfg("data", "dataset_name", "dataset1"),
        "n_keypoints": int(_cfg("data", "n_keypoints", 111)),
        "augment": augment,
        "normalization": _cfg("data", "normalization", "minmax"),
        "augmentation_profile": _cfg("data", "augmentation_profile", "legacy"),
        "trim_active": bool(_cfg("data", "trim_active", False)),
        "target_frames": _cfg("data", "target_frames", None),
        "downsample": int(_cfg("data", "downsample", 1)),
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = top1_hits = top5_hits = n = 0.0
    batches = 0
    with torch.no_grad():
        for keypoints, target in loader:
            target = target.to(device)
            logits = _forward_batch(model, keypoints, device)
            total_loss += criterion(logits, target).item()
            bs = target.size(0)
            top1_hits += top_k_accuracy(logits, target, 1) * bs
            top5_hits += top_k_accuracy(logits, target, 5) * bs
            n += bs
            batches += 1
    return {
        "loss": total_loss / max(1, batches),
        "top1": top1_hits / max(1, n),
        "top5": top5_hits / max(1, n),
        "samples": int(n),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.mslm.utils.paths import path_vars

    seed = _seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    h5_filename = ARGS.h5_filename or _cfg("data", "h5_filename", "dataset1_isolated.hdf5")
    h5_file = path_vars.data_path / "processed" / h5_filename
    records = list_clip_records(h5_file, _cfg("data", "dataset_name", "dataset1"))
    max_samples = _cfg("data", "max_samples", None)
    if max_samples and int(max_samples) < len(records):
        rng = random.Random(seed)
        keep = sorted(rng.sample(range(len(records)), int(max_samples)))
        records = [records[i] for i in keep]

    heldout_signer = (
        ARGS.heldout_signer
        if ARGS.heldout_signer is not None
        else _cfg("evaluation", "heldout_signer", None)
    )
    train_idx, val_idx = split_records(
        records,
        seed,
        int(_cfg("data", "n_val_per_class", 10)),
        int(heldout_signer) if heldout_signer is not None else None,
        int(ARGS.exclude_signer) if ARGS.exclude_signer is not None else None,
    )
    if ARGS.exclude_signer is not None:
        train_signers = {records[i]["signer_id"] for i in train_idx}
        val_signers = {records[i]["signer_id"] for i in val_idx}
        if ARGS.exclude_signer in train_signers or ARGS.exclude_signer in val_signers:
            raise RuntimeError(
                f"contamination guard: exclude_signer={ARGS.exclude_signer} leaked into split"
            )
    train_labels = sorted({records[i]["label"] for i in train_idx})
    label_to_idx = {label: i for i, label in enumerate(train_labels)}
    val_idx = [i for i in val_idx if records[i]["label"] in label_to_idx]

    def make_dataset(indices, augment):
        return IsolatedKeypointDataset(
            h5_file,
            [records[i]["clip_id"] for i in indices],
            [records[i]["label"] for i in indices],
            **_dataset_kwargs(augment),
        )

    collate = functools.partial(isolated_collate_fn, label_to_idx=label_to_idx)
    batch_size = int(_cfg("training", "batch_size", 16))
    train_dl = DataLoader(
        make_dataset(train_idx, augment=True),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    train_eval_dl = DataLoader(
        make_dataset(train_idx, augment=False),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    val_dl = DataLoader(
        make_dataset(val_idx, augment=False),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    model_cfg = dict(cfg.model)
    model = IsolatedSignClassifier(
        A=np.load(path_vars.data_path / "processed" / "adjacency_matrix.npy", allow_pickle=True),
        input_size=int(_cfg("data", "n_keypoints", 111)),
        gcn_channels=tuple(model_cfg.get("gcn_channels", [32, 64, 128])),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_classes=len(label_to_idx),
        norm_type=model_cfg.get("norm_type", "batch"),
        norm_groups=int(model_cfg.get("norm_groups", 16)),
        temporal_head=bool(model_cfg.get("temporal_head", False)),
        use_motion_stream=bool(model_cfg.get("use_motion_stream", False)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(_cfg("training", "label_smoothing", 0.1))
    )
    weight_decay = (
        ARGS.weight_decay
        if ARGS.weight_decay is not None
        else float(_cfg("training", "weight_decay", 1e-4))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(_cfg("training", "learning_rate", 1e-3)),
        weight_decay=weight_decay,
    )

    version = int(_cfg("training", "model_version", 121))
    run_id = ARGS.run_id if ARGS.run_id is not None else int(_cfg("training", "run_id", seed))
    output_root = Path(_cfg("training", "output_root", "../outputs"))
    writer = SummaryWriter(
        output_root
        / "reports"
        / str(version)
        / str(run_id)
        / datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    )
    checkpoint = CheckpointManager(str(output_root / "checkpoints"), version, run_id)
    checkpoint_dir = Path(checkpoint._path())
    with open(checkpoint_dir / "label_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, indent=2, ensure_ascii=False)

    lineage = {
        "fold_outer_test_signer": ARGS.exclude_signer,
        "fold_inner_val_signer": heldout_signer,
        "train_signers": sorted({records[i]["signer_id"] for i in train_idx}),
        "val_signers": sorted({records[i]["signer_id"] for i in val_idx}),
        "git_commit": _git_commit_hash(),
        "argv": sys.argv[1:],
    }

    stopper = EarlyStopping(
        patience=int(_cfg("training", "early_stopping_patience", 20)),
        threshold=1e-4,
        verbose=True,
    )
    history = []
    best_top1 = -1.0
    epochs = int(_cfg("training", "epochs", 80))
    checkpoint_interval = int(_cfg("training", "checkpoint_interval", 10))

    print(
        f"[v{version}] seed={seed} train={len(train_idx)} val={len(val_idx)} "
        f"signer={heldout_signer} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M"
    )
    for epoch in range(epochs):
        model.train()
        train_optim_loss = 0.0
        steps = 0
        for keypoints, target in tqdm(
            train_dl, desc=f"ep{epoch} train", leave=False, mininterval=10.0
        ):
            target = target.to(device)
            logits = _forward_batch(model, keypoints, device)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_optim_loss += loss.item()
            steps += 1

        train_metrics = evaluate(model, train_eval_dl, criterion, device)
        val_metrics = evaluate(model, val_dl, criterion, device)
        row = {
            "epoch": epoch,
            "train_optim_loss": train_optim_loss / max(1, steps),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        for split, metrics in (("train_eval", train_metrics), ("val", val_metrics)):
            writer.add_scalar(f"Loss/{split}", metrics["loss"], epoch)
            writer.add_scalar(f"Acc/{split}_top1", metrics["top1"], epoch)
            writer.add_scalar(f"Acc/{split}_top5", metrics["top5"], epoch)
        writer.add_scalar("Loss/train_optim", row["train_optim_loss"], epoch)
        print(
            f"ep{epoch:>3} | optim {row['train_optim_loss']:.3f} | "
            f"train {train_metrics['top1']:.1%} | val {val_metrics['top1']:.1%} "
            f"| gap {train_metrics['top1'] - val_metrics['top1']:.1%}"
        )

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            checkpoint.save_checkpoint(model, epoch, optimizer, None, tag="best_top1", metadata=lineage)
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint.save_checkpoint(model, epoch, optimizer, None, metadata=lineage)
        stopper(-val_metrics["top1"], epoch)
        if stopper.stop:
            break

    best_row = max(history, key=lambda row: row["val"]["top1"])
    summary = {
        "config": ARGS.config,
        "version": version,
        "run_id": run_id,
        "seed": seed,
        "heldout_signer": heldout_signer,
        "weight_decay": weight_decay,
        "best_epoch": best_row["epoch"],
        "best_train_top1": best_row["train"]["top1"],
        "best_val_top1": best_row["val"]["top1"],
        "best_val_top5": best_row["val"]["top5"],
        "generalization_gap": best_row["train"]["top1"] - best_row["val"]["top1"],
        "history": history,
    }
    summary_path = ARGS.summary_path or checkpoint_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    writer.close()
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
