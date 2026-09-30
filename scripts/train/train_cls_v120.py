"""v120 — reconocimiento de señas aislado sobre dataset1 (64 glosas, 50
ejemplos c/u). Reemplaza el objetivo de traducción de v113-v119 (dataset2: 5600
frases, 96% labels únicas, sin glosas -- CTC necesita repetición por token,
imposible con ~1 ejemplo por frase). dataset1 es la tarea que SÍ resuelve la
literatura cargada (LSA64 99.94%, AUTSL 90.5%, "Less is More" PUCP/LSP) y la
que encaja con los datos reales que tenemos. Azar = 1/64 = 1.56%.

Uso:
    PYTHONPATH=. python scripts/train/train_cls_v120.py
"""
import os

os.environ.setdefault("MSLM_EXPERIMENT_CONFIG", "experiments/v120_isolated/cls_v120.toml")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from settings import initialize

initialize()

import functools
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Orden de imports: utils ANTES de models (import circular real entre
# src.mslm.models.__init__ y src.mslm.utils.__init__, ver nota en
# train_ctc_v119.py -- mismo patrón).
from src.mslm.utils.setup_train import setup_paths
from src.mslm.utils.config_loader import cfg
from src.mslm.utils.early_stopping import EarlyStopping
from src.mslm.utils.metrics import top_k_accuracy
from src.mslm.models.isolated_classifier import IsolatedSignClassifier
from src.mslm.dataloader.isolated_keypoint_dataset import (
    IsolatedKeypointDataset, isolated_collate_fn, list_clips,
)
from src.mslm.checkpoint.manager import CheckpointManager

torch.manual_seed(23)
random.seed(23)


def _forward_batch(model, keypoints, device):
    """Procesa cada clip por separado (B=1, sin padding) y concatena logits --
    mismo motivo que `_encode_batch` en train_ctc_v119.py: batchear con
    padding+máscara filtra entre samples en un STGCN multi-capa (ver docstring
    de IsolatedSignClassifier.forward)."""
    logits = [model(kp.unsqueeze(0).to(device)) for kp in keypoints]
    return torch.cat(logits, dim=0)


def _cfg(section, key, default):
    s = getattr(cfg, section, {}) or {}
    return s.get(key, default)


def stratified_split(clip_ids, labels, n_val_per_class, seed):
    """Separa n_val_per_class ejemplos de cada clase para val; el resto a train."""
    by_label = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_label[lbl].append(i)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for lbl, idxs in by_label.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        val_idx.extend(idxs[:n_val_per_class])
        train_idx.extend(idxs[n_val_per_class:])
    return train_idx, val_idx


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, _ = setup_paths()
    from src.mslm.utils.paths import path_vars

    adj_path = path_vars.data_path / "processed" / "adjacency_matrix.npy"
    A = np.load(adj_path, allow_pickle=True)

    h5_filename = _cfg("data", "h5_filename", "dataset1_isolated.hdf5")
    h5_file = path_vars.data_path / "processed" / h5_filename
    dataset_name = _cfg("data", "dataset_name", "dataset1")
    n_keypoints = int(_cfg("data", "n_keypoints", 111))
    n_val_per_class = int(_cfg("data", "n_val_per_class", 10))

    model_cfg = dict(cfg.model)
    epochs = int(_cfg("training", "epochs", 60))
    batch_size = int(_cfg("training", "batch_size", 16))
    lr = float(_cfg("training", "learning_rate", 1e-3))
    wd = float(_cfg("training", "weight_decay", 1e-4))
    label_smoothing = float(_cfg("training", "label_smoothing", 0.1))
    version = int(_cfg("training", "model_version", 120))
    run_id = int(_cfg("training", "run_id", 1))
    patience = int(_cfg("training", "early_stopping_patience", 20))
    checkpoint_interval = int(_cfg("training", "checkpoint_interval", 10))
    seed = int(_cfg("experiment", "seed", 23))

    clip_ids, labels = list_clips(h5_file, dataset_name=dataset_name)
    print(f"[v120] {len(clip_ids)} clips cargados de {dataset_name} ({h5_file}).")

    max_samples = _cfg("data", "max_samples", None)
    if max_samples and int(max_samples) < len(clip_ids):
        rng = random.Random(seed)
        idx = list(range(len(clip_ids)))
        rng.shuffle(idx)
        keep = sorted(idx[: int(max_samples)])
        clip_ids = [clip_ids[i] for i in keep]
        labels = [labels[i] for i in keep]
        print(f"[v120] Subset (smoke): {len(clip_ids)} clips (max_samples={max_samples}).")

    train_idx, val_idx = stratified_split(clip_ids, labels, n_val_per_class, seed)
    train_labels = sorted(set(labels[i] for i in train_idx))
    label_to_idx = {lbl: i for i, lbl in enumerate(train_labels)}
    num_classes = len(label_to_idx)
    # Una clase con menos de n_val_per_class+1 ejemplos manda todos sus
    # ejemplos a val y ninguno a train (stratified_split) -- con dataset1
    # (exactamente 50/clase) esto no ocurre nunca, pero sí puede pasar con
    # subsets chicos (smoke run); esas clases no se pueden evaluar (el
    # modelo nunca las vio), así que se excluyen de val en vez de crashear.
    val_idx = [i for i in val_idx if labels[i] in label_to_idx]
    print(f"[v120] {num_classes} clases | train={len(train_idx)} val={len(val_idx)}")

    train_ds = IsolatedKeypointDataset(
        h5_file, [clip_ids[i] for i in train_idx], [labels[i] for i in train_idx],
        dataset_name=dataset_name, n_keypoints=n_keypoints, augment=True,
    )
    val_ds = IsolatedKeypointDataset(
        h5_file, [clip_ids[i] for i in val_idx], [labels[i] for i in val_idx],
        dataset_name=dataset_name, n_keypoints=n_keypoints, augment=False,
    )

    collate = functools.partial(isolated_collate_fn, label_to_idx=label_to_idx)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = IsolatedSignClassifier(
        A=A, input_size=n_keypoints,
        gcn_channels=tuple(model_cfg.get("gcn_channels", [32, 64, 128])),
        hidden_size=model_cfg.get("hidden_size", 128),
        num_classes=num_classes,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[v120] IsolatedSignClassifier: {n_params:.2f} M params entrenables")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    writer = SummaryWriter(
        f"../outputs/reports/{version}/{run_id}/{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
    ckpt = CheckpointManager("../outputs/checkpoints", version, run_id)
    import json
    Path(ckpt._path()).mkdir(parents=True, exist_ok=True)
    with open(Path(ckpt._path()) / "label_to_idx.json", "w") as f:
        json.dump(label_to_idx, f, indent=2, ensure_ascii=False)

    stopper = EarlyStopping(patience=patience, threshold=1e-4, verbose=True)
    best_top1 = 0.0
    for epoch in range(epochs):
        model.train()
        tot, n_steps = 0.0, 0
        for keypoints, target in tqdm(train_dl, desc=f"ep{epoch} train", leave=False, mininterval=10.0):
            target = target.to(device)
            logits = _forward_batch(model, keypoints, device)
            loss = criterion(logits, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            n_steps += 1
        train_loss = tot / max(1, n_steps)

        model.eval()
        val_tot, vb = 0.0, 0
        top1_hits, top5_hits, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for keypoints, target in val_dl:
                target = target.to(device)
                logits = _forward_batch(model, keypoints, device)
                loss = criterion(logits, target)
                val_tot += loss.item()
                vb += 1
                bs = target.size(0)
                top1_hits += top_k_accuracy(logits, target, k=1) * bs
                top5_hits += top_k_accuracy(logits, target, k=5) * bs
                n_val += bs
        val_loss = val_tot / max(1, vb)
        val_top1 = top1_hits / max(1, n_val)
        val_top5 = top5_hits / max(1, n_val)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/val_top1", val_top1, epoch)
        writer.add_scalar("Acc/val_top5", val_top5, epoch)
        tqdm.write(f"ep{epoch:>3} | train {train_loss:.3f} | val {val_loss:.3f} | "
                   f"top1 {val_top1:.1%} | top5 {val_top5:.1%}")

        if val_top1 > best_top1:
            best_top1 = val_top1
            ckpt.save_checkpoint(model, epoch, opt, None, tag="best_top1")
            tqdm.write(f"  ↑ best top1: {best_top1:.1%} (ep {epoch})")
        if (epoch + 1) % checkpoint_interval == 0:
            ckpt.save_checkpoint(model, epoch, opt, None)

        stopper(-val_top1, epoch)  # EarlyStopping minimiza -> -top1 para maximizar accuracy
        if stopper.stop:
            tqdm.write(f"Early stopping: sin mejora de top1 en {patience} épocas.")
            break

    writer.close()
    print(f"[v120] FIN. Mejor top1 val = {best_top1:.1%} (azar = {1/num_classes:.1%}).")


if __name__ == "__main__":
    main()
