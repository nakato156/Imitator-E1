from settings import initialize
initialize()
import torch

import random

torch.manual_seed(23)
random.seed(23)

from src.mslm.utils.setup_train import setup_paths
from src.mslm.utils import create_dataloaders, build_model, run_training, prepare_datasets, check_checkpoint
from src.mslm.utils.config_loader import cfg

def run(
    epochs: int,
    batch_size: int,
    batch_sample: int,
    checkpoint_interval: int,
    log_interval: int,
    train_ratio: float = 0.8,
    key_points: int = 111,
    batch_sampling: bool = True,
):   
    _, _, h5_file = setup_paths()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- config de entrenamiento ---
    training_cfg:dict = cfg.training
    model_cfg = cfg.model
    
    train_ratio = training_cfg.get("train_ratio", train_ratio)
    training_cfg.update({
        "epochs": epochs if epochs else training_cfg.get("epochs", 100),
        "batch_size": batch_size if batch_size else training_cfg.get("batch_size", 32),
        "batch_sample": batch_sample if batch_sample else training_cfg.get("sub_batch_size", 32),
        "checkpoint_interval": checkpoint_interval if checkpoint_interval else training_cfg.get("checkpoint_interval", 5),
        "log_interval": log_interval if log_interval else training_cfg.get("log_interval", 2),
        "train_ratio": train_ratio,
        "validation_ratio": round(1 - train_ratio, 2),
        "device": device if model_cfg.get("device") == "auto" else model_cfg.get("device", device),
        "n_keypoints": key_points,
    })
    print(training_cfg)


    if batch_sampling:
        if batch_size%batch_sample != 0 or batch_size < batch_sample:
            raise ValueError(f"The sub_batch {batch_sample} needs to be divisible the batch size {batch_size}")

    training_cfg["batch_sampling"] = batch_sampling
    training_cfg["batch_sample"] = batch_sample
    training_cfg["compile"] = False

    # --- config del experimento (opcional): datasets a incluir + diagnóstico de colapso ---
    data_cfg = getattr(cfg, "data", None)
    include_datasets = data_cfg.get("primary_datasets") if data_cfg else None
    diag_cfg = getattr(cfg, "diagnostics", None)
    if diag_cfg is not None:
        training_cfg["diagnostics"] = dict(diag_cfg)
    sigreg_cfg = getattr(cfg, "sigreg", None)
    if sigreg_cfg is not None:
        training_cfg["sigreg"] = dict(sigreg_cfg)
    loss_cfg = getattr(cfg, "loss", None)
    if loss_cfg is not None:
        training_cfg["loss"] = dict(loss_cfg)
    # CE-vocab (v115) y CE-AR (v116) necesitan los token IDs en los batches.
    return_token_ids = bool(loss_cfg) and loss_cfg.get("type") in ("ce_vocab", "ce_ar")

    print(f"Batch size: {batch_size}, batch sample: {batch_sample}")
    print(f"using dataset {h5_file} | include_datasets={include_datasets} | token_ids={return_token_ids}")
    tr_ds, val_ds, _, _ = prepare_datasets(h5_file, train_ratio, key_points, include_datasets=include_datasets, return_token_ids=return_token_ids)
    tr_dl, val_dl = create_dataloaders(tr_ds, val_ds, batch_size, num_workers=10)

    model = build_model(**model_cfg)
    # v116 ce_ar: wrap Imitator in PrefixImitator to add the LayerNorm+Linear adapter
    # that projects from the Imitator's embedding space into Gemma's embedding space.
    if bool(loss_cfg) and loss_cfg.get("type") == "ce_ar":
        from src.mslm.models.imitator import PrefixImitator
        model = PrefixImitator(model)
        print(f"[v116] Modelo envuelto en PrefixImitator: {sum(p.numel() for p in model.parameters())/1e6:.2f} M params totales")
    load_previous = check_checkpoint(model, training_cfg)
    training_cfg["load_previous_model"] = load_previous

    run_training(training_cfg, tr_dl, val_dl, model)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--epochs", type=int, default=0, help="Number of epochs to train (0 = use config TOML).")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size (0 = use config TOML).")
    parser.add_argument("--batch_sample", type=int, default=0, help="Sub-batch size (0 = use config TOML).")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Interval for saving checkpoints.")
    parser.add_argument("--log_interval", type=int, default=2, help="Interval for logging training progress.")
    parser.add_argument("--num_keypoints", type=int, default=111, help="Number of keypoints to use in the model.")
    parser.add_argument("--batch_sampling", type=bool, default=False, help="Enables batch sampling for training.")
    args = parser.parse_args()

    run(args.epochs, args.batch_size, args.batch_sample, args.checkpoint_interval, args.log_interval, args.batch_sampling, args.num_keypoints)