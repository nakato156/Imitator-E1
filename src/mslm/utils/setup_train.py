import os
import torch
import random
from io import StringIO

torch.manual_seed(23)
random.seed(23)

from torch.utils.data import DataLoader
import numpy as np

#Imported Classes
from src.mslm.models import Imitator, SignAlignGCN_KeypointsOnly, AlignmentDecoder, KeypointEncoder
from src.mslm.training import Trainer
from src.mslm.dataloader import KeypointDataset, collate_fn, GRPCDataset, BatchSampler
from src.mslm.utils.paths import path_vars

#Profilers
from torch.profiler import profile, ProfilerActivity, record_function
import datetime

def setup_paths():
    """Define y retorna las rutas necesarias para datos y modelos."""
    data_path = path_vars.data_path
    model_path = path_vars.model_path
    h5_file = path_vars.h5_file
    return data_path, model_path, h5_file

def prepare_datasets(h5File, train_ratio, n_keypoints=111, include_datasets=None, return_token_ids=False, text_group="embeddings"):
    """Carga el dataset base, lo envuelve y lo divide en entrenamiento y validación.

    include_datasets: lista opcional de grupos del HDF5 a usar (p.ej. ["dataset2"] para
    el experimento SIGReg). Si es None se usan los datasets por defecto.
    return_token_ids: incluye los token IDs en los batches (experimento CE-vocab, v115).
    text_group: grupo HDF5 de donde leer el embedding de texto ("embeddings" = tabla de
    entrada de Gemma per-token; "text_ctx" = último hidden state contextual, v118e).
    """
    keypoint_reader = KeypointDataset(h5Path=h5File, return_label=False, n_keypoints=n_keypoints, data_augmentation=False, max_length=4000, include_datasets=include_datasets, return_token_ids=return_token_ids, text_group=text_group)
    train_dataset, validation_dataset, train_length, val_length = keypoint_reader.split_dataset(train_ratio)

    print(f"Train size:\t{len(train_dataset)}\nValidation size:\t{len(validation_dataset)}")
    return train_dataset, validation_dataset, train_length, val_length

def create_dataloaders(train_dataset, validation_dataset, batch_size, num_workers=4, use_grpc=False, grpc_address=None, rank = 4, world_size = 4):
    """Crea y retorna los DataLoaders para entrenamiento y validación."""
    if use_grpc:
        gtrain_dataset = GRPCDataset(grpc_address, rank, world_size, split="train")
        gval_dataset = GRPCDataset(grpc_address, rank, world_size, split="val")
        
        batch_per_worker = int(batch_size/world_size)

        train_dataloader = DataLoader(gtrain_dataset,
        batch_size=batch_per_worker,           
        num_workers=4,           
        pin_memory=True)

        val_dataloader = DataLoader(gval_dataset,
        batch_size=batch_per_worker,           
        num_workers=4,           
        pin_memory=True)

    else:
        train_sampler = BatchSampler(train_dataset, batch_size)
        val_sampler   = BatchSampler(validation_dataset, batch_size)

        train_dataloader = DataLoader(
            train_dataset,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler
        )
        val_dataloader = DataLoader(
            validation_dataset,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            batch_sampler=val_sampler
        )
    return train_dataloader, val_dataloader

def build_model(input_size, output_size, hidden_size=512, nhead=8, ff_dim=1024, n_layers=2,
                encoder_dropout=0.4, multihead_dropout=0.1, pool_dim=256, max_seq_length=20,
                **kwargs):
    """Construye y retorna el modelo Imitator (STGCN + Transformer + cross-attn).

    La matriz de adyacencia (111x111) se genera con scripts/data/build_adjacency.py y se carga
    desde data/processed/ (ruta derivada de path_vars, no hardcodeada).
    """
    adj_path = path_vars.data_path / "processed" / "adjacency_matrix.npy"
    A = np.load(adj_path, allow_pickle=True)

    model = Imitator(
        A=A,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        nhead=nhead,
        ff_dim=ff_dim,
        n_layers=n_layers,
        max_seq_length=max_seq_length,
        encoder_dropout=encoder_dropout,
        multihead_dropout=multihead_dropout,
        pool_dim=pool_dim,
    )
    print(f"Imitator: {sum(p.numel() for p in model.parameters())/1e6:.2f} M parámetros | "
          f"adyacencia {A.shape} desde {adj_path}")
    return model

def check_checkpoint(model, params):
    """
    Verifica que el checkpoint pedido sea válido.
    """
    checkpoint = int(params["checkpoint"])
    version   = str(params["model_version"])

    root      = os.path.join(path_vars._base_path.parent / "outputs" / "checkpoints", version)

    if checkpoint == 1:
        return False

    cp_path = os.path.join(root, str(checkpoint))
    if os.path.exists(cp_path):
        raise ValueError(f"El checkpoint {checkpoint} ya existe en {cp_path!r}.")

    prev_path = os.path.join(root, str(checkpoint - 1))
    if not os.path.exists(prev_path):
        raise ValueError(f"Checkpoint anterior {checkpoint-1} no encontrado en {prev_path!r}.")

    arch_file = os.path.join(root, str(1), "model_architecture.txt")
    if not os.path.isfile(arch_file):
        raise FileNotFoundError(f"No se encontró {arch_file!r}.")

    with open(arch_file, "r") as f:
        saved_arch = f.read()

    buf = StringIO()
    print(model, file=buf)
    current_arch = buf.getvalue()

    if saved_arch != current_arch:
        raise ValueError("La arquitectura del modelo no coincide con la guardada.")
    
    return True

def run_training(params, train_dataloader, val_dataloader, model):
    """Configura y ejecuta el entrenamiento."""
    print("Training Parameters: ", params)

    trainer = Trainer(model, train_dataloader, val_dataloader,save_tb_model=False , **params)
    trainer.ckpt_mgr.save_params(params)
    trainer.ckpt_mgr.save_model_architecture(model)

    print("Starting training...")
    return trainer.train()

def profile_training(params, train_dataloader, val_dataloader, model, profile_mode: str):
    trainer = Trainer(model, train_dataloader, val_dataloader,save_tb_model=False, **params)
    trainer.ckpt_mgr.save_params(params)
    trainer.ckpt_mgr.save_model_architecture(model)

    if profile_mode == "nvidia":
        print("Starting training with profiling nvidia...")
        return trainer.train(prof=True)
    elif profile_mode == "pytorch_model":
        print("Starting training with profiling pytorch...")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=True) as p:
            with record_function("train"):
                trainer.train()
            file_path = f"{path_vars.report_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"Saved at: {file_path}")    
            p.export_chrome_trace(f"{file_path}.json.gz")
            p.export_stacks(f"{file_path}_stacks.txt", "self_cpu_time_total")
    elif profile_mode == "pytorch_memory":
        print("Starting training with profiling memory torch...")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=True,
                    profile_memory=True) as p:
            with record_function("train"):
                trainer.train()
            file_path = f"{path_vars.report_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"Saved at: {file_path}")    
            p.export_chrome_trace(f"{file_path}.json.gz")
            p.export_stacks(f"{file_path}_stacks.txt", "self_cpu_time_total")
            p.export_memory_profile(f"{file_path}_memory.txt")
    else:
        raise ValueError("Unsupported profiling mode. Use 'nvidia' or 'pytorch'.")

    def run_dt_training(params, train_dataloader, val_dataloader, model, rank, channel, dist, stub):
        """Configura y ejecuta el entrenamiento."""
        trainer = Trainer(model, train_dataloader, val_dataloader,save_tb_model=False, **params)
        trainer.ckpt_mgr.save_params(params)
        trainer.ckpt_mgr.save_model_architecture(model)

        print("Starting training...")
        return trainer.train_dist(rank, channel, dist, stub)
