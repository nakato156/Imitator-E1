"""v118 — Pre-entrenamiento contrastivo del encoder (CLIP-style), SIN LLM.

Alinea el embedding del vídeo (PrefixImitator) con el de la frase (embeddings de
Gemma ya cacheados en el HDF5) mediante InfoNCE simétrico. Selecciona el checkpoint
por retrieval@1 sobre val — la métrica que mide grounding directamente.

Uso:
    MSLM_EXPERIMENT_CONFIG=experiments/v118_contrastive/contrastive_v118.toml \
        PYTHONPATH=. python scripts/train/train_contrastive_v118.py

Contexto: la línea CE-AR soft-prefix (v116/v117) producía español fluido sin
traducir (retr@1=0%). Esta etapa valida de forma barata si el encoder puede
aterrizar seña con 600 clips antes de pagar el coste de una etapa generativa.
"""
import os
os.environ.setdefault("MSLM_EXPERIMENT_CONFIG", "experiments/v118_contrastive/contrastive_v118.toml")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from settings import initialize
initialize()

import math
import random
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.mslm.utils.setup_train import setup_paths, build_model, prepare_datasets, create_dataloaders
from src.mslm.utils.config_loader import cfg
from src.mslm.models.imitator import PrefixImitator
from src.mslm.models.contrastive import ContrastiveAligner
from src.mslm.training.loss_contrastive import clip_contrastive_loss, vicreg_loss, retrieval_metrics
from src.mslm.dataloader.augmentations import make_augment_fn
from src.mslm.checkpoint.manager import CheckpointManager

torch.manual_seed(23)
random.seed(23)


def _cfg(section, key, default):
    s = getattr(cfg, section, {}) or {}
    return s.get(key, default)


def _truncate(keypoint, frames_mask, max_frames):
    """Recorta la dimensión temporal para acotar memoria del STGCN."""
    if max_frames and keypoint.size(1) > max_frames:
        keypoint = keypoint[:, :max_frames]
        frames_mask = frames_mask[:, :max_frames]
    return keypoint, frames_mask


def encode_video_batch(model, keypoint, frames_mask, grad_ckpt: bool,
                       augment_fn=None, normalize: bool = True):
    """Encodea cada vídeo por separado (con gradient checkpointing) y apila los
    vectores proyectados, de modo que todos los negativos del batch viven en el
    grafo sin necesidad de un forward conjunto que haría OOM con vídeos largos.
    Si augment_fn != None se aplica ANTES de encodear (solo en train).
    normalize=False entrega la proyección cruda (necesaria para VICReg)."""
    vecs = []
    B = keypoint.size(0)
    for i in range(B):
        kp = keypoint[i:i + 1]
        fm = frames_mask[i:i + 1]
        if augment_fn is not None:
            kp, fm = augment_fn(kp, fm)
        if grad_ckpt and model.training:
            v = checkpoint(lambda k, f: model.encode_video(k, f, normalize=normalize),
                            kp, fm, use_reentrant=False)
        else:
            v = model.encode_video(kp, fm, normalize=normalize)
        vecs.append(v)
    return torch.cat(vecs, dim=0)                       # [B, d]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, h5_file = setup_paths()

    model_cfg = dict(cfg.model)
    proj_dim = int(_cfg("loss", "proj_dim", 256))
    temperature = float(_cfg("loss", "temperature", 0.07))
    loss_type = _cfg("loss", "type", "infonce")            # "infonce" | "vicreg"
    vicreg_lambda = float(_cfg("loss", "vicreg_lambda_inv", 25.0))
    vicreg_mu = float(_cfg("loss", "vicreg_mu_var", 25.0))
    vicreg_nu = float(_cfg("loss", "vicreg_nu_cov", 1.0))

    epochs = int(_cfg("training", "epochs", 60))
    batch_size = int(_cfg("training", "batch_size", 16))
    lr = float(_cfg("training", "learning_rate", 3e-4))
    wd = float(_cfg("training", "weight_decay", 0.05))
    warmup = int(_cfg("training", "warmup_epochs", 5))
    max_frames = int(_cfg("training", "max_frames", 768))
    grad_ckpt = bool(_cfg("training", "grad_checkpoint_encoder", True))
    version = int(_cfg("training", "model_version", 118))
    run_id = int(_cfg("training", "run_id", 1))
    patience = int(_cfg("training", "early_stopping_patience", 20))

    aug_crop_min = float(_cfg("augmentation", "temporal_crop_min", 1.0))
    aug_noise_std = float(_cfg("augmentation", "coord_noise_std", 0.0))
    augment_fn = None
    if aug_crop_min < 1.0 or aug_noise_std > 0.0:
        augment_fn = make_augment_fn(aug_crop_min, aug_noise_std)
        print(f"[v118] Augmentations: temporal_crop_min={aug_crop_min} coord_noise_std={aug_noise_std}")

    include = _cfg("data", "primary_datasets", ["dataset2"])
    train_ratio = float(_cfg("data", "train_ratio", 0.8))
    text_group = _cfg("data", "text_group", "embeddings")

    tr_ds, val_ds, _, _ = prepare_datasets(
        h5_file, train_ratio, model_cfg.get("input_size", 111),
        include_datasets=include, return_token_ids=False, text_group=text_group)
    tr_dl, val_dl = create_dataloaders(tr_ds, val_ds, batch_size, num_workers=8)

    aligner = ContrastiveAligner(
        PrefixImitator(build_model(**model_cfg)),
        hidden=model_cfg.get("output_size", 2048),
        proj_dim=proj_dim, init_temperature=temperature,
    ).to(device)

    n_params = sum(p.numel() for p in aligner.parameters() if p.requires_grad) / 1e6
    print(f"[v118] ContrastiveAligner: {n_params:.2f} M params entrenables | "
          f"loss={loss_type} proj_dim={proj_dim} batch={batch_size} "
          f"max_frames={max_frames} grad_ckpt={grad_ckpt} text_group={text_group}")

    # logit_scale solo lo usa InfoNCE; excluirlo evita que el weight decay
    # desacoplado de AdamW lo arrastre sin gradiente cuando se usa VICReg.
    if loss_type == "vicreg":
        opt_params = [p for n, p in aligner.named_parameters() if n != "logit_scale"]
    else:
        opt_params = aligner.parameters()
    opt = torch.optim.AdamW(opt_params, lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, len(tr_dl))

    def lr_at(step):
        warm = warmup * steps_per_epoch
        total = epochs * steps_per_epoch
        if step < warm:
            return step / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    writer = SummaryWriter(
        f"../outputs/reports/{version}/{run_id}/{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
    ckpt = CheckpointManager("../outputs/checkpoints", version, run_id)

    best_r1, since_improve, gstep = -1.0, 0, 0
    for epoch in range(epochs):
        # ---- train ----
        aligner.train()
        tot, n_steps = 0.0, 0
        for batch in tqdm(tr_dl, desc=f"ep{epoch} train", leave=False):
            if batch[0].size(0) < 2:
                continue  # var/cov (VICReg) y negativos in-batch (InfoNCE) no existen con B=1
            keypoint, frames_mask = batch[0].to(device), batch[1].to(device)
            text_emb, text_mask = batch[2].to(device), batch[3].to(device)
            keypoint, frames_mask = _truncate(keypoint, frames_mask, max_frames)

            normalize = (loss_type != "vicreg")
            v = encode_video_batch(aligner, keypoint, frames_mask, grad_ckpt,
                                   augment_fn=augment_fn, normalize=normalize)
            t = aligner.encode_text(text_emb, text_mask, normalize=normalize)
            if loss_type == "vicreg":
                loss = vicreg_loss(v, t, vicreg_lambda, vicreg_mu, vicreg_nu)
            else:
                loss = clip_contrastive_loss(v, t, aligner.logit_scale)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aligner.parameters(), 1.0)
            opt.step()
            sched.step()
            gstep += 1
            tot += loss.item()
            n_steps += 1
        train_loss = tot / max(1, n_steps)

        # ---- val: retrieval sobre TODO el val set ----
        aligner.eval()
        V, T, vtot, vb = [], [], 0.0, 0
        normalize = (loss_type != "vicreg")
        with torch.no_grad():
            for batch in val_dl:
                if batch[0].size(0) < 2:
                    continue
                keypoint, frames_mask = batch[0].to(device), batch[1].to(device)
                text_emb, text_mask = batch[2].to(device), batch[3].to(device)
                keypoint, frames_mask = _truncate(keypoint, frames_mask, max_frames)
                v = encode_video_batch(aligner, keypoint, frames_mask, grad_ckpt=False,
                                       normalize=normalize)
                t = aligner.encode_text(text_emb, text_mask, normalize=normalize)
                if loss_type == "vicreg":
                    vtot += vicreg_loss(v, t, vicreg_lambda, vicreg_mu, vicreg_nu).item()
                else:
                    vtot += clip_contrastive_loss(v, t, aligner.logit_scale).item()
                vb += 1
                V.append(v); T.append(t)
        V, T = torch.cat(V), torch.cat(T)
        # retrieval@k necesita vectores L2-normalizados (similitud coseno),
        # independientemente de si la pérdida de entrenamiento los usó crudos.
        V_n = F.normalize(V, dim=-1) if loss_type == "vicreg" else V
        T_n = F.normalize(T, dim=-1) if loss_type == "vicreg" else T
        m = retrieval_metrics(V_n, T_n)
        val_loss = vtot / max(1, vb)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Retrieval/R@1", m["R@1"], epoch)
        writer.add_scalar("Retrieval/R@5", m["R@5"], epoch)
        writer.add_scalar("Retrieval/R@10", m["R@10"], epoch)
        writer.add_scalar("Retrieval/median_rank", m["median_rank"], epoch)
        writer.add_scalar("Misc/temperature", (1.0 / aligner.logit_scale.exp()).item(), epoch)
        tqdm.write(
            f"ep{epoch:>3} | train {train_loss:.3f} | val {val_loss:.3f} | "
            f"R@1 {m['R@1']:.1%} R@5 {m['R@5']:.1%} R@10 {m['R@10']:.1%} "
            f"medR {m['median_rank']:.0f} (chance R@1={m['chance']:.1%})")

        improved = m["R@1"] > best_r1
        if improved:
            best_r1, since_improve = m["R@1"], 0
            ckpt.save_checkpoint(aligner, epoch, opt, sched, tag="best_r1")
            tqdm.write(f"  ↑ best R@1: {best_r1:.1%} (ep {epoch})")
        else:
            since_improve += 1
        if (epoch + 1) % int(_cfg("training", "checkpoint_interval", 10)) == 0:
            ckpt.save_checkpoint(aligner, epoch, opt, sched)
        if since_improve >= patience:
            tqdm.write(f"Early stopping: sin mejora de R@1 en {patience} épocas.")
            break

    writer.close()
    print(f"[v118] FIN. Mejor R@1 = {best_r1:.1%} (chance = {1.0/len(val_ds):.1%}).")


if __name__ == "__main__":
    main()
