"""v119 — CTC sobre secuencia (reemplaza el objetivo contrastivo global de v118).

Tres formulaciones contrastivas (InfoNCE, InfoNCE+aug, VICReg) convergieron al
mismo techo (R@1 ≈ 2.5-2.9%, train≈val -> no es sobreajuste, ver
outputs/diag_v118_train_val_gap.json). La causa no es la pérdida ni los datos,
sino comprimir la frase entera en un vector y rankearla. Aquí el encoder emite
una distribución por paso temporal y CTC aprende el alineamiento implícito
contra la secuencia de palabras -- igual que LiftSign (CVPRW 2026) formula CSLR.

La primera versión de v119 (1 STGCNBlock + Transformer + 1 sola cabeza CTC)
colapsó a blank tanto en 1000 como en 5600 clips reales (ver report.md). La
arquitectura actual está PORTADA de Min et al. ("A Closer Look at Skeleton-based
CSLR", ICCVW 2025) / LiftSign: GCN multi-capa -> TCN (K3-P2-K3-P2, P2=TLP) ->
BiLSTM -> clasificador compartido con supervisión CTC dual (Y_s sobre el TCN,
Y_l sobre el BiLSTM) -- exactamente la respuesta que esa literatura da al mismo
síntoma (el módulo de contexto largo no generaliza con pocos datos).

Vocabulario propio (word-level, NO el BPE de Gemma): se construye desde las
labels de TRAIN únicamente (evita fuga val->vocab), blank=0 (convención CTC).

use_motion_stream (flag en [model] del toml) sigue siendo el único toggle de
ablation -- Min et al. Table 6/8 muestran que fusionar motion+skeleton da una
mejora adicional pero menor que el resto de la arquitectura.

Uso:
    MSLM_EXPERIMENT_CONFIG=experiments/v119_ctc/ctc_v119.toml \
        PYTHONPATH=. python scripts/train/train_ctc_v119.py
"""
import os

os.environ.setdefault("MSLM_EXPERIMENT_CONFIG", "experiments/v119_ctc/ctc_v119.toml")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from settings import initialize

initialize()

import functools
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Orden de imports importante: utils.setup_train ANTES de models.ctc_encoder.
# src/mslm/models/__init__.py <-> src/mslm/utils/__init__.py tienen un import
# circular real (models importa utils.early_stopping, utils importa setup_train,
# que importa `from src.mslm.models import Imitator`); cargar utils primero deja
# que `models` termine de inicializarse antes de que setup_train lo necesite. Es
# el mismo orden que ya usa scripts/train/train_contrastive_v118.py.
from src.mslm.utils.setup_train import setup_paths
from src.mslm.utils.config_loader import cfg
from src.mslm.utils.wer import greedy_ctc_decode, word_error_rate
from src.mslm.models.ctc_encoder import CTCEncoder
from src.mslm.dataloader import KeypointDataset, BatchSampler
from src.mslm.dataloader.vocab import Vocab, collect_labels, tokenize
from src.mslm.training.loss_ctc import ctc_loss
from src.mslm.checkpoint.manager import CheckpointManager

torch.manual_seed(23)
random.seed(23)


def _cfg(section, key, default):
    s = getattr(cfg, section, {}) or {}
    return s.get(key, default)


def _truncate(keypoint, frames_mask, max_frames):
    if max_frames and keypoint.size(1) > max_frames:
        keypoint = keypoint[:, :max_frames]
        frames_mask = frames_mask[:, :max_frames]
    return keypoint, frames_mask


def ctc_collate_fn(batch, vocab: Vocab):
    """Pad-ea keypoints igual que components.collate_fn y arma los targets
    concatenados que pide nn.CTCLoss a partir de las labels crudas."""
    keypoints_list = [item[0] for item in batch]
    labels = [item[2] for item in batch]

    frame_lengths = torch.tensor([kp.size(0) for kp in keypoints_list], dtype=torch.long)
    keypoints_padded = pad_sequence(keypoints_list, batch_first=True, padding_value=0.0).float()

    B, T_max = keypoints_padded.shape[:2]
    arange_frames = torch.arange(T_max).unsqueeze(0).expand(B, -1)
    frames_mask = arange_frames >= frame_lengths.unsqueeze(1)

    target_ids = [torch.tensor(vocab.encode(lbl), dtype=torch.long) for lbl in labels]
    target_lengths = torch.tensor([len(t) for t in target_ids], dtype=torch.long)
    targets_concat = torch.cat(target_ids)

    return keypoints_padded, frames_mask.bool(), targets_concat, target_lengths, labels


def _pad_cat(tensors):
    max_t = max(t.size(1) for t in tensors)
    padded = [F.pad(t, (0, 0, 0, max_t - t.size(1))) for t in tensors]
    return torch.cat(padded, dim=0)


def _encode_batch(model, keypoint, frames_mask, grad_ckpt: bool):
    """Codifica cada vídeo del batch por separado (con gradient checkpointing),
    igual que encode_video_batch en train_contrastive_v118.py: un forward batcheado
    del STGCN sobre frames de padding (T_max compartido por todo el batch) agota la
    VRAM con vídeos largos -- mismo problema que motivó ese patrón en v118. Procesar
    cada muestra a su longitud REAL (sin padding) evita ese desperdicio de memoria.
    """
    B = keypoint.size(0)
    lengths = (~frames_mask).sum(dim=1)
    short_list, long_list, seq_lengths_list, aux_list = [], [], [], []
    for i in range(B):
        L = int(lengths[i].item())
        kp = keypoint[i : i + 1, :L]
        fm = torch.zeros(1, L, dtype=torch.bool, device=keypoint.device)
        if grad_ckpt and model.training:
            lp_s, lp_l, sl, aux = checkpoint(model, kp, fm, use_reentrant=False)
        else:
            lp_s, lp_l, sl, aux = model(kp, fm)
        short_list.append(lp_s)
        long_list.append(lp_l)
        seq_lengths_list.append(sl)
        aux_list.append(aux)

    log_probs_short = _pad_cat(short_list)
    log_probs_long = _pad_cat(long_list)
    seq_lengths = torch.cat(seq_lengths_list)
    aux = {k: sum(a[k] for a in aux_list) / len(aux_list) for k in aux_list[0]}
    return log_probs_short, log_probs_long, seq_lengths, aux


def _forward_batch(model, keypoint, frames_mask, targets, target_lengths, device, max_frames,
                    grad_ckpt: bool = True):
    keypoint, frames_mask = keypoint.to(device), frames_mask.to(device)
    keypoint, frames_mask = _truncate(keypoint, frames_mask, max_frames)
    log_probs_short, log_probs_long, seq_lengths, aux = _encode_batch(
        model, keypoint, frames_mask, grad_ckpt)
    targets, target_lengths = targets.to(device), target_lengths.to(device)
    loss_short = ctc_loss(log_probs_short, targets, seq_lengths, target_lengths)
    loss_long = ctc_loss(log_probs_long, targets, seq_lengths, target_lengths)
    loss = loss_short + loss_long
    return log_probs_short, log_probs_long, seq_lengths, aux, loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, h5_file = setup_paths()
    from src.mslm.utils.paths import path_vars

    adj_path = path_vars.data_path / "processed" / "adjacency_matrix.npy"
    A = np.load(adj_path, allow_pickle=True)

    model_cfg = dict(cfg.model)
    use_motion_stream = bool(model_cfg.get("use_motion_stream", False))
    lambda_u = float(_cfg("loss", "lambda_u", 0.1))
    lambda_p = float(_cfg("loss", "lambda_p", 0.1))

    epochs = int(_cfg("training", "epochs", 60))
    batch_size = int(_cfg("training", "batch_size", 16))
    lr = float(_cfg("training", "learning_rate", 3e-4))
    wd = float(_cfg("training", "weight_decay", 0.05))
    warmup = int(_cfg("training", "warmup_epochs", 5))
    max_frames = int(_cfg("training", "max_frames", 1024))
    version = int(_cfg("training", "model_version", 119))
    run_id = int(_cfg("training", "run_id", 1))
    patience = int(_cfg("training", "early_stopping_patience", 25))

    include = _cfg("data", "primary_datasets", ["dataset2"])
    train_ratio = float(_cfg("data", "train_ratio", 0.8))
    text_group = _cfg("data", "text_group", "text_ctx")
    n_keypoints = int(_cfg("data", "n_keypoints", model_cfg.get("input_size", 111)))
    max_samples = _cfg("data", "max_samples", None)

    # Palancas data-oriented (v119, "Less is More" generalizado a la restricción
    # real de nn.CTCLoss en esta arquitectura): filtro de calidad + augmentation.
    # Ver report.md para el detalle del criterio y el bug de TransformedSubset
    # que motivó probarlos recién ahora.
    min_frames = int(_cfg("data", "min_frames", 16))
    filter_invalid_labels = bool(_cfg("data", "filter_invalid_labels", True))
    data_augmentation = bool(_cfg("data", "data_augmentation", True))

    with h5py.File(h5_file, "r") as f:
        raw_clip_count = sum(len(f[d]["embeddings"].keys()) for d in include if d in f)

    ds = KeypointDataset(
        h5Path=h5_file, n_keypoints=n_keypoints, return_label=True,
        text_group=text_group, include_datasets=include,
        data_augmentation=data_augmentation, max_length=4000,
        min_frames=min_frames, filter_invalid_labels=filter_invalid_labels,
    )
    print(f"[v119] Filtro de calidad: {raw_clip_count - len(ds.valid_index)} de {raw_clip_count} "
          f"clips excluidos (min_frames={min_frames}, filter_invalid_labels={filter_invalid_labels}).")

    if max_samples and int(max_samples) < len(ds.valid_index):
        # Muestra aleatoria determinista (seed=23): dataset2 mezcla los 600 clips
        # originales con los 5000 añadidos en v118f bajo el mismo índice secuencial
        # (verificado: las claves "0".."599" NO corresponden a los 600 originales,
        # su media de frames no coincide con la documentada en report.md), así que
        # no hay forma de recuperar esa identidad por id -- se usa un subconjunto
        # aleatorio del tamaño pedido en vez de asumir una identidad incorrecta.
        rng = random.Random(23)
        idx = list(range(len(ds.valid_index)))
        rng.shuffle(idx)
        keep = sorted(idx[: int(max_samples)])
        ds.valid_index = [ds.valid_index[i] for i in keep]
        ds.video_lengths = [ds.video_lengths[i] for i in keep]
        ds.dataset_length = len(ds.valid_index)
        print(f"[v119] Subset aleatorio: {ds.dataset_length} clips (max_samples={max_samples}).")

    train_subset, val_subset, train_lengths, val_length = ds.split_dataset(train_ratio)

    # Vocab SOLO con labels de train (evita fuga val->vocab). Con
    # data_augmentation=True, split_dataset() devuelve un ConcatDataset (la
    # copia original + 4 aumentadas) que no tiene `.indices` -- el subconjunto
    # original (sin augmentation) siempre es el primer elemento de `.datasets`.
    base_train_subset = train_subset.datasets[0] if hasattr(train_subset, "datasets") else train_subset
    train_clip_ids = [ds.valid_index[i][1] for i in base_train_subset.indices]
    train_labels = collect_labels(h5_file, include[0], train_clip_ids)
    vocab = Vocab.build_from_labels(train_labels)
    print(f"[v119] Vocab: {len(vocab)} tokens (incl. blank+unk) desde {len(train_labels)} labels de train.")

    collate = functools.partial(ctc_collate_fn, vocab=vocab)
    train_dl = DataLoader(
        train_subset, num_workers=8, pin_memory=True, persistent_workers=True,
        collate_fn=collate, batch_sampler=BatchSampler(train_subset, batch_size, lengths=train_lengths),
    )
    val_dl = DataLoader(
        val_subset, num_workers=8, pin_memory=True, persistent_workers=True,
        collate_fn=collate, batch_sampler=BatchSampler(val_subset, batch_size, lengths=val_length),
    )

    # CTCEncoder.vocab_size = clases SIN contar blank (suma +1 internamente para
    # blank=0); Vocab ya reserva el id 0 para blank, así que restamos 1 para que
    # el índice 0 del clasificador siga correspondiendo a blank.
    model = CTCEncoder(
        A=A, input_size=model_cfg.get("input_size", n_keypoints),
        gcn_channels=tuple(model_cfg.get("gcn_channels", [32, 64, 128])),
        hidden_size=model_cfg.get("hidden_size", 256),
        lstm_layers=model_cfg.get("lstm_layers", 2),
        vocab_size=len(vocab) - 1,
        use_motion_stream=use_motion_stream,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[v119] CTCEncoder: {n_params:.2f} M params entrenables | "
          f"use_motion_stream={use_motion_stream} (GCN+TCN/TLP+BiLSTM+CTC dual) "
          f"batch={batch_size} max_frames={max_frames}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, len(train_dl))

    # Step-decay /10 en lr_decay_epochs (default 20/35) -- portado literal de
    # Min et al. §4.1 ("reduced by a factor of 10 at 20 and 35 epochs"), no un
    # cosine inventado: es parte de la metodología que ya funciona (WER
    # 4.6%/41.0% en Isharah), no solo la arquitectura.
    lr_decay_epochs = _cfg("training", "lr_decay_epochs", [20, 35])

    def lr_at(step):
        warm = warmup * steps_per_epoch
        if step < warm:
            return step / max(1, warm)
        epoch_now = step / steps_per_epoch
        factor = 1.0
        for de in lr_decay_epochs:
            if epoch_now >= de:
                factor *= 0.1
        return factor

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    writer = SummaryWriter(
        f"../outputs/reports/{version}/{run_id}/{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
    ckpt = CheckpointManager("../outputs/checkpoints", version, run_id)
    vocab.save(Path(ckpt._path()) / "vocab.json")

    best_wer, since_improve, gstep = float("inf"), 0, 0
    for epoch in range(epochs):
        # ---- train ----
        model.train()
        tot, n_steps = 0.0, 0
        for keypoint, frames_mask, targets, target_lengths, _ in tqdm(
            train_dl, desc=f"ep{epoch} train", leave=False, mininterval=10.0
        ):
            _, _, _, aux, loss = _forward_batch(
                model, keypoint, frames_mask, targets, target_lengths, device, max_frames)
            loss = loss + lambda_u * aux["L_u"] + lambda_p * aux["L_p"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            gstep += 1
            tot += loss.item()
            n_steps += 1
        train_loss = tot / max(1, n_steps)

        # ---- val: WER sobre TODO el val (greedy decode), ambas cabezas ----
        # Min et al. (ICCVW 2025) §5.3: el módulo de contexto largo (Y_l,
        # BiLSTM) generaliza peor que el de corto plazo (Y_s, TCN) con pocos
        # datos de train -- eligen la cabeza según la tarea en vez de
        # hardcodear largo plazo. Acá se decodifican ambas y se selecciona
        # checkpoint por la mejor de las dos, en vez de descartar Y_s.
        model.eval()
        val_tot, vb = 0.0, 0
        hyp_short_all, hyp_long_all, ref_words_all = [], [], []
        with torch.no_grad():
            for keypoint, frames_mask, targets, target_lengths, labels in val_dl:
                log_probs_short, log_probs_long, seq_lengths, aux, loss = _forward_batch(
                    model, keypoint, frames_mask, targets, target_lengths, device, max_frames)
                loss = loss + lambda_u * aux["L_u"] + lambda_p * aux["L_p"]
                val_tot += loss.item()
                vb += 1

                decoded_short = greedy_ctc_decode(log_probs_short.cpu(), seq_lengths.cpu(), blank=vocab.blank_id)
                decoded_long = greedy_ctc_decode(log_probs_long.cpu(), seq_lengths.cpu(), blank=vocab.blank_id)
                for ids_s, ids_l, label in zip(decoded_short, decoded_long, labels):
                    hyp_short_all.append(vocab.decode(ids_s))
                    hyp_long_all.append(vocab.decode(ids_l))
                    ref_words_all.append(tokenize(label))
        val_loss = val_tot / max(1, vb)
        wers_short = [word_error_rate(h, r) for h, r in zip(hyp_short_all, ref_words_all) if r]
        wers_long = [word_error_rate(h, r) for h, r in zip(hyp_long_all, ref_words_all) if r]
        wer_short = sum(wers_short) / max(1, len(wers_short))
        wer_long = sum(wers_long) / max(1, len(wers_long))
        val_wer = min(wer_short, wer_long)
        best_head = "short" if wer_short <= wer_long else "long"

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("WER/val_short", wer_short, epoch)
        writer.add_scalar("WER/val_long", wer_long, epoch)
        tqdm.write(f"ep{epoch:>3} | train {train_loss:.3f} | val {val_loss:.3f} | "
                   f"WER short {wer_short:.1%} | WER long {wer_long:.1%} | best={best_head}")

        improved = val_wer < best_wer
        if improved:
            best_wer, since_improve = val_wer, 0
            ckpt.save_checkpoint(model, epoch, opt, sched, tag="best_wer")
            tqdm.write(f"  ↓ best WER: {best_wer:.1%} (ep {epoch}, cabeza={best_head})")
        else:
            since_improve += 1
        if (epoch + 1) % int(_cfg("training", "checkpoint_interval", 10)) == 0:
            ckpt.save_checkpoint(model, epoch, opt, sched)
        if since_improve >= patience:
            tqdm.write(f"Early stopping: sin mejora de WER en {patience} épocas.")
            break

    writer.close()
    print(f"[v119] FIN. Mejor WER val = {best_wer:.1%}.")


if __name__ == "__main__":
    main()
