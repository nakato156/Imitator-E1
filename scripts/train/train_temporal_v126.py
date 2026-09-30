"""Train v126 synthetic temporal pretraining model."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", ".hf_cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from settings import initialize
from src.mslm.dataloader.isolated_keypoint_dataset import list_clip_records
from src.mslm.dataloader.synthetic_temporal import (
    SyntheticTemporalSignDataset,
    permute_video_segments,
    synthetic_temporal_collate,
)
from src.mslm.inference.imitator_tokens import make_token_predictions
from src.mslm.models.temporal_sign_prompt import (
    CIFAggregator,
    STGCNTemporalFrameEncoder,
    TemporalSignPromptModel,
    alpha_diagnostics,
    alpha_schedule_weights,
    boundary_error_mae,
    clamp_predicted_lengths,
    length_mask_from_lengths,
    load_visual_low_level_weights,
    rescale_alphas_to_predicted_lengths,
    rescale_alphas_to_rounded_count,
    rescale_alphas_to_target_lengths,
    set_cif_diagnostic_freeze,
)
from src.mslm.dataloader.data_augmentation import normalize_augment_data, remove_keypoints
from src.mslm.utils.text_metrics import bleu_score, chrf_score
from src.mslm.utils.sequence_metrics import (
    legacy_prefix_exact,
    strict_exact,
    token_edit_similarity,
    token_error_rate,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha-loss-weight", type=float, default=5.0)
    parser.add_argument("--qty-loss-weight", type=float, default=1.0)
    parser.add_argument("--emb-loss-weight", type=float, default=0.05)
    parser.add_argument("--length-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--max-len-class", type=int, default=16)
    parser.add_argument("--min-clips", type=int, default=2)
    parser.add_argument("--max-clips", type=int, default=8)
    parser.add_argument("--min-neutral-frames", type=int, default=0)
    parser.add_argument("--max-neutral-frames", type=int, default=8)
    parser.add_argument("--prediction-samples", type=int, default=8)
    parser.add_argument(
        "--heldout-signer",
        type=int,
        help="Leave one signer out: train on signer_id != N and validate/evaluate only signer_id == N.",
    )
    parser.add_argument(
        "--exclude-signer",
        type=int,
        default=None,
        help="Outer-test signer for clean LOSO: excluded from both train and val/--heldout-signer "
        "(never loaded by this process). Requires --heldout-signer.",
    )
    parser.add_argument(
        "--token-head",
        choices=["linear", "contextual"],
        default="contextual",
        help="linear: pre-Etapa-2 per-slot Linear (no cross-slot context). "
        "contextual: position embedding + TransformerEncoderLayer (current default).",
    )
    parser.add_argument(
        "--length-head",
        choices=["mean", "attention"],
        default="attention",
        help="mean: pre-Etapa-2 mean-pooled frame summary. "
        "attention: 1-query attention pooling (current default).",
    )
    parser.add_argument(
        "--token-label-smoothing",
        type=float,
        default=0.1,
        help="label_smoothing for the token cross-entropy loss (current default preserves v126b).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for train/val split, independent of --seed (init/sampling). "
        "Defaults to --seed when omitted, preserving existing run reproducibility.",
    )
    parser.add_argument(
        "--prediction-alpha-mode",
        choices=[
            "teacher_alpha",
            "pred_raw",
            "pred_rescaled_to_target_len",
            "pred_rescaled_to_pred_len",
        ],
        default="teacher_alpha",
        help="Alpha source used for predictions.jsonl reports.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--phase",
        choices=["teacher_forced", "learned_cif"],
        default="teacher_forced",
        help="teacher_forced: v126 baseline (alpha=alpha_target always). "
        "learned_cif: v126b, freeze/teacher-forcing curriculum to train CIF.alpha.",
    )
    parser.add_argument("--stgcn-lr-scale", type=float, default=0.1)
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help="Load model weights from --resume but reset optimizer, epoch, and best metric.",
    )
    parser.add_argument(
        "--alpha-schedule",
        choices=["current", "target_only", "linear_pred_mix"],
        default="current",
        help="Override learned-CIF alpha blending schedule. Default preserves v126b.",
    )
    parser.add_argument("--mix-start-epoch", type=int, default=0)
    parser.add_argument("--mix-ramp-epochs", type=int, default=10)
    parser.add_argument("--mix-w-pred-start", type=float, default=0.05)
    parser.add_argument("--mix-w-pred-end", type=float, default=0.15)
    parser.add_argument(
        "--diag-alpha-loss",
        choices=["current", "qty_only", "kl_quantity", "logit_l1"],
        default="current",
        help="Diagnostic CIF-alpha objective. Default preserves the current v126b loss.",
    )
    parser.add_argument(
        "--diag-freeze",
        choices=["alpha_only", "target_only_stage1", "heads_only", "tcn_heads", "full_current"],
        default="full_current",
        help="Diagnostic freezing regime. Default preserves the current v126b schedule.",
    )
    parser.add_argument(
        "--diag-eval-force-count",
        action="store_true",
        help="Use target-length-rescaled predicted CIF for val_predicted/select_metric.",
    )
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("data/processed/dataset1_isolated_v122.hdf5"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(
            ".hf_cache/hub/models--unsloth--gemma-3n-E2B-it-unsloth-bnb-4bit/"
            "snapshots/3d26ffdd2276698f582562bf01511aa625bc6f30"
        ),
    )
    parser.add_argument(
        "--embedding-table",
        type=Path,
        default=Path("data/processed/gemma3n_embed_table.pt"),
    )
    parser.add_argument(
        "--checkpoint-v121",
        type=Path,
        default=Path("../outputs/checkpoints/121/23/best_top1/checkpoint.pth"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../outputs/v126_temporal"),
    )
    return parser.parse_args()


def scheduled_alpha_weights(args, epoch: int) -> tuple[float, float]:
    """Return ``(w_target, w_pred)`` while preserving the historical default."""
    if args.alpha_schedule == "target_only":
        return 1.0, 0.0
    if args.alpha_schedule == "current":
        return alpha_schedule_weights(epoch)

    if epoch < args.mix_start_epoch:
        w_pred = 0.0
    else:
        ramp = max(1, args.mix_ramp_epochs)
        progress = min(1.0, (epoch - args.mix_start_epoch) / ramp)
        w_pred = args.mix_w_pred_start + progress * (
            args.mix_w_pred_end - args.mix_w_pred_start
        )
    w_pred = float(max(0.0, min(1.0, w_pred)))
    return 1.0 - w_pred, w_pred


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).strip().lower()
    return " ".join(text.split())


def stratified_clip_split(records, seed: int, n_val_per_class: int = 10):
    by_label = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)
    rng = random.Random(seed)
    train, val = [], []
    for rows in by_label.values():
        rows = rows[:]
        rng.shuffle(rows)
        val.extend(rows[:n_val_per_class])
        train.extend(rows[n_val_per_class:])
    return train, val


def split_records(
    records,
    seed: int,
    heldout_signer: int | None = None,
    exclude_signer: int | None = None,
):
    if exclude_signer is not None and heldout_signer is None:
        raise ValueError("--exclude-signer requires --heldout-signer")
    if exclude_signer is not None and exclude_signer == heldout_signer:
        raise ValueError("--exclude-signer must differ from --heldout-signer")
    if heldout_signer is None:
        return stratified_clip_split(records, seed)
    excluded = {heldout_signer, exclude_signer} - {None}
    train = [record for record in records if record["signer_id"] not in excluded]
    val = [record for record in records if record["signer_id"] == heldout_signer]
    if not train or not val:
        raise ValueError(f"--heldout-signer {heldout_signer} did not produce both train and val splits")
    return train, val


def effective_split_seed(args) -> int:
    return args.split_seed if args.split_seed is not None else args.seed


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def assert_resume_lineage_clean(checkpoint_state: dict, exclude_signer: int | None) -> None:
    """Refuse to resume from a checkpoint whose lineage trained/validated on exclude_signer."""
    if exclude_signer is None:
        return
    lineage = checkpoint_state.get("lineage")
    if not lineage:
        return
    contaminated = set(lineage.get("train_signers", [])) | set(lineage.get("val_signers", []))
    if exclude_signer in contaminated:
        raise RuntimeError(
            f"refusing to resume: parent checkpoint lineage trained/validated on "
            f"exclude_signer={exclude_signer} (contaminated_signers={sorted(contaminated)})"
        )


def build_lineage(args, train_records, val_records) -> dict:
    """Per-checkpoint provenance: signers in each role, parent hash, commit, args."""
    train_signers = sorted({r["signer_id"] for r in train_records})
    val_signers = sorted({r["signer_id"] for r in val_records})
    return {
        "fold_outer_test_signer": args.exclude_signer,
        "fold_inner_val_signer": args.heldout_signer,
        "train_signers": train_signers,
        "val_signers": val_signers,
        "parent_checkpoint": str(args.resume) if args.resume else None,
        "parent_checkpoint_sha256": sha256_of(args.resume) if args.resume else None,
        "git_commit": git_commit_hash(),
        "argv": sys.argv[1:],
    }


def make_label_tokens(records, tokenizer):
    labels = sorted({record["label"] for record in records})
    return {
        label: tokenizer(normalize_text(label), add_special_tokens=False).input_ids
        for label in labels
    }


def load_embedding_rows(table_path: Path, token_ids: set[int], embedding_dim: int) -> torch.Tensor:
    table = torch.load(table_path, map_location="cpu")
    if isinstance(table, dict):
        for key in ("weight", "embeddings", "embedding_table"):
            if key in table:
                table = table[key]
                break
    rows = torch.zeros(max(token_ids) + 1, embedding_dim, dtype=table.dtype)
    ids = torch.tensor(sorted(token_ids), dtype=torch.long)
    rows[ids] = table[ids]
    return rows


class IsolatedTokenEvalDataset(Dataset):
    """Deterministic one-pass isolated clip dataset for LOSO/A1 evaluation."""

    def __init__(self, h5_path: Path, records: list[dict], token_ids_by_label, embedding_table):
        self.h5_path = h5_path
        self.records = list(records)
        self.token_ids_by_label = token_ids_by_label
        self.embedding_table = embedding_table

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        with h5py.File(self.h5_path, "r") as f:
            keypoints = f["dataset1"]["keypoints"][record["clip_id"]][:]
        keypoints = remove_keypoints(keypoints)
        keypoints = normalize_augment_data(keypoints, "Original", 111)
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
        token_ids = torch.tensor(self.token_ids_by_label[record["label"]], dtype=torch.long)
        return {
            "keypoints": keypoints,
            "token_ids": token_ids,
            "target_embeddings": self.embedding_table[token_ids],
            "boundary": torch.tensor([[0, keypoints.size(0)]], dtype=torch.long),
            "token_span": torch.tensor([[0, token_ids.numel()]], dtype=torch.long),
            "clip_ids": (str(record["clip_id"]),),
            "glosses": (record["label"],),
            "signer_id": int(record["signer_id"]) if record["signer_id"] is not None else None,
            "video_id": record["video_id"],
            "repetition": record["repetition"],
        }


def isolated_eval_collate(batch):
    return {
        "keypoints": pad_sequence([item["keypoints"] for item in batch], batch_first=True),
        "frame_lengths": torch.tensor([item["keypoints"].size(0) for item in batch], dtype=torch.long),
        "token_ids": pad_sequence([item["token_ids"] for item in batch], batch_first=True, padding_value=-100),
        "token_lengths": torch.tensor([item["token_ids"].numel() for item in batch], dtype=torch.long),
        "target_embeddings": pad_sequence(
            [item["target_embeddings"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        ),
        "boundaries": torch.stack([item["boundary"] for item in batch]),
        "token_spans": torch.stack([item["token_span"] for item in batch]),
        "clip_ids": [item["clip_ids"] for item in batch],
        "glosses": [item["glosses"] for item in batch],
        "signer_ids": [item["signer_id"] for item in batch],
        "video_ids": [item["video_id"] for item in batch],
        "repetitions": [item["repetition"] for item in batch],
    }


def gather_logits(logits, targets):
    logits = align_time_to_targets(logits, targets.size(1))
    mask = targets.ne(-100)
    return logits[mask], targets[mask]


def align_time_to_targets(values, target_steps: int):
    if values.size(1) < target_steps:
        pad_shape = (values.size(0), target_steps - values.size(1), *values.shape[2:])
        pad = values.new_zeros(pad_shape)
        values = torch.cat([values, pad], dim=1)
    elif values.size(1) > target_steps:
        values = values[:, :target_steps]
    return values


def gather_embeddings(embeddings, targets, target_embeddings):
    embeddings = align_time_to_targets(embeddings, targets.size(1))
    mask = targets.ne(-100)
    return embeddings[mask], target_embeddings[mask]


def forward_from_cif(model, frame_features, frame_lengths, **cif_kwargs):
    cif_out = model.cif(frame_features, frame_lengths, **cif_kwargs)
    return {
        "cif": cif_out,
        "token_logits": model.token_head(cif_out.embeddings, cif_out.padding_mask),
        "embeddings": model.embedding_head(cif_out.embeddings),
        "length_logits": model.predict_length_logits(frame_features, frame_lengths),
    }


def compute_alpha_loss(mode, alpha_logits, alpha_pred, alpha_target, frame_mask):
    if mode == "qty_only":
        return alpha_pred.new_zeros(())

    target = alpha_target[frame_mask]
    if mode == "current":
        frame_loss = F.smooth_l1_loss(
            alpha_pred[frame_mask],
            target,
            reduction="none",
        )
        positive_weight = 1.0 + 20.0 * (target > 0).float()
        return (frame_loss * positive_weight).mean()

    if mode == "kl_quantity":
        eps = 1e-8
        pred_dist = alpha_pred / alpha_pred.sum(dim=1, keepdim=True).clamp(min=eps)
        target_dist = alpha_target / alpha_target.sum(dim=1, keepdim=True).clamp(min=eps)
        return F.kl_div(
            (pred_dist[frame_mask] + eps).log(),
            target_dist[frame_mask],
            reduction="batchmean",
        )

    if mode == "logit_l1":
        eps = 1e-4
        target_logits = torch.logit(alpha_target.clamp(min=eps, max=1.0 - eps))
        frame_loss = F.smooth_l1_loss(
            alpha_logits[frame_mask],
            target_logits[frame_mask],
            reduction="none",
        )
        positive_weight = 1.0 + 20.0 * (alpha_target[frame_mask] > 0).float()
        return (frame_loss * positive_weight).mean()

    raise ValueError(f"unknown diagnostic alpha loss: {mode}")


def _module_grad_norm(module):
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().float().norm(2).item()
        total += grad_norm * grad_norm
    return total ** 0.5


def compute_grad_norms(model):
    encoder = model.frame_encoder
    groups = {
        "stgcn": encoder.stgcn_layers,
        "linear_hidden": encoder.linear_hidden,
        "tcn": encoder.tcn,
        "transformer": encoder.transformer,
        "cif": model.cif,
        "token_head": model.token_head,
        "embedding_head": model.embedding_head,
        "length_head": model.length_head,
        "all": model,
    }
    return {name: _module_grad_norm(module) for name, module in groups.items()}


def _classification_metrics(out, batch, centers=None):
    token_logits, token_targets = gather_logits(out["token_logits"], batch["token_ids"])
    ce = F.cross_entropy(token_logits, token_targets)
    pred = token_logits.argmax(dim=-1)
    top1 = pred.eq(token_targets).float().mean()
    top5 = token_logits.topk(5, dim=-1).indices.eq(token_targets.unsqueeze(1)).any(dim=1).float().mean()
    legacy_prefix_exact_value = sequence_accuracy(
        out["token_logits"], batch["token_ids"], batch["token_lengths"]
    )
    strict_metrics = strict_sequence_metrics(
        out["token_logits"], out["cif"].counts, batch["token_ids"], batch["token_lengths"]
    )
    mae = (out["cif"].quantity - batch["token_lengths"].float()).abs().mean()
    count_mae = (out["cif"].counts.float() - batch["token_lengths"].float()).abs().mean()
    count_match = out["cif"].counts.eq(batch["token_lengths"])
    logits_aligned = align_time_to_targets(out["token_logits"], batch["token_ids"].size(1))
    pred_aligned = logits_aligned.argmax(dim=-1)
    per_sample_acc = []
    for i, length in enumerate(batch["token_lengths"].tolist()):
        target = batch["token_ids"][i, :length]
        per_sample_acc.append(pred_aligned[i, :length].eq(target).float().mean())
    per_sample_acc = torch.stack(per_sample_acc) if per_sample_acc else token_logits.new_zeros(0)
    length_targets = batch["token_lengths"].clamp(max=out["length_logits"].size(1) - 1)
    length_loss = F.cross_entropy(out["length_logits"], length_targets)
    pred_len = clamp_predicted_lengths(
        out["length_logits"].argmax(dim=-1),
        max_len=out["length_logits"].size(1) - 1,
    )
    pred_len_mae = (pred_len.float() - batch["token_lengths"].float()).abs().mean()
    return {
        "loss": ce.item(),
        "top1": top1.item(),
        "top5": top5.item(),
        "exact": strict_metrics["exact"],
        "legacy_prefix_exact": legacy_prefix_exact_value,
        "token_error_rate": strict_metrics["token_error_rate"],
        "token_edit_similarity": strict_metrics["token_edit_similarity"],
        "quantity_mae": mae.item(),
        "mae_len": mae.item(),
        "count_mae": count_mae.item(),
        "count_match_rate": count_match.float().mean().item(),
        "token_accuracy_when_count_correct": (
            per_sample_acc[count_match].mean().item() if count_match.any() else 0.0
        ),
        "token_accuracy_when_count_wrong": (
            per_sample_acc[~count_match].mean().item() if (~count_match).any() else 0.0
        ),
        "length_loss": length_loss.item(),
        "pred_len_mae": pred_len_mae.item(),
        "pred_len_match_rate": pred_len.eq(batch["token_lengths"]).float().mean().item(),
        "pred_count_mean": out["cif"].counts.float().mean().item(),
        "quantity_mean": out["cif"].quantity.float().mean().item(),
        "pred_len_mean": pred_len.float().mean().item(),
        "target_len_mean": batch["token_lengths"].float().mean().item(),
    }


def evaluate_alpha_mode(
    model, loader, device, mode, w_target=1.0, w_pred=0.0, rounded_count_bias=0.0
):
    """Evaluate a concrete CIF alpha source."""
    model.eval()
    totals = defaultdict(float)
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            alpha_target = CIFAggregator.boundary_targets(
                batch["boundaries"],
                batch["keypoints"].size(1),
                token_spans=batch["token_spans"],
            )
            frame_features = model.frame_encoder(batch["keypoints"], batch["frame_lengths"])
            alpha_logits = model.cif.predict_alpha_logits(frame_features, batch["frame_lengths"])
            alpha_pred = torch.sigmoid(alpha_logits).masked_fill(
                ~length_mask_from_lengths(batch["frame_lengths"], alpha_logits.size(1)),
                0.0,
            )
            alpha_rescaled = rescale_alphas_to_target_lengths(alpha_pred, batch["token_lengths"])
            length_logits = model.predict_length_logits(frame_features, batch["frame_lengths"])
            pred_lengths = clamp_predicted_lengths(
                length_logits.argmax(dim=-1),
                max_len=model.max_len_class,
            )
            alpha_rescaled_pred_len, pred_lengths = rescale_alphas_to_predicted_lengths(
                alpha_pred,
                pred_lengths,
                max_len=model.max_len_class,
            )
            if mode == "teacher_alpha":
                alphas = alpha_target
            elif mode == "pred_raw":
                alphas = alpha_pred
            elif mode == "pred_rescaled_to_target_len":
                alphas = alpha_rescaled
            elif mode == "pred_rescaled_to_pred_len":
                alphas = alpha_rescaled_pred_len
            elif mode == "pred_rescaled_to_rounded_count":
                alphas, rounded_lengths = rescale_alphas_to_rounded_count(
                    alpha_pred,
                    max_len=model.max_len_class,
                    bias=rounded_count_bias,
                )
            elif mode == "blended_alpha":
                alphas = w_target * alpha_target + w_pred * alpha_rescaled
            else:
                raise ValueError(f"unknown eval alpha mode: {mode}")

            out = forward_from_cif(model, frame_features, batch["frame_lengths"], alphas=alphas)
            for key, value in _classification_metrics(out, batch).items():
                totals[key] += value
            if mode == "pred_rescaled_to_pred_len":
                totals["rescale_length_mae"] += (
                    pred_lengths.float() - batch["token_lengths"].float()
                ).abs().mean().item()
            elif mode == "pred_rescaled_to_rounded_count":
                totals["rescale_length_mae"] += (
                    rounded_lengths.float() - batch["token_lengths"].float()
                ).abs().mean().item()
            totals["alpha_eval_sum_mean"] += alphas.sum(dim=1).mean().item()
            totals["target_alpha_sum_mean"] += alpha_target.sum(dim=1).mean().item()
            if mode != "teacher_alpha":
                for key, value in alpha_diagnostics(
                    frame_features,
                    alpha_logits,
                    alpha_pred,
                    batch["frame_lengths"],
                ).items():
                    totals[key] += value
            centers = CIFAggregator.token_centers(batch["boundaries"], batch["token_spans"]).to(device)
            boundary_mae = boundary_error_mae(
                out["cif"].fire_positions, out["cif"].counts, centers, batch["token_lengths"]
            )
            totals["boundary_mae"] += boundary_mae
            count_match = out["cif"].counts.eq(batch["token_lengths"])
            if count_match.any():
                totals["boundary_mae_when_count_correct"] += boundary_error_mae(
                    out["cif"].fire_positions[count_match],
                    out["cif"].counts[count_match],
                    centers[count_match],
                    batch["token_lengths"][count_match],
                    missing_penalty=0.0,
                )
                totals["boundary_mae_when_count_correct_batches"] += 1
            batches += 1
    result = {}
    for key, value in totals.items():
        denom = (
            totals["boundary_mae_when_count_correct_batches"]
            if key == "boundary_mae_when_count_correct"
            else batches
        )
        if key == "boundary_mae_when_count_correct_batches":
            continue
        result[key] = value / max(1, denom)
    if "boundary_mae_when_count_correct" not in result:
        # No batch had even one sample with a correct fired count this pass: undefined,
        # not zero. Same sentinel boundary_error_mae() itself uses when nothing fires.
        result["boundary_mae_when_count_correct"] = 1000.0
    return result


def evaluate_loso_isolated(model, loader, tokenizer, device):
    """Evaluate every isolated held-out clip exactly once with teacher alphas."""
    model.eval()
    totals = defaultdict(float)
    by_gloss = defaultdict(lambda: defaultdict(float))
    prediction_rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            alpha_target = CIFAggregator.boundary_targets(
                batch["boundaries"],
                batch["keypoints"].size(1),
                token_spans=batch["token_spans"],
            )
            out = model(
                batch["keypoints"],
                batch["frame_lengths"],
                alphas=alpha_target,
                target_lengths=batch["token_lengths"],
            )
            logits = align_time_to_targets(out["token_logits"], batch["token_ids"].size(1))
            mask = batch["token_ids"].ne(-100)
            pred = logits.argmax(dim=-1)
            top1 = pred[mask].eq(batch["token_ids"][mask]).float()
            top5 = logits.topk(5, dim=-1).indices.eq(batch["token_ids"].unsqueeze(-1)).any(dim=-1)
            top5 = top5[mask].float()

            rows = make_token_predictions(
                token_logits=logits,
                token_ids=batch["token_ids"],
                token_lengths=batch["token_lengths"],
                clip_ids=batch["clip_ids"],
                glosses=batch["glosses"],
                tokenizer=tokenizer,
            )
            lengths = batch["token_lengths"].detach().cpu().tolist()
            pred_cpu = pred.detach().cpu()
            target_cpu = batch["token_ids"].detach().cpu()
            for i, row in enumerate(rows):
                length = int(lengths[i])
                target_ids = target_cpu[i, :length].tolist()
                pred_ids = pred_cpu[i, :length].tolist()
                token_hits = sum(int(a == b) for a, b in zip(pred_ids, target_ids))
                token_acc = token_hits / max(1, length)
                # teacher_alpha rescales alpha mass to target_lengths exactly, so the
                # fired count matches target length here; prefix and strict coincide.
                legacy_prefix_exact_value = int(legacy_prefix_exact(pred_ids, target_ids))
                exact = int(strict_exact(pred_ids, target_ids))
                ter = token_error_rate(pred_ids, target_ids)
                row_dict = row.as_dict()
                row_dict.update(
                    {
                        "signer_id": batch["signer_ids"][i],
                        "video_id": batch["video_ids"][i],
                        "repetition": batch["repetitions"][i],
                        "token_accuracy": token_acc,
                        "exact": exact,
                        "legacy_prefix_exact": legacy_prefix_exact_value,
                        "token_error_rate": ter,
                        "token_edit_similarity": token_edit_similarity(ter),
                    }
                )
                prediction_rows.append(row_dict)

                gloss = row.glosses[0]
                by_gloss[gloss]["samples"] += 1
                by_gloss[gloss]["exact"] += exact
                by_gloss[gloss]["token_hits"] += token_hits
                by_gloss[gloss]["token_count"] += length
                by_gloss[gloss]["chrf"] += chrf_score(row.predicted_text, row.target_text)
                by_gloss[gloss]["bleu"] += bleu_score(row.predicted_text, row.target_text)

            totals["token_top1_sum"] += top1.sum().item()
            totals["token_top5_sum"] += top5.sum().item()
            totals["token_count"] += top1.numel()
            totals["samples"] += len(rows)

    sample_count = max(1, len(prediction_rows))
    exact_sum = sum(row["exact"] for row in prediction_rows)
    summary = {
        "samples": len(prediction_rows),
        "token_top1": totals["token_top1_sum"] / max(1.0, totals["token_count"]),
        "token_top5": totals["token_top5_sum"] / max(1.0, totals["token_count"]),
        "exact": exact_sum / sample_count,
        "chrf": sum(chrf_score(row["predicted_text"], row["target_text"]) for row in prediction_rows) / sample_count,
        "bleu": sum(bleu_score(row["predicted_text"], row["target_text"]) for row in prediction_rows) / sample_count,
    }
    gloss_rows = []
    for gloss, values in by_gloss.items():
        samples = max(1.0, values["samples"])
        gloss_rows.append(
            {
                "gloss": gloss,
                "samples": int(values["samples"]),
                "exact": values["exact"] / samples,
                "token_accuracy": values["token_hits"] / max(1.0, values["token_count"]),
                "chrf": values["chrf"] / samples,
                "bleu": values["bleu"] / samples,
            }
        )
    gloss_rows = sorted(gloss_rows, key=lambda row: (row["exact"], row["token_accuracy"], row["gloss"]))
    correct_examples = [row for row in prediction_rows if row["exact"]][:10]
    incorrect_examples = [row for row in prediction_rows if not row["exact"]][:10]
    return {
        **summary,
        "worst_glosses": gloss_rows[:10],
        "examples": {
            "correct": correct_examples,
            "incorrect": incorrect_examples,
        },
    }, prediction_rows


def evaluate_teacher(model, loader, device):
    """Forward with alpha=alpha_target (perfect injected boundaries). Comparable ceiling."""
    return evaluate_alpha_mode(model, loader, device, "teacher_alpha")


def evaluate_predicted(model, loader, device):
    """Forward using only CIF.alpha (no injected boundaries). Official v126b gate."""
    return evaluate_alpha_mode(model, loader, device, "pred_raw")


def evaluate_permuted(model, loader, device, seed: int):
    """Predicted-mode top1 after shuffling whole gloss segments; targets stay unpermuted."""
    model.eval()
    totals = defaultdict(float)
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            permuted = batch["keypoints"].clone()
            for i in range(permuted.size(0)):
                length = int(batch["frame_lengths"][i].item())
                rng = random.Random(seed + i)
                permuted[i, :length] = permute_video_segments(
                    batch["keypoints"][i, :length].cpu(),
                    batch["boundaries"][i].cpu(),
                    length,
                    rng,
                ).to(device)
            out = model(permuted, batch["frame_lengths"])
            token_logits, token_targets = gather_logits(out["token_logits"], batch["token_ids"])
            top1 = token_logits.argmax(dim=-1).eq(token_targets).float().mean()
            totals["top1"] += top1.item()
            batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def write_prediction_report(
    *,
    model,
    loader,
    tokenizer,
    device,
    path: Path,
    max_rows: int,
    alpha_mode: str,
    rounded_count_bias: float = 0.0,
):
    """Write prototype rows: Gemma token IDs, decoded text, correction prompt."""
    if max_rows <= 0:
        return
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            alpha_target = CIFAggregator.boundary_targets(
                batch["boundaries"],
                batch["keypoints"].size(1),
                token_spans=batch["token_spans"],
            )
            frame_features = model.frame_encoder(batch["keypoints"], batch["frame_lengths"])
            if alpha_mode == "teacher_alpha":
                alphas = alpha_target
            else:
                alpha_pred = model.cif.predict_alpha(
                    frame_features,
                    batch["frame_lengths"],
                )
                if alpha_mode == "pred_raw":
                    alphas = alpha_pred
                elif alpha_mode == "pred_rescaled_to_target_len":
                    alphas = rescale_alphas_to_target_lengths(
                        alpha_pred,
                        batch["token_lengths"],
                    )
                elif alpha_mode == "pred_rescaled_to_pred_len":
                    pred_lengths = model.predict_lengths(
                        frame_features,
                        batch["frame_lengths"],
                    )
                    alphas, _ = rescale_alphas_to_predicted_lengths(
                        alpha_pred,
                        pred_lengths,
                        max_len=model.max_len_class,
                    )
                elif alpha_mode == "pred_rescaled_to_rounded_count":
                    alphas, _ = rescale_alphas_to_rounded_count(
                        alpha_pred,
                        max_len=model.max_len_class,
                        bias=rounded_count_bias,
                    )
                else:
                    raise ValueError(f"unknown prediction alpha mode: {alpha_mode}")
            out = forward_from_cif(
                model,
                frame_features,
                batch["frame_lengths"],
                alphas=alphas,
            )
            token_logits = align_time_to_targets(
                out["token_logits"],
                batch["token_ids"].size(1),
            )
            rows.extend(
                make_token_predictions(
                    token_logits=token_logits,
                    token_ids=batch["token_ids"],
                    token_lengths=batch["token_lengths"],
                    clip_ids=batch["clip_ids"],
                    glosses=batch["glosses"],
                    tokenizer=tokenizer,
                )
            )
            if len(rows) >= max_rows:
                break

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows[:max_rows]:
            f.write(json.dumps(row.as_dict(), ensure_ascii=False) + "\n")


def sequence_accuracy(logits, targets, lengths):
    """Historical metric: prefix-of-target-length match, ignoring predicted count."""
    logits = align_time_to_targets(logits, targets.size(1))
    pred = logits.argmax(dim=-1)
    hits = 0
    for i, length in enumerate(lengths.tolist()):
        hits += bool(torch.equal(pred[i, :length].cpu(), targets[i, :length].cpu()))
    return hits / max(1, targets.size(0))


def strict_sequence_metrics(token_logits, pred_counts, targets, lengths):
    """Etapa 4 correction: strict_exact requires pred_count == target_len, plus TER.

    ``token_logits`` are the raw per-fired-slot logits (one row per CIF slot,
    not padded/truncated to the target length), so the predicted sequence used
    for TER/strict-exact is exactly what the model fired, of length
    ``pred_counts[i]`` -- not the prefix-aligned view ``sequence_accuracy`` uses.
    """
    pred_ids_full = token_logits.argmax(dim=-1)
    batch_size = targets.size(0)
    strict_hits = 0
    ter_sum = 0.0
    sim_sum = 0.0
    for i in range(batch_size):
        p_len = int(pred_counts[i].item())
        t_len = int(lengths[i].item())
        pred_seq = pred_ids_full[i, :p_len].tolist()
        target_seq = targets[i, :t_len].tolist()
        strict_hits += int(strict_exact(pred_seq, target_seq))
        ter = token_error_rate(pred_seq, target_seq)
        ter_sum += ter
        sim_sum += token_edit_similarity(ter)
    return {
        "exact": strict_hits / max(1, batch_size),
        "token_error_rate": ter_sum / max(1, batch_size),
        "token_edit_similarity": sim_sum / max(1, batch_size),
    }


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def main():
    args = parse_args()
    if args.min_clips < 1 or args.max_clips < args.min_clips:
        raise ValueError("--min-clips/--max-clips must satisfy 1 <= min <= max")
    initialize(seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    diagnostic_mode = (
        args.diag_alpha_loss != "current"
        or args.diag_freeze != "full_current"
        or args.diag_eval_force_count
    )
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    if diagnostic_mode and not run_name.startswith("diag_"):
        run_name = f"diag_{run_name}"
    out_dir = args.output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    metrics_path = out_dir / "metrics.jsonl"
    ckpt_path = out_dir / "checkpoint_latest.pt"
    best_path = out_dir / "checkpoint_best.pt"

    records = list_clip_records(args.h5, "dataset1")
    train_records, val_records = split_records(
        records, effective_split_seed(args), args.heldout_signer, args.exclude_signer
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    token_ids_by_label = make_label_tokens(records, tokenizer)
    all_token_ids = {tid for ids in token_ids_by_label.values() for tid in ids}
    embedding_rows = load_embedding_rows(args.embedding_table, all_token_ids, args.embedding_dim)

    def make_ds(rows, samples, seed):
        return SyntheticTemporalSignDataset(
            args.h5,
            [r["clip_id"] for r in rows],
            {r["clip_id"]: r["label"] for r in rows},
            token_ids_by_label,
            min_clips=args.min_clips,
            max_clips=args.max_clips,
            min_neutral_frames=args.min_neutral_frames,
            max_neutral_frames=args.max_neutral_frames,
            samples_per_epoch=samples,
            seed=seed,
            embedding_table=embedding_rows,
            apply_remove_keypoints=True,
            normalize=True,
            n_keypoints=111,
        )

    train_loader = DataLoader(
        make_ds(train_records, args.samples_per_epoch, args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=synthetic_temporal_collate,
        pin_memory=torch.cuda.is_available(),
    )
    if args.heldout_signer is None:
        val_loader = DataLoader(
            make_ds(val_records, args.val_samples, args.seed + 100_000),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=synthetic_temporal_collate,
            pin_memory=torch.cuda.is_available(),
        )
        loso_loader = None
    else:
        val_loader = DataLoader(
            IsolatedTokenEvalDataset(args.h5, val_records, token_ids_by_label, embedding_rows),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=isolated_eval_collate,
            pin_memory=torch.cuda.is_available(),
        )
        loso_loader = val_loader

    A = np.load("data/processed/adjacency_matrix.npy", allow_pickle=True)
    encoder = STGCNTemporalFrameEncoder(A, hidden_size=args.hidden_size)
    load_info = load_visual_low_level_weights(encoder, args.checkpoint_v121)
    model = TemporalSignPromptModel(
        encoder,
        hidden_size=args.hidden_size,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        max_len_class=args.max_len_class,
        token_head_variant=args.token_head,
        length_head_variant=args.length_head,
    ).to(device)
    if args.phase == "learned_cif":
        stgcn_params = list(encoder.stgcn_layers.parameters()) + list(encoder.linear_hidden.parameters())
        stgcn_ids = {id(p) for p in stgcn_params}
        other_params = [p for p in model.parameters() if id(p) not in stgcn_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": stgcn_params, "lr": args.lr * args.stgcn_lr_scale},
                {"params": other_params, "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    best_select_metric = (-1.0, -1.0)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        assert_resume_lineage_clean(state, args.exclude_signer)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if missing or unexpected:
            print(
                f"[v126] non-strict load missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
        same_stage = state.get("phase", "teacher_forced") == args.phase and not args.resume_weights_only
        if same_stage:
            try:
                optimizer.load_state_dict(state["optimizer"])
            except ValueError:
                same_stage = False
        if same_stage:
            start_epoch = state["epoch"] + 1
            best_select_metric = tuple(state.get("best_select_metric", best_select_metric))
        else:
            print(
                f"[v126] stage transition: checkpoint phase={state.get('phase', 'teacher_forced')!r} "
                f"-> {args.phase!r}; loaded model weights only, starting fresh epoch/optimizer",
                flush=True,
            )

    split_config = {
        "split_protocol": "loso" if args.heldout_signer is not None else "stratified",
        "heldout_signer": args.heldout_signer,
        "train_records": len(train_records),
        "val_records": len(val_records),
    }
    lineage = build_lineage(args, train_records, val_records)
    if args.exclude_signer is not None and args.exclude_signer in lineage["train_signers"]:
        raise RuntimeError(
            f"contamination guard: exclude_signer={args.exclude_signer} leaked into train_signers"
        )
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {**vars(args), **split_config, "lineage": lineage, "load_info": load_info, "device": device},
            f,
            indent=2,
            default=str,
        )
    print(
        f"[v126] out={out_dir} device={device} load={load_info} "
        f"split={split_config['split_protocol']} train={len(train_records)} val={len(val_records)}",
        flush=True,
    )

    final_gates = None
    for epoch in range(start_epoch, args.epochs):
        if args.phase == "learned_cif":
            phase_state = set_cif_diagnostic_freeze(model, epoch, args.diag_freeze)
            w_target, w_pred = scheduled_alpha_weights(args, epoch)
            if args.diag_freeze == "target_only_stage1" and args.alpha_schedule == "current":
                w_target, w_pred = 1.0, 0.0
        else:
            phase_state, w_target, w_pred = None, 1.0, 0.0

        model.train()
        totals = defaultdict(float)
        steps = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            alpha_target = CIFAggregator.boundary_targets(
                batch["boundaries"],
                batch["keypoints"].size(1),
                token_spans=batch["token_spans"],
            )

            if args.phase == "learned_cif":
                frame_features = model.frame_encoder(batch["keypoints"], batch["frame_lengths"])
                frame_mask = length_mask_from_lengths(batch["frame_lengths"], frame_features.size(1))
                alpha_logits = model.cif.predict_alpha_logits(frame_features, batch["frame_lengths"])
                alpha_pred = torch.sigmoid(alpha_logits).masked_fill(~frame_mask, 0.0)
                pred_quantity = alpha_pred.sum(dim=1)
                length_logits = model.predict_length_logits(frame_features, batch["frame_lengths"])
                pred_scale = (
                    batch["token_lengths"].float()
                    / pred_quantity.detach().clamp(min=1e-6)
                ).unsqueeze(1)
                alpha_pred_for_tokens = alpha_pred * pred_scale
                blended_alpha = w_target * alpha_target + w_pred * alpha_pred_for_tokens
                out = forward_from_cif(
                    model,
                    frame_features,
                    batch["frame_lengths"],
                    alphas=blended_alpha,
                )
            else:
                out = model(
                    batch["keypoints"],
                    batch["frame_lengths"],
                    alphas=alpha_target,
                    target_lengths=batch["token_lengths"],
                )
                length_logits = out["length_logits"]

            token_logits, token_targets = gather_logits(out["token_logits"], batch["token_ids"])
            token_loss = F.cross_entropy(
                token_logits, token_targets, label_smoothing=args.token_label_smoothing
            )
            length_targets = batch["token_lengths"].clamp(max=args.max_len_class)
            length_loss = F.cross_entropy(length_logits, length_targets)
            pred_emb, target_emb = gather_embeddings(
                out["embeddings"], batch["token_ids"], batch["target_embeddings"]
            )
            emb_loss = F.smooth_l1_loss(pred_emb, target_emb)

            if args.phase == "learned_cif":
                alpha_loss = compute_alpha_loss(
                    args.diag_alpha_loss,
                    alpha_logits,
                    alpha_pred,
                    alpha_target,
                    frame_mask,
                )
                qty_loss = F.l1_loss(pred_quantity, batch["token_lengths"].float())
                if epoch < 3 or args.diag_freeze == "alpha_only":
                    loss = (
                        args.alpha_loss_weight * alpha_loss
                        + args.qty_loss_weight * qty_loss
                        + args.length_loss_weight * length_loss
                    )
                else:
                    loss = (
                        token_loss
                        + args.emb_loss_weight * emb_loss
                        + args.length_loss_weight * length_loss
                        + args.alpha_loss_weight * alpha_loss
                        + args.qty_loss_weight * qty_loss
                    )
            else:
                alpha_loss = token_loss.new_zeros(())
                qty_loss = F.l1_loss(out["cif"].quantity, batch["token_lengths"].float())
                loss = (
                    token_loss
                    + args.emb_loss_weight * emb_loss
                    + args.length_loss_weight * length_loss
                    + 0.1 * qty_loss
                )

            optimizer.zero_grad()
            loss.backward()
            grad_norms = compute_grad_norms(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            totals["loss"] += loss.item()
            totals["token_loss"] += token_loss.item()
            totals["emb_loss"] += emb_loss.item()
            totals["length_loss"] += length_loss.item()
            totals["qty_loss"] += qty_loss.item()
            totals["alpha_loss"] += alpha_loss.item()
            if args.phase == "learned_cif":
                for key, value in alpha_diagnostics(
                    frame_features,
                    alpha_logits,
                    alpha_pred,
                    batch["frame_lengths"],
                ).items():
                    totals[key] += value
            for key, value in grad_norms.items():
                totals[f"grad_norm_{key}"] += value
            steps += 1

        train_metrics = {key: value / max(1, steps) for key, value in totals.items()}

        if args.phase == "learned_cif":
            val_teacher = evaluate_teacher(model, val_loader, device)
            val_pred_raw = evaluate_predicted(model, val_loader, device)
            val_pred_rescaled = evaluate_alpha_mode(
                model,
                val_loader,
                device,
                "pred_rescaled_to_target_len",
            )
            val_pred_rescaled_to_pred_len = evaluate_alpha_mode(
                model,
                val_loader,
                device,
                "pred_rescaled_to_pred_len",
            )
            val_blended = evaluate_alpha_mode(
                model,
                val_loader,
                device,
                "blended_alpha",
                w_target=w_target,
                w_pred=w_pred,
            )
            val_predicted = val_pred_rescaled if args.diag_eval_force_count else val_pred_raw
            val_permuted = evaluate_permuted(model, val_loader, device, seed=args.seed + epoch)
            permuted_drop = val_pred_raw["top1"] - val_permuted["top1"]
            row = {
                "epoch": epoch,
                "phase_state": phase_state,
                "tf_weights": [w_target, w_pred],
                "diagnostics": {
                    "alpha_loss": args.diag_alpha_loss,
                    "freeze": args.diag_freeze,
                    "eval_force_count": args.diag_eval_force_count,
                },
                "train": train_metrics,
                "val_teacher": val_teacher,
                "val_pred_raw": val_pred_raw,
                "val_pred_rescaled_to_target_len": val_pred_rescaled,
                "val_pred_rescaled_to_pred_len": val_pred_rescaled_to_pred_len,
                "val_blended_alpha": val_blended,
                "val_predicted": val_predicted,
                "val_permuted_top1": val_permuted["top1"],
                "permuted_top1_drop": permuted_drop,
            }
            select_metric = (
                val_pred_rescaled_to_pred_len["exact"],
                val_pred_rescaled_to_pred_len["top1"],
            )
            print(
                f"ep{epoch:03d} loss={train_metrics['loss']:.4f} tf=({w_target:.2f},{w_pred:.2f}) "
                f"teacher_top1={val_teacher['top1']:.3f} pred_raw_top1={val_pred_raw['top1']:.3f} "
                f"pred_rescaled_top1={val_pred_rescaled['top1']:.3f} "
                f"pred_len_top1={val_pred_rescaled_to_pred_len['top1']:.3f} "
                f"pred_exact={val_predicted['exact']:.3f} "
                f"pred_mae_len={val_predicted['mae_len']:.3f} pred_boundary_mae={val_predicted['boundary_mae']:.3f} "
                f"permuted_drop={permuted_drop:.3f}",
                flush=True,
            )
            # Etapa 1 acceptance gates from ROADMAP_A3_CIF_LENGTH_CONDITIONED.md,
            # evaluated on the deployable pred_rescaled_to_pred_len mode.
            final_gates = {
                "pred_len_mae<=0.10": val_pred_rescaled_to_pred_len["pred_len_mae"] <= 0.10,
                "count_match_rate>=0.93": val_pred_rescaled_to_pred_len["count_match_rate"] >= 0.93,
                "top1>=0.85": val_pred_rescaled_to_pred_len["top1"] >= 0.85,
                "top5>=0.94": val_pred_rescaled_to_pred_len["top5"] >= 0.94,
                "exact>=0.79": val_pred_rescaled_to_pred_len["exact"] >= 0.79,
                "boundary_mae_when_count_correct<=1.0": (
                    val_pred_rescaled_to_pred_len["boundary_mae_when_count_correct"] <= 1.0
                ),
            }
        else:
            val_metrics = evaluate_teacher(model, val_loader, device)
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            select_metric = (val_metrics["top1"], val_metrics["top1"])
            print(
                f"ep{epoch:03d} loss={train_metrics['loss']:.4f} "
                f"val_top1={val_metrics['top1']:.3f} val_top5={val_metrics['top5']:.3f} "
                f"exact={val_metrics['exact']:.3f} mae_len={val_metrics['mae_len']:.3f}",
                flush=True,
            )

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_select_metric": best_select_metric,
            "token_ids_by_label": token_ids_by_label,
            "load_info": load_info,
            "phase": args.phase,
            "arch_config": {"token_head": args.token_head, "length_head": args.length_head},
            "lineage": lineage,
        }
        torch.save(state, ckpt_path)
        if select_metric > best_select_metric:
            best_select_metric = select_metric
            state["best_select_metric"] = best_select_metric
            torch.save(state, best_path)
            if final_gates is not None:
                with open(out_dir / "gates.json", "w", encoding="utf-8") as f:
                    json.dump(final_gates, f, indent=2)
                print(f"[v126b] gates (checkpoint_best epoch={epoch})={final_gates}", flush=True)

    write_prediction_report(
        model=model,
        loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        path=out_dir / "predictions.jsonl",
        max_rows=args.prediction_samples,
        alpha_mode=args.prediction_alpha_mode,
    )

    if args.heldout_signer is not None and loso_loader is not None:
        best_state = torch.load(best_path, map_location=device)
        model.load_state_dict(best_state["model"])
        loso_eval, loso_rows = evaluate_loso_isolated(model, loso_loader, tokenizer, device)
        loso_eval.update(
            {
                "checkpoint": str(best_path),
                "split_protocol": "loso",
                "heldout_signer": args.heldout_signer,
                "train_records": len(train_records),
                "val_records": len(val_records),
                "alpha": "teacher_alpha/oracle boundaries",
            }
        )
        with open(out_dir / "loso_eval.json", "w", encoding="utf-8") as f:
            json.dump(loso_eval, f, indent=2, ensure_ascii=False)
        with open(out_dir / "loso_predictions.jsonl", "w", encoding="utf-8") as f:
            for row in loso_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[v126] loso signer={args.heldout_signer} samples={loso_eval['samples']} "
            f"token_top1={loso_eval['token_top1']:.3f} token_top5={loso_eval['token_top5']:.3f} "
            f"exact={loso_eval['exact']:.3f} chrf={loso_eval['chrf']:.2f} bleu={loso_eval['bleu']:.2f}",
            flush=True,
        )

    print(f"[v126] done best_select_metric={best_select_metric} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
