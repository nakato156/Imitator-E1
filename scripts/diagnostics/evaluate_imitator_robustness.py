"""Robustness diagnostics for the dataset1 -> Gemma-token Imitator."""
from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from settings import initialize
from src.mslm.dataloader.data_augmentation import normalize_augment_data, remove_keypoints
from src.mslm.dataloader.isolated_keypoint_dataset import list_clip_records
from src.mslm.inference.imitator_tokens import (
    build_gemma_correction_prompt,
    make_token_predictions,
)
from src.mslm.models.temporal_sign_prompt import (
    CIFAggregator,
    STGCNTemporalFrameEncoder,
    TemporalSignPromptModel,
    load_visual_low_level_weights,
)
from src.mslm.utils.text_metrics import bleu_score, chrf_score


DEFAULT_TOKENIZER = (
    ".hf_cache/hub/"
    "models--unsloth--gemma-3n-E2B-it-unsloth-bnb-4bit/"
    "snapshots/3d26ffdd2276698f582562bf01511aa625bc6f30"
)
DEFAULT_H5 = "data/processed/dataset1_isolated_v122.hdf5"
DEFAULT_EMBEDDING_TABLE = "data/processed/gemma3n_embed_table.pt"
DEFAULT_V121 = "../outputs/checkpoints/121/23/best_top1/checkpoint.pth"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h5", type=Path, default=Path(DEFAULT_H5))
    parser.add_argument("--tokenizer", type=Path, default=Path(DEFAULT_TOKENIZER))
    parser.add_argument("--embedding-table", type=Path, default=Path(DEFAULT_EMBEDDING_TABLE))
    parser.add_argument("--checkpoint-v121", type=Path, default=Path(DEFAULT_V121))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--heldout-signer", type=int)
    parser.add_argument(
        "--temporal-scales",
        default="0.5,0.75,1.0,1.25,1.5,2.0",
        help="Comma-separated output length multipliers. <1 is faster/shorter; >1 is slower/longer.",
    )
    parser.add_argument(
        "--noise-stds",
        default="0.0,0.01,0.03,0.05",
        help="Gaussian keypoint noise std after normalization.",
    )
    parser.add_argument(
        "--keypoint-drop-rates",
        default="0.0,0.05,0.10,0.20",
        help="Rates for dropping full keypoint trajectories to zero.",
    )
    parser.add_argument("--prediction-samples", type=int, default=24)
    return parser.parse_args()


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


def split_records(records, seed: int, heldout_signer: int | None = None):
    if heldout_signer is None:
        return stratified_clip_split(records, seed)
    train = [record for record in records if record["signer_id"] != heldout_signer]
    val = [record for record in records if record["signer_id"] == heldout_signer]
    if not train or not val:
        raise ValueError(f"--heldout-signer {heldout_signer} did not produce both train and val splits")
    return train, val


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


def temporal_rescale_keypoints(keypoints: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return keypoints
    length = max(1, int(round(keypoints.size(0) * scale)))
    x = keypoints.permute(1, 2, 0).reshape(1, -1, keypoints.size(0))
    y = F.interpolate(x, size=length, mode="linear", align_corners=False)
    return y.reshape(keypoints.size(1), keypoints.size(2), length).permute(2, 0, 1).contiguous()


def perturb_keypoints(
    keypoints: torch.Tensor,
    *,
    noise_std: float,
    keypoint_drop_rate: float,
    seed: int,
) -> torch.Tensor:
    if noise_std > 0:
        generator = torch.Generator(device=keypoints.device).manual_seed(seed)
        keypoints = keypoints + torch.randn(
            keypoints.shape,
            generator=generator,
            dtype=keypoints.dtype,
            device=keypoints.device,
        ) * float(noise_std)
    if keypoint_drop_rate > 0:
        generator = torch.Generator(device=keypoints.device).manual_seed(seed + 10_000)
        keep = torch.rand(
            keypoints.size(1),
            generator=generator,
            dtype=keypoints.dtype,
            device=keypoints.device,
        ) >= float(keypoint_drop_rate)
        keypoints = keypoints * keep.view(1, -1, 1)
    return keypoints


class IsolatedTokenDataset(Dataset):
    def __init__(
        self,
        h5_path: Path,
        records: list[dict],
        token_ids_by_label: dict[str, list[int]],
        embedding_table: torch.Tensor,
        temporal_scale: float = 1.0,
        noise_std: float = 0.0,
        keypoint_drop_rate: float = 0.0,
        seed: int = 23,
        max_samples: int | None = None,
    ):
        self.h5_path = h5_path
        self.records = records[: max_samples or len(records)]
        self.token_ids_by_label = token_ids_by_label
        self.embedding_table = embedding_table
        self.temporal_scale = float(temporal_scale)
        self.noise_std = float(noise_std)
        self.keypoint_drop_rate = float(keypoint_drop_rate)
        self.seed = int(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        with h5py.File(self.h5_path, "r") as f:
            keypoints = f["dataset1"]["keypoints"][record["clip_id"]][:]
        keypoints = remove_keypoints(keypoints)
        keypoints = normalize_augment_data(keypoints, "Original", 111)
        if not isinstance(keypoints, torch.Tensor):
            keypoints = torch.as_tensor(keypoints)
        keypoints = temporal_rescale_keypoints(keypoints.float(), self.temporal_scale)
        keypoints = perturb_keypoints(
            keypoints,
            noise_std=self.noise_std,
            keypoint_drop_rate=self.keypoint_drop_rate,
            seed=self.seed + idx,
        )
        token_ids = torch.tensor(self.token_ids_by_label[record["label"]], dtype=torch.long)
        return {
            "keypoints": keypoints,
            "token_ids": token_ids,
            "target_embeddings": self.embedding_table[token_ids],
            "boundary": torch.tensor([[0, keypoints.size(0)]], dtype=torch.long),
            "token_span": torch.tensor([[0, token_ids.numel()]], dtype=torch.long),
            "clip_id": str(record["clip_id"]),
            "gloss": record["label"],
            "signer_id": int(record["signer_id"]) if record["signer_id"] is not None else -1,
        }


def collate(batch):
    keypoints = pad_sequence([item["keypoints"] for item in batch], batch_first=True)
    token_ids = pad_sequence([item["token_ids"] for item in batch], batch_first=True, padding_value=-100)
    target_embeddings = pad_sequence(
        [item["target_embeddings"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    boundaries = torch.stack([item["boundary"] for item in batch])
    token_spans = torch.stack([item["token_span"] for item in batch])
    return {
        "keypoints": keypoints,
        "frame_lengths": torch.tensor([item["keypoints"].size(0) for item in batch], dtype=torch.long),
        "token_ids": token_ids,
        "token_lengths": torch.tensor([item["token_ids"].numel() for item in batch], dtype=torch.long),
        "target_embeddings": target_embeddings,
        "boundaries": boundaries,
        "token_spans": token_spans,
        "clip_ids": [(item["clip_id"],) for item in batch],
        "glosses": [(item["gloss"],) for item in batch],
        "signer_ids": torch.tensor([item["signer_id"] for item in batch], dtype=torch.long),
    }


def align_time_to_targets(values, target_steps: int):
    if values.size(1) < target_steps:
        pad_shape = (values.size(0), target_steps - values.size(1), *values.shape[2:])
        values = torch.cat([values, values.new_zeros(pad_shape)], dim=1)
    elif values.size(1) > target_steps:
        values = values[:, :target_steps]
    return values


def evaluate(model, loader, tokenizer, device, prediction_limit=0):
    model.eval()
    totals = defaultdict(float)
    by_signer = defaultdict(lambda: defaultdict(float))
    prediction_rows = []
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            alpha = CIFAggregator.boundary_targets(
                batch["boundaries"],
                batch["keypoints"].size(1),
                token_spans=batch["token_spans"],
            )
            out = model(
                batch["keypoints"],
                batch["frame_lengths"],
                alphas=alpha,
                target_lengths=batch["token_lengths"],
            )
            logits = align_time_to_targets(out["token_logits"], batch["token_ids"].size(1))
            mask = batch["token_ids"].ne(-100)
            pred = logits.argmax(dim=-1)
            top1 = pred[mask].eq(batch["token_ids"][mask]).float()
            top5 = logits.topk(5, dim=-1).indices.eq(batch["token_ids"].unsqueeze(-1)).any(dim=-1)
            top5 = top5[mask].float()

            exact_hits = []
            chrf_values = []
            bleu_values = []
            signer_ids = batch["signer_ids"].detach().cpu().tolist()
            for i, length in enumerate(batch["token_lengths"].detach().cpu().tolist()):
                target_ids = batch["token_ids"][i, :length].detach().cpu().tolist()
                pred_ids = pred[i, :length].detach().cpu().tolist()
                exact = int(pred_ids == target_ids)
                target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
                pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                chrf = chrf_score(pred_text, target_text)
                bleu = bleu_score(pred_text, target_text)
                exact_hits.append(exact)
                chrf_values.append(chrf)
                bleu_values.append(bleu)
                signer = signer_ids[i]
                by_signer[signer]["samples"] += 1
                by_signer[signer]["exact"] += exact
                by_signer[signer]["chrf"] += chrf
                by_signer[signer]["bleu"] += bleu

            totals["token_top1_sum"] += top1.sum().item()
            totals["token_top5_sum"] += top5.sum().item()
            totals["token_count"] += top1.numel()
            totals["exact_sum"] += sum(exact_hits)
            totals["chrf_sum"] += sum(chrf_values)
            totals["bleu_sum"] += sum(bleu_values)
            totals["samples"] += len(exact_hits)
            batches += 1

            if len(prediction_rows) < prediction_limit:
                rows = make_token_predictions(
                    token_logits=logits,
                    token_ids=batch["token_ids"],
                    token_lengths=batch["token_lengths"],
                    clip_ids=batch["clip_ids"],
                    glosses=batch["glosses"],
                    tokenizer=tokenizer,
                )
                prediction_rows.extend(row.as_dict() for row in rows)

    samples = max(1.0, totals["samples"])
    summary = {
        "samples": int(totals["samples"]),
        "token_top1": totals["token_top1_sum"] / max(1.0, totals["token_count"]),
        "token_top5": totals["token_top5_sum"] / max(1.0, totals["token_count"]),
        "exact": totals["exact_sum"] / samples,
        "chrf": totals["chrf_sum"] / samples,
        "bleu": totals["bleu_sum"] / samples,
    }
    signer_rows = []
    for signer, values in sorted(by_signer.items()):
        n = max(1.0, values["samples"])
        signer_rows.append(
            {
                "signer_id": signer,
                "samples": int(values["samples"]),
                "exact": values["exact"] / n,
                "chrf": values["chrf"] / n,
                "bleu": values["bleu"] / n,
            }
        )
    return summary, signer_rows, prediction_rows[:prediction_limit]


def main():
    args = parse_args()
    initialize(seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = list_clip_records(args.h5, "dataset1")
    train_records, val_records = split_records(records, args.seed, args.heldout_signer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    token_ids_by_label = make_label_tokens(records, tokenizer)
    all_token_ids = {tid for ids in token_ids_by_label.values() for tid in ids}
    embedding_rows = load_embedding_rows(args.embedding_table, all_token_ids, 2048)

    A = np.load("data/processed/adjacency_matrix.npy", allow_pickle=True)
    encoder = STGCNTemporalFrameEncoder(A, hidden_size=128)
    load_visual_low_level_weights(encoder, args.checkpoint_v121)
    model = TemporalSignPromptModel(
        encoder,
        hidden_size=128,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=2048,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    scales = [float(item) for item in args.temporal_scales.split(",") if item.strip()]
    noise_stds = [float(item) for item in args.noise_stds.split(",") if item.strip()]
    drop_rates = [float(item) for item in args.keypoint_drop_rates.split(",") if item.strip()]
    scale_reports = []
    prediction_rows = []
    for scale in scales:
        ds = IsolatedTokenDataset(
            args.h5,
            val_records,
            token_ids_by_label,
            embedding_rows,
            temporal_scale=scale,
            seed=args.seed,
            max_samples=args.max_samples,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        summary, signer_rows, rows = evaluate(
            model,
            loader,
            tokenizer,
            device,
            prediction_limit=args.prediction_samples if scale == 1.0 else 0,
        )
        scale_reports.append({"temporal_scale": scale, **summary})
        if scale == 1.0:
            prediction_rows = rows
            signer_report = signer_rows

    noise_reports = []
    for noise_std in noise_stds:
        ds = IsolatedTokenDataset(
            args.h5,
            val_records,
            token_ids_by_label,
            embedding_rows,
            noise_std=noise_std,
            seed=args.seed + 100_000,
            max_samples=args.max_samples,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        summary, _, _ = evaluate(model, loader, tokenizer, device)
        noise_reports.append({"noise_std": noise_std, **summary})

    drop_reports = []
    for drop_rate in drop_rates:
        ds = IsolatedTokenDataset(
            args.h5,
            val_records,
            token_ids_by_label,
            embedding_rows,
            keypoint_drop_rate=drop_rate,
            seed=args.seed + 200_000,
            max_samples=args.max_samples,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        summary, _, _ = evaluate(model, loader, tokenizer, device)
        drop_reports.append({"keypoint_drop_rate": drop_rate, **summary})

    result = {
        "checkpoint": str(args.checkpoint),
        "protocol": {
            "split": (
                f"dataset1 LOSO signer {args.heldout_signer}"
                if args.heldout_signer is not None
                else "dataset1 stratified validation, seed 23"
            ),
            "heldout_signer": args.heldout_signer,
            "train_records": len(train_records),
            "val_records": len(val_records),
            "alpha": "teacher_alpha/oracle boundaries",
            "temporal_scale": "<1 faster-shorter, >1 slower-longer",
            "note": (
                "LOSO evaluation over the held-out signer clips."
                if args.heldout_signer is not None
                else "Signer rows are diagnostic slices of the standard val split, not LOSO retraining."
            ),
        },
        "temporal_scale": scale_reports,
        "spatial_noise": noise_reports,
        "keypoint_dropout": drop_reports,
        "signer_slices_scale_1_0": signer_report,
        "worst_signers_by_exact": sorted(signer_report, key=lambda row: row["exact"])[:5],
        "sample_predictions": prediction_rows,
        "gemma_prompt_template": build_gemma_correction_prompt("[SECUENCIA]"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "temporal_scale",
                    "spatial_noise",
                    "keypoint_dropout",
                    "worst_signers_by_exact",
                )
            },
            indent=2,
        )
    )
    print(f"[robustness] wrote {args.output}")


if __name__ == "__main__":
    main()
