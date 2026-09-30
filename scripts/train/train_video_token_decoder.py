#!/usr/bin/env python3
"""Train/evaluate the CIF-independent autoregressive video token decoder.

Training never touches the outer-test signer.  Outer evaluation is a distinct
subcommand and only accepts a checkpoint that the train command has closed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.mslm.dataloader.isolated_keypoint_dataset import list_clip_records
from src.mslm.dataloader.synthetic_temporal import (
    SyntheticTemporalSignDataset,
    synthetic_temporal_collate,
)
from src.mslm.models.temporal_sign_prompt import (
    STGCNTemporalFrameEncoder,
    TemporalSignPromptModel,
    rescale_alphas_to_predicted_lengths,
)
from src.mslm.models.video_token_decoder import (
    BOS_ID,
    EOS_ID,
    GEMMA_VOCAB_SIZE,
    MAX_CONTENT_TOKENS,
    PAD_ID,
    VideoTokenDecoder,
    build_teacher_forcing,
    initialize_from_cif_checkpoint,
    set_decoder_training_stage,
    sha256_file,
    validate_checkpoint_lineage,
)
from src.mslm.utils.sequence_metrics import strict_exact, token_edit_similarity, token_error_rate


DEFAULT_H5 = Path("data/processed/dataset1_isolated_v122.hdf5")
DEFAULT_TOKENIZER = Path(
    ".hf_cache/hub/"
    "models--unsloth--gemma-3n-E2B-it-unsloth-bnb-4bit/"
    "snapshots/3d26ffdd2276698f582562bf01511aa625bc6f30"
)
DEFAULT_MANIFEST = ROOT / "experiments/a3_etapa4_clean_loso/manifest.json"
DEFAULT_SOURCE_ROOT = ROOT.parent / "outputs/loso_clean"
DEFAULT_OUTPUT_ROOT = ROOT.parent / "outputs/video_token_decoder"
DEFAULT_ADJACENCY = ROOT.parent / "data/processed/adjacency_matrix.npy"


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).strip().lower().split())


def canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    recorded = manifest.get("manifest_sha256")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not recorded or canonical_hash(payload) != recorded:
        raise RuntimeError(f"invalid manifest hash: {path}")
    return manifest


def fold_spec(manifest: dict, fold: int) -> dict:
    if fold not in range(1, 11):
        raise ValueError("this protocol permits only folds 1-10")
    try:
        spec = next(row for row in manifest["folds"] if int(row["fold"]) == fold)
    except StopIteration as exc:
        raise ValueError(f"fold {fold} is absent from manifest") from exc
    roles = [
        {int(spec["outer_test_signer"])},
        {int(spec["inner_val_signer"])},
        set(map(int, spec["train_signers"])),
    ]
    if roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2]:
        raise RuntimeError("manifest contains signer-role contamination")
    return spec


def default_source_checkpoint(root: Path, fold: int) -> Path:
    candidates = (
        root / f"fold{fold}_promotion/checkpoint_best.pt",
        root / f"diag_fold{fold}_promotion/checkpoint_best.pt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def recorded_source_hash(checkpoint: Path) -> str:
    done_path = checkpoint.parent / "stage_done.json"
    if not done_path.exists():
        raise RuntimeError(f"missing hash record: {done_path}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    expected = done.get("checkpoint_sha256")
    if not expected:
        raise RuntimeError(f"stage record has no checkpoint_sha256: {done_path}")
    return str(expected)


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.vocab_size != GEMMA_VOCAB_SIZE:
        raise RuntimeError(
            f"expected full Gemma vocabulary {GEMMA_VOCAB_SIZE}, got {tokenizer.vocab_size}"
        )
    return tokenizer


def label_tokens(records: list[dict], tokenizer) -> dict[str, list[int]]:
    result = {}
    for label in sorted({row["label"] for row in records}):
        ids = tokenizer(normalize_text(label), add_special_tokens=False).input_ids
        if len(ids) > 4:
            raise RuntimeError(
                f"label {label!r} uses {len(ids)} tokens; protocol assumes at most 4"
            )
        result[label] = list(map(int, ids))
    return result


def rows_for_signers(records: list[dict], signers) -> list[dict]:
    allowed = set(map(int, signers))
    return [row for row in records if int(row["signer_id"]) in allowed]


def make_dataset(
    args,
    rows: list[dict],
    tokens: dict[str, list[int]],
    *,
    samples: int,
    seed: int,
    rescue: bool = False,
):
    return SyntheticTemporalSignDataset(
        args.h5,
        [row["clip_id"] for row in rows],
        {row["clip_id"]: row["label"] for row in rows},
        tokens,
        min_clips=2,
        max_clips=8,
        min_neutral_frames=0,
        max_neutral_frames=8,
        samples_per_epoch=samples,
        seed=seed,
        apply_remove_keypoints=True,
        normalize=True,
        n_keypoints=111,
        rescue_augmentation=rescue,
        jitter_std=0.01,
        temporal_drop_rate=0.15,
        augmentation_probability=0.5,
    )


def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=synthetic_temporal_collate,
        pin_memory=torch.cuda.is_available(),
    )


def make_vocab_map(tokens: dict[str, list[int]]) -> list[int]:
    """Dense output vocabulary: specials first, then the effective Gemma ids."""
    effective = sorted({t for ids in tokens.values() for t in ids})
    if set(effective) & {PAD_ID, BOS_ID, EOS_ID}:
        raise RuntimeError("special ids collide with effective label tokens")
    return [PAD_ID, BOS_ID, EOS_ID, *effective]


def build_model(
    adjacency: Path,
    *,
    encoder_pe: bool = False,
    vocab_map: list[int] | None = None,
    with_ctc: bool = False,
) -> VideoTokenDecoder:
    matrix = np.load(adjacency, allow_pickle=True)
    encoder = STGCNTemporalFrameEncoder(
        matrix,
        hidden_size=128,
        tcn_layers=2,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.1,
        use_positional_encoding=encoder_pe,
    )
    return VideoTokenDecoder(
        encoder,
        hidden_size=128,
        vocab_size=GEMMA_VOCAB_SIZE if vocab_map is None else len(vocab_map),
        num_layers=2,
        num_heads=4,
        ffn_size=512,
        dropout=0.1,
        max_content_tokens=MAX_CONTENT_TOKENS,
        vocab_map=vocab_map,
        with_ctc=with_ctc,
    )


def move_core_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    # Deliberately extract only the three fields allowed by the AR model/loss.
    return (
        batch["keypoints"].to(device),
        batch["frame_lengths"].to(device),
        batch["token_ids"].to(device),
    )


def sequence_metrics(predictions, emitted_eos, targets, tokenizer=None) -> tuple[dict, list[dict]]:
    totals = {
        "strict_exact": 0.0,
        "token_error_rate": 0.0,
        "token_edit_similarity": 0.0,
        "text_exact": 0.0,
        "predicted_length": 0.0,
        "length_mae": 0.0,
        "eos_rate": 0.0,
        "no_eos_rate": 0.0,
    }
    rows = []
    for pred, has_eos, target in zip(predictions, emitted_eos, targets):
        target = list(map(int, target))
        pred = list(map(int, pred))
        exact = bool(has_eos) and strict_exact(pred, target)
        ter = token_error_rate(pred, target)
        text_pred = tokenizer.decode(pred, skip_special_tokens=True).strip() if tokenizer else ""
        text_target = tokenizer.decode(target, skip_special_tokens=True).strip() if tokenizer else ""
        text_exact = bool(has_eos) and text_pred == text_target if tokenizer else exact
        values = {
            "strict_exact": float(exact),
            "token_error_rate": ter,
            "token_edit_similarity": token_edit_similarity(ter),
            "text_exact": float(text_exact),
            "predicted_length": float(len(pred) if has_eos else MAX_CONTENT_TOKENS + 1),
            "length_mae": float(abs((len(pred) if has_eos else MAX_CONTENT_TOKENS + 1) - len(target))),
            "eos_rate": float(bool(has_eos)),
            "no_eos_rate": float(not has_eos),
        }
        for key, value in values.items():
            totals[key] += value
        rows.append({"pred": pred, "target": target, "emitted_eos": bool(has_eos), **values})
    count = max(1, len(rows))
    return {key: value / count for key, value in totals.items()}, rows


@torch.no_grad()
def evaluate_model(model, loader, device, tokenizer=None) -> tuple[dict, list[dict]]:
    model.eval()
    predictions, eos_flags, targets = [], [], []
    for batch in loader:
        keypoints, frame_lengths, token_ids = move_core_batch(batch, device)
        decoded = model.greedy_decode(keypoints, frame_lengths)
        predictions.extend(decoded.token_ids)
        eos_flags.extend(decoded.emitted_eos.cpu().tolist())
        targets.extend(row[row.ne(-100)].cpu().tolist() for row in token_ids)
    return sequence_metrics(predictions, eos_flags, targets, tokenizer)


def optimizer_for(model: VideoTokenDecoder) -> torch.optim.Optimizer:
    visual_ids = {
        id(parameter)
        for module in (model.frame_encoder.tcn, model.frame_encoder.transformer)
        for parameter in module.parameters()
    }
    stgcn_ids = {
        id(parameter)
        for module in (model.frame_encoder.stgcn_layers, model.frame_encoder.linear_hidden)
        for parameter in module.parameters()
    }
    decoder = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in visual_ids and id(parameter) not in stgcn_ids
    ]
    visual = [parameter for parameter in model.parameters() if id(parameter) in visual_ids]
    stgcn = [parameter for parameter in model.parameters() if id(parameter) in stgcn_ids]
    return torch.optim.AdamW(
        [
            {"params": decoder, "lr": 3e-4, "name": "decoder"},
            {"params": visual, "lr": 3e-5, "name": "visual_tcn_transformer"},
            # Inert while frozen; lr 1e-5 only matters under --unfreeze-stgcn-epoch.
            {"params": stgcn, "lr": 1e-5, "name": "stgcn_linear_hidden"},
        ],
        weight_decay=1e-4,
    )


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    accumulation: int,
    *,
    label_smoothing: float = 0.1,
    ctc_weight: float = 0.0,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, batches = 0.0, 0
    for step, batch in enumerate(loader):
        keypoints, frame_lengths, token_ids = move_core_batch(batch, device)
        dense_ids = model.to_dense(token_ids)
        decoder_input, labels = build_teacher_forcing(
            dense_ids, bos_id=model.bos_id, eos_id=model.eos_id, pad_id=model.pad_id
        )
        features = model.frame_encoder(keypoints, frame_lengths)
        logits = model.decode_features(features, frame_lengths, decoder_input)
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            labels.flatten(),
            ignore_index=model.pad_id,
            label_smoothing=label_smoothing,
        )
        if ctc_weight > 0.0:
            rows = [row[row.ge(0)] for row in dense_ids]
            ctc_log_probs = model.ctc_head(features).log_softmax(-1).transpose(0, 1)
            ctc = F.ctc_loss(
                ctc_log_probs,
                torch.cat(rows),
                frame_lengths,
                torch.tensor([r.numel() for r in rows], device=device),
                blank=model.ctc_blank_id,
                zero_infinity=True,
            )
            loss = (1.0 - ctc_weight) * loss + ctc_weight * ctc
        (loss / accumulation).backward()
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach())
        batches += 1
    return total_loss / max(1, batches)


@torch.no_grad()
def ctc_greedy_decode(model, keypoints, frame_lengths) -> list[list[int]]:
    """Collapse argmax frames (remove repeats, drop blanks) -> Gemma ids."""
    features = model.frame_encoder(keypoints, frame_lengths)
    frame_ids = model.ctc_head(features).argmax(-1)
    sequences = []
    for row, length in zip(frame_ids, frame_lengths.tolist()):
        collapsed, previous = [], model.ctc_blank_id
        for token in row[:length].tolist():
            if token != previous and token != model.ctc_blank_id:
                collapsed.append(token)
            previous = token
        sequences.append(model.to_gemma(collapsed))
    return sequences


@torch.no_grad()
def evaluate_model_ctc(model, loader, device, tokenizer=None) -> tuple[dict, list[dict]]:
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        keypoints, frame_lengths, token_ids = move_core_batch(batch, device)
        predictions.extend(ctc_greedy_decode(model, keypoints, frame_lengths))
        targets.extend(row[row.ne(-100)].cpu().tolist() for row in token_ids)
    return sequence_metrics(predictions, [True] * len(predictions), targets, tokenizer)


def checkpoint_payload(model, optimizer, epoch, metric, provenance, args, *, closed=False):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_select_metric": list(metric),
        "closed": bool(closed),
        "provenance": provenance,
        "config": {
            "hidden_size": 128,
            "decoder_layers": 2,
            "heads": 4,
            "ffn_size": 512,
            "dropout": 0.1,
            "vocab_size": model.vocab_size,
            "bos_id": BOS_ID,
            "eos_id": EOS_ID,
            "pad_id": PAD_ID,
            "max_content_tokens": MAX_CONTENT_TOKENS,
            "epochs": args.epochs,
            "samples_per_epoch": 2048,
            "seed": args.seed,
            "rescue_augmentation": bool(args.rescue_augmentation),
            "encoder_pe": bool(args.encoder_pe),
            "vocab_map": model.vocab_map,
            "ctc_weight": float(args.ctc_weight),
            "label_smoothing": float(args.label_smoothing),
            "unfreeze_stgcn_epoch": args.unfreeze_stgcn_epoch,
            "select": args.select,
            "run_tag": args.run_tag,
        },
    }


def train_once(args, batch_size: int, accumulation: int) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.manifest)
    spec = fold_spec(manifest, args.fold)
    source = args.source_checkpoint or default_source_checkpoint(args.source_root, args.fold)
    expected_hash = recorded_source_hash(source)

    records = list_clip_records(args.h5, "dataset1")
    train_rows = rows_for_signers(records, spec["train_signers"])
    val_rows = rows_for_signers(records, [spec["inner_val_signer"]])
    outer = int(spec["outer_test_signer"])
    if any(int(row["signer_id"]) == outer for row in train_rows + val_rows):
        raise RuntimeError("outer signer leaked into train/inner-val rows")
    tokenizer = load_tokenizer(args.tokenizer)
    tokens = label_tokens(records, tokenizer)

    train_ds = make_dataset(
        args, train_rows, tokens, samples=2048, seed=args.seed,
        rescue=args.rescue_augmentation,
    )
    val_ds = make_dataset(args, val_rows, tokens, samples=args.val_samples, seed=args.seed + 100_000)
    train_loader = make_loader(train_ds, batch_size, True)
    val_loader = make_loader(val_ds, batch_size, False)

    vocab_map = make_vocab_map(tokens) if args.restricted_vocab else None
    model = build_model(
        args.adjacency,
        encoder_pe=args.encoder_pe,
        vocab_map=vocab_map,
        with_ctc=args.ctc_weight > 0,
    )
    source_info = initialize_from_cif_checkpoint(
        model, source, outer_signer=outer, expected_sha256=expected_hash
    )
    model.to(device)
    optimizer = optimizer_for(model)
    out_dir = args.output_root / f"fold{args.fold}_seed{args.seed}" / (
        args.run_tag or ("rescue" if args.rescue_augmentation else "base")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    best_path = out_dir / "checkpoint_best.pt"
    best_metric = (-1.0, -1.0)
    best_epoch = -1
    provenance = {
        "fold": args.fold,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "outer_test_signer": outer,
        "inner_val_signer": int(spec["inner_val_signer"]),
        "train_signers": list(map(int, spec["train_signers"])),
        "source_checkpoint": str(source),
        **source_info,
        "batch_size": batch_size,
        "gradient_accumulation": accumulation,
    }
    (out_dir / "config.json").write_text(
        json.dumps({"provenance": provenance, "args": vars(args)}, indent=2, default=str),
        encoding="utf-8",
    )

    for epoch in range(args.epochs):
        stage = set_decoder_training_stage(
            model, epoch, unfreeze_stgcn_epoch=args.unfreeze_stgcn_epoch
        )
        loss = train_epoch(
            model, train_loader, optimizer, device, accumulation,
            label_smoothing=args.label_smoothing, ctc_weight=args.ctc_weight,
        )
        val_metrics, _ = evaluate_model(model, val_loader, device, tokenizer)
        if args.select == "edit":
            metric = (val_metrics["token_edit_similarity"], val_metrics["strict_exact"])
        else:
            metric = (val_metrics["strict_exact"], val_metrics["token_edit_similarity"])
        row = {"epoch": epoch, "train_loss": loss, "stage": stage, "inner_val": val_metrics}
        if args.ctc_weight > 0:
            ctc_metrics, _ = evaluate_model_ctc(model, val_loader, device, tokenizer)
            row["inner_val_ctc"] = ctc_metrics
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"fold={args.fold} epoch={epoch:02d} loss={loss:.4f} "
            f"exact={val_metrics['strict_exact']:.4f} "
            f"edit_sim={val_metrics['token_edit_similarity']:.4f}", flush=True
        )
        if metric > best_metric:
            best_metric, best_epoch = metric, epoch
            torch.save(
                checkpoint_payload(model, optimizer, epoch, metric, provenance, args), best_path
            )

    best = torch.load(best_path, map_location="cpu")
    best["closed"] = True
    best["closed_after_epoch"] = args.epochs - 1
    closed_path = out_dir / "checkpoint_closed.pt"
    torch.save(best, closed_path)
    (out_dir / "closed.json").write_text(
        json.dumps(
            {
                "checkpoint": str(closed_path),
                "checkpoint_sha256": sha256_file(closed_path),
                "best_epoch": best_epoch,
                "best_select_metric": list(best_metric),
                "outer_test_evaluated": False,
            }, indent=2
        ), encoding="utf-8"
    )
    return closed_path


def command_train(args) -> None:
    try:
        path = train_once(args, batch_size=2, accumulation=2)
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM: restarting with batch=1, accumulation=4", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        path = train_once(args, batch_size=1, accumulation=4)
    print(json.dumps({"closed_checkpoint": str(path), "sha256": sha256_file(path)}, indent=2))


def load_closed_decoder(args, spec, device):
    state = torch.load(args.checkpoint, map_location="cpu")
    if not state.get("closed"):
        raise RuntimeError("outer evaluation requires a closed checkpoint")
    provenance = state.get("provenance", {})
    if int(provenance.get("fold", -1)) != args.fold:
        raise RuntimeError("checkpoint fold does not match requested fold")
    if int(provenance.get("outer_test_signer", -1)) != int(spec["outer_test_signer"]):
        raise RuntimeError("checkpoint outer signer does not match manifest")
    config = state.get("config", {})
    model = build_model(
        args.adjacency,
        encoder_pe=bool(config.get("encoder_pe", False)),
        vocab_map=config.get("vocab_map"),
        with_ctc=float(config.get("ctc_weight", 0.0)) > 0,
    )
    model.load_state_dict(state["model"], strict=True)
    return model.to(device), state


def command_evaluate(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.manifest)
    spec = fold_spec(manifest, args.fold)
    model, state = load_closed_decoder(args, spec, device)
    records = list_clip_records(args.h5, "dataset1")
    tokenizer = load_tokenizer(args.tokenizer)
    tokens = label_tokens(records, tokenizer)
    outer_rows = rows_for_signers(records, [spec["outer_test_signer"]])
    dataset = make_dataset(args, outer_rows, tokens, samples=args.eval_samples, seed=args.seed + 200_000)
    evaluate_fn = evaluate_model_ctc if args.decode == "ctc" else evaluate_model
    if args.decode == "ctc" and model.ctc_head is None:
        raise RuntimeError("checkpoint has no CTC head; cannot decode with --decode ctc")
    metrics, rows = evaluate_fn(model, make_loader(dataset, args.eval_batch_size, False), device, tokenizer)
    result = {
        "fold": args.fold,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "outer_test_signer": int(spec["outer_test_signer"]),
        "metrics": metrics,
        "predictions": rows,
    }
    if args.cif_comparator:
        cif = json.loads(args.cif_comparator.read_text(encoding="utf-8"))
        if int(cif["fold"]) != args.fold:
            raise RuntimeError("CIF comparator fold mismatch")
        cif_rows = cif["outer_test"]["predictions"]
        if len(cif_rows) != len(rows):
            raise RuntimeError("paired CIF/decoder sample counts differ")
        differences = [
            int(ar["strict_exact"]) - int(cr["strict_exact"])
            for ar, cr in zip(rows, cif_rows)
        ]
        result["paired_vs_affine_cif"] = {
            "strict_exact_delta": sum(differences) / max(1, len(differences)),
            "wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
        }
    default_name = "outer_test_ctc.json" if args.decode == "ctc" else "outer_test.json"
    output = args.output or args.checkpoint.with_name(default_name)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))


def cif_model_from_checkpoint(args, source: Path, tokenizer, outer: int):
    state = torch.load(source, map_location="cpu")
    validate_checkpoint_lineage(state, outer)
    matrix = np.load(args.adjacency, allow_pickle=True)
    encoder = STGCNTemporalFrameEncoder(matrix, hidden_size=128)
    arch = state.get("arch_config", {})
    source_model = state["model"]
    max_len = source_model["length_head.classifier.1.weight"].shape[0] - 1
    model = TemporalSignPromptModel(
        encoder,
        hidden_size=128,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=source_model["embedding_head.weight"].shape[0],
        max_len_class=max_len,
        token_head_variant=arch.get("token_head", "contextual"),
        length_head_variant=arch.get("length_head", "attention"),
    )
    model.load_state_dict(source_model, strict=True)
    return model


@torch.no_grad()
def collect_cif_counts(model, loader, device):
    model.eval()
    raw, head, targets = [], [], []
    cached = []
    for batch in loader:
        keypoints, frame_lengths, token_ids = move_core_batch(batch, device)
        features = model.frame_encoder(keypoints, frame_lengths)
        alphas = model.cif.predict_alpha(features, frame_lengths)
        raw.extend(alphas.sum(1).cpu().tolist())
        head.extend(model.predict_lengths(features, frame_lengths).cpu().tolist())
        targets.extend(row[row.ne(-100)].cpu().tolist() for row in token_ids)
        cached.append((features.cpu(), frame_lengths.cpu(), alphas.cpu(), token_ids.cpu()))
    return raw, head, targets, cached


def count_report(raw, head, targets, a=1.0, b=0.0):
    target_lengths = [len(row) for row in targets]
    rounded = [max(1, round(value)) for value in raw]
    affine = [max(1, min(MAX_CONTENT_TOKENS, round(a * value + b))) for value in raw]
    def summary(pred):
        return {
            "count_match": sum(p == t for p, t in zip(pred, target_lengths)) / max(1, len(pred)),
            "length_mae": sum(abs(p - t) for p, t in zip(pred, target_lengths)) / max(1, len(pred)),
        }
    return {
        "length_head": summary(list(map(int, head))),
        "raw": {"length_mae": sum(abs(p-t) for p,t in zip(raw,target_lengths))/max(1,len(raw))},
        "rounded": summary(rounded),
        "affine": {**summary(affine), "a": a, "b": b},
    }


def fit_affine(raw, targets):
    target_lengths = [len(row) for row in targets]
    best = None
    for ai in range(70, 131, 5):
        for bi in range(-60, 61, 5):
            a, b = ai / 100, bi / 100
            pred = [max(1, min(MAX_CONTENT_TOKENS, round(a * value + b))) for value in raw]
            match = sum(p == t for p, t in zip(pred, target_lengths)) / max(1, len(pred))
            mae = sum(abs(p - t) for p, t in zip(pred, target_lengths)) / max(1, len(pred))
            candidate = (match, -mae, -abs(a - 1.0), -abs(b), a, b)
            if best is None or candidate > best:
                best = candidate
    return best[-2], best[-1]


@torch.no_grad()
def cif_affine_predictions(model, cached, a, b, tokenizer):
    predictions, targets = [], []
    device = next(model.parameters()).device
    for features, lengths, alphas, token_ids in cached:
        features, lengths, alphas = features.to(device), lengths.to(device), alphas.to(device)
        pred_lengths = (a * alphas.sum(1) + b).round().clamp(1, MAX_CONTENT_TOKENS)
        scaled, decided = rescale_alphas_to_predicted_lengths(
            alphas, pred_lengths, max_len=MAX_CONTENT_TOKENS
        )
        cif = model.cif(features, lengths, alphas=scaled)
        ids = model.token_head(cif.embeddings, cif.padding_mask).argmax(-1)
        for i, count in enumerate(decided.tolist()):
            predictions.append(ids[i, :count].cpu().tolist())
            targets.append(token_ids[i][token_ids[i].ne(-100)].tolist())
    metrics, rows = sequence_metrics(predictions, [True] * len(predictions), targets, tokenizer)
    return metrics, rows


def command_cif_comparator(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.manifest)
    spec = fold_spec(manifest, args.fold)
    source = args.source_checkpoint or default_source_checkpoint(args.source_root, args.fold)
    expected = recorded_source_hash(source)
    if sha256_file(source) != expected:
        raise RuntimeError("source checkpoint hash differs from stage_done.json")
    records = list_clip_records(args.h5, "dataset1")
    tokenizer = load_tokenizer(args.tokenizer)
    tokens = label_tokens(records, tokenizer)
    model = cif_model_from_checkpoint(
        args, source, tokenizer, int(spec["outer_test_signer"])
    ).to(device)
    inner_ds = make_dataset(
        args, rows_for_signers(records, [spec["inner_val_signer"]]), tokens,
        samples=args.eval_samples, seed=args.seed + 100_000,
    )
    outer_ds = make_dataset(
        args, rows_for_signers(records, [spec["outer_test_signer"]]), tokens,
        samples=args.eval_samples, seed=args.seed + 200_000,
    )
    inner = collect_cif_counts(model, make_loader(inner_ds, args.eval_batch_size, False), device)
    a, b = fit_affine(inner[0], inner[2])
    outer = collect_cif_counts(model, make_loader(outer_ds, args.eval_batch_size, False), device)
    inner_seq, _ = cif_affine_predictions(model, inner[3], a, b, tokenizer)
    outer_seq, outer_rows = cif_affine_predictions(model, outer[3], a, b, tokenizer)
    result = {
        "fold": args.fold,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": expected,
        "calibration_frozen_on": "inner_val",
        "inner_val": {"counts": count_report(inner[0], inner[1], inner[2], a, b), "sequence": inner_seq},
        "outer_test": {
            "counts": count_report(outer[0], outer[1], outer[2], a, b),
            "sequence": outer_seq,
            "predictions": outer_rows,
        },
    }
    output = args.output or args.output_root / f"fold{args.fold}_cif_comparator.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "a": a, "b": b}, indent=2))


def shared_parser(parser):
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--adjacency", type=Path, default=DEFAULT_ADJACENCY)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    shared_parser(train)
    train.add_argument("--val-samples", type=int, default=512)
    train.add_argument("--rescue-augmentation", action="store_true")
    train.add_argument("--epochs", type=int, default=15)
    train.add_argument("--encoder-pe", action="store_true",
                       help="sinusoidal positional encoding before the encoder transformer")
    train.add_argument("--restricted-vocab", action="store_true",
                       help="dense output layer over the effective Gemma ids (reversible map)")
    train.add_argument("--ctc-weight", type=float, default=0.0,
                       help="auxiliary frame-level CTC loss weight (0 disables)")
    train.add_argument("--label-smoothing", type=float, default=0.1)
    train.add_argument("--unfreeze-stgcn-epoch", type=int, default=None,
                       help="unfreeze ST-GCN+linear_hidden (lr 1e-5) from this epoch; default never")
    train.add_argument("--select", choices=("exact", "edit"), default="exact",
                       help="primary inner-val checkpoint selection metric")
    train.add_argument("--run-tag", default=None,
                       help="output subdirectory name (default: base/rescue)")
    train.set_defaults(func=command_train)
    evaluate = sub.add_parser("evaluate")
    shared_parser(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--eval-samples", type=int, default=896)
    evaluate.add_argument("--eval-batch-size", type=int, default=2)
    evaluate.add_argument("--cif-comparator", type=Path)
    evaluate.add_argument("--decode", choices=("ar", "ctc"), default="ar")
    evaluate.add_argument("--output", type=Path)
    evaluate.set_defaults(func=command_evaluate)
    cif = sub.add_parser("cif-comparator")
    shared_parser(cif)
    cif.add_argument("--eval-samples", type=int, default=896)
    cif.add_argument("--eval-batch-size", type=int, default=2)
    cif.add_argument("--output", type=Path)
    cif.set_defaults(func=command_cif_comparator)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    arguments.func(arguments)
