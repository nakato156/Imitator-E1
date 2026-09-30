"""Build the 10-fold clean LOSO manifest for the A3/CIF Etapa 4 correction.

Each fold holds out one signer as the outer test signer (never seen by any
training stage) and a second signer as the inner validation signer (used for
checkpoint selection, never for gradient updates). The remaining eight
signers train. Folds are fully determined by ``SIGNERS`` and ``SEED``, so the
manifest is reproducible byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SEED = 23
SIGNERS = tuple(range(1, 11))  # dataset1 has 10 signers, ids 1..10


def build_folds() -> list[dict]:
    folds = []
    for outer_test_signer in SIGNERS:
        inner_val_signer = outer_test_signer % len(SIGNERS) + 1
        train_signers = sorted(
            s for s in SIGNERS if s not in (outer_test_signer, inner_val_signer)
        )
        folds.append(
            {
                "fold": outer_test_signer,
                "outer_test_signer": outer_test_signer,
                "inner_val_signer": inner_val_signer,
                "train_signers": train_signers,
            }
        )
    return folds


def validate_folds(folds: list[dict]) -> None:
    if len(folds) != len(SIGNERS):
        raise ValueError(f"expected {len(SIGNERS)} folds, got {len(folds)}")
    for fold in folds:
        roles = (
            {fold["outer_test_signer"]},
            {fold["inner_val_signer"]},
            set(fold["train_signers"]),
        )
        union = roles[0] | roles[1] | roles[2]
        if union != set(SIGNERS):
            raise ValueError(f"fold {fold['fold']}: signer union != all signers ({union})")
        if len(roles[2]) != 8:
            raise ValueError(f"fold {fold['fold']}: expected 8 train signers, got {len(roles[2])}")
        if roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2]:
            raise ValueError(f"fold {fold['fold']}: train/val/test signer sets intersect")
    seen_outer = {fold["outer_test_signer"] for fold in folds}
    if seen_outer != set(SIGNERS):
        raise ValueError("folds do not cover every signer as outer_test_signer exactly once")


def manifest_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest() -> dict:
    folds = build_folds()
    validate_folds(folds)
    payload = {"seed": SEED, "signers": list(SIGNERS), "folds": folds}
    return {**payload, "manifest_sha256": manifest_hash(payload)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/a3_etapa4_clean_loso/manifest.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(args.output), "manifest_sha256": manifest["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
