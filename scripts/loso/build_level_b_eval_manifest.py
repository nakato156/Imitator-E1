"""Build a fixed Level B evaluation manifest: 128 reproducible sequences per
sign-count (2..8) for one signer, used to validate synthetic temporal
composition independent of the per-epoch random training sampler.

Each sequence picks ``n_signs`` *distinct* clip ids belonging to the given
signer (no repeats within a sequence) and a neutral-frame gap (0-8 by
default) between each consecutive pair. This only validates composition of
isolated clips -- it is not an evaluation on real continuous video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from src.mslm.dataloader.isolated_keypoint_dataset import list_clip_records

SIGN_COUNTS = tuple(range(2, 9))  # 2..8 inclusive
SEQUENCES_PER_COUNT = 128


def signer_clip_ids(records: list[dict], signer: int) -> list[dict]:
    return [r for r in records if r["signer_id"] == signer]


def build_sequences(
    clips: list[dict],
    signer: int,
    seed: int,
    min_neutral_frames: int = 0,
    max_neutral_frames: int = 8,
    sequences_per_count: int = SEQUENCES_PER_COUNT,
) -> list[dict]:
    if len(clips) < max(SIGN_COUNTS):
        raise ValueError(
            f"signer={signer} has only {len(clips)} clips, need at least {max(SIGN_COUNTS)}"
        )
    sequences = []
    for n_signs in SIGN_COUNTS:
        for i in range(sequences_per_count):
            rng = random.Random(f"{seed}-{signer}-{n_signs}-{i}")
            chosen = rng.sample(clips, n_signs)
            gaps = [rng.randint(min_neutral_frames, max_neutral_frames) for _ in range(n_signs - 1)]
            sequences.append(
                {
                    "n_signs": n_signs,
                    "index": i,
                    "clip_ids": [c["clip_id"] for c in chosen],
                    "labels": [c["label"] for c in chosen],
                    "neutral_gaps": gaps,
                }
            )
    return sequences


def manifest_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest(
    h5_path: Path,
    signer: int,
    seed: int,
    min_neutral_frames: int = 0,
    max_neutral_frames: int = 8,
    sequences_per_count: int = SEQUENCES_PER_COUNT,
) -> dict:
    records = list_clip_records(h5_path, "dataset1")
    clips = signer_clip_ids(records, signer)
    sequences = build_sequences(
        clips, signer, seed, min_neutral_frames, max_neutral_frames, sequences_per_count
    )
    payload = {
        "signer": signer,
        "seed": seed,
        "sign_counts": list(SIGN_COUNTS),
        "sequences_per_count": sequences_per_count,
        "total_sequences": len(sequences),
        "min_neutral_frames": min_neutral_frames,
        "max_neutral_frames": max_neutral_frames,
        "sequences": sequences,
    }
    return {**payload, "manifest_sha256": manifest_hash(payload)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5", type=Path, default=Path("data/processed/dataset1_isolated_v122.hdf5")
    )
    parser.add_argument("--signer", type=int, required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--min-neutral-frames", type=int, default=0)
    parser.add_argument("--max-neutral-frames", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_manifest(
        args.h5, args.signer, args.seed, args.min_neutral_frames, args.max_neutral_frames
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(args.output),
                "total_sequences": manifest["total_sequences"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
