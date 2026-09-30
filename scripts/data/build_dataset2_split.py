"""Split persistente de dataset2 (v125): solo sobre clips que pasaron el
manifiesto de integridad (Task 4), agrupado por source_group (Task 2)."""
import argparse
import json
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.mslm.dataloader.grouped_split import make_grouped_split  # noqa: E402


def main(h5_path: Path, manifest_path: Path, out_path: Path, seed: int) -> None:
    manifest = json.loads(manifest_path.read_text())
    kept_ids = [m["clip_id"] for m in manifest["manifest"] if m["kept"]]

    with h5py.File(h5_path, "r") as f:
        g = f["dataset2"]["source_group"]
        source_groups = [g[cid][0].decode() for cid in kept_ids]

    split = make_grouped_split(kept_ids, source_groups, seed=seed, ratios=(0.8, 0.1, 0.1))
    payload = {"seed": seed, "ratios": [0.8, 0.1, 0.1], **split}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[split] train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=Path("data/processed/dataset_v6_unsloth.hdf5"))
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/dataset2_manifest_v125.json"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/dataset2_split_v125.json"))
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    main(
        args.h5 if args.h5.is_absolute() else root / args.h5,
        args.manifest if args.manifest.is_absolute() else root / args.manifest,
        args.out if args.out.is_absolute() else root / args.out,
        args.seed,
    )
