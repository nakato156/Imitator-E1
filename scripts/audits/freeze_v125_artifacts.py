"""Congela hashes y metadata de los artefactos que gobiernan v126+."""

import argparse
import hashlib
import json
from pathlib import Path

MODEL_ID = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"
ARTIFACTS = {
    "dataset_h5": "data/processed/dataset_v6_unsloth.hdf5",
    "embedding_table": "data/processed/gemma3n_embed_table.pt",
    "manifest": "data/processed/dataset2_manifest_v125.json",
    "split": "data/processed/dataset2_split_v125.json",
    "anisotropy_gate": "data/processed/v125_anisotropy_gate.json",
    "embedding_transform": "data/processed/v125_anisotropy_gate.transform.pt",
    "gemma_oracle": "data/processed/v125_gemma_oracle.json",
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_record(root: Path) -> dict:
    missing = [relative for relative in ARTIFACTS.values() if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"faltan artefactos v125: {missing}")

    artifacts = {}
    for name, relative in ARTIFACTS.items():
        path = root / relative
        artifacts[name] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
        }

    anisotropy = json.loads((root / ARTIFACTS["anisotropy_gate"]).read_text())
    oracle = json.loads((root / ARTIFACTS["gemma_oracle"]).read_text())["summary"]
    model_cache = (
        root
        / ".hf_cache/hub/models--unsloth--gemma-3n-E2B-it-unsloth-bnb-4bit"
    )
    revision_path = model_cache / "refs/main"
    revision = revision_path.read_text().strip() if revision_path.is_file() else None
    model_files = {}
    if revision:
        snapshot = model_cache / "snapshots" / revision
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            path = snapshot / filename
            if path.is_file():
                model_files[filename] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_of(path),
                }
    return {
        "schema_version": 1,
        "tokenizer_model_id": MODEL_ID,
        "model_revision": revision,
        "model_files": model_files,
        "selected_embedding_mode": anisotropy["selected_mode"],
        "gates": {
            "pca_cosine_mean": anisotropy["gate_cosine_mean_ge_0.99"],
            "pca_cosine_p5": anisotropy["gate_cosine_p5_ge_0.97"],
            "gemma_oracle": oracle["all_gates_pass"],
        },
        "artifacts": artifacts,
    }


def main(root: Path, out_path: Path) -> None:
    record = build_record(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"[freeze] {len(record['artifacts'])}/{len(ARTIFACTS)} artefactos: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/v125_artifacts.json")
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output = args.out if args.out.is_absolute() else project_root / args.out
    main(project_root, output)
