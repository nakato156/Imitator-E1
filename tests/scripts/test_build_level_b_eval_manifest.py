import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_level_b_eval_manifest", ROOT / "scripts" / "loso" / "build_level_b_eval_manifest.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["build_level_b_eval_manifest"] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _fixture_h5(tmp_path, n_clips_per_signer=10, n_signers=2):
    path = tmp_path / "dataset1.hdf5"
    with h5py.File(path, "w") as f:
        g = f.create_group("dataset1")
        kg = g.create_group("keypoints")
        labels_g = g.create_group("labels")
        signer_g = g.create_group("signer_id")
        clip_id = 0
        for signer in range(1, n_signers + 1):
            for i in range(n_clips_per_signer):
                cid = str(clip_id)
                kg.create_dataset(cid, data=np.ones((3, 2, 2), dtype=np.float32))
                labels_g.create_dataset(cid, data=[f"gloss{i}".encode()])
                signer_g.create_dataset(cid, data=[signer])
                clip_id += 1
    return path


def test_build_manifest_has_896_sequences_with_correct_bucket_sizes(tmp_path):
    h5_path = _fixture_h5(tmp_path, n_clips_per_signer=10, n_signers=1)
    manifest = MODULE.build_manifest(h5_path, signer=1, seed=23)
    assert manifest["total_sequences"] == 896
    counts = {}
    for seq in manifest["sequences"]:
        counts[seq["n_signs"]] = counts.get(seq["n_signs"], 0) + 1
    assert counts == {n: 128 for n in range(2, 9)}


def test_sequences_never_repeat_a_clip_within_themselves(tmp_path):
    h5_path = _fixture_h5(tmp_path, n_clips_per_signer=10, n_signers=1)
    manifest = MODULE.build_manifest(h5_path, signer=1, seed=23)
    for seq in manifest["sequences"]:
        assert len(set(seq["clip_ids"])) == len(seq["clip_ids"]) == seq["n_signs"]


def test_sequences_only_use_clips_from_the_requested_signer(tmp_path):
    h5_path = _fixture_h5(tmp_path, n_clips_per_signer=10, n_signers=2)
    records = MODULE.list_clip_records(h5_path, "dataset1")
    signer1_clips = {r["clip_id"] for r in records if r["signer_id"] == 1}

    manifest = MODULE.build_manifest(h5_path, signer=1, seed=23)

    for seq in manifest["sequences"]:
        assert set(seq["clip_ids"]) <= signer1_clips


def test_neutral_gaps_within_requested_range(tmp_path):
    h5_path = _fixture_h5(tmp_path, n_clips_per_signer=10, n_signers=1)
    manifest = MODULE.build_manifest(h5_path, signer=1, seed=23, min_neutral_frames=2, max_neutral_frames=5)
    for seq in manifest["sequences"]:
        assert len(seq["neutral_gaps"]) == seq["n_signs"] - 1
        assert all(2 <= g <= 5 for g in seq["neutral_gaps"])


def test_manifest_is_deterministic_given_same_seed(tmp_path):
    h5_path = _fixture_h5(tmp_path, n_clips_per_signer=10, n_signers=1)
    a = MODULE.build_manifest(h5_path, signer=1, seed=23)
    b = MODULE.build_manifest(h5_path, signer=1, seed=23)
    assert a == b


def test_different_seeds_produce_different_sequences(tmp_path):
    h5_path = _fixture_h5(tmp_path, n_clips_per_signer=10, n_signers=1)
    a = MODULE.build_manifest(h5_path, signer=1, seed=23)
    b = MODULE.build_manifest(h5_path, signer=1, seed=99)
    assert a["sequences"] != b["sequences"]
    assert a["manifest_sha256"] != b["manifest_sha256"]


def test_raises_when_signer_has_too_few_clips(tmp_path):
    h5_path = _fixture_h5(tmp_path, n_clips_per_signer=5, n_signers=1)
    with pytest.raises(ValueError):
        MODULE.build_manifest(h5_path, signer=1, seed=23)


if __name__ == "__main__":
    print("run via pytest")
