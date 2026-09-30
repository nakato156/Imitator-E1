import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_clean_loso_orchestrator", ROOT / "scripts" / "loso" / "run_clean_loso_orchestrator.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["run_clean_loso_orchestrator"] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "build_clean_loso_manifest", ROOT / "scripts" / "loso" / "build_clean_loso_manifest.py"
)
MANIFEST_MODULE = importlib.util.module_from_spec(MANIFEST_SPEC)
sys.modules["build_clean_loso_manifest"] = MANIFEST_MODULE
assert MANIFEST_SPEC.loader is not None
MANIFEST_SPEC.loader.exec_module(MANIFEST_MODULE)


def test_expected_run_name_adds_diag_prefix_for_diagnostic_flags():
    extra = ["--phase", "learned_cif", "--diag-alpha-loss", "logit_l1", "--diag-freeze", "target_only_stage1"]
    assert MODULE.expected_run_name("fold1_A2", extra) == "diag_fold1_A2"


def test_expected_run_name_keeps_plain_name_without_diagnostic_flags():
    extra = ["--phase", "teacher_forced", "--token-head", "linear"]
    assert MODULE.expected_run_name("fold1_A1", extra) == "fold1_A1"


def test_sha256_of_dict_is_deterministic_regardless_of_key_order():
    a = MODULE.sha256_of_dict({"b": 2, "a": 1})
    b = MODULE.sha256_of_dict({"a": 1, "b": 2})
    assert a == b


def test_load_manifest_rejects_tampered_file(tmp_path):
    manifest = MANIFEST_MODULE.build_manifest()
    manifest["folds"][0]["train_signers"] = [99]  # tamper without updating hash
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError):
        MODULE.load_manifest(path)


def test_load_manifest_accepts_untampered_file(tmp_path):
    manifest = MANIFEST_MODULE.build_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = MODULE.load_manifest(path)
    assert loaded["manifest_sha256"] == manifest["manifest_sha256"]


def test_try_skip_returns_none_when_checkpoint_hash_changed(tmp_path):
    checkpoint = tmp_path / "checkpoint_best.pt"
    checkpoint.write_bytes(b"v1")
    done_path = tmp_path / "stage_done.json"
    done_path.write_text(
        json.dumps(
            {
                "stage_hash": "abc",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": MODULE.sha256_of(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"v2-mutated")  # simulate stale/corrupted state
    assert MODULE._try_skip(done_path, "abc") is None


def test_try_skip_returns_checkpoint_when_everything_matches(tmp_path):
    checkpoint = tmp_path / "checkpoint_best.pt"
    checkpoint.write_bytes(b"stable")
    done_path = tmp_path / "stage_done.json"
    done_path.write_text(
        json.dumps(
            {
                "stage_hash": "abc",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": MODULE.sha256_of(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    result = MODULE._try_skip(done_path, "abc")
    assert result == checkpoint


def test_resolved_parent_sha256_hashes_file_when_present(tmp_path):
    checkpoint = tmp_path / "checkpoint_best.pt"
    checkpoint.write_bytes(b"weights")
    assert MODULE.resolved_parent_sha256(checkpoint) == MODULE.sha256_of(checkpoint)


def test_resolved_parent_sha256_falls_back_to_recorded_hash_for_pruned_v126_stage(tmp_path):
    out_dir = tmp_path / "fold1_A1"
    out_dir.mkdir()
    checkpoint = out_dir / "checkpoint_best.pt"
    checkpoint.write_bytes(b"weights")
    recorded_hash = MODULE.sha256_of(checkpoint)
    (out_dir / "stage_done.json").write_text(
        json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": recorded_hash}),
        encoding="utf-8",
    )
    checkpoint.unlink()  # pruned

    assert MODULE.resolved_parent_sha256(checkpoint) == recorded_hash


def test_resolved_parent_sha256_falls_back_for_pruned_v121_layout(tmp_path):
    # v121's stage_done.json lives one level above best_top1/checkpoint.pth.
    run_dir = tmp_path / "9001"
    (run_dir / "best_top1").mkdir(parents=True)
    checkpoint = run_dir / "best_top1" / "checkpoint.pth"
    checkpoint.write_bytes(b"weights")
    recorded_hash = MODULE.sha256_of(checkpoint)
    (run_dir / "stage_done.json").write_text(
        json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": recorded_hash}),
        encoding="utf-8",
    )
    checkpoint.unlink()  # pruned

    assert MODULE.resolved_parent_sha256(checkpoint) == recorded_hash


def test_resolved_parent_sha256_prefers_attested_hash_over_a_reappeared_file(tmp_path):
    # Regression: an interrupted rerun can recreate a file at the pruned path
    # (e.g. a killed retrain that got partway through and saved a checkpoint)
    # before being killed. That file is NOT the checkpoint this hash chain
    # was built on -- the "pruned" marker must win over re-hashing it.
    out_dir = tmp_path / "fold1_A1"
    out_dir.mkdir()
    checkpoint = out_dir / "checkpoint_best.pt"
    checkpoint.write_bytes(b"original weights")
    original_hash = MODULE.sha256_of(checkpoint)
    (out_dir / "stage_done.json").write_text(
        json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": original_hash, "pruned": True}),
        encoding="utf-8",
    )
    checkpoint.write_bytes(b"corrupted partial retrain weights")  # reappeared with different content

    assert MODULE.resolved_parent_sha256(checkpoint) == original_hash
    assert MODULE.resolved_parent_sha256(checkpoint) != MODULE.sha256_of(checkpoint)


def test_resolved_parent_sha256_raises_when_pruned_and_unrecorded(tmp_path):
    checkpoint = tmp_path / "checkpoint_best.pt"
    with pytest.raises(FileNotFoundError):
        MODULE.resolved_parent_sha256(checkpoint)


def test_prune_checkpoint_files_deletes_best_and_latest_but_keeps_other_files(tmp_path):
    (tmp_path / "checkpoint_best.pt").write_bytes(b"a")
    (tmp_path / "checkpoint_latest.pt").write_bytes(b"b")
    (tmp_path / "stage_done.json").write_text(json.dumps({"stage_hash": "x"}), encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    MODULE.prune_checkpoint_files(tmp_path)

    assert not (tmp_path / "checkpoint_best.pt").exists()
    assert not (tmp_path / "checkpoint_latest.pt").exists()
    assert (tmp_path / "stage_done.json").exists()
    assert (tmp_path / "config.json").exists()


def test_prune_checkpoint_files_marks_stage_done_as_pruned(tmp_path):
    (tmp_path / "checkpoint_best.pt").write_bytes(b"a")
    (tmp_path / "stage_done.json").write_text(
        json.dumps({"stage_hash": "x", "checkpoint": str(tmp_path / "checkpoint_best.pt")}),
        encoding="utf-8",
    )

    MODULE.prune_checkpoint_files(tmp_path)

    done = json.loads((tmp_path / "stage_done.json").read_text(encoding="utf-8"))
    assert done["pruned"] is True


def test_try_skip_trusts_pruned_stage_without_requiring_the_file(tmp_path):
    checkpoint = tmp_path / "checkpoint_best.pt"  # deliberately never created
    done_path = tmp_path / "stage_done.json"
    done_path.write_text(
        json.dumps({"stage_hash": "abc", "checkpoint": str(checkpoint), "pruned": True}),
        encoding="utf-8",
    )

    assert MODULE._try_skip(done_path, "abc") == checkpoint


def test_try_skip_still_rejects_hash_mismatch_even_when_pruned(tmp_path):
    checkpoint = tmp_path / "checkpoint_best.pt"
    done_path = tmp_path / "stage_done.json"
    done_path.write_text(
        json.dumps({"stage_hash": "abc", "checkpoint": str(checkpoint), "pruned": True}),
        encoding="utf-8",
    )

    assert MODULE._try_skip(done_path, "different-hash") is None


def test_prune_checkpoint_files_is_idempotent_on_already_pruned_dir(tmp_path):
    MODULE.prune_checkpoint_files(tmp_path)  # no files at all; must not raise
    MODULE.prune_checkpoint_files(tmp_path)


def test_prune_v121_checkpoint_deletes_all_checkpoint_pth_but_keeps_sidecars(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "V121_CHECKPOINTS_ROOT", tmp_path)
    run_dir = tmp_path / "9001"
    (run_dir / "best_top1").mkdir(parents=True)
    (run_dir / "best_top1" / "checkpoint.pth").write_bytes(b"a")
    (run_dir / "10").mkdir()
    (run_dir / "10" / "checkpoint.pth").write_bytes(b"b")
    (run_dir / "label_to_idx.json").write_text("{}", encoding="utf-8")
    (run_dir / "stage_done.json").write_text(
        json.dumps({"stage_hash": "x", "checkpoint": str(run_dir / "best_top1" / "checkpoint.pth")}),
        encoding="utf-8",
    )

    MODULE.prune_v121_checkpoint(9001)

    assert not (run_dir / "best_top1" / "checkpoint.pth").exists()
    assert not (run_dir / "10" / "checkpoint.pth").exists()
    assert (run_dir / "label_to_idx.json").exists()
    done = json.loads((run_dir / "stage_done.json").read_text(encoding="utf-8"))
    assert done["pruned"] is True


def test_prune_v121_checkpoint_handles_missing_run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "V121_CHECKPOINTS_ROOT", tmp_path)
    MODULE.prune_v121_checkpoint(424242)  # no such run dir; must not raise


def test_selected_folds_preserves_requested_order_and_rejects_unknown():
    manifest = {"folds": [{"fold": 7}, {"fold": 8}, {"fold": 9}]}
    assert MODULE.selected_folds(manifest, fold=None, folds=[8, 7]) == [
        {"fold": 8}, {"fold": 7}
    ]
    with pytest.raises(ValueError, match="not found"):
        MODULE.selected_folds(manifest, fold=None, folds=[10])


def test_prepare_only_cli_is_explicit_and_supports_folds_seven_eight():
    args = MODULE.parse_args(["--folds", "7", "8", "--prepare-only"])
    assert args.folds == [7, 8]
    assert args.prepare_only is True


if __name__ == "__main__":
    test_expected_run_name_adds_diag_prefix_for_diagnostic_flags()
    test_expected_run_name_keeps_plain_name_without_diagnostic_flags()
    test_sha256_of_dict_is_deterministic_regardless_of_key_order()
    print("ok")
