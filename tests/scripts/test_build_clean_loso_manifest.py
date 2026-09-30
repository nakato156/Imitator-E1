import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_clean_loso_manifest", ROOT / "scripts" / "loso" / "build_clean_loso_manifest.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["build_clean_loso_manifest"] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_builds_ten_folds_covering_every_signer_as_test_exactly_once():
    manifest = MODULE.build_manifest()
    assert len(manifest["folds"]) == 10
    outer = [f["outer_test_signer"] for f in manifest["folds"]]
    assert sorted(outer) == list(range(1, 11))


def test_each_fold_has_disjoint_train_val_test_signers():
    manifest = MODULE.build_manifest()
    for fold in manifest["folds"]:
        train = set(fold["train_signers"])
        val = {fold["inner_val_signer"]}
        test = {fold["outer_test_signer"]}
        assert not (train & val)
        assert not (train & test)
        assert not (val & test)
        assert train | val | test == set(range(1, 11))
        assert len(train) == 8


def test_inner_val_signer_never_equals_outer_test_signer():
    manifest = MODULE.build_manifest()
    for fold in manifest["folds"]:
        assert fold["inner_val_signer"] != fold["outer_test_signer"]


def test_manifest_is_deterministic():
    a = MODULE.build_manifest()
    b = MODULE.build_manifest()
    assert a == b
    assert a["manifest_sha256"] == b["manifest_sha256"]


def test_seed_is_23():
    manifest = MODULE.build_manifest()
    assert manifest["seed"] == 23


def test_validate_folds_rejects_intersection():
    folds = MODULE.build_folds()
    folds[0]["train_signers"] = list(folds[0]["train_signers"]) + [folds[0]["inner_val_signer"]]
    with pytest.raises(ValueError):
        MODULE.validate_folds(folds)


if __name__ == "__main__":
    test_builds_ten_folds_covering_every_signer_as_test_exactly_once()
    test_each_fold_has_disjoint_train_val_test_signers()
    test_inner_val_signer_never_equals_outer_test_signer()
    test_manifest_is_deterministic()
    test_seed_is_23()
    print("ok")
