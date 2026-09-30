import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "train_temporal_v126_clean_loso", ROOT / "scripts" / "train" / "train_temporal_v126.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["train_temporal_v126_clean_loso"] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _records():
    return [
        {"signer_id": signer, "clip_id": f"s{signer}_{i}", "label": "x"}
        for signer in range(1, 11)
        for i in range(3)
    ]


def test_exclude_signer_never_appears_in_train_or_val():
    train, val = MODULE.split_records(_records(), seed=23, heldout_signer=2, exclude_signer=1)
    train_signers = {r["signer_id"] for r in train}
    val_signers = {r["signer_id"] for r in val}
    assert 1 not in train_signers
    assert 1 not in val_signers
    assert val_signers == {2}
    assert train_signers == set(range(3, 11))


def test_exclude_signer_requires_heldout_signer():
    with pytest.raises(ValueError):
        MODULE.split_records(_records(), seed=23, heldout_signer=None, exclude_signer=1)


def test_exclude_signer_must_differ_from_heldout_signer():
    with pytest.raises(ValueError):
        MODULE.split_records(_records(), seed=23, heldout_signer=1, exclude_signer=1)


def test_resume_refuses_checkpoint_whose_lineage_trained_on_test_signer():
    contaminated_state = {"lineage": {"train_signers": [3, 4, 5], "val_signers": [1]}}
    with pytest.raises(RuntimeError):
        MODULE.assert_resume_lineage_clean(contaminated_state, exclude_signer=1)


def test_resume_allows_checkpoint_whose_lineage_excludes_test_signer():
    clean_state = {"lineage": {"train_signers": [3, 4, 5], "val_signers": [2]}}
    MODULE.assert_resume_lineage_clean(clean_state, exclude_signer=1)  # no raise


def test_resume_allows_checkpoints_without_lineage_metadata():
    # Legacy/v121 checkpoints predating this fix carry no lineage field.
    MODULE.assert_resume_lineage_clean({}, exclude_signer=1)  # no raise


if __name__ == "__main__":
    test_exclude_signer_never_appears_in_train_or_val()
    test_exclude_signer_requires_heldout_signer()
    test_exclude_signer_must_differ_from_heldout_signer()
    test_resume_refuses_checkpoint_whose_lineage_trained_on_test_signer()
    test_resume_allows_checkpoint_whose_lineage_excludes_test_signer()
    test_resume_allows_checkpoints_without_lineage_metadata()
    print("ok")
