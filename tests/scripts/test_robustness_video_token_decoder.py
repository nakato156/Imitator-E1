import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "robustness_video_token_decoder_test_module",
    ROOT / "scripts/eval/robustness_video_token_decoder.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_gaussian_jitter_is_deterministic_and_zero_is_noop():
    sequence = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    assert torch.equal(MODULE.gaussian_jitter(sequence, 0.0, 11), sequence)
    first = MODULE.gaussian_jitter(sequence, 0.02, 11)
    second = MODULE.gaussian_jitter(sequence, 0.02, 11)
    assert torch.equal(first, second)
    assert not torch.equal(first, sequence)


def test_frame_dropout_is_deterministic_noop_at_zero_and_never_empty():
    sequence = torch.arange(40, dtype=torch.float32).reshape(10, 2, 2)
    assert torch.equal(MODULE.drop_frames(sequence, 0.0, 12), sequence)
    assert torch.equal(
        MODULE.drop_frames(sequence, 0.2, 12), MODULE.drop_frames(sequence, 0.2, 12)
    )
    assert MODULE.drop_frames(sequence[:1], 0.9, 12).shape[0] == 1


def test_temporal_resampling_uses_playback_speed_semantics():
    sequence = torch.arange(40, dtype=torch.float32).reshape(10, 2, 2)
    assert torch.equal(MODULE.temporal_resample(sequence, 1.0), sequence)
    assert MODULE.temporal_resample(sequence, 0.75).shape[0] == 13
    assert MODULE.temporal_resample(sequence, 1.25).shape[0] == 8


def test_bootstrap_ci_is_deterministic_and_contains_constant_mean():
    first = MODULE.bootstrap_mean_ci([0.5] * 8, seed=7, replicates=50)
    second = MODULE.bootstrap_mean_ci([0.5] * 8, seed=7, replicates=50)
    assert first == second
    assert first["lower"] == first["estimate"] == first["upper"] == 0.5


def test_two_sided_sign_test_handles_ties_and_direction_elsewhere():
    assert MODULE.two_sided_sign_test(0, 0) == 1.0
    assert MODULE.two_sided_sign_test(10, 0) == pytest.approx(2 / 2**10)
    assert MODULE.two_sided_sign_test(7, 3) == MODULE.two_sided_sign_test(3, 7)


def test_validate_e1_checkpoint_rejects_unfreeze_or_wrong_selection():
    state = {
        "provenance": {"fold": 7},
        "config": {
            "epochs": 30,
            "seed": 23,
            "encoder_pe": True,
            "ctc_weight": 0.0,
            "label_smoothing": 0.0,
            "unfreeze_stgcn_epoch": 10,
            "select": "exact",
            "vocab_map": [0, 2, 106, 7],
        },
    }
    with pytest.raises(RuntimeError, match="not frozen E1"):
        MODULE.validate_e1_checkpoint(state, 7, 23)


def test_condition_subset_requires_both_pre_registered_primary_conditions():
    selected = MODULE.condition_transforms(["clean", "segment_permutation"])
    assert list(selected) == ["clean", "segment_permutation"]
    with pytest.raises(ValueError, match="requires both"):
        MODULE.condition_transforms(["clean"])
    with pytest.raises(ValueError, match="unknown"):
        MODULE.condition_transforms(["clean", "segment_permutation", "rain"])
