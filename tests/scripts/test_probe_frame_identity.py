import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "probe_frame_identity_test_module", ROOT / "scripts/eval/probe_frame_identity.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_clip_balanced_moments_do_not_weight_long_clip_by_frame_count():
    clips = [torch.zeros(1, 1), torch.full((9, 1), 10.0)]
    mean, std = MODULE.clip_balanced_moments(clips)
    assert mean.item() == 5.0
    assert std.item() == 5.0


def test_clip_balanced_cross_entropy_is_mean_of_clip_means():
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0], [0.0, 5.0]])
    labels = torch.tensor([0, 0, 0])
    loss = MODULE.clip_balanced_cross_entropy(logits, labels, [1, 2])
    per_frame = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    assert loss == pytest.approx((per_frame[0] + per_frame[1:].mean()).item() / 2)


def test_clip_balanced_cross_entropy_rejects_bad_lengths():
    with pytest.raises(ValueError, match="do not match"):
        MODULE.clip_balanced_cross_entropy(
            torch.zeros(3, 2), torch.zeros(3, dtype=torch.long), [1, 1]
        )
