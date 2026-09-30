import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "train_temporal_v126", ROOT / "scripts" / "train" / "train_temporal_v126.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["train_temporal_v126"] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _parse(extra_argv):
    sys.argv = ["train_temporal_v126.py"] + extra_argv
    return MODULE.parse_args()


def test_token_head_length_head_label_smoothing_defaults_preserve_current_behavior():
    args = _parse([])
    assert args.token_head == "contextual"
    assert args.length_head == "attention"
    assert args.token_label_smoothing == 0.1
    assert args.split_seed is None


def test_ablation_flags_are_settable():
    args = _parse(
        [
            "--token-head", "linear",
            "--length-head", "mean",
            "--token-label-smoothing", "0.0",
            "--split-seed", "23",
            "--seed", "42",
        ]
    )
    assert args.token_head == "linear"
    assert args.length_head == "mean"
    assert args.token_label_smoothing == 0.0
    assert args.split_seed == 23
    assert args.seed == 42


def test_effective_split_seed_falls_back_to_seed_when_unset():
    args = _parse(["--seed", "101"])
    assert MODULE.effective_split_seed(args) == 101


def test_effective_split_seed_uses_explicit_value():
    args = _parse(["--seed", "101", "--split-seed", "23"])
    assert MODULE.effective_split_seed(args) == 23
