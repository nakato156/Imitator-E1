import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_ablation_etapa2", ROOT / "scripts" / "train" / "run_ablation_etapa2.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_generate_variants_is_full_factorial_24_runs():
    variants = MODULE.generate_variants()
    assert len(variants) == 24
    assert len({tuple(sorted(v.items())) for v in variants}) == 24
    for v in variants:
        assert v["split_seed"] == 23
    seeds = sorted({v["seed"] for v in variants})
    assert seeds == [23, 42, 101]
    token_heads = sorted({v["token_head"] for v in variants})
    assert token_heads == ["contextual", "linear"]
    length_heads = sorted({v["length_head"] for v in variants})
    assert length_heads == ["attention", "mean"]
    smoothings = sorted({v["token_label_smoothing"] for v in variants})
    assert smoothings == [0.0, 0.1]


def test_run_name_is_deterministic_and_unique_per_variant():
    variants = MODULE.generate_variants()
    names = {MODULE.run_name_for(v, run_tag="TAG") for v in variants}
    assert len(names) == 24


def test_train_command_includes_all_ablation_flags():
    variant = {
        "token_head": "linear", "length_head": "mean",
        "token_label_smoothing": 0.0, "seed": 42, "split_seed": 23,
    }
    cmd = MODULE.train_command_for(variant, run_name="run1", resume_ckpt=Path("ckpt.pt"))
    cmd_str = " ".join(cmd)
    assert "--token-head linear" in cmd_str
    assert "--length-head mean" in cmd_str
    assert "--token-label-smoothing 0.0" in cmd_str
    assert "--seed 42" in cmd_str
    assert "--split-seed 23" in cmd_str
    assert "--resume-weights-only" in cmd_str
    assert "--epochs 15" in cmd_str
    assert f"--output-root {MODULE.DEFAULT_OUT_ROOT}" in cmd_str


def test_train_command_forwards_custom_out_root():
    variant = {
        "token_head": "linear", "length_head": "mean",
        "token_label_smoothing": 0.0, "seed": 42, "split_seed": 23,
    }
    custom_root = Path("/tmp/custom_out_root")
    cmd = MODULE.train_command_for(
        variant, run_name="run1", resume_ckpt=Path("ckpt.pt"), out_root=custom_root
    )
    assert "--output-root" in cmd
    assert cmd[cmd.index("--output-root") + 1] == str(custom_root)


def test_is_valid_reuse_requires_matching_config_and_checkpoint_hash(tmp_path):
    ckpt = tmp_path / "checkpoint_best.pt"
    ckpt.write_bytes(b"weights-v1")
    variant = {"token_head": "linear", "length_head": "mean", "token_label_smoothing": 0.0, "seed": 23, "split_seed": 23}
    entry = {"variant": variant, "checkpoint_sha256": hashlib.sha256(b"weights-v1").hexdigest()}

    assert MODULE.is_valid_reuse(entry, variant, ckpt) is True

    ckpt.write_bytes(b"weights-v2-overwritten")
    assert MODULE.is_valid_reuse(entry, variant, ckpt) is False

    different_variant = {**variant, "seed": 999}
    ckpt.write_bytes(b"weights-v1")
    assert MODULE.is_valid_reuse(entry, different_variant, ckpt) is False


def test_is_valid_reuse_false_when_checkpoint_missing(tmp_path):
    variant = {"token_head": "linear", "length_head": "mean", "token_label_smoothing": 0.0, "seed": 23, "split_seed": 23}
    entry = {"variant": variant, "checkpoint_sha256": "deadbeef"}
    assert MODULE.is_valid_reuse(entry, variant, tmp_path / "missing.pt") is False


def test_main_skips_valid_reused_runs_and_runs_missing_ones(tmp_path):
    calls = []

    def fake_runner(cmd, **kwargs):
        # Simulate the train + audit commands writing their checkpoint.
        if "train_temporal_v126.py" in cmd[1]:
            run_name = cmd[cmd.index("--run-name") + 1]
            out_dir = tmp_path / "out" / f"diag_{run_name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "checkpoint_best.pt").write_bytes(run_name.encode())
        calls.append(cmd)
        return None

    registry_dir = tmp_path / "artifacts"
    registry_dir.mkdir()
    variants = MODULE.generate_variants()[:2]  # keep the test fast
    MODULE.main(
        variants=variants,
        run_tag="TESTTAG",
        out_root=tmp_path / "out",
        registry_dir=registry_dir,
        resume_ckpt=tmp_path / "etapa1.pt",
        python_bin="python3",
        runner=fake_runner,
    )
    first_run_calls = len(calls)
    assert first_run_calls == 4  # 2 variants x (train + audit)

    # Second invocation: registries now match real checkpoints -> must skip both.
    calls.clear()
    MODULE.main(
        variants=variants,
        run_tag="TESTTAG",
        out_root=tmp_path / "out",
        registry_dir=registry_dir,
        resume_ckpt=tmp_path / "etapa1.pt",
        python_bin="python3",
        runner=fake_runner,
    )
    assert calls == []
