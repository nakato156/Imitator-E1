import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "analyze_imitator_a2", ROOT / "scripts" / "diagnostics" / "analyze_imitator_a2.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_resolve_arch_config_uses_checkpoint_metadata_when_present():
    state = {"arch_config": {"token_head": "linear", "length_head": "mean"}}
    resolved = MODULE.resolve_arch_config(state, cli_token_head=None, cli_length_head=None)
    assert resolved == {"token_head": "linear", "length_head": "mean"}


def test_resolve_arch_config_defaults_to_legacy_when_checkpoint_lacks_metadata():
    state = {}
    resolved = MODULE.resolve_arch_config(state, cli_token_head=None, cli_length_head=None)
    assert resolved == {"token_head": "contextual", "length_head": "attention"}


def test_resolve_arch_config_explicit_cli_override_wins():
    state = {"arch_config": {"token_head": "linear", "length_head": "mean"}}
    resolved = MODULE.resolve_arch_config(state, cli_token_head="contextual", cli_length_head=None)
    assert resolved == {"token_head": "contextual", "length_head": "mean"}


def test_effective_split_seed_falls_back_to_seed():
    assert MODULE.effective_split_seed(seed=23, split_seed=None) == 23
    assert MODULE.effective_split_seed(seed=23, split_seed=101) == 101


def test_round_trip_checkpoint_with_arch_config_loads_without_shape_mismatch(tmp_path):
    import torch.nn as nn

    from src.mslm.models.temporal_sign_prompt import TemporalSignPromptModel

    class TinyEncoder(nn.Module):
        def forward(self, keypoints, frame_lengths):
            return keypoints

    model = TemporalSignPromptModel(
        TinyEncoder(), hidden_size=4, vocab_size=12, embedding_dim=4,
        token_head_variant="linear", length_head_variant="mean",
    )
    ckpt_path = tmp_path / "checkpoint_best.pt"
    torch.save(
        {"model": model.state_dict(), "arch_config": {"token_head": "linear", "length_head": "mean"}},
        ckpt_path,
    )

    state = torch.load(ckpt_path, map_location="cpu")
    arch_config = MODULE.resolve_arch_config(state, None, None)
    rebuilt = TemporalSignPromptModel(
        TinyEncoder(), hidden_size=4, vocab_size=12, embedding_dim=4,
        token_head_variant=arch_config["token_head"], length_head_variant=arch_config["length_head"],
    )
    missing, unexpected = rebuilt.load_state_dict(state["model"], strict=True)
    assert not missing
    assert not unexpected
