import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "expand_max_len_class", ROOT / "scripts" / "loso" / "expand_max_len_class.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

from src.mslm.models.temporal_sign_prompt import TemporalSignPromptModel


class TinyEncoder(nn.Module):
    def forward(self, keypoints, frame_lengths):
        return keypoints


def _model(max_len_class, token_head="contextual", length_head="attention"):
    return TemporalSignPromptModel(
        TinyEncoder(),
        hidden_size=8,
        vocab_size=12,
        embedding_dim=4,
        max_len_class=max_len_class,
        token_head_variant=token_head,
        length_head_variant=length_head,
    )


def test_expand_state_dict_copies_old_rows_and_keeps_new_rows_from_scaffold():
    old_model = _model(max_len_class=4)
    new_scaffold = _model(max_len_class=8)
    old_state = old_model.state_dict()
    scaffold_state = new_scaffold.state_dict()

    merged, expanded = expand_max_len_class_call(old_state, scaffold_state)

    pos_name = "token_head.position_embedding.weight"
    assert merged[pos_name].shape == (8, 8)
    assert torch.equal(merged[pos_name][:4], old_state[pos_name])
    # new rows came from the freshly-initialized scaffold, not zeros and not a copy of row 0
    assert not torch.equal(merged[pos_name][4:], old_state[pos_name][:4])

    len_name = "length_head.classifier.1.weight"
    assert merged[len_name].shape == (9, 8)  # max_len_class+1 = 9
    assert torch.equal(merged[len_name][:5], old_state[len_name])

    expanded_names = {p["name"] for p in expanded}
    assert pos_name in expanded_names
    assert len_name in expanded_names
    assert "length_head.classifier.1.bias" in expanded_names


def test_expand_state_dict_leaves_unrelated_tensors_unchanged():
    old_model = _model(max_len_class=4)
    new_scaffold = _model(max_len_class=8)
    old_state = old_model.state_dict()
    scaffold_state = new_scaffold.state_dict()

    merged, _ = expand_max_len_class_call(old_state, scaffold_state)

    assert torch.equal(merged["embedding_head.weight"], old_state["embedding_head.weight"])
    assert torch.equal(merged["cif.alpha.weight"], old_state["cif.alpha.weight"])


def test_infer_model_kwargs_reads_shapes_and_arch_config():
    model = _model(max_len_class=4, token_head="linear", length_head="mean")
    state = model.state_dict()
    kwargs = MODULE.infer_model_kwargs(state, {"token_head": "linear", "length_head": "mean"})
    assert kwargs == {
        "hidden_size": 8,
        "vocab_size": 12,
        "embedding_dim": 4,
        "token_head_variant": "linear",
        "length_head_variant": "mean",
        "old_max_len_class": 4,
    }


def expand_max_len_class_call(old_state, scaffold_state):
    return MODULE.expand_state_dict(old_state, scaffold_state)


if __name__ == "__main__":
    test_expand_state_dict_copies_old_rows_and_keeps_new_rows_from_scaffold()
    test_expand_state_dict_leaves_unrelated_tensors_unchanged()
    test_infer_model_kwargs_reads_shapes_and_arch_config()
    print("ok")
