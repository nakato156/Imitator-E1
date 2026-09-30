import inspect

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mslm.models.video_token_decoder import (
    VideoTokenDecoder,
    build_teacher_forcing,
    causal_mask,
    initialize_from_cif_checkpoint,
    set_decoder_training_stage,
)
from src.mslm.utils.sequence_metrics import strict_exact


class IdentityFrameEncoder(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.stgcn_layers = nn.Sequential(nn.Linear(hidden, hidden))
        self.linear_hidden = nn.Sequential(nn.Linear(hidden, hidden))
        self.tcn = nn.Sequential(nn.Linear(hidden, hidden))
        self.transformer = nn.Sequential(nn.Linear(hidden, hidden))

    def forward(self, keypoints, frame_lengths):
        return keypoints


def tiny_model(vocab=16, max_content=4):
    return VideoTokenDecoder(
        IdentityFrameEncoder(),
        hidden_size=8,
        vocab_size=vocab,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
        max_content_tokens=max_content,
        bos_id=2,
        eos_id=3,
        pad_id=0,
    )


def test_logits_shape_and_causal_mask():
    model = tiny_model()
    logits = model(torch.randn(2, 5, 8), torch.tensor([5, 3]), torch.tensor([[2, 4, 5], [2, 6, 0]]))
    assert logits.shape == (2, 3, 16)
    assert torch.equal(
        causal_mask(3),
        torch.tensor([[False, True, True], [False, False, True], [False, False, False]]),
    )


def test_cross_attention_ignores_padded_frames():
    torch.manual_seed(2)
    model = tiny_model().eval()
    frames = torch.randn(1, 5, 8)
    changed = frames.clone()
    changed[:, 3:] = 10_000
    lengths = torch.tensor([3])
    decoder_input = torch.tensor([[2, 4]])
    with torch.no_grad():
        first = model(frames, lengths, decoder_input)
        second = model(changed, lengths, decoder_input)
    torch.testing.assert_close(first, second)


def test_teacher_forcing_builds_bos_tokens_to_tokens_eos():
    inputs, labels = build_teacher_forcing(
        torch.tensor([[7, 8, -100], [9, -100, -100]]),
        bos_id=2,
        eos_id=3,
        pad_id=0,
        max_content_tokens=4,
    )
    assert inputs.tolist() == [[2, 7, 8], [2, 9, 0]]
    assert labels.tolist() == [[7, 8, 3], [9, 3, 0]]


def test_greedy_forbids_empty_and_stops_at_first_eos(monkeypatch):
    model = tiny_model()

    def fake_decode(features, lengths, inputs):
        logits = torch.zeros(inputs.size(0), inputs.size(1), model.vocab_size)
        if inputs.size(1) == 1:
            logits[:, -1, 3] = 100  # forbidden EOS
            logits[:, -1, 7] = 90
        else:
            logits[:, -1, 3] = 100
        return logits

    monkeypatch.setattr(model, "decode_features", fake_decode)
    output = model.greedy_decode(torch.randn(1, 2, 8), torch.tensor([2]))
    assert output.token_ids == [[7]]
    assert output.emitted_eos.tolist() == [True]
    assert output.lengths.tolist() == [1]


def test_no_eos_gets_maximum_length_penalty(monkeypatch):
    model = tiny_model(max_content=4)

    def fake_decode(features, lengths, inputs):
        logits = torch.zeros(inputs.size(0), inputs.size(1), model.vocab_size)
        logits[:, -1, 7] = 1
        return logits

    monkeypatch.setattr(model, "decode_features", fake_decode)
    output = model.greedy_decode(torch.randn(1, 2, 8), torch.tensor([2]))
    assert not output.emitted_eos.item()
    assert output.lengths.item() == 5
    assert len(output.token_ids[0]) == 5


def test_cif_initialization_is_exact_and_lineage_is_clean(tmp_path):
    source_encoder = IdentityFrameEncoder()
    weight = torch.randn(16, 8)
    bias = torch.randn(16)
    source = {
        **{f"frame_encoder.{key}": value.clone() for key, value in source_encoder.state_dict().items()},
        "token_head.classifier.weight": weight,
        "token_head.classifier.bias": bias,
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": source,
            "lineage": {
                "fold_outer_test_signer": 1,
                "train_signers": [3, 4],
                "val_signers": [2],
            },
        },
        checkpoint,
    )
    model = tiny_model()
    initialize_from_cif_checkpoint(model, checkpoint, outer_signer=1)
    torch.testing.assert_close(model.token_embedding.weight, weight, rtol=0, atol=0)
    torch.testing.assert_close(model.output_bias, bias, rtol=0, atol=0)
    for key, value in source_encoder.state_dict().items():
        torch.testing.assert_close(model.frame_encoder.state_dict()[key], value, rtol=0, atol=0)


def test_lineage_rejects_outer_signer_contamination(tmp_path):
    model = tiny_model()
    source = {
        **{f"frame_encoder.{key}": value for key, value in model.frame_encoder.state_dict().items()},
        "token_head.classifier.weight": torch.randn(16, 8),
        "token_head.classifier.bias": torch.randn(16),
    }
    checkpoint = tmp_path / "bad.pt"
    torch.save(
        {"model": source, "lineage": {"fold_outer_test_signer": 1, "train_signers": [1], "val_signers": [2]}},
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="contaminates"):
        initialize_from_cif_checkpoint(model, checkpoint, outer_signer=1)


def test_api_has_no_cif_or_target_length_inputs():
    parameters = set(inspect.signature(VideoTokenDecoder.forward).parameters)
    assert parameters == {"self", "keypoints", "frame_lengths", "decoder_input_ids"}
    assert not parameters & {"boundaries", "alphas", "target_lengths", "token_spans"}


def test_fixed_freeze_schedule_keeps_low_level_visual_frozen():
    model = tiny_model()
    early = set_decoder_training_stage(model, 4)
    assert early["decoder"] and not early["frame_encoder.tcn"]
    late = set_decoder_training_stage(model, 5)
    assert late["frame_encoder.tcn"] and late["frame_encoder.transformer"]
    assert all(not p.requires_grad for p in model.frame_encoder.stgcn_layers.parameters())
    assert all(not p.requires_grad for p in model.frame_encoder.linear_hidden.parameters())


def test_unfreeze_stgcn_epoch_flag_controls_low_level_visual():
    model = tiny_model()
    before = set_decoder_training_stage(model, 9, unfreeze_stgcn_epoch=10)
    assert not before["frame_encoder.stgcn_layers"]
    assert all(not p.requires_grad for p in model.frame_encoder.stgcn_layers.parameters())
    after = set_decoder_training_stage(model, 10, unfreeze_stgcn_epoch=10)
    assert after["frame_encoder.stgcn_layers"] and after["frame_encoder.linear_hidden"]
    assert all(p.requires_grad for p in model.frame_encoder.stgcn_layers.parameters())
    assert all(p.requires_grad for p in model.frame_encoder.linear_hidden.parameters())


def test_optimizer_puts_stgcn_in_low_lr_group():
    import importlib
    train_mod = importlib.import_module("scripts.train.train_video_token_decoder")
    model = tiny_model()
    optimizer = train_mod.optimizer_for(model)
    groups = {g["name"]: g for g in optimizer.param_groups}
    assert groups["stgcn_linear_hidden"]["lr"] == 1e-5
    stgcn_ids = {
        id(p)
        for m in (model.frame_encoder.stgcn_layers, model.frame_encoder.linear_hidden)
        for p in m.parameters()
    }
    assert stgcn_ids == {id(p) for p in groups["stgcn_linear_hidden"]["params"]}
    for name in ("decoder", "visual_tcn_transformer"):
        assert not stgcn_ids & {id(p) for p in groups[name]["params"]}


def test_embedding_and_output_projection_are_tied():
    model = tiny_model()
    decoder_input = torch.tensor([[2, 4]])
    features = torch.randn(1, 3, 8)
    logits = model.decode_features(features, torch.tensor([3]), decoder_input)
    assert logits.shape[-1] == model.token_embedding.num_embeddings
    assert not hasattr(model, "output_projection")
    logits.sum().backward()
    # The projection path writes gradients directly into the embedding table.
    assert model.token_embedding.weight.grad is not None


def test_overfits_one_synthetic_minibatch_to_strict_exact_one():
    torch.manual_seed(7)
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    frames = torch.randn(1, 3, 8)
    frame_lengths = torch.tensor([3])
    target = torch.tensor([[7, -100]])
    decoder_input, labels = build_teacher_forcing(
        target, bos_id=2, eos_id=3, pad_id=0, max_content_tokens=4
    )
    model.train()
    for _ in range(60):
        logits = model(frames, frame_lengths, decoder_input)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    decoded = model.greedy_decode(frames, frame_lengths)
    exact = decoded.emitted_eos.item() and strict_exact(decoded.token_ids[0], [7])
    assert exact
