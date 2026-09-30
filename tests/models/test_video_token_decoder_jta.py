"""Imitator-JTA additions: encoder PE, restricted reversible vocab, CTC head."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.mslm.models.temporal_sign_prompt import (
    STGCNTemporalFrameEncoder,
    sinusoidal_positional_encoding,
)
from src.mslm.models.video_token_decoder import (
    VideoTokenDecoder,
    initialize_from_cif_checkpoint,
    set_decoder_training_stage,
)
from tests.models.test_video_token_decoder import IdentityFrameEncoder

VOCAB_MAP = [0, 2, 106, 500, 501, 502]  # pad, bos, eos + 3 effective Gemma ids


def mapped_model(**kwargs):
    return VideoTokenDecoder(
        IdentityFrameEncoder(),
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
        dropout=0.0,
        max_content_tokens=4,
        bos_id=2,
        eos_id=106,
        pad_id=0,
        vocab_map=VOCAB_MAP,
        **kwargs,
    )


def test_vocab_map_round_trip_preserves_ignore_index_and_rejects_unknown():
    model = mapped_model()
    gemma = torch.tensor([[500, 502, -100], [501, -100, -100]])
    dense = model.to_dense(gemma)
    assert dense.tolist() == [[3, 5, -100], [4, -100, -100]]
    assert model.to_gemma([3, 5]) == [500, 502]
    with pytest.raises(ValueError, match="not in vocab_map"):
        model.to_dense(torch.tensor([[999]]))


def test_dense_special_ids_and_output_width():
    model = mapped_model()
    assert (model.pad_id, model.bos_id, model.eos_id) == (0, 1, 2)
    logits = model(torch.randn(1, 3, 8), torch.tensor([3]), torch.tensor([[1, 3]]))
    assert logits.shape == (1, 2, len(VOCAB_MAP))


def test_greedy_decode_returns_gemma_ids(monkeypatch):
    model = mapped_model()

    def fake_decode(features, lengths, inputs):
        logits = torch.zeros(inputs.size(0), inputs.size(1), model.vocab_size)
        logits[:, -1, 3 if inputs.size(1) == 1 else model.eos_id] = 100
        return logits

    monkeypatch.setattr(model, "decode_features", fake_decode)
    output = model.greedy_decode(torch.randn(1, 2, 8), torch.tensor([2]))
    assert output.token_ids == [[500]]  # Gemma id, not dense id 3
    assert output.emitted_eos.tolist() == [True]


def test_cif_initialization_copies_only_mapped_rows(tmp_path):
    source_encoder = IdentityFrameEncoder()
    weight = torch.randn(107, 8)  # any table covering the mapped Gemma ids
    bias = torch.randn(107)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": {
                **{f"frame_encoder.{k}": v.clone() for k, v in source_encoder.state_dict().items()},
                "token_head.classifier.weight": weight,
                "token_head.classifier.bias": bias,
            },
            "lineage": {"fold_outer_test_signer": 1, "train_signers": [3], "val_signers": [2]},
        },
        checkpoint,
    )
    map_small = [0, 2, 106, 50, 51]
    model = VideoTokenDecoder(
        IdentityFrameEncoder(), hidden_size=8, num_layers=1, num_heads=2, ffn_size=16,
        dropout=0.0, max_content_tokens=4, bos_id=2, eos_id=106, pad_id=0,
        vocab_map=map_small,
    )
    initialize_from_cif_checkpoint(model, checkpoint, outer_signer=1)
    torch.testing.assert_close(model.token_embedding.weight, weight[torch.tensor(map_small)])
    torch.testing.assert_close(model.output_bias, bias[torch.tensor(map_small)])


def test_ctc_head_shapes_loss_finite_and_trains_with_decoder_stage():
    model = mapped_model(with_ctc=True)
    assert model.ctc_blank_id == len(VOCAB_MAP)
    features = torch.randn(2, 6, 8)
    log_probs = model.ctc_head(features).log_softmax(-1).transpose(0, 1)
    assert log_probs.shape == (6, 2, len(VOCAB_MAP) + 1)
    loss = F.ctc_loss(
        log_probs,
        torch.tensor([3, 4, 5]),
        torch.tensor([6, 6]),
        torch.tensor([2, 1]),
        blank=model.ctc_blank_id,
        zero_infinity=True,
    )
    assert torch.isfinite(loss)
    stage = set_decoder_training_stage(model, 0)
    assert stage["decoder"]
    assert all(p.requires_grad for p in model.ctc_head.parameters())


def tiny_encoder(use_pe: bool) -> STGCNTemporalFrameEncoder:
    return STGCNTemporalFrameEncoder(
        np.ones((3, 3), dtype=np.float32),
        gcn_channels=(4,),
        hidden_size=8,
        transformer_heads=2,
        dropout=0.0,
        use_positional_encoding=use_pe,
        max_positions=64,
    )


def test_encoder_pe_makes_features_position_dependent():
    """Regression test for the v125 order-blindness root cause: on a
    time-constant input, interior frames (away from conv edge padding) get
    identical features without PE — the encoder provably carries no global
    temporal position — and distinct features with PE."""
    torch.manual_seed(0)
    keypoints = torch.randn(1, 1, 3, 2).expand(1, 32, 3, 2).contiguous()
    lengths = torch.tensor([32])

    for use_pe, should_differ in ((False, False), (True, True)):
        torch.manual_seed(1)
        encoder = tiny_encoder(use_pe).eval()
        with torch.no_grad():
            features = encoder(keypoints, lengths)
        # frames 8 and 24 are >3 frames from both temporal edges (receptive
        # field: one ST-GCN block ±1 + TCN 2×k3 ±2)
        differs = not torch.allclose(features[0, 8], features[0, 24], atol=1e-5)
        assert differs == should_differ, f"use_pe={use_pe}"


def test_sinusoidal_table_shape_and_range():
    table = sinusoidal_positional_encoding(16, 8)
    assert table.shape == (16, 8)
    assert float(table.abs().max()) <= 1.0
    assert not torch.allclose(table[0], table[5])
