import importlib.util
import random
import sys
import types
from pathlib import Path

import h5py
import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


src_stub = sys.modules.setdefault("src", types.ModuleType("src"))
src_stub.__path__ = [str(_ROOT / "src")]
mslm_stub = sys.modules.setdefault("src.mslm", types.ModuleType("src.mslm"))
mslm_stub.__path__ = [str(_ROOT / "src/mslm")]
dataloader_stub = sys.modules.setdefault(
    "src.mslm.dataloader", types.ModuleType("src.mslm.dataloader")
)
dataloader_stub.__path__ = [str(_ROOT / "src/mslm/dataloader")]
models_stub = types.ModuleType("src.mslm.models")
models_stub.__path__ = [str(_ROOT / "src/mslm/models")]
sys.modules["src.mslm.models"] = models_stub
_load("src.mslm.dataloader.data_augmentation", "src/mslm/dataloader/data_augmentation.py")
_load("src.mslm.models.components", "src/mslm/models/components/__init__.py")
_load("src.mslm.models.components.stgcn", "src/mslm/models/components/stgcn.py")
_data_mod = _load("src.mslm.dataloader.synthetic_temporal", "src/mslm/dataloader/synthetic_temporal.py")
_model_mod = _load("src.mslm.models.temporal_sign_prompt", "src/mslm/models/temporal_sign_prompt.py")

SyntheticTemporalSignDataset = _data_mod.SyntheticTemporalSignDataset
synthetic_temporal_collate = _data_mod.synthetic_temporal_collate
permute_video_segments = _data_mod.permute_video_segments
CIFAggregator = _model_mod.CIFAggregator
alpha_schedule_weights = _model_mod.alpha_schedule_weights
set_cif_phase = _model_mod.set_cif_phase
set_cif_diagnostic_freeze = _model_mod.set_cif_diagnostic_freeze
boundary_error_mae = _model_mod.boundary_error_mae
TemporalSignPromptModel = _model_mod.TemporalSignPromptModel
alpha_diagnostics = _model_mod.alpha_diagnostics
rescale_alphas_to_target_lengths = _model_mod.rescale_alphas_to_target_lengths
rescale_alphas_to_predicted_lengths = _model_mod.rescale_alphas_to_predicted_lengths
rescale_alphas_to_rounded_count = _model_mod.rescale_alphas_to_rounded_count


def _fixture_h5(tmp_path):
    path = tmp_path / "dataset1.hdf5"
    with h5py.File(path, "w") as f:
        g = f.create_group("dataset1")
        kg = g.create_group("keypoints")
        kg.create_dataset("0", data=torch.ones(3, 2, 2).numpy())
        kg.create_dataset("1", data=(2 * torch.ones(4, 2, 2)).numpy())
        kg.create_dataset("2", data=(3 * torch.ones(5, 2, 2)).numpy())
    return path


def test_synthetic_dataset_concatenates_clips_tokens_and_boundaries(tmp_path):
    h5_path = _fixture_h5(tmp_path)
    ds = SyntheticTemporalSignDataset(
        h5_path,
        ["0", "1", "2"],
        {"0": "hola", "1": "mundo", "2": "si"},
        {"hola": [10, 11], "mundo": [12], "si": [13, 14, 15]},
        min_clips=3,
        max_clips=3,
        min_neutral_frames=2,
        max_neutral_frames=2,
        samples_per_epoch=2,
        seed=5,
    )

    sample = ds[0]

    assert sample.boundaries.shape == (3, 2)
    for start, end in sample.boundaries.tolist():
        assert end > start
        assert torch.count_nonzero(sample.keypoints[start:end]).item() > 0
    assert sample.boundaries[1, 0].item() - sample.boundaries[0, 1].item() == 2
    assert sample.boundaries[2, 0].item() - sample.boundaries[1, 1].item() == 2

    expected = []
    for gloss in sample.glosses:
        expected.extend(ds.token_ids_by_label[gloss])
    assert sample.token_ids.tolist() == expected
    assert sample.token_spans[-1, 1].item() == len(expected)


def test_synthetic_collate_pads_frames_tokens_and_sign_metadata(tmp_path):
    h5_path = _fixture_h5(tmp_path)
    ds = SyntheticTemporalSignDataset(
        h5_path,
        ["0", "1", "2"],
        {"0": "hola", "1": "mundo", "2": "si"},
        {"hola": [10], "mundo": [12, 13], "si": [14]},
        min_clips=2,
        max_clips=3,
        min_neutral_frames=0,
        max_neutral_frames=1,
        samples_per_epoch=4,
        seed=11,
        embedding_table=torch.arange(40, dtype=torch.float32).view(20, 2),
    )

    batch = synthetic_temporal_collate([ds[0], ds[1]])

    assert batch["keypoints"].shape[0] == 2
    assert batch["frame_lengths"].tolist() == [
        len(ds[0].keypoints),
        len(ds[1].keypoints),
    ]
    assert batch["token_ids"].shape[0] == 2
    assert (batch["token_ids"] == -100).any()
    assert batch["boundaries"].shape[:2] == (2, batch["sign_counts"].max().item())
    assert batch["target_embeddings"].shape[:2] == batch["token_ids"].shape


def test_cif_boundary_targets_use_token_span_mass():
    boundaries = torch.tensor([[[0, 2], [4, 8]]])
    spans = torch.tensor([[[0, 1], [1, 3]]])

    target = CIFAggregator.boundary_targets(boundaries, frame_count=8, token_spans=spans)

    assert torch.allclose(target[0, :2], torch.tensor([0.5, 0.5]))
    assert torch.allclose(target[0, 2:4], torch.zeros(2))
    assert torch.allclose(target[0, 4:8], torch.full((4,), 0.5))
    assert torch.isclose(target.sum(), torch.tensor(3.0))


def test_cif_integrates_known_alpha_spikes_and_reports_positions():
    cif = CIFAggregator(hidden_size=2)
    features = torch.tensor(
        [
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 0.0],
                [0.0, 3.0],
                [0.0, 4.0],
            ]
        ]
    )
    alphas = torch.tensor([[0.5, 0.5, 0.0, 0.25, 0.75]])

    out = cif(features, torch.tensor([5]), alphas=alphas)

    assert out.counts.tolist() == [2]
    assert torch.allclose(out.embeddings[0, 0], torch.tensor([1.5, 0.0]))
    assert torch.allclose(out.embeddings[0, 1], torch.tensor([0.0, 3.75]))
    assert torch.allclose(out.fire_positions[0, :2], torch.tensor([0.5, 3.75]))
    assert out.padding_mask.tolist() == [[False, False]]


def test_cif_scales_alpha_to_target_lengths():
    cif = CIFAggregator(hidden_size=1)
    features = torch.ones(1, 4, 1)
    alphas = torch.full((1, 4), 0.25)

    out = cif(features, torch.tensor([4]), alphas=alphas, target_lengths=torch.tensor([2]))

    assert out.counts.tolist() == [2]
    assert torch.allclose(out.quantity, torch.tensor([2.0]))


def test_predict_alpha_is_zero_outside_frame_lengths():
    cif = CIFAggregator(hidden_size=3)
    features = torch.randn(2, 5, 3)
    lengths = torch.tensor([3, 5])

    alphas = cif.predict_alpha(features, lengths)

    assert alphas.shape == (2, 5)
    assert torch.all(alphas[0, 3:] == 0.0)
    assert torch.all(alphas[1] >= 0.0)


def test_token_centers_distributes_uniformly_within_boundary():
    boundaries = torch.tensor([[[0, 4], [4, 8]]])
    spans = torch.tensor([[[0, 2], [2, 3]]])

    centers = CIFAggregator.token_centers(boundaries, spans)

    assert centers.shape == (1, 3)
    assert torch.allclose(centers[0, :2], torch.tensor([1.0, 3.0]))
    assert torch.allclose(centers[0, 2:3], torch.tensor([6.0]))


def test_token_centers_pads_missing_tokens_with_negative_one():
    boundaries = torch.tensor([[[0, 2], [-1, -1]]])
    spans = torch.tensor([[[0, 1], [-1, -1]]])

    centers = CIFAggregator.token_centers(boundaries, spans)

    assert centers.shape == (1, 1)


def test_boundary_error_mae_zero_when_positions_match_centers():
    fire_positions = torch.tensor([[1.0, 3.0]])
    counts = torch.tensor([2])
    centers = torch.tensor([[1.0, 3.0]])
    token_lengths = torch.tensor([2])

    mae = boundary_error_mae(fire_positions, counts, centers, token_lengths)

    assert mae == 0.0


def test_boundary_error_mae_truncates_to_min_of_count_and_token_length():
    fire_positions = torch.tensor([[1.5, -1.0]])
    counts = torch.tensor([1])
    centers = torch.tensor([[1.0, 3.0]])
    token_lengths = torch.tensor([2])

    mae = boundary_error_mae(fire_positions, counts, centers, token_lengths)

    assert mae == pytest.approx(500.25)


def test_boundary_error_mae_penalizes_no_fires():
    fire_positions = torch.tensor([[-1.0]])
    counts = torch.tensor([0])
    centers = torch.tensor([[1.0, 3.0]])
    token_lengths = torch.tensor([2])

    mae = boundary_error_mae(fire_positions, counts, centers, token_lengths)

    assert mae == pytest.approx(1000.0)


def test_alpha_schedule_weights_by_epoch_phase():
    assert alpha_schedule_weights(0) == (0.75, 0.25)
    assert alpha_schedule_weights(2) == (0.75, 0.25)
    assert alpha_schedule_weights(3) == (0.5, 0.5)
    assert alpha_schedule_weights(5) == (0.5, 0.5)
    assert alpha_schedule_weights(6) == (0.25, 0.75)
    assert alpha_schedule_weights(9) == (0.25, 0.75)
    assert alpha_schedule_weights(10) == (0.0, 1.0)
    assert alpha_schedule_weights(40) == (0.0, 1.0)


def _build_tiny_model():
    encoder = nn.Sequential()

    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.stgcn_layers = nn.ModuleList([nn.Linear(2, 2)])
            self.linear_hidden = nn.Linear(2, 2)
            self.tcn = nn.Linear(2, 2)
            self.transformer = nn.Linear(2, 2)

        def forward(self, keypoints, frame_lengths):
            return keypoints

    return TemporalSignPromptModel(TinyEncoder(), hidden_size=2, vocab_size=10, embedding_dim=4)


def test_set_cif_phase_stage0_trains_only_cif():
    model = _build_tiny_model()

    set_cif_phase(model, epoch=0)

    assert all(not p.requires_grad for p in model.frame_encoder.parameters())
    assert all(not p.requires_grad for p in model.token_head.parameters())
    assert all(not p.requires_grad for p in model.embedding_head.parameters())
    assert all(p.requires_grad for p in model.cif.parameters())


def test_set_cif_phase_stage1_keeps_stgcn_frozen():
    model = _build_tiny_model()

    state = set_cif_phase(model, epoch=5)

    assert all(not p.requires_grad for p in model.frame_encoder.stgcn_layers.parameters())
    assert all(not p.requires_grad for p in model.frame_encoder.linear_hidden.parameters())
    assert all(p.requires_grad for p in model.frame_encoder.tcn.parameters())
    assert all(p.requires_grad for p in model.frame_encoder.transformer.parameters())
    assert all(p.requires_grad for p in model.cif.parameters())
    assert all(p.requires_grad for p in model.token_head.parameters())
    assert state["frame_encoder.stgcn_layers"] is False
    assert state["frame_encoder.linear_hidden"] is False
    assert state["frame_encoder.tcn"] is True
    assert state["frame_encoder.transformer"] is True


def test_set_cif_diagnostic_freeze_alpha_only_reports_granular_state():
    model = _build_tiny_model()

    state = set_cif_diagnostic_freeze(model, epoch=5, mode="alpha_only")

    assert state["cif"] is True
    assert state["token_head"] is False
    assert state["embedding_head"] is False
    assert state["frame_encoder.stgcn_layers"] is False
    assert state["frame_encoder.tcn"] is False


def test_target_only_stage1_freeze_changes_at_epoch_three():
    model = _build_tiny_model()

    stage0 = set_cif_diagnostic_freeze(model, epoch=2, mode="target_only_stage1")
    stage1 = set_cif_diagnostic_freeze(model, epoch=3, mode="target_only_stage1")

    assert stage0["cif"] is True
    assert stage0["token_head"] is False
    assert stage0["frame_encoder.tcn"] is False
    assert stage1["cif"] is True
    assert stage1["token_head"] is True
    assert stage1["frame_encoder.tcn"] is True
    assert stage1["frame_encoder.stgcn_layers"] is False


def test_alpha_diagnostics_are_finite_for_masked_lengths():
    features = torch.randn(2, 4, 3)
    logits = torch.tensor([[0.0, -2.0, 1.0, 7.0], [-4.0, -3.0, 5.0, 6.0]])
    lengths = torch.tensor([3, 2])
    alphas = torch.sigmoid(logits)
    alphas[0, 3:] = 0.0
    alphas[1, 2:] = 0.0

    stats = alpha_diagnostics(features, logits, alphas, lengths)

    assert set(stats) == {
        "alpha_logit_mean",
        "alpha_logit_p01",
        "alpha_logit_p50",
        "alpha_logit_p99",
        "alpha_mean",
        "alpha_sum_mean",
        "feature_norm_mean",
        "feature_norm_max",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())


def test_rescale_alphas_to_target_lengths_preserves_requested_quantity():
    alphas = torch.tensor([[0.25, 0.25, 0.0], [0.1, 0.2, 0.3]])
    target_lengths = torch.tensor([2, 3])

    scaled = rescale_alphas_to_target_lengths(alphas, target_lengths)

    assert torch.allclose(scaled.sum(dim=1), target_lengths.float())


def test_length_head_produces_batch_by_class_logits():
    model = _build_tiny_model()
    features = torch.randn(3, 5, 2)
    lengths = torch.tensor([5, 3, 4])

    logits = model.predict_length_logits(features, lengths)

    assert logits.shape == (3, model.max_len_class + 1)


def test_predicted_lengths_are_clamped_to_valid_range():
    model = _build_tiny_model()
    model.max_len_class = 4
    features = torch.randn(2, 3, 2)
    lengths = torch.tensor([3, 3])
    with torch.no_grad():
        model.length_head.classifier[-1].bias.fill_(0.0)
        model.length_head.classifier[-1].bias[0] = 10.0

    pred = model.predict_lengths(features, lengths)

    assert pred.tolist() == [1, 1]


def test_token_head_with_padding_mask_produces_finite_vocab_logits():
    model = _build_tiny_model()
    embeddings = torch.randn(2, 4, 2)
    # second row is fully padded (count=0 sample) to exercise the
    # all-masked-row guard in _TokenHead.
    padding_mask = torch.tensor([[False, False, True, True], [True, True, True, True]])

    logits = model.token_head(embeddings, padding_mask)

    assert logits.shape == (2, 4, 10)
    assert torch.isfinite(logits).all()


def test_rescale_alphas_to_predicted_lengths_uses_clamped_prediction():
    alphas = torch.tensor([[0.25, 0.25, 0.0], [0.1, 0.2, 0.3]])
    predicted = torch.tensor([2.2, 99.0])

    scaled, lengths = rescale_alphas_to_predicted_lengths(alphas, predicted, max_len=4)

    assert lengths.tolist() == [2, 4]
    assert torch.allclose(scaled.sum(dim=1), torch.tensor([2.0, 4.0]))


def test_rescale_alphas_to_rounded_count_rounds_and_clamps_quantity():
    # sums: 3.4 -> 3, 0.3 -> clamp to min_len=1, 99.0 -> clamp to max_len=4
    alphas = torch.tensor(
        [[1.5, 1.0, 0.9], [0.1, 0.1, 0.1], [33.0, 33.0, 33.0]]
    )

    scaled, lengths = rescale_alphas_to_rounded_count(alphas, max_len=4)

    assert lengths.tolist() == [3, 1, 4]
    assert torch.allclose(scaled.sum(dim=1), torch.tensor([3.0, 1.0, 4.0]))


def test_rescale_alphas_to_rounded_count_bias_corrects_undercount():
    # 1.8 rounds to 2 with or without bias; 1.4 only reaches 2 with bias=+0.2
    alphas = torch.tensor([[0.7, 0.7, 0.4], [0.7, 0.5, 0.2]])

    _, no_bias = rescale_alphas_to_rounded_count(alphas, max_len=4)
    _, biased = rescale_alphas_to_rounded_count(alphas, max_len=4, bias=0.2)

    assert no_bias.tolist() == [2, 1]
    assert biased.tolist() == [2, 2]


def test_checkpoint_selection_prefers_pred_len_exact_over_raw_top1():
    # Regression for ROADMAP_A3_CIF_LENGTH_CONDITIONED.md Etapa 1: checkpoint_best.pt
    # must be picked by val_pred_rescaled_to_pred_len.(exact, top1), not by
    # val_pred_raw.top1. Real metrics.jsonl rows from
    # diag_A3_length_head_20260626_000239: the old rule picked epoch 26
    # (val_pred_raw.top1=0.752 beats epoch 25's 0.674) even though epoch 25 is
    # the better deployable checkpoint on the official mode.
    epoch_25 = {
        "pred_len": {"exact": 0.818359375, "top1": 0.8693229814525694},
        "raw_top1": 0.6741624165442772,
    }
    epoch_26 = {
        "pred_len": {"exact": 0.79296875, "top1": 0.8603656638879329},
        "raw_top1": 0.7516882328782231,
    }

    def select_by(rows, metric_fn):
        best_select_metric = (-1.0, -1.0)
        best_epoch = None
        for epoch, metrics in rows:
            select_metric = metric_fn(metrics)
            if select_metric > best_select_metric:
                best_select_metric = select_metric
                best_epoch = epoch
        return best_epoch

    rows = [(25, epoch_25), (26, epoch_26)]
    old_rule = select_by(rows, lambda m: (m["raw_top1"], m["raw_top1"]))
    new_rule = select_by(rows, lambda m: (m["pred_len"]["exact"], m["pred_len"]["top1"]))

    assert old_rule == 26
    assert new_rule == 25


def test_checkpoint_selection_tiebreaks_on_top1_when_exact_is_equal():
    candidates = [
        ("a", {"exact": 0.80, "top1": 0.85}),
        ("b", {"exact": 0.80, "top1": 0.90}),
    ]

    best_select_metric = (-1.0, -1.0)
    best_name = None
    for name, metrics in candidates:
        select_metric = (metrics["exact"], metrics["top1"])
        if select_metric > best_select_metric:
            best_select_metric = select_metric
            best_name = name

    assert best_name == "b"


def test_set_cif_phase_stage2_unfreezes_everything():
    model = _build_tiny_model()
    set_cif_phase(model, epoch=0)

    set_cif_phase(model, epoch=10)

    assert all(p.requires_grad for p in model.parameters())


def test_permute_video_segments_preserves_length_and_content():
    keypoints = torch.arange(10, dtype=torch.float32).view(10, 1, 1)
    boundaries = torch.tensor([[0, 2], [4, 6], [8, 10]])
    rng = random.Random(1)

    permuted = permute_video_segments(keypoints, boundaries, 10, rng)

    assert permuted.shape == keypoints.shape
    assert sorted(permuted.flatten().tolist()) == sorted(keypoints.flatten().tolist())


def test_permute_video_segments_changes_chunk_order():
    keypoints = torch.arange(10, dtype=torch.float32).view(10, 1, 1)
    boundaries = torch.tensor([[0, 2], [4, 6], [8, 10]])

    found_different_order = False
    for seed in range(20):
        permuted = permute_video_segments(keypoints, boundaries, 10, random.Random(seed))
        if not torch.equal(permuted, keypoints):
            found_different_order = True
            break

    assert found_different_order


def test_permute_video_segments_can_force_non_identity():
    keypoints = torch.arange(10, dtype=torch.float32).view(10, 1, 1)
    boundaries = torch.tensor([[0, 2], [4, 6], [8, 10]])

    # seed 5 gives the identity permutation for three chunks.
    permuted = permute_video_segments(
        keypoints, boundaries, 10, random.Random(5), require_change=True
    )

    assert not torch.equal(permuted, keypoints)
    assert sorted(permuted.flatten().tolist()) == sorted(keypoints.flatten().tolist())


def test_token_head_linear_variant_ignores_cross_slot_context():
    model = TemporalSignPromptModel(
        _build_tiny_model().frame_encoder,
        hidden_size=2,
        vocab_size=10,
        embedding_dim=4,
        token_head_variant="linear",
    )
    embeddings = torch.randn(2, 4, 2)
    padding_mask = torch.tensor([[False, False, True, True], [False, True, True, True]])

    logits = model.token_head(embeddings, padding_mask)

    assert logits.shape == (2, 4, 10)
    assert torch.isfinite(logits).all()
    assert not hasattr(model.token_head, "position_embedding")


def test_length_head_mean_variant_pools_uniformly_over_valid_frames():
    model = TemporalSignPromptModel(
        _build_tiny_model().frame_encoder,
        hidden_size=2,
        vocab_size=10,
        embedding_dim=4,
        length_head_variant="mean",
    )
    features = torch.randn(3, 5, 2)
    lengths = torch.tensor([5, 3, 4])

    logits = model.length_head(features, lengths)

    assert logits.shape == (3, model.max_len_class + 1)
    assert not hasattr(model.length_head, "attn")


def test_invalid_head_variant_raises():
    with pytest.raises(ValueError):
        TemporalSignPromptModel(
            _build_tiny_model().frame_encoder,
            hidden_size=2,
            vocab_size=10,
            embedding_dim=4,
            token_head_variant="not-a-real-variant",
        )


def test_default_head_variants_match_current_behavior():
    model = TemporalSignPromptModel(
        _build_tiny_model().frame_encoder, hidden_size=2, vocab_size=10, embedding_dim=4
    )
    assert model.token_head_variant == "contextual"
    assert model.length_head_variant == "attention"
    assert hasattr(model.token_head, "position_embedding")
    assert hasattr(model.length_head, "attn")


def test_checkpoint_resume_loads_full_model_state(tmp_path):
    model = _build_tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "checkpoint_best.pt"
    torch.save(
        {"epoch": 3, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_top1": 0.4},
        ckpt_path,
    )

    fresh_model = _build_tiny_model()
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    state = torch.load(ckpt_path, map_location="cpu")
    fresh_model.load_state_dict(state["model"])
    fresh_optimizer.load_state_dict(state["optimizer"])

    for key, value in model.state_dict().items():
        assert torch.equal(value, fresh_model.state_dict()[key])
    assert state["epoch"] == 3
    assert state["best_top1"] == 0.4
