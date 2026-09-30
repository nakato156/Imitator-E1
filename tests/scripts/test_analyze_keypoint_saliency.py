from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "analyze_keypoint_saliency",
    ROOT / "scripts/interpretability/analyze_keypoint_saliency.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DummyDecoder(torch.nn.Module):
    vocab_map = None

    def to_dense(self, ids):
        return ids

    def forward(self, keypoints, frame_lengths, decoder_input_ids):
        batch, length = decoder_input_ids.shape
        weights = torch.linspace(-1.0, 1.0, 128, device=keypoints.device)
        signal = keypoints.sum(dim=(1, 2, 3))
        return signal[:, None, None] * weights[None, None, :].expand(batch, length, -1)


def sample(sequence_id="s1", frames=3):
    return MODULE.SequenceInput(
        sequence_id=sequence_id,
        clip_ids=("1",),
        keypoints=torch.ones(frames, 111, 2),
        token_ids=torch.tensor([7, 8]),
        glosses=("test",),
    )


def test_analyze_sequence_shapes_and_normalization():
    result = MODULE.analyze_sequence(DummyDecoder(), sample(), torch.device("cpu"))
    assert result["reference_raw"].shape == (2, 3, 111)
    assert result["predicted_raw"].shape == (2, 3, 111)
    assert result["reference_ablation"].shape == (2, 4)
    assert result["predicted_ablation"].shape == (2, 4)
    np.testing.assert_allclose(result["reference_normalized"].sum(axis=(1, 2)), 1.0, atol=2e-7)
    np.testing.assert_allclose(result["predicted_normalized"].sum(axis=(1, 2)), 1.0, atol=2e-7)


def test_pack_uses_offsets_without_object_arrays():
    model = DummyDecoder()
    samples = [sample("short", 2), sample("long", 4)]
    results = [MODULE.analyze_sequence(model, row, torch.device("cpu")) for row in samples]
    packed = MODULE.pack_results(samples, results)
    assert packed["sequence_offsets"].tolist() == [0, 4, 12]
    assert packed["reference_normalized"].shape == (12, 111)
    assert all(value.dtype != object for value in packed.values())
    assert MODULE.unpack_map(packed, 1, "reference_normalized").shape == (2, 4, 111)


def test_similarity_identical_maps_is_one():
    values = np.arange(12, dtype=np.float64)
    cosine, spearman = MODULE.similarity(values, values)
    assert np.isclose(cosine, 1.0)
    assert np.isclose(spearman, 1.0)


def test_sequence_spec_validation(tmp_path):
    path = tmp_path / "sequences.json"
    path.write_text('{"sequences":[{"id":"x","clip_ids":[1,2]}]}')
    assert MODULE.load_sequence_spec(path) == [("x", ("1", "2"))]

