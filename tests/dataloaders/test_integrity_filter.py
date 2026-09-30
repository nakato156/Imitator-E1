import math

from src.mslm.dataloader.integrity_filter import ClipRecord, filter_clip_records


def _rec(**kwargs):
    base = dict(
        clip_id="0",
        has_keypoints=True,
        has_label=True,
        has_embeddings=True,
        has_token_ids=True,
        frame_count=40,
        token_count=8,
        embedding_rows=8,
        truncated=False,
        keypoints_have_nan_inf=False,
        embeddings_have_nan_inf=False,
    )
    base.update(kwargs)
    return ClipRecord(**base)


def test_keeps_healthy_clip():
    kept, manifest = filter_clip_records([_rec()])
    assert kept == ["0"]
    assert manifest[0] == {"clip_id": "0", "kept": True, "cause": None}


def test_excludes_missing_embeddings():
    kept, manifest = filter_clip_records([_rec(has_embeddings=False)])
    assert kept == []
    assert manifest[0]["cause"] == "missing_embeddings"


def test_excludes_too_few_frames():
    kept, manifest = filter_clip_records([_rec(frame_count=7)])
    assert kept == []
    assert manifest[0]["cause"] == "min_frames"


def test_excludes_too_few_visual_tokens():
    kept, manifest = filter_clip_records([_rec(token_count=1, embedding_rows=1)])
    assert kept == []
    assert manifest[0]["cause"] == "min_tokens"


def test_excludes_low_frames_per_token_ratio():
    # 10 frames / 6 tokens = 1.67 < 2
    kept, manifest = filter_clip_records([_rec(frame_count=10, token_count=6, embedding_rows=6)])
    assert kept == []
    assert manifest[0]["cause"] == "frames_per_token"


def test_excludes_truncated_clip():
    kept, manifest = filter_clip_records([_rec(truncated=True)])
    assert kept == []
    assert manifest[0]["cause"] == "truncated"


def test_excludes_token_embedding_row_mismatch():
    kept, manifest = filter_clip_records([_rec(token_count=8, embedding_rows=7)])
    assert kept == []
    assert manifest[0]["cause"] == "token_embedding_mismatch"


def test_excludes_nan_inf():
    kept, manifest = filter_clip_records([_rec(keypoints_have_nan_inf=True)])
    assert kept == []
    assert manifest[0]["cause"] == "nan_inf_keypoints"

    kept, manifest = filter_clip_records([_rec(embeddings_have_nan_inf=True)])
    assert manifest[0]["cause"] == "nan_inf_embeddings"


def test_first_applicable_cause_wins_deterministically():
    # un clip puede violar varias reglas a la vez; el orden de chequeo debe ser fijo
    rec = _rec(has_keypoints=False, frame_count=1, token_count=0)
    _, manifest = filter_clip_records([rec])
    assert manifest[0]["cause"] == "missing_keypoints"


def test_manifest_covers_every_input_clip():
    records = [_rec(clip_id=str(i)) for i in range(5)]
    records[2] = _rec(clip_id="2", has_label=False)
    kept, manifest = filter_clip_records(records)
    assert len(manifest) == 5
    assert kept == ["0", "1", "3", "4"]
