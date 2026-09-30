import json
from pathlib import Path

import pytest

from scripts.diagnostics.aggregate_loso_etapa4 import (
    abandon_signals,
    average,
    gate_report,
    load_per_signer,
    validate_rows,
    worst_glosses,
)


def _write_audit(path: Path, signer: int, by_gloss=None, **metrics):
    payload = {
        "heldout_signer": signer,
        "samples": 50,
        "mode_summaries": {"pred_rescaled_to_pred_len": metrics},
        "mode_cuts": {
            "pred_rescaled_to_pred_len": {"by_gloss": by_gloss or {}},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


_OK = dict(
    top1=0.9, top5=0.95, exact=0.5, pred_len_mae=0.1,
    count_match_rate=0.85, boundary_mae_when_count_correct=0.5,
)
_BAD = dict(
    top1=0.3, top5=0.4, exact=0.2, pred_len_mae=0.4,
    count_match_rate=0.5, boundary_mae_when_count_correct=2.0,
)


def test_average_and_gates_pass(tmp_path):
    paths = [
        _write_audit(tmp_path / "s0.json", 0, **_OK),
        _write_audit(tmp_path / "s1.json", 1, **_OK),
    ]
    rows = load_per_signer(paths)
    avg = average(rows)
    assert avg["exact"] == 0.5
    assert gate_report(avg) == {
        "pred_len_mae<=0.25": True,
        "count_match_rate>=0.80": True,
        "exact>=0.45": True,
    }
    assert abandon_signals(avg) == {"exact<0.40": False, "count_match_rate<0.75": False}


def test_gates_fail_when_metrics_below_threshold(tmp_path):
    paths = [_write_audit(tmp_path / "s0.json", 0, **_BAD)]
    rows = load_per_signer(paths)
    avg = average(rows)
    assert gate_report(avg) == {
        "pred_len_mae<=0.25": False,
        "count_match_rate>=0.80": False,
        "exact>=0.45": False,
    }
    assert abandon_signals(avg) == {"exact<0.40": True, "count_match_rate<0.75": True}


def test_worst_glosses_sorted_ascending_by_exact(tmp_path):
    by_gloss = {
        "bien": {"samples": 5, "exact": 0.9, "token_accuracy": 0.9, "count_match": 1.0},
        "víveres": {"samples": 5, "exact": 0.1, "token_accuracy": 0.4, "count_match": 0.2},
    }
    path = _write_audit(tmp_path / "s0.json", 0, by_gloss=by_gloss, **_OK)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = worst_glosses(data, top_n=2)
    assert [row["gloss"] for row in rows] == ["víveres", "bien"]


def test_average_is_weighted_by_samples():
    rows = [
        {"signer_id": 0, "samples": 10, "exact": 0.2, "top1": 0.0, "top5": 0.0,
         "pred_len_mae": 0.0, "count_match_rate": 0.0, "boundary_mae_when_count_correct": 0.0},
        {"signer_id": 1, "samples": 90, "exact": 0.8, "top1": 0.0, "top5": 0.0,
         "pred_len_mae": 0.0, "count_match_rate": 0.0, "boundary_mae_when_count_correct": 0.0},
    ]
    avg = average(rows)
    # weighted: (10*0.2 + 90*0.8) / 100 = 0.74, vs naive (0.2+0.8)/2 = 0.5
    assert avg["exact"] == pytest.approx(0.74)


def test_validate_rows_rejects_empty_list():
    with pytest.raises(ValueError, match="empty"):
        validate_rows([])


def test_validate_rows_rejects_duplicate_ids():
    rows = [{"signer_id": 1, "samples": 5}, {"signer_id": 1, "samples": 5}]
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows(rows)


def test_validate_rows_rejects_zero_sample_entries():
    rows = [{"signer_id": 1, "samples": 0}]
    with pytest.raises(ValueError, match="empty entry"):
        validate_rows(rows)


def test_load_per_signer_raises_on_duplicate_signer(tmp_path):
    p1 = _write_audit(tmp_path / "s0.json", 0, **_OK)
    p2 = _write_audit(tmp_path / "s0_dup.json", 0, **_OK)
    with pytest.raises(ValueError, match="duplicate"):
        load_per_signer([p1, p2])
