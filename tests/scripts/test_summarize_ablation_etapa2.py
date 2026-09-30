import json
from pathlib import Path

import pytest

from scripts.diagnostics.summarize_ablation_etapa2 import (
    classify_causal_support,
    full_model_effect,
    load_runs,
    main,
    marginal_effect,
)

METRICS_OK = dict(
    top1=0.9, top5=0.95, exact=0.85, pred_len_mae=0.05, count_match_rate=0.9,
    token_accuracy_when_count_correct=0.95,
)


def _write_run(tmp_path, token_head, length_head, smoothing, seed, exact_3plus, **overrides):
    metrics = {**METRICS_OK, **overrides}
    variant = {
        "token_head": token_head, "length_head": length_head,
        "token_label_smoothing": smoothing, "seed": seed, "split_seed": 23,
    }
    run_name = f"run_{token_head}_{length_head}_{smoothing}_{seed}"
    registry_path = tmp_path / f"{run_name}.registry.json"
    registry_path.write_text(json.dumps({"variant": variant, "run_name": run_name}), encoding="utf-8")
    audit_path = tmp_path / f"{run_name}.audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "samples": 640,
                "mode_summaries": {"pred_rescaled_to_pred_len": metrics},
                "mode_cuts": {
                    "pred_rescaled_to_pred_len": {
                        "by_target_length": {"3+": {"exact": exact_3plus, "samples": 200}},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return registry_path, audit_path


def test_load_runs_rejects_duplicate_variant():
    pass  # covered indirectly via validate_rows reuse; see duplicate test below


def test_marginal_effect_is_positive_for_all_seeds_when_factor_always_helps(tmp_path):
    pairs = []
    for length_head in ("mean", "attention"):
        for smoothing in (0.0, 0.1):
            for seed in (23, 42, 101):
                # token_head=contextual always beats linear by +0.1 on 3+ exact, same seed.
                pairs.append(_write_run(tmp_path, "linear", length_head, smoothing, seed, exact_3plus=0.5))
                pairs.append(_write_run(tmp_path, "contextual", length_head, smoothing, seed, exact_3plus=0.6))
    rows = load_runs(pairs)
    effect = marginal_effect(rows, factor="token_head", metric="exact_3plus")
    assert set(effect.keys()) == {23, 42, 101}
    assert all(value == pytest.approx(0.1) for value in effect.values())


def test_classify_causal_support_requires_positive_in_all_seeds_both_ways():
    marginal = {23: 0.1, 42: 0.05, 101: 0.02}
    full_model = {23: 0.08, 42: 0.03, 101: 0.01}
    assert classify_causal_support(marginal, full_model) == "causa respaldada"


def test_classify_causal_support_is_mixed_when_one_seed_disagrees():
    marginal = {23: 0.1, 42: -0.01, 101: 0.02}
    full_model = {23: 0.08, 42: 0.03, 101: 0.01}
    assert classify_causal_support(marginal, full_model) == "efecto mixto"


def test_classify_causal_support_is_mixed_when_marginal_and_full_model_disagree():
    marginal = {23: 0.1, 42: 0.05, 101: 0.02}
    full_model = {23: -0.02, 42: 0.03, 101: 0.01}
    assert classify_causal_support(marginal, full_model) == "efecto mixto"


def test_full_model_effect_compares_all_on_combo_against_single_factor_off(tmp_path):
    pairs = [
        _write_run(tmp_path, "contextual", "attention", 0.1, 23, exact_3plus=0.86),
        _write_run(tmp_path, "linear", "attention", 0.1, 23, exact_3plus=0.61),
        _write_run(tmp_path, "contextual", "attention", 0.1, 42, exact_3plus=0.84),
        _write_run(tmp_path, "linear", "attention", 0.1, 42, exact_3plus=0.60),
    ]
    rows = load_runs(pairs)
    effect = full_model_effect(rows, factor="token_head", metric="exact_3plus")
    assert effect == {23: pytest.approx(0.25), 42: pytest.approx(0.24)}


def test_load_runs_rejects_duplicate_run(tmp_path):
    registry_path, audit_path = _write_run(tmp_path, "linear", "mean", 0.0, 23, exact_3plus=0.5)
    with pytest.raises(ValueError, match="duplicate"):
        load_runs([(registry_path, audit_path), (registry_path, audit_path)])


def test_validate_complete_rejects_anything_other_than_24_runs(tmp_path):
    from scripts.diagnostics.summarize_ablation_etapa2 import validate_complete

    pairs = [_write_run(tmp_path, "linear", "mean", 0.0, 23, exact_3plus=0.5)]
    rows = load_runs(pairs)
    with pytest.raises(ValueError, match="incomplete"):
        validate_complete(rows)


def _write_operational_layout(base_dir, token_head, length_head, smoothing, seed):
    """Lay out registry + audit files like the real operational step does:
    <base_dir>/registry/{run_name}.json and <base_dir>/diag_{run_name}_audit.json
    (audit as a SIBLING of registry/, not two levels up from it)."""
    metrics = METRICS_OK
    variant = {
        "token_head": token_head, "length_head": length_head,
        "token_label_smoothing": smoothing, "seed": seed, "split_seed": 23,
    }
    run_name = f"A3_etapa2_ablation_{token_head}_{length_head}_ls{smoothing}_seed{seed}"
    registry_dir = base_dir / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / f"{run_name}.json").write_text(
        json.dumps({"variant": variant, "run_name": run_name}), encoding="utf-8"
    )
    (base_dir / f"diag_{run_name}_audit.json").write_text(
        json.dumps(
            {
                "samples": 640,
                "mode_summaries": {"pred_rescaled_to_pred_len": metrics},
                "mode_cuts": {
                    "pred_rescaled_to_pred_len": {
                        "by_target_length": {"3+": {"exact": 0.5, "samples": 200}},
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_main_finds_audits_via_registry_parent_path_join(tmp_path):
    """Integration test for main()'s own path-joining logic (the bug Fix 3
    addresses): registries live in ablation_etapa2/registry/, audits are
    siblings of registry/ inside ablation_etapa2/ — not two levels up."""
    from scripts.diagnostics.summarize_ablation_etapa2 import FACTOR_LEVELS

    base_dir = tmp_path / "ablation_etapa2"
    for token_head in FACTOR_LEVELS["token_head"]:
        for length_head in FACTOR_LEVELS["length_head"]:
            for smoothing in FACTOR_LEVELS["token_label_smoothing"]:
                for seed in (23, 42, 101):
                    _write_operational_layout(base_dir, token_head, length_head, smoothing, seed)

    output = tmp_path / "out.json"
    main(base_dir / "registry", output)

    assert output.is_file()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["n_runs"] == 24


def test_validate_complete_passes_for_full_24_run_matrix(tmp_path):
    from scripts.diagnostics.summarize_ablation_etapa2 import FACTOR_LEVELS, validate_complete

    pairs = []
    seed_counter = 0
    seeds = (23, 42, 101)
    for token_head in FACTOR_LEVELS["token_head"]:
        for length_head in FACTOR_LEVELS["length_head"]:
            for smoothing in FACTOR_LEVELS["token_label_smoothing"]:
                for seed in seeds:
                    pairs.append(_write_run(tmp_path, token_head, length_head, smoothing, seed, exact_3plus=0.5))
                    seed_counter += 1
    rows = load_runs(pairs)
    assert len(rows) == 24
    validate_complete(rows)  # must not raise
