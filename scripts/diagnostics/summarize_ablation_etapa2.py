"""Mean/std/paired-seed-delta summary + causal attribution for the Etapa 2 ablation."""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

from scripts.diagnostics.aggregate_loso_etapa4 import validate_rows

FACTORS = ("token_head", "length_head", "token_label_smoothing")
FACTOR_LEVELS = {
    "token_head": ("linear", "contextual"),
    "length_head": ("mean", "attention"),
    "token_label_smoothing": (0.0, 0.1),
}
FULL_MODEL = {"token_head": "contextual", "length_head": "attention", "token_label_smoothing": 0.1}
GLOBAL_METRICS = ("top1", "top5", "exact", "pred_len_mae", "count_match_rate", "token_accuracy_when_count_correct")


def _run_id(variant: dict) -> tuple:
    return tuple(variant[key] for key in (*FACTORS, "seed"))


def load_runs(pairs: list[tuple[Path, Path]]) -> list[dict]:
    rows = []
    seen = set()
    for registry_path, audit_path in pairs:
        variant = json.loads(Path(registry_path).read_text(encoding="utf-8"))["variant"]
        run_id = _run_id(variant)
        if run_id in seen:
            raise ValueError(f"duplicate ablation run for variant={variant}")
        seen.add(run_id)
        audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        summary = audit["mode_summaries"]["pred_rescaled_to_pred_len"]
        row = {
            "variant": variant,
            "seed": variant["seed"],
            "samples": audit["samples"],
            "exact_3plus": audit["mode_cuts"]["pred_rescaled_to_pred_len"]["by_target_length"]["3+"]["exact"],
            **{key: summary[key] for key in GLOBAL_METRICS},
        }
        rows.append(row)
    validate_rows([{"signer_id": _run_id(r["variant"]), "samples": r["samples"]} for r in rows])
    return rows


def _match(row: dict, **fixed) -> bool:
    return all(row["variant"][key] == value for key, value in fixed.items())


def marginal_effect(rows: list[dict], factor: str, metric: str) -> dict[int, float]:
    """Per-seed delta for `factor`'s high level vs low level, averaged over the other two factors."""
    other_factors = [f for f in FACTORS if f != factor]
    low, high = FACTOR_LEVELS[factor]
    seeds = sorted({row["seed"] for row in rows})
    result = {}
    for seed in seeds:
        deltas = []
        for combo in itertools.product(*(FACTOR_LEVELS[f] for f in other_factors)):
            fixed_other = dict(zip(other_factors, combo))
            low_rows = [r for r in rows if r["seed"] == seed and _match(r, **{factor: low}, **fixed_other)]
            high_rows = [r for r in rows if r["seed"] == seed and _match(r, **{factor: high}, **fixed_other)]
            if not low_rows or not high_rows:
                continue
            deltas.append(high_rows[0][metric] - low_rows[0][metric])
        result[seed] = statistics.mean(deltas) if deltas else 0.0
    return result


def full_model_effect(rows: list[dict], factor: str, metric: str) -> dict[int, float]:
    """Per-seed delta between the all-on combo and the same combo with `factor` switched off."""
    low, high = FACTOR_LEVELS[factor]
    seeds = sorted({row["seed"] for row in rows})
    result = {}
    for seed in seeds:
        on_fixed = {**FULL_MODEL, factor: high}
        off_fixed = {**FULL_MODEL, factor: low}
        on_rows = [r for r in rows if r["seed"] == seed and _match(r, **on_fixed)]
        off_rows = [r for r in rows if r["seed"] == seed and _match(r, **off_fixed)]
        if on_rows and off_rows:
            result[seed] = on_rows[0][metric] - off_rows[0][metric]
    return result


def classify_causal_support(marginal: dict[int, float], full_model: dict[int, float]) -> str:
    if not marginal or not full_model:
        return "efecto mixto"
    if all(value > 0 for value in marginal.values()) and all(value > 0 for value in full_model.values()):
        return "causa respaldada"
    return "efecto mixto"


def variant_summary(rows: list[dict]) -> list[dict]:
    """Mean + stdev across seeds, grouped by the 8 non-seed variant configs."""
    by_config: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["variant"]["token_head"], row["variant"]["length_head"], row["variant"]["token_label_smoothing"])
        by_config.setdefault(key, []).append(row)
    summary_metrics = ("exact_3plus", "token_accuracy_when_count_correct", "count_match_rate", *GLOBAL_METRICS)
    out = []
    for (token_head, length_head, smoothing), group in sorted(by_config.items()):
        entry = {"token_head": token_head, "length_head": length_head, "token_label_smoothing": smoothing, "n_seeds": len(group)}
        for metric in dict.fromkeys(summary_metrics):
            values = [row[metric] for row in group]
            entry[f"{metric}_mean"] = statistics.mean(values)
            entry[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        out.append(entry)
    return out


def validate_complete(rows: list[dict]) -> None:
    """Reject anything but the full 2x2x2x3-seed=24 matrix — partial summaries must not be reported."""
    expected = {
        (token_head, length_head, smoothing, seed)
        for token_head in FACTOR_LEVELS["token_head"]
        for length_head in FACTOR_LEVELS["length_head"]
        for smoothing in FACTOR_LEVELS["token_label_smoothing"]
        for seed in (23, 42, 101)
    }
    actual = {
        (row["variant"]["token_head"], row["variant"]["length_head"], row["variant"]["token_label_smoothing"], row["seed"])
        for row in rows
    }
    missing = expected - actual
    if missing:
        raise ValueError(f"incomplete ablation matrix: missing {len(missing)}/24 runs: {sorted(missing)}")


def causal_report(rows: list[dict]) -> dict:
    metrics = ("exact_3plus", "token_accuracy_when_count_correct", "count_match_rate", "exact")
    report = {}
    for factor in FACTORS:
        report[factor] = {}
        for metric in metrics:
            marginal = marginal_effect(rows, factor, metric)
            full_model = full_model_effect(rows, factor, metric)
            report[factor][metric] = {
                "marginal_by_seed": marginal,
                "full_model_by_seed": full_model,
                "verdict": classify_causal_support(marginal, full_model),
            }
    return report


def main(registry_dir: Path, output: Path) -> None:
    registries = sorted(registry_dir.glob("*.json"))
    pairs = []
    for registry_path in registries:
        run_name = json.loads(registry_path.read_text(encoding="utf-8"))["run_name"]
        audit_path = registry_path.parents[1] / f"diag_{run_name}_audit.json"
        pairs.append((registry_path, audit_path))
    rows = load_runs(pairs)
    validate_complete(rows)
    result = {
        "n_runs": len(rows),
        "variant_summary": variant_summary(rows),
        "causal_report": causal_report(rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"wrote": str(output), "n_runs": len(rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.registry_dir, args.output)
