"""Aggregate per-signer Etapa 4 LOSO audits (from analyze_imitator_a2.py --heldout-signer) into one report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = (
    "top1",
    "top5",
    "exact",
    "pred_len_mae",
    "count_match_rate",
    "boundary_mae_when_count_correct",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit", type=Path, nargs="+", required=True,
        help="One audit JSON per signer, produced by analyze_imitator_a2.py --heldout-signer.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def worst_glosses(data: dict, top_n: int = 5) -> list[dict]:
    by_gloss = data["mode_cuts"]["pred_rescaled_to_pred_len"]["by_gloss"]
    rows = [{"gloss": gloss, **stats} for gloss, stats in by_gloss.items()]
    rows.sort(key=lambda row: (row["exact"], row["samples"]))
    return rows[:top_n]


def validate_rows(rows: list[dict], id_key: str = "signer_id") -> None:
    if not rows:
        raise ValueError("aggregator received an empty row list")
    seen_ids = set()
    for row in rows:
        row_id = row[id_key]
        if row_id in seen_ids:
            raise ValueError(f"duplicate {id_key}={row_id} in aggregator input")
        seen_ids.add(row_id)
        if row.get("samples", 0) <= 0:
            raise ValueError(f"empty entry ({id_key}={row_id} has samples<=0)")


def load_per_signer(audit_paths: list[Path]) -> list[dict]:
    rows = []
    for path in audit_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        summary = data["mode_summaries"]["pred_rescaled_to_pred_len"]
        rows.append(
            {
                "signer_id": data["heldout_signer"],
                "samples": data["samples"],
                "worst_glosses": worst_glosses(data),
                **{key: summary[key] for key in METRICS},
            }
        )
    validate_rows(rows)
    return rows


def average(rows: list[dict], weight_key: str = "samples") -> dict:
    total_weight = sum(row[weight_key] for row in rows)
    return {
        key: sum(row[key] * row[weight_key] for row in rows) / total_weight
        for key in METRICS
    }


def gate_report(avg: dict) -> dict:
    return {
        "pred_len_mae<=0.25": avg["pred_len_mae"] <= 0.25,
        "count_match_rate>=0.80": avg["count_match_rate"] >= 0.80,
        "exact>=0.45": avg["exact"] >= 0.45,
    }


def abandon_signals(avg: dict) -> dict:
    return {
        "exact<0.40": avg["exact"] < 0.40,
        "count_match_rate<0.75": avg["count_match_rate"] < 0.75,
    }


def _write_markdown(path: Path, rows: list[dict], avg: dict, gates: dict, abandon: dict) -> None:
    lines = [
        "# Etapa 4 LOSO — resumen por signante",
        "",
        "| signer | samples | top1 | top5 | exact | pred_len_mae | count_match_rate | boundary_mae_when_count_correct |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['signer_id']} | {row['samples']} | {row['top1']:.4f} | {row['top5']:.4f} | "
            f"{row['exact']:.4f} | {row['pred_len_mae']:.4f} | {row['count_match_rate']:.4f} | "
            f"{row['boundary_mae_when_count_correct']:.4f} |"
        )
    lines.append(
        f"| **avg** | - | {avg['top1']:.4f} | {avg['top5']:.4f} | {avg['exact']:.4f} | "
        f"{avg['pred_len_mae']:.4f} | {avg['count_match_rate']:.4f} | {avg['boundary_mae_when_count_correct']:.4f} |"
    )
    lines += ["", "## Gates Etapa 4 (ROADMAP_A3_CIF_LENGTH_CONDITIONED.md)", ""]
    for key, passed in gates.items():
        lines.append(f"- {key}: {'PASA' if passed else 'FALLA'}")
    lines += ["", "## Señales de abandono CIF (promedio LOSO)", ""]
    for key, triggered in abandon.items():
        lines.append(f"- {key}: {'SI (alarma)' if triggered else 'no'}")
    lines += ["", "## Peores glosas por signante", ""]
    for row in rows:
        lines.append(f"### signer {row['signer_id']}")
        for gloss_row in row["worst_glosses"]:
            lines.append(
                f"- {gloss_row['gloss']}: exact={gloss_row['exact']:.2f} "
                f"token_accuracy={gloss_row.get('token_accuracy', 0):.2f} "
                f"samples={int(gloss_row['samples'])}"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows = load_per_signer(args.audit)
    rows.sort(key=lambda row: row["signer_id"])
    avg = average(rows)
    gates = gate_report(avg)
    abandon = abandon_signals(avg)
    result = {"per_signer": rows, "average": avg, "gates": gates, "abandon_signals": abandon}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    _write_markdown(md_path, rows, avg, gates, abandon)
    print(json.dumps({"wrote": str(args.output), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
