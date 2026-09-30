"""Orquesta réplicas, ablations v122, sweep v124 y leave-one-signer-out."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "scripts" / "train" / "train_isolated_staged.py"
SUMMARY_ROOT = ROOT.parent / "outputs" / "experiment_summaries"


def run_one(config, seed, run_id, weight_decay=None, heldout_signer=None):
    name = Path(config).stem
    suffix = f"s{seed}_r{run_id}"
    if weight_decay is not None:
        suffix += f"_wd{weight_decay:g}"
    if heldout_signer is not None:
        suffix += f"_signer{heldout_signer}"
    summary = SUMMARY_ROOT / name / f"{suffix}.json"
    if summary.is_file():
        print(f"[resume] reutilizando {summary}", flush=True)
        return json.loads(summary.read_text())
    command = [
        sys.executable,
        str(TRAIN),
        "--config",
        config,
        "--seed",
        str(seed),
        "--run-id",
        str(run_id),
        "--summary-path",
        str(summary),
    ]
    if weight_decay is not None:
        command += ["--weight-decay", str(weight_decay)]
    if heldout_signer is not None:
        command += ["--heldout-signer", str(heldout_signer)]
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(summary.read_text())


def aggregate(rows):
    scores = [row["best_val_top1"] for row in rows]
    gaps = [row["generalization_gap"] for row in rows]
    return {
        "runs": len(rows),
        "mean_top1": statistics.mean(scores),
        "std_top1": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "worst_top1": min(scores),
        "mean_gap": statistics.mean(gaps),
        "success": (
            statistics.mean(scores) >= 0.80
            and (statistics.pstdev(scores) if len(scores) > 1 else 0.0) <= 0.05
            and statistics.mean(gaps) <= 0.10
        ),
        "details": rows,
    }


def save_report(name, rows):
    report = aggregate(rows)
    path = SUMMARY_ROOT / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("staged")
    sub.add_parser("v121")
    sub.add_parser("v122")
    sub.add_parser("v124-sweep")
    loso = sub.add_parser("loso")
    loso.add_argument("--config", default="experiments/v121_v124_isolated_staged/cls_v123.toml")
    args = parser.parse_args()

    if args.command == "staged":
        v121_rows = [run_one("experiments/v121_v124_isolated_staged/cls_v121.toml", 23, 23)]
        if v121_rows[0]["best_val_top1"] < 0.70:
            save_report("v121_three_seeds", v121_rows)
            raise SystemExit("v121 no alcanzó 70%; se detiene la progresión")
        v121_rows += [
            run_one("experiments/v121_v124_isolated_staged/cls_v121.toml", 42, 42),
            run_one("experiments/v121_v124_isolated_staged/cls_v121.toml", 101, 101),
        ]
        v121_report = aggregate(v121_rows)
        save_report("v121_three_seeds", v121_rows)
        if not v121_report["success"]:
            raise SystemExit("v121 no cumple media/std/gap; se detiene la progresión")

        h5_v122 = ROOT.parent / "data" / "processed" / "dataset1_isolated_v122.hdf5"
        if not h5_v122.exists():
            raise SystemExit(
                "Falta dataset1_isolated_v122.hdf5; ejecutar scripts/data/build_dataset1_v122_h5.py"
            )
        v122_configs = [
            "experiments/v121_v124_isolated_staged/cls_v122_no_trim.toml",
            "experiments/v121_v124_isolated_staged/cls_v122_trim.toml",
            "experiments/v121_v124_isolated_staged/cls_v122_trim_downsample.toml",
        ]
        v122_rows = [
            run_one(config, 23, i) for i, config in enumerate(v122_configs)
        ]
        save_report("v122_ablation", v122_rows)
        best_v122 = max(v122_rows, key=lambda row: row["best_val_top1"])
        if best_v122["best_val_top1"] <= v121_report["mean_top1"]:
            raise SystemExit("v122 no supera v121; no se avanza a v123")

        v123_rows = [
            run_one("experiments/v121_v124_isolated_staged/cls_v123.toml", seed, 300 + i)
            for i, seed in enumerate([23, 42, 101])
        ]
        save_report("v123_three_seeds", v123_rows)
        v123_report = aggregate(v123_rows)
        if v123_report["mean_top1"] <= best_v122["best_val_top1"]:
            raise SystemExit("v123 no supera la mejor ablation v122; no se avanza a v124")

        v124_rows = []
        for model_index, config in enumerate(
            [
                "experiments/v121_v124_isolated_staged/cls_v124_base.toml",
                "experiments/v121_v124_isolated_staged/cls_v124_reduced.toml",
            ]
        ):
            for wd_index, weight_decay in enumerate([1e-4, 1e-3, 5e-3]):
                for seed_index, seed in enumerate([23, 42, 101]):
                    run_id = model_index * 100 + wd_index * 10 + seed_index
                    v124_rows.append(
                        run_one(config, seed, run_id, weight_decay=weight_decay)
                    )
        save_report("v124_sweep", v124_rows)
    elif args.command == "v121":
        rows = [run_one("experiments/v121_v124_isolated_staged/cls_v121.toml", 23, 23)]
        if rows[0]["best_val_top1"] >= 0.70:
            rows += [
                run_one("experiments/v121_v124_isolated_staged/cls_v121.toml", 42, 42),
                run_one("experiments/v121_v124_isolated_staged/cls_v121.toml", 101, 101),
            ]
        save_report("v121_three_seeds", rows)
    elif args.command == "v122":
        configs = [
            "experiments/v121_v124_isolated_staged/cls_v122_no_trim.toml",
            "experiments/v121_v124_isolated_staged/cls_v122_trim.toml",
            "experiments/v121_v124_isolated_staged/cls_v122_trim_downsample.toml",
        ]
        rows = [run_one(config, 23, i) for i, config in enumerate(configs)]
        save_report("v122_ablation", rows)
    elif args.command == "v124-sweep":
        rows = []
        for model_index, config in enumerate(
            [
                "experiments/v121_v124_isolated_staged/cls_v124_base.toml",
                "experiments/v121_v124_isolated_staged/cls_v124_reduced.toml",
            ]
        ):
            for wd_index, weight_decay in enumerate([1e-4, 1e-3, 5e-3]):
                for seed_index, seed in enumerate([23, 42, 101]):
                    run_id = model_index * 100 + wd_index * 10 + seed_index
                    rows.append(run_one(config, seed, run_id, weight_decay=weight_decay))
        save_report("v124_sweep", rows)
    else:
        rows = [
            run_one(args.config, 23, 200 + signer, heldout_signer=signer)
            for signer in range(1, 11)
        ]
        report = aggregate(rows)
        report["worst_signer"] = min(rows, key=lambda row: row["best_val_top1"])[
            "heldout_signer"
        ]
        path = SUMMARY_ROOT / f"{Path(args.config).stem}_loso.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
