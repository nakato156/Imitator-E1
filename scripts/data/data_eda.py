#!/usr/bin/env python3
"""EDA reproducible para los datasets crudos de lengua de senas."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


VIDEO_EXTENSIONS = {".mp4", ".flv", ".mov", ".avi", ".mkv", ".webm"}
SPANISH_HINTS = {
    "a",
    "al",
    "como",
    "con",
    "de",
    "del",
    "el",
    "ella",
    "en",
    "es",
    "esta",
    "estaba",
    "la",
    "las",
    "le",
    "lo",
    "los",
    "no",
    "para",
    "pero",
    "por",
    "que",
    "se",
    "si",
    "su",
    "un",
    "una",
    "y",
    "ya",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root.parent / "data" / "raw",
        help="Directorio que contiene dataset1, dataset2, etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "data_eda",
        help="Directorio para CSV, JSON y reporte Markdown.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Cantidad de procesos ffprobe concurrentes.",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="No ejecutar ffprobe; reutiliza la cache disponible.",
    )
    return parser.parse_args()


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(v for v in values if v is not None and math.isfinite(v))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def safe_round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)?", text.casefold(), re.UNICODE)


def foreign_script_count(text: str) -> int:
    count = 0
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "LATIN" not in name and "ORDINAL INDICATOR" not in name:
            count += 1
    return count


def annotation_type(word_counts: list[int]) -> str:
    median = percentile(word_counts, 0.5) or 0
    p90 = percentile(word_counts, 0.9) or 0
    if median <= 1 and p90 <= 2:
        return "glosas/signos aislados"
    if median <= 5 and p90 <= 10:
        return "frases cortas o glosas compuestas"
    if median <= 30:
        return "frases"
    return "narraciones/transcripciones largas"


def inferred_language(labels: list[str]) -> str:
    tokens = [token for label in labels for token in words(label)]
    hint_count = sum(token in SPANISH_HINTS for token in tokens)
    foreign_count = sum(foreign_script_count(label) for label in labels)
    if foreign_count:
        return "espanol con ruido multilingue"
    if hint_count or labels:
        return "espanol"
    return "sin texto"


def read_metadata(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_fps(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return None


def probe_video(path: Path, data_root: Path) -> dict[str, Any]:
    stat = path.stat()
    base = {
        "path": str(path.relative_to(data_root)),
        "dataset": path.relative_to(data_root).parts[0],
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "is_appledouble": path.name.startswith("._"),
        "duration_seconds": None,
        "width": None,
        "height": None,
        "fps": None,
        "video_codec": "",
        "audio_codec": "",
        "has_audio": False,
        "probe_error": "",
    }
    if base["is_appledouble"]:
        base["probe_error"] = "AppleDouble sidecar; no es un video real"
        return base
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
        if result.returncode:
            base["probe_error"] = result.stderr.strip() or f"ffprobe rc={result.returncode}"
            return base
        payload = json.loads(result.stdout)
        format_data = payload.get("format", {})
        duration = format_data.get("duration")
        base["duration_seconds"] = float(duration) if duration else None
        for stream in payload.get("streams", []):
            if stream.get("codec_type") == "video" and not base["video_codec"]:
                base["video_codec"] = stream.get("codec_name", "")
                base["width"] = stream.get("width")
                base["height"] = stream.get("height")
                base["fps"] = parse_fps(stream.get("avg_frame_rate"))
            elif stream.get("codec_type") == "audio" and not base["audio_codec"]:
                base["audio_codec"] = stream.get("codec_name", "")
                base["has_audio"] = True
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError) as exc:
        base["probe_error"] = str(exc)
    return base


def load_probe_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["path"]: row for row in csv.DictReader(handle)}


def cache_is_current(cached: dict[str, Any], path: Path) -> bool:
    stat = path.stat()
    return (
        int(cached.get("size_bytes", -1)) == stat.st_size
        and int(cached.get("mtime_ns", -1)) == stat.st_mtime_ns
    )


def coerce_probe_row(row: dict[str, Any]) -> dict[str, Any]:
    for field in ("size_bytes", "mtime_ns", "width", "height"):
        row[field] = int(row[field]) if row.get(field) not in ("", None) else None
    for field in ("duration_seconds", "fps"):
        row[field] = float(row[field]) if row.get(field) not in ("", None) else None
    row["is_appledouble"] = str(row.get("is_appledouble", "")).lower() == "true"
    row["has_audio"] = str(row.get("has_audio", "")).lower() == "true"
    return row


def collect_video_metadata(
    paths: list[Path],
    data_root: Path,
    cache_path: Path,
    workers: int,
    skip_probe: bool,
) -> list[dict[str, Any]]:
    cache = load_probe_cache(cache_path)
    rows: list[dict[str, Any]] = []
    pending: list[Path] = []
    for path in paths:
        key = str(path.relative_to(data_root))
        if key in cache and cache_is_current(cache[key], path):
            rows.append(coerce_probe_row(cache[key]))
        elif skip_probe:
            rows.append(probe_video(path, data_root) if path.name.startswith("._") else {
                "path": key,
                "dataset": key.split("/", 1)[0],
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "is_appledouble": False,
                "duration_seconds": None,
                "width": None,
                "height": None,
                "fps": None,
                "video_codec": "",
                "audio_codec": "",
                "has_audio": False,
                "probe_error": "No cache; --skip-probe activo",
            })
        else:
            pending.append(path)

    if pending:
        print(f"Ejecutando ffprobe sobre {len(pending):,} archivos...")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(probe_video, path, data_root): path for path in pending
            }
            completed = 0
            for future in as_completed(futures):
                rows.append(future.result())
                completed += 1
                if completed % 500 == 0 or completed == len(pending):
                    print(f"  {completed:,}/{len(pending):,}")
    return sorted(rows, key=lambda row: row["path"])


def describe(values: list[float]) -> dict[str, Any]:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return {
        "count": len(clean),
        "min": safe_round(min(clean) if clean else None),
        "p25": safe_round(percentile(clean, 0.25)),
        "median": safe_round(percentile(clean, 0.5)),
        "mean": safe_round(statistics.fmean(clean) if clean else None),
        "p75": safe_round(percentile(clean, 0.75)),
        "p95": safe_round(percentile(clean, 0.95)),
        "max": safe_round(max(clean) if clean else None),
        "total": safe_round(sum(clean) if clean else None),
    }


def iqr_bounds(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 4:
        return None, None
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    if q1 is None or q3 is None:
        return None, None
    spread = q3 - q1
    return q1 - 1.5 * spread, q3 + 1.5 * spread


def parse_srt(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\r?\n\r?\n+", text.strip())
    entries = []
    timestamp = re.compile(
        r"(\d+):(\d+):(\d+)[,.](\d+)\s+-->\s+"
        r"(\d+):(\d+):(\d+)[,.](\d+)"
    )
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        time_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if time_index is None:
            continue
        match = timestamp.search(lines[time_index])
        if not match:
            continue
        values = [int(value) for value in match.groups()]
        start = values[0] * 3600 + values[1] * 60 + values[2] + values[3] / 1000
        end = values[4] * 3600 + values[5] * 60 + values[6] + values[7] / 1000
        label = " ".join(lines[time_index + 1 :])
        entries.append({"label": label, "duration_seconds": end - start})
    return entries


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/d"
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_root.exists():
        raise SystemExit(f"No existe el directorio de datos: {data_root}")

    datasets = sorted(path for path in data_root.glob("dataset*") if path.is_dir())
    all_files = [path for path in data_root.rglob("*") if path.is_file()]
    video_paths = sorted(
        path for path in all_files if path.suffix.casefold() in VIDEO_EXTENSIONS
    )
    video_rows = collect_video_metadata(
        video_paths,
        data_root,
        output_dir / "video_metadata.csv",
        args.workers,
        args.skip_probe,
    )
    write_csv(output_dir / "video_metadata.csv", video_rows)

    videos_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in video_rows:
        videos_by_dataset[row["dataset"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    duration_outliers: list[dict[str, Any]] = []
    label_outliers: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []

    for dataset_path in datasets:
        dataset = dataset_path.name
        metadata = read_metadata(dataset_path / "meta.csv")
        labels = [(row.get("label") or "").strip() for row in metadata]
        ids = [(row.get("id") or "").strip() for row in metadata]
        word_counts = [len(words(label)) for label in labels]
        char_counts = [len(label) for label in labels]
        all_dataset_videos = videos_by_dataset[dataset]
        usable_videos = [
            row for row in all_dataset_videos if not row["is_appledouble"]
        ]
        probed_videos = [
            row
            for row in usable_videos
            if row["duration_seconds"] is not None and not row["probe_error"]
        ]
        durations = [row["duration_seconds"] for row in probed_videos]

        stems = Counter(Path(row["filename"]).stem for row in usable_videos)
        normalized_stems: dict[str, list[str]] = defaultdict(list)
        for row in usable_videos:
            normalized_stems[normalize_name(Path(row["filename"]).stem)].append(
                row["filename"]
            )
        exact_matches = sum(stems.get(identifier, 0) > 0 for identifier in ids)
        normalized_matches = sum(
            normalize_name(identifier) in normalized_stems for identifier in ids
        )
        id_norms = {normalize_name(identifier) for identifier in ids}
        for identifier in ids:
            if normalize_name(identifier) not in normalized_stems:
                unmatched_rows.append(
                    {
                        "dataset": dataset,
                        "kind": "metadata_without_normalized_video",
                        "value": identifier,
                    }
                )
        for row in usable_videos:
            stem = Path(row["filename"]).stem
            if normalize_name(stem) not in id_norms:
                unmatched_rows.append(
                    {
                        "dataset": dataset,
                        "kind": "video_without_normalized_metadata",
                        "value": row["filename"],
                    }
                )

        duration_low, duration_high = iqr_bounds(durations)
        if duration_low is not None and duration_high is not None:
            for row in probed_videos:
                duration = row["duration_seconds"]
                if duration < duration_low or duration > duration_high:
                    duration_outliers.append(
                        {
                            "dataset": dataset,
                            "filename": row["filename"],
                            "duration_seconds": safe_round(duration),
                            "iqr_lower": safe_round(duration_low),
                            "iqr_upper": safe_round(duration_high),
                            "direction": "short" if duration < duration_low else "long",
                        }
                    )

        word_low, word_high = iqr_bounds([float(value) for value in word_counts])
        if word_low is not None and word_high is not None:
            for row, count in zip(metadata, word_counts):
                if count < word_low or count > word_high:
                    label_outliers.append(
                        {
                            "dataset": dataset,
                            "id": row.get("id", ""),
                            "word_count": count,
                            "iqr_lower": safe_round(word_low),
                            "iqr_upper": safe_round(word_high),
                            "label": row.get("label", ""),
                        }
                    )

        duration_stats = describe(durations)
        word_stats = describe([float(value) for value in word_counts])
        summary_rows.append(
            {
                "dataset": dataset,
                "metadata_rows": len(metadata),
                "unique_ids": len(set(ids)),
                "video_extension_files": len(all_dataset_videos),
                "appledouble_sidecars": sum(
                    row["is_appledouble"] for row in all_dataset_videos
                ),
                "usable_videos": len(usable_videos),
                "probe_failures": sum(
                    bool(row["probe_error"]) for row in usable_videos
                ),
                "exact_id_matches": exact_matches,
                "normalized_id_matches": normalized_matches,
                "duplicate_video_stems": sum(value > 1 for value in stems.values()),
                "annotation_type": annotation_type(word_counts),
                "language": inferred_language(labels),
                "unique_labels": len(set(labels)),
                "duplicate_label_rows": len(labels) - len(set(labels)),
                "empty_labels": sum(not label for label in labels),
                "label_words_median": word_stats["median"],
                "label_words_p95": word_stats["p95"],
                "label_words_max": word_stats["max"],
                "label_chars_median": describe([float(v) for v in char_counts])["median"],
                "foreign_script_characters": sum(
                    foreign_script_count(label) for label in labels
                ),
                "duration_min_seconds": duration_stats["min"],
                "duration_median_seconds": duration_stats["median"],
                "duration_mean_seconds": duration_stats["mean"],
                "duration_p95_seconds": duration_stats["p95"],
                "duration_max_seconds": duration_stats["max"],
                "duration_total_seconds": duration_stats["total"],
                "duration_outliers_iqr": sum(
                    row["dataset"] == dataset for row in duration_outliers
                ),
                "label_outliers_iqr": sum(
                    row["dataset"] == dataset for row in label_outliers
                ),
                "resolutions": "; ".join(
                    f"{width}x{height} ({count})"
                    for (width, height), count in Counter(
                        (row["width"], row["height"]) for row in probed_videos
                    ).most_common()
                ),
                "video_codecs": "; ".join(
                    f"{codec} ({count})"
                    for codec, count in Counter(
                        row["video_codec"] for row in probed_videos
                    ).most_common()
                ),
                "videos_with_audio": sum(row["has_audio"] for row in probed_videos),
            }
        )

    srt_rows: list[dict[str, Any]] = []
    for path in sorted(data_root.rglob("*.srt")):
        entries = parse_srt(path)
        entry_words = [len(words(entry["label"])) for entry in entries]
        entry_durations = [entry["duration_seconds"] for entry in entries]
        srt_rows.append(
            {
                "path": str(path.relative_to(data_root)),
                "entries": len(entries),
                "unique_labels": len({entry["label"] for entry in entries}),
                "median_words": safe_round(percentile(entry_words, 0.5)),
                "max_words": max(entry_words, default=0),
                "median_entry_seconds": safe_round(percentile(entry_durations, 0.5)),
                "max_entry_seconds": safe_round(max(entry_durations, default=0)),
            }
        )

    inventory_counts = Counter(path.suffix.casefold() or "[sin_extension]" for path in all_files)
    inventory_bytes = Counter()
    for path in all_files:
        inventory_bytes[path.suffix.casefold() or "[sin_extension]"] += path.stat().st_size
    inventory_rows = [
        {
            "extension": extension,
            "files": count,
            "size_bytes": inventory_bytes[extension],
        }
        for extension, count in inventory_counts.most_common()
    ]

    write_csv(output_dir / "dataset_summary.csv", summary_rows)
    write_csv(
        output_dir / "duration_outliers.csv",
        duration_outliers,
        ["dataset", "filename", "duration_seconds", "iqr_lower", "iqr_upper", "direction"],
    )
    write_csv(
        output_dir / "label_outliers.csv",
        label_outliers,
        ["dataset", "id", "word_count", "iqr_lower", "iqr_upper", "label"],
    )
    write_csv(
        output_dir / "metadata_unmatched.csv",
        unmatched_rows,
        ["dataset", "kind", "value"],
    )
    write_csv(output_dir / "srt_summary.csv", srt_rows)
    write_csv(output_dir / "file_inventory.csv", inventory_rows)

    usable_rows = [row for row in video_rows if not row["is_appledouble"]]
    valid_duration_rows = [
        row for row in usable_rows if row["duration_seconds"] is not None
    ]
    overall_seconds = sum(row["duration_seconds"] for row in valid_duration_rows)
    payload = {
        "data_root": str(data_root),
        "file_count": len(all_files),
        "size_bytes": sum(path.stat().st_size for path in all_files),
        "video_extension_files": len(video_rows),
        "appledouble_video_sidecars": sum(row["is_appledouble"] for row in video_rows),
        "usable_videos": len(usable_rows),
        "probed_videos": len(valid_duration_rows),
        "probe_failures": sum(bool(row["probe_error"]) for row in usable_rows),
        "total_video_seconds": safe_round(overall_seconds),
        "datasets": summary_rows,
        "srt_files": len(srt_rows),
        "srt_entries": sum(row["entries"] for row in srt_rows),
    }
    (output_dir / "eda_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# EDA de datos crudos",
        "",
        f"- Raiz analizada: `{data_root}`",
        f"- Archivos totales: **{len(all_files):,}** "
        f"({payload['size_bytes'] / 1024**3:.2f} GiB).",
        f"- Archivos con extension de video: **{len(video_rows):,}**.",
        f"- Sidecars AppleDouble `._*` descartados: "
        f"**{payload['appledouble_video_sidecars']:,}**.",
        f"- Videos candidatos reales: **{len(usable_rows):,}**.",
        f"- Videos con duracion valida: **{len(valid_duration_rows):,}**; "
        f"fallos de lectura: **{payload['probe_failures']:,}**.",
        f"- Duracion total: **{format_duration(overall_seconds)}**.",
        "",
        "## Resumen por dataset",
        "",
        markdown_table(
            [
                "Dataset",
                "Videos",
                "Filas meta",
                "Tipo de texto",
                "Idioma",
                "Mediana video (s)",
                "P95 video (s)",
                "Max video (s)",
                "Outliers",
            ],
            [
                [
                    row["dataset"],
                    row["usable_videos"],
                    row["metadata_rows"],
                    row["annotation_type"],
                    row["language"],
                    row["duration_median_seconds"],
                    row["duration_p95_seconds"],
                    row["duration_max_seconds"],
                    row["duration_outliers_iqr"],
                ]
                for row in summary_rows
            ],
        ),
        "",
        "## Texto y correspondencia",
        "",
        markdown_table(
            [
                "Dataset",
                "Mediana palabras",
                "P95 palabras",
                "Max palabras",
                "Labels unicos",
                "Match exacto",
                "Match normalizado",
            ],
            [
                [
                    row["dataset"],
                    row["label_words_median"],
                    row["label_words_p95"],
                    row["label_words_max"],
                    row["unique_labels"],
                    f"{row['exact_id_matches']}/{row['metadata_rows']}",
                    f"{row['normalized_id_matches']}/{row['metadata_rows']}",
                ]
                for row in summary_rows
            ],
        ),
        "",
        "## Notas de interpretacion",
        "",
        "- `dataset1`, `dataset3`, `dataset5` y `dataset7` contienen principalmente "
        "glosas o signos aislados.",
        "- `dataset2` contiene frases segmentadas; `dataset4` y `dataset6` contienen "
        "narraciones o transcripciones largas.",
        "- El idioma dominante de las anotaciones es espanol. La etiqueta "
        "`espanol con ruido multilingue` indica caracteres de otros alfabetos, "
        "probablemente errores de transcripcion automatica.",
        "- Los outliers se calculan dentro de cada dataset con la regla IQR "
        "(fuera de Q1 - 1.5*IQR o Q3 + 1.5*IQR).",
        "- Los matches normalizados ignoran mayusculas, tildes, espacios y signos "
        "para detectar inconsistencias de nombres sin confundirlas con faltantes.",
        "- Revisar `duration_outliers.csv`, `label_outliers.csv` y "
        "`metadata_unmatched.csv` antes de entrenar.",
        "",
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(f"EDA generado en {output_dir}")


if __name__ == "__main__":
    main()
