"""Canonical Level-A Imitator run: dataset1 isolated clip -> Gemma token IDs."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


DEFAULT_ARGS = [
    "--phase",
    "teacher_forced",
    "--min-clips",
    "1",
    "--max-clips",
    "1",
    "--min-neutral-frames",
    "0",
    "--max-neutral-frames",
    "0",
    "--run-name",
    "imitator_dataset1_tokens",
    "--prediction-alpha-mode",
    "teacher_alpha",
]


def main():
    script = Path(__file__).with_name("train_temporal_v126.py")
    sys.argv = [str(script), *DEFAULT_ARGS, *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
