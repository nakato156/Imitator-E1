"""Construye el artefacto v122 sin sobrescribir el HDF5 de v120.

Uso:
    PYTHONPATH=. python scripts/data/build_dataset1_v122_h5.py
"""
import sys
from pathlib import Path


if __name__ == "__main__":
    # Reutiliza el builder probado, fijando los defaults específicos de v122.
    import subprocess

    args = [
        sys.executable,
        str(Path(__file__).with_name("build_dataset1_h5.py")),
        "--max-frames",
        "240",
        "--out",
        "data/processed/dataset1_isolated_v122.hdf5",
        "--include-metadata",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(args))
