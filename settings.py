"""Inicialización del entorno de ejecución del proyecto.

`scripts/train/train.py` (y otros entrypoints) hacen `from settings import initialize; initialize()`.
Este archivo no estaba versionado (estaba en la máquina del dev). Aquí se centraliza:
  - semillas reproducibles,
  - variables de entorno (tokenizers, etc.),
  - instanciación del singleton de rutas `PathVariables` con base_path = raíz del repo.
"""
import os
import random
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

# El HDF5 de entrenamiento por defecto que espera el pipeline (ver src/mslm/utils/paths.py).
DATASET_FILENAME = os.environ.get("MSLM_DATASET", "dataset_v6_unsloth.hdf5")


def initialize(seed: int = 23, dataset_filename: str | None = None):
    """Prepara entorno reproducible y rutas. Devuelve el singleton `path_vars`."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    # Instancia el singleton de rutas anclado a la raíz del repo (data/ está en el padre).
    from src.mslm.utils.paths import PathVariables
    path_vars = PathVariables(
        base_path=_REPO_ROOT,
        dataset_filename=dataset_filename or DATASET_FILENAME,
    )
    return path_vars


if __name__ == "__main__":
    pv = initialize()
    print("Repo root:", _REPO_ROOT)
    print("data_path:", pv.data_path)
    print("h5_file:", pv.h5_file)
