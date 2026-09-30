"""Utilidades públicas de MSLM con imports pesados bajo demanda.

`setup_train` depende de `src.mslm.models`; importar todo aquí al cargar el
paquete crea un ciclo (`utils -> setup_train -> models -> utils`) que rompe
módulos puros como `embedding_space` y `text_metrics`.  Se conserva la API
histórica mediante `__getattr__`.
"""

from .early_stopping import EarlyStopping
from .config_loader import ConfigLoader

_SETUP_TRAIN_EXPORTS = {
    "setup_paths",
    "prepare_datasets",
    "create_dataloaders",
    "build_model",
    "check_checkpoint",
    "profile_training",
    "run_training",
    "run_dt_training",
}

__all__ = ["EarlyStopping", "ConfigLoader", *_SETUP_TRAIN_EXPORTS]


def __getattr__(name: str):
    if name in _SETUP_TRAIN_EXPORTS:
        from . import setup_train

        return getattr(setup_train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
