import os
import tomllib
from pathlib import Path

class ConfigDict(dict):
    def __getattr__(self, name):
        try:
            value = self[name]
            return ConfigDict(value) if isinstance(value, dict) else value
        except KeyError as e:
            raise AttributeError(f"No attribute or key '{name}'") from e

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]
        
def load_toml(path: str | Path) -> ConfigDict:
    with open(path, "rb") as f:
        return ConfigDict(tomllib.load(f))

class ConfigLoader:
    _instance = None

    def __new__(cls, *paths: str | Path):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.config = ConfigDict()
            for path in paths:
                cls._instance._merge(load_toml(path))
        return cls._instance

    def _merge(self, other: dict):
        for k, v in other.items():
            if k in self.config and isinstance(self.config[k], dict) and isinstance(v, dict):
                self.config[k].update(v)
            else:
                self.config[k] = v

    def __getattr__(self, item):
        return getattr(self.config, item)

# Instancia global. El config del experimento añade las secciones [experiment], [data],
# [diagnostics], [sigreg] y [loss]. En esta rama el experimento por defecto es ce_vocab
# (v115); se puede cambiar con MSLM_EXPERIMENT_CONFIG=experiments/<familia>/<otro>.toml.
_paths = ["config/model/config.toml", "config/training/train_config.toml"]
_experiment = os.environ.get("MSLM_EXPERIMENT_CONFIG", "experiments/v115_ce_vocab/ce_vocab.toml")
if Path(_experiment).exists():
    _paths.append(_experiment)

cfg = ConfigLoader(*_paths)
