from pathlib import Path
from os import getcwd


class PathVariables:
    __slots__ = (
        "_base_path",
        "model_path",
        "report_path",
        "logs_path",
        "study_path",
        "data_path",
        "h5_file",
    )
    _instance = None

    def __new__(cls, base_path: str | Path = None, dataset_filename: str = "dataset_v4.hdf5"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init(base_path, dataset_filename)
        return cls._instance

    def _init(self, base_path: str | Path, dataset_filename: str = "dataset_v4.hdf5"):
        bp = Path(base_path) if base_path else Path(getcwd())

        self._base_path = bp

        out = bp / "outputs"
        out.mkdir(parents=True, exist_ok=True)

        self.model_path = out / "checkpoints"
        self.model_path.mkdir(exist_ok=True)

        self.report_path = out / "reports"
        self.report_path.mkdir(exist_ok=True)

        self.logs_path = out / "logs"
        self.logs_path.mkdir(exist_ok=True)

        self.study_path = out / "studies"
        self.study_path.mkdir(exist_ok=True)

        # Datos
        dp = bp.parent / "data"
        self.data_path = dp
        self.h5_file = dp / "processed" / dataset_filename

path_vars = PathVariables(dataset_filename="dataset_v6_unsloth.hdf5")
