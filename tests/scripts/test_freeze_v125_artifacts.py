import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audits" / "freeze_v125_artifacts.py"
SPEC = importlib.util.spec_from_file_location("freeze_v125_artifacts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_sha256_of(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"v125")
    assert MODULE.sha256_of(path) == hashlib.sha256(b"v125").hexdigest()


def test_build_record_is_strict_and_captures_gate_decisions(tmp_path, monkeypatch):
    artifacts = {
        "anisotropy_gate": "anisotropy.json",
        "gemma_oracle": "oracle.json",
        "other": "other.bin",
    }
    monkeypatch.setattr(MODULE, "ARTIFACTS", artifacts)
    (tmp_path / "anisotropy.json").write_text(
        json.dumps(
            {
                "selected_mode": "full_standardized",
                "gate_cosine_mean_ge_0.99": False,
                "gate_cosine_p5_ge_0.97": False,
            }
        )
    )
    (tmp_path / "oracle.json").write_text(
        json.dumps({"summary": {"all_gates_pass": False}})
    )
    (tmp_path / "other.bin").write_bytes(b"x")

    record = MODULE.build_record(tmp_path)
    assert record["selected_embedding_mode"] == "full_standardized"
    assert record["gates"]["gemma_oracle"] is False
    assert record["artifacts"]["other"]["bytes"] == 1

    (tmp_path / "other.bin").unlink()
    with pytest.raises(FileNotFoundError):
        MODULE.build_record(tmp_path)
