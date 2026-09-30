"""Tests para Vocab y collect_labels (v119 — CTC sobre secuencia).

Vocab se construye SOLO con las labels de train (sin fuga val->vocab).
blank=0 (convención CTC), <unk>=1 para palabras de val no vistas en train.
"""
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

# Carga directa del archivo: vocab.py no tiene imports relativos, pero un import
# dotted (`from src.mslm.dataloader.vocab import ...`) dispararía src/mslm/__init__
# y quedaría expuesto a la polución global de sys.modules que test_contrastive_aligner.py
# (y otros tests con el mismo patrón) hacen sobre "src"/"src.mslm" para evitar el
# import circular real de models/__init__ <-> utils/__init__. Mismo workaround que
# ya usa el resto de la suite (test_loss_infonce.py, etc.).
_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("vocab", _ROOT / "src/mslm/dataloader/vocab.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Vocab = _mod.Vocab
collect_labels = _mod.collect_labels


def test_build_from_labels_reserves_blank_and_unk():
    vocab = Vocab.build_from_labels(["hola mundo"])
    assert vocab.blank_id == 0
    assert vocab.unk_id == 1
    assert vocab.id_to_token[0] == "<blank>"
    assert vocab.id_to_token[1] == "<unk>"


def test_build_from_labels_includes_train_words():
    vocab = Vocab.build_from_labels(["hola mundo", "hola amigos"])
    assert "hola" in vocab.token_to_id
    assert "mundo" in vocab.token_to_id
    assert "amigos" in vocab.token_to_id
    # blank + unk + 3 palabras únicas
    assert len(vocab) == 5


def test_encode_known_words():
    vocab = Vocab.build_from_labels(["hola mundo"])
    ids = vocab.encode("hola mundo")
    assert ids == [vocab.token_to_id["hola"], vocab.token_to_id["mundo"]]


def test_encode_unknown_word_maps_to_unk():
    vocab = Vocab.build_from_labels(["hola mundo"])
    ids = vocab.encode("hola marciano")
    assert ids[0] == vocab.token_to_id["hola"]
    assert ids[1] == vocab.unk_id


def test_encode_lowercases_and_strips_punctuation():
    vocab = Vocab.build_from_labels(["Hola, mundo!"])
    ids = vocab.encode("¿Hola? ¡mundo!")
    assert ids == [vocab.token_to_id["hola"], vocab.token_to_id["mundo"]]


def test_decode_roundtrip():
    vocab = Vocab.build_from_labels(["hola mundo amigo"])
    ids = vocab.encode("hola mundo")
    assert vocab.decode(ids) == ["hola", "mundo"]


def test_save_load_roundtrip(tmp_path):
    vocab = Vocab.build_from_labels(["hola mundo", "buenas tardes"])
    path = tmp_path / "vocab.json"
    vocab.save(path)

    loaded = Vocab.load(path)
    assert loaded.token_to_id == vocab.token_to_id
    assert loaded.id_to_token == vocab.id_to_token
    assert loaded.blank_id == vocab.blank_id
    assert loaded.unk_id == vocab.unk_id


def test_save_writes_valid_json(tmp_path):
    vocab = Vocab.build_from_labels(["hola mundo"])
    path = tmp_path / "vocab.json"
    vocab.save(path)
    with open(path) as f:
        data = json.load(f)
    assert "token_to_id" in data


def test_collect_labels_reads_from_hdf5(tmp_path):
    h5_path = tmp_path / "fixture.hdf5"
    with h5py.File(h5_path, "w") as f:
        g = f.require_group("dataset2").require_group("labels")
        g.create_dataset("0", data=np.array([b"hola mundo"]))
        g.create_dataset("5", data=np.array([b"buenas tardes"]))

    labels = collect_labels(h5_path, "dataset2", ["0", "5"])
    assert labels == ["hola mundo", "buenas tardes"]
