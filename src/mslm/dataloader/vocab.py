"""Vocabulario word-level para CTC (v119), construido SOLO desde labels de train.

A diferencia del BPE de Gemma (~256k tokens, demasiado esparso para 481 clips de
train y desalineado con WER, que la literatura mide a nivel palabra), este
vocabulario es chico y específico de la tarea: blank=0 (convención CTC) y
unk=1 para palabras de val no vistas en train (evita fuga val->vocab).
"""
import json
import re
from pathlib import Path

import h5py

BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"

_PUNCT_RE = re.compile(r"[^\w\sáéíóúñü]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Normalización pública: lowercase + strip de puntuación + split por espacios.

    Compartida entre Vocab.encode/build_from_labels y el cálculo de WER en el
    training loop (el ground-truth de WER debe normalizarse igual que el vocab,
    pero SIN pasar por encode/decode -- mapear a <unk> falsearía la métrica para
    palabras de val no vistas en train).
    """
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    return text.split()


class Vocab:
    def __init__(self, token_to_id: dict[str, int], id_to_token: dict[int, str]):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token

    @property
    def blank_id(self) -> int:
        return self.token_to_id[BLANK_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_id)

    @classmethod
    def build_from_labels(cls, labels: list[str]) -> "Vocab":
        words: set[str] = set()
        for label in labels:
            words.update(tokenize(label))

        token_to_id = {BLANK_TOKEN: 0, UNK_TOKEN: 1}
        for word in sorted(words):
            token_to_id[word] = len(token_to_id)
        id_to_token = {i: t for t, i in token_to_id.items()}
        return cls(token_to_id, id_to_token)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(w, self.unk_id) for w in tokenize(text)]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[i] for i in ids]

    def save(self, path: str | Path) -> None:
        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with open(path) as f:
            data = json.load(f)
        token_to_id = data["token_to_id"]
        id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        return cls(token_to_id, id_to_token)


def collect_labels(h5_path: str | Path, dataset_name: str, clip_ids: list[str]) -> list[str]:
    """Lee labels crudas desde el HDF5 sin pasar por KeypointDataset.__getitem__
    (evita cargar keypoints/embeddings, que no hacen falta para construir el vocab)."""
    with h5py.File(h5_path, "r") as f:
        g = f[dataset_name]["labels"]
        return [g[cid][:][0].decode() for cid in clip_ids]
