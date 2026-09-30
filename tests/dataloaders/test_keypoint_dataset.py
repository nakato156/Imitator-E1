"""Tests para KeypointDataset (v119 — filtro de calidad de datos + augmentation).

Cubre dos cambios sobre `src/mslm/dataloader/keypoint_dataset.py`:
1. Bug de `TransformedSubset` que ignoraba `return_label` real del dataset padre
   (hardcodeado a False en `split_dataset()`), lo que rompería silenciosamente
   las labels del 80% del train set si se activa `data_augmentation=True` junto
   con `return_label=True` (combinación nunca antes ejercitada en el repo).
2. Filtro de calidad de datos opcional (`min_frames`, `filter_invalid_labels`),
   generalizando "Less is More" (PUCP) a la restricción real de `nn.CTCLoss`
   en esta arquitectura: la longitud post-downsampling (T // 4, por los 2
   `TemporalLiftPooling` en cascada) debe alcanzar para el largo del label.
"""
import importlib.util
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# `keypoint_dataset.py` hace `from .data_augmentation import ...` (import relativo):
# precargamos data_augmentation.py bajo su nombre dotted real y stubeamos los
# paquetes padre para que ese import relativo resuelva contra el módulo ya
# cargado, sin disparar src/mslm/__init__ (mismo patrón que test_contrastive_aligner.py).
sys.modules.setdefault("src", types.ModuleType("src"))
sys.modules.setdefault("src.mslm", types.ModuleType("src.mslm"))
sys.modules.setdefault("src.mslm.dataloader", types.ModuleType("src.mslm.dataloader"))
_load("src.mslm.dataloader.data_augmentation", "src/mslm/dataloader/data_augmentation.py")
_load("src.mslm.dataloader.vocab", "src/mslm/dataloader/vocab.py")

_mod = _load("src.mslm.dataloader.keypoint_dataset", "src/mslm/dataloader/keypoint_dataset.py")
KeypointDataset = _mod.KeypointDataset
TransformedSubset = _mod.TransformedSubset


def _write_clip(group, clip_id, n_frames, label, n_keypoints=133, n_channels=2, embed_dim=8):
    rng = np.random.default_rng(0)
    group["keypoints"].create_dataset(clip_id, data=rng.random((n_frames, n_keypoints, n_channels), dtype=np.float32))
    group["embeddings"].create_dataset(clip_id, data=rng.random((4, embed_dim), dtype=np.float32))
    group["labels"].create_dataset(clip_id, data=np.array([label.encode()]))


def _make_hdf5(tmp_path, clips, dataset_name="dataset2", filename="fixture.hdf5"):
    h5_path = tmp_path / filename
    with h5py.File(h5_path, "w") as f:
        g = f.require_group(dataset_name)
        g.require_group("keypoints")
        g.require_group("embeddings")
        g.require_group("labels")
        for clip_id, n_frames, label in clips:
            _write_clip(g, clip_id, n_frames, label)
    return h5_path


# --- 1. Bug de TransformedSubset.return_label ------------------------------

def test_transformed_subset_with_augmentation_preserves_labels(tmp_path):
    clips = [(str(i), 40, f"palabra{i} mundo") for i in range(6)]
    h5_path = _make_hdf5(tmp_path, clips)

    ds = KeypointDataset(h5Path=h5_path, return_label=True, data_augmentation=True,
                          include_datasets=["dataset2"], max_length=4000)
    train_subset, _, _, _ = ds.split_dataset(train_ratio=0.8)

    for i in range(len(train_subset)):
        _, _, label = train_subset[i]
        assert label is not None


# --- 2. Filtro de calidad ----------------------------------------------------

def test_min_frames_excludes_below_threshold_includes_at_threshold(tmp_path):
    clips = [
        ("below", 15, "hola"),
        ("at", 16, "hola"),
    ]
    h5_path = _make_hdf5(tmp_path, clips)

    ds = KeypointDataset(h5Path=h5_path, include_datasets=["dataset2"],
                          max_length=4000, min_frames=16)

    kept_clip_ids = {clip for _, clip in ds.valid_index}
    assert kept_clip_ids == {"at"}


def test_filter_invalid_labels_excludes_empty_label(tmp_path):
    clips = [
        ("empty", 40, "   "),
        ("nonempty", 40, "hola"),
    ]
    h5_path = _make_hdf5(tmp_path, clips)

    ds = KeypointDataset(h5Path=h5_path, include_datasets=["dataset2"],
                          max_length=4000, filter_invalid_labels=True)

    kept_clip_ids = {clip for _, clip in ds.valid_index}
    assert kept_clip_ids == {"nonempty"}


def test_filter_invalid_labels_excludes_when_downsampled_length_below_label_length(tmp_path):
    clips = [
        # frames=8 -> frames//4=2, 3 palabras: 2 < 3, debe excluirse.
        ("too_short", 8, "una dos tres"),
        # frames=8 -> frames//4=2, 2 palabras: 2 >= 2, caso borde, debe incluirse.
        ("border_ok", 8, "una dos"),
    ]
    h5_path = _make_hdf5(tmp_path, clips)

    ds = KeypointDataset(h5Path=h5_path, include_datasets=["dataset2"],
                          max_length=4000, filter_invalid_labels=True)

    kept_clip_ids = {clip for _, clip in ds.valid_index}
    assert kept_clip_ids == {"border_ok"}


def test_filters_disabled_by_default_keeps_legacy_behavior(tmp_path):
    """Sin pasar min_frames/filter_invalid_labels, valid_index no cambia (protege
    a otros call-sites como setup_train.py que no pasan estos kwargs)."""
    clips = [
        ("tiny", 2, ""),
        ("normal", 40, "hola mundo"),
    ]
    h5_path = _make_hdf5(tmp_path, clips)

    ds = KeypointDataset(h5Path=h5_path, include_datasets=["dataset2"], max_length=4000)

    kept_clip_ids = {clip for _, clip in ds.valid_index}
    assert kept_clip_ids == {"tiny", "normal"}
