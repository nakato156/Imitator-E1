"""Tests CPU para ContrastiveAligner (v118c: soporte de embeddings sin normalizar
para VICReg, que a diferencia de InfoNCE opera sobre vectores crudos)."""
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Registra paquetes-stub vacíos para que el `from src.mslm.training.loss_contrastive
# import masked_mean` de contrastive.py resuelva contra el módulo ya cargado en vez
# de disparar src.mslm.training.__init__ (que importa Trainer -> dependencia circular
# pesada, mismo problema que en tests/test_loss_contrastive.py).
sys.modules.setdefault("src", types.ModuleType("src"))
sys.modules.setdefault("src.mslm", types.ModuleType("src.mslm"))
sys.modules.setdefault("src.mslm.training", types.ModuleType("src.mslm.training"))
_load("src.mslm.training.loss_contrastive", "src/mslm/training/loss_contrastive.py")

_contrastive = _load("contrastive_aligner_mod", "src/mslm/models/contrastive.py")
ContrastiveAligner = _contrastive.ContrastiveAligner


class _FakeEncoder(nn.Module):
    """Sustituye a PrefixImitator: proyecta keypoints a [B, K, H] con una Linear."""

    def __init__(self, input_dim: int, hidden: int, k: int = 4):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(input_dim, hidden)

    def forward(self, keypoints, frames_padding_mask):
        # keypoints: [B, T, input_dim] -> toma los primeros k frames como "prefix"
        prefix = self.proj(keypoints[:, : self.k])  # [B, k, hidden]
        return prefix, None


def _make_aligner(hidden=8, proj_dim=6, input_dim=5):
    encoder = _FakeEncoder(input_dim=input_dim, hidden=hidden)
    return ContrastiveAligner(encoder, hidden=hidden, proj_dim=proj_dim)


def test_encode_video_default_is_l2_normalized():
    aligner = _make_aligner()
    keypoints = torch.randn(3, 10, 5)
    mask = torch.zeros(3, 10, dtype=torch.bool)
    v = aligner.encode_video(keypoints, mask)
    norms = v.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_encode_video_normalize_false_returns_raw_projection():
    aligner = _make_aligner()
    keypoints = torch.randn(3, 10, 5)
    mask = torch.zeros(3, 10, dtype=torch.bool)
    v_raw = aligner.encode_video(keypoints, mask, normalize=False)
    norms = v_raw.norm(dim=-1)
    # Una proyección cruda (LayerNorm->Linear->GELU->Linear) no cae en la esfera unidad.
    assert not torch.allclose(norms, torch.ones(3), atol=1e-3)


def test_encode_text_default_is_l2_normalized():
    aligner = _make_aligner(hidden=8)
    text_embeds = torch.randn(3, 6, 8)
    text_mask = torch.zeros(3, 6, dtype=torch.bool)
    t = aligner.encode_text(text_embeds, text_mask)
    norms = t.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_encode_text_normalize_false_returns_raw_projection():
    aligner = _make_aligner(hidden=8)
    text_embeds = torch.randn(3, 6, 8)
    text_mask = torch.zeros(3, 6, dtype=torch.bool)
    t_raw = aligner.encode_text(text_embeds, text_mask, normalize=False)
    norms = t_raw.norm(dim=-1)
    assert not torch.allclose(norms, torch.ones(3), atol=1e-3)
