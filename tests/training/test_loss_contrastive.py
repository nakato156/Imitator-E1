"""Tests CPU para la pérdida contrastiva y las métricas de retrieval (v118)."""
import importlib.util
import math
from pathlib import Path

import torch

# Carga directa del módulo para evitar el import circular de src.mslm.training.__init__
# (que arrastra al Trainer). Mismo patrón que tests/test_loss_infonce.py.
_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "loss_contrastive", _ROOT / "src/mslm/training/loss_contrastive.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
masked_mean = _mod.masked_mean
clip_contrastive_loss = _mod.clip_contrastive_loss
retrieval_metrics = _mod.retrieval_metrics
vicreg_loss = _mod.vicreg_loss


def _normed(x):
    return torch.nn.functional.normalize(x, dim=-1)


def test_masked_mean_ignores_padding():
    x = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])  # [1, 3, 2]
    mask = torch.tensor([[False, False, True]])                  # 3ª posición = padding
    out = masked_mean(x, mask)
    assert torch.allclose(out, torch.tensor([[2.0, 2.0]]))


def test_perfect_alignment_low_loss_and_full_recall():
    torch.manual_seed(0)
    B, d = 16, 32
    v = _normed(torch.randn(B, d))
    t = v.clone()  # alineación perfecta: vídeo i idéntico a texto i
    logit_scale = torch.tensor(math.log(1 / 0.07))
    loss = clip_contrastive_loss(v, t, logit_scale)
    m = retrieval_metrics(v, t)
    assert loss.item() < 0.01
    assert m["R@1"] == 1.0


def test_shuffled_alignment_chance_recall():
    torch.manual_seed(1)
    B, d = 64, 32
    v = _normed(torch.randn(B, d))
    t = _normed(torch.randn(B, d))  # sin relación vídeo-texto
    m = retrieval_metrics(v, t)
    # R@1 cercano al azar (1/B); con holgura por ruido finito
    assert m["R@1"] < 5 * m["chance"] + 0.02


def test_loss_is_differentiable_wrt_inputs():
    B, d = 8, 16
    v = _normed(torch.randn(B, d, requires_grad=True))
    t = _normed(torch.randn(B, d, requires_grad=True))
    logit_scale = torch.tensor(math.log(1 / 0.07), requires_grad=True)
    loss = clip_contrastive_loss(v, t, logit_scale)
    loss.backward()
    assert logit_scale.grad is not None and torch.isfinite(loss)


def test_collapse_gives_high_loss():
    # Todos los vídeos colapsados al mismo vector -> loss ~ log(B), no un mínimo.
    B, d = 32, 16
    v = _normed(torch.ones(B, d))           # idénticos = colapso
    t = _normed(torch.randn(B, d))
    logit_scale = torch.tensor(0.0)         # exp(0)=1
    loss = clip_contrastive_loss(v, t, logit_scale)
    assert loss.item() > math.log(B) * 0.5  # claramente alto


def test_retrieval_symmetric_and_bounded():
    torch.manual_seed(2)
    B, d = 20, 8
    v = _normed(torch.randn(B, d))
    t = _normed(torch.randn(B, d))
    m = retrieval_metrics(v, t)
    assert 0.0 <= m["R@1"] <= m["R@5"] <= m["R@10"] <= 1.0
    assert m["median_rank"] >= 1.0


def test_vicreg_differentiable_wrt_inputs():
    B, d = 8, 16
    v = torch.randn(B, d, requires_grad=True)
    t = torch.randn(B, d, requires_grad=True)
    loss = vicreg_loss(v, t)
    loss.backward()
    assert v.grad is not None and torch.isfinite(loss)


def test_vicreg_aligned_pairs_lower_loss_than_misaligned():
    # Mismas estadísticas marginales (var/cov) en ambos casos: solo cambia si
    # texto_i corresponde a vídeo_i (alineado) o a una permutación (desalineado).
    torch.manual_seed(3)
    B, d = 32, 16
    v = torch.randn(B, d)
    aligned_t = v + 0.01 * torch.randn(B, d)
    misaligned_t = v[torch.randperm(B)] + 0.01 * torch.randn(B, d)
    loss_aligned = vicreg_loss(v, aligned_t)
    loss_misaligned = vicreg_loss(v, misaligned_t)
    assert loss_aligned.item() < loss_misaligned.item()


def test_vicreg_collapse_gives_high_variance_penalty():
    # Vídeos colapsados al mismo vector -> std=0 en cada dimensión -> penaliza
    # el término de varianza aunque la invariancia con el texto sea baja.
    B, d = 32, 16
    v_collapsed = torch.ones(B, d) + 1e-8 * torch.randn(B, d)
    v_healthy = torch.randn(B, d)
    t = torch.randn(B, d)
    loss_collapsed = vicreg_loss(v_collapsed, t)
    loss_healthy = vicreg_loss(v_healthy, t)
    assert loss_collapsed.item() > loss_healthy.item()


def test_vicreg_loss_is_nonnegative():
    torch.manual_seed(4)
    B, d = 16, 8
    v = torch.randn(B, d)
    t = torch.randn(B, d)
    assert vicreg_loss(v, t).item() >= 0.0
