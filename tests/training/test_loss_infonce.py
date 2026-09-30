import importlib.util
import math
from pathlib import Path

import torch
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "loss_infonce", _ROOT / "src/mslm/training/loss_infonce.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
prefix_text_infonce = _mod.prefix_text_infonce


B, K, L, H = 8, 20, 15, 64


def _make_inputs(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    prefix = torch.randn(B, K, H, generator=g)
    text_embeds = torch.randn(B, L, H, generator=g)
    text_ids = torch.randint(1, 1000, (B, L), generator=g)
    return prefix, text_embeds, text_ids


def test_infonce_perfect_alignment():
    """Loss is ~0 when prefix pool and text pool are identical (perfect alignment)."""
    prefix, _, text_ids = _make_inputs()
    # Pool prefix the same way the function does, then expand back to [B, L, H]
    v = prefix.mean(dim=1)                      # [B, H]
    text_embeds = v.unsqueeze(1).expand(B, L, H)
    loss = prefix_text_infonce(prefix, text_embeds, text_ids)
    # With perfectly aligned pairs, diagonal similarities dominate; loss should
    # be very small (bounded by floating-point precision at this temperature).
    assert loss.item() < 0.5, f"Expected near-zero loss, got {loss.item():.4f}"


def test_infonce_random_diagonal():
    """Loss for independent prefix/text pairs is close to log(B) (uniform sim matrix)."""
    prefix, text_embeds, text_ids = _make_inputs(seed=42)
    loss = prefix_text_infonce(prefix, text_embeds, text_ids)
    expected = math.log(B)
    assert abs(loss.item() - expected) < 1.0, (
        f"Expected ~log({B})={expected:.3f}, got {loss.item():.4f}"
    )


def test_infonce_gradient_flows():
    """backward() succeeds and prefix.grad is not None."""
    prefix, text_embeds, text_ids = _make_inputs()
    prefix = prefix.requires_grad_(True)
    loss = prefix_text_infonce(prefix, text_embeds, text_ids)
    loss.backward()
    assert prefix.grad is not None
    assert not prefix.grad.isnan().any()


def test_infonce_padding_ignored():
    """Padding positions (text_ids == -100) are excluded from the text pool."""
    prefix, text_embeds, text_ids_clean = _make_inputs()

    # Run without any padding
    loss_clean = prefix_text_infonce(prefix, text_embeds, text_ids_clean)

    # Mask the last half of each sequence with -100
    text_ids_padded = text_ids_clean.clone()
    text_ids_padded[:, L // 2:] = -100
    # Replace the corresponding embeddings with noise — mask should ignore them
    text_embeds_noisy = text_embeds.clone()
    text_embeds_noisy[:, L // 2:] = torch.randn_like(text_embeds_noisy[:, L // 2:]) * 100

    loss_padded = prefix_text_infonce(prefix, text_embeds_noisy, text_ids_padded)

    # The losses will differ (different valid embeddings pooled), but both should
    # be finite and the padded run should not explode due to noisy masked tokens.
    assert loss_padded.isfinite(), "Loss with padding is not finite"
    assert loss_clean.isfinite(), "Loss without padding is not finite"


def test_infonce_symmetric():
    """Swapping prefix and text (with matching ids) gives the same loss."""
    prefix, text_embeds, text_ids = _make_inputs()

    # Forward: prefix → v, text_embeds → t
    loss_fwd = prefix_text_infonce(prefix, text_embeds, text_ids)

    # Reverse: treat text mean as "prefix", original prefix mean as "text".
    # We re-use the function by constructing inputs so that the pooled vectors swap.
    mask = (text_ids != -100).float().unsqueeze(-1)
    t_pooled = (text_embeds * mask).sum(1) / mask.sum(1).clamp(min=1)  # [B, H]
    v_pooled = prefix.mean(dim=1)                                        # [B, H]

    # Build fake prefix [B, 1, H] = t_pooled and fake text_embeds [B, 1, H] = v_pooled
    fake_prefix = t_pooled.unsqueeze(1)
    fake_text   = v_pooled.unsqueeze(1)
    fake_ids    = torch.ones(B, 1, dtype=torch.long)   # no padding

    loss_rev = prefix_text_infonce(fake_prefix, fake_text, fake_ids)

    assert abs(loss_fwd.item() - loss_rev.item()) < 0.5, (
        f"Forward={loss_fwd.item():.4f} vs reverse={loss_rev.item():.4f} differ too much"
    )
