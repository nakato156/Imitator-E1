"""Augmentaciones de keypoints para pre-entrenamiento contrastivo (v118b+).

Se aplican muestra-a-muestra durante el train loop, nunca en val.
Las entradas son tensores individuales ya trasladados a GPU.
"""
import random
import torch
import torch.nn.functional as F


def temporal_crop(
    kp: torch.Tensor,
    mask: torch.Tensor,
    min_ratio: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recorta una ventana contigua aleatoria de [min_ratio, 1.0] de los frames válidos.

    kp   : [1, T, K, D]
    mask : [1, T]   (True = padding)
    """
    valid_len = int((~mask[0]).sum().item())
    if valid_len < 2:
        return kp, mask

    crop_len = max(1, int(valid_len * random.uniform(min_ratio, 1.0)))
    start = random.randint(0, valid_len - crop_len)

    new_kp = torch.zeros_like(kp)
    new_mask = torch.ones_like(mask)          # todo padding por defecto
    new_kp[0, :crop_len] = kp[0, start : start + crop_len]
    new_mask[0, :crop_len] = False
    return new_kp, new_mask


def coord_noise(
    kp: torch.Tensor,
    mask: torch.Tensor,
    std: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Agrega ruido Gaussiano a las coordenadas de keypoints válidos.

    kp   : [1, T, K, D]
    mask : [1, T]   (True = padding)
    """
    valid = (~mask[0]).float().view(1, -1, 1, 1)   # [1, T, 1, 1]
    noise = torch.randn_like(kp) * std * valid
    return kp + noise, mask


def make_augment_fn(
    temporal_crop_min: float = 0.7,
    coord_noise_std: float = 0.02,
) -> "Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]":
    """Devuelve una función que compone las augmentaciones activas."""

    def augment(kp: torch.Tensor, mask: torch.Tensor):
        if temporal_crop_min < 1.0:
            kp, mask = temporal_crop(kp, mask, min_ratio=temporal_crop_min)
        if coord_noise_std > 0.0:
            kp, mask = coord_noise(kp, mask, std=coord_noise_std)
        return kp, mask

    return augment
