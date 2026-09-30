import numpy as np
import random
import torch
import torch.nn.functional as F

def scaling(keypoint):
    # Escalar los keypoints sin cambiar su cantidad
    scale = random.uniform(0.9, 1.1)
    return keypoint * scale

def rotation_2D(keypoint):
    # Aseguramos que la rotación no cambie la cantidad de keypoints
    angle = random.uniform(-15, 15)  # Rotación aleatoria en grados
    angle_rad = torch.tensor(angle * np.pi / 180.0, dtype=torch.float32)  # Convertir a radianes
    rotation_matrix = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad)], 
                                    [torch.sin(angle_rad), torch.cos(angle_rad)]], dtype=torch.float32)
    # El recorte/downsampling/remuestreo de v122 puede producir tensores no
    # contiguos. reshape conserva el comportamiento sin exigir strides contiguos.
    keypoint_rotated = torch.matmul(keypoint.reshape(-1, 2), rotation_matrix)
    return keypoint_rotated.reshape(keypoint.shape)

def length_variance(keypoint, scale=0.8):
    T, J, C = keypoint.shape
    T_new = int(round(T * scale))
    
    orig_times = np.linspace(0, T-1, num=T)
    new_times = np.linspace(0, T-1, num=T_new)

    flat = keypoint.reshape(T, J*C)
    streched = np.stack([
        np.interp(new_times, orig_times, flat[:, d])
        for d in range(J*C)
    ], axis = 1)
    keypoints_streched = streched.reshape(T_new, J, C)
    return keypoints_streched

def gaussian_jitter(keypoint, sigma=0.0014, clip=3.0):
    keypoint_jitter = np.random.normal(loc=0.0, scale=sigma, size=keypoint.shape)

    if clip is not None:
        np.clip(keypoint_jitter, -clip, clip, out=keypoint_jitter)

    return keypoint + keypoint_jitter

# Min et al. (ICCVW 2025) Tabla 5: temporal drop (~15% de frames) reduce WER
# en el escenario de generalización limitada del backend (BiLSTM).
TEMPORAL_DROP_PROB = 0.15

def temporal_drop(keypoint, drop_prob=TEMPORAL_DROP_PROB):
    # Elimina ~drop_prob de los frames al azar, conservando el orden temporal.
    T = keypoint.shape[0]
    n_drop = int(round(T * drop_prob))
    if n_drop <= 0 or T - n_drop < 1:
        return keypoint
    drop_idx = set(random.sample(range(T), n_drop))
    keep_idx = [i for i in range(T) if i not in drop_idx]
    return keypoint[keep_idx]

def apply_augmentation(keypoint, augmentation_type):
    """
    Aplica diferentes augmentaciones de acuerdo al tipo.
    """

    if augmentation_type == "Gaussian_jitter":
        return gaussian_jitter(keypoint)
    elif augmentation_type == "Length_variance":
        return length_variance(keypoint)
    elif augmentation_type == "Rotation_2D":
        return rotation_2D(keypoint)
    elif augmentation_type == "Scaling":
        return scaling(keypoint)
    elif augmentation_type == "Temporal_drop":
        return temporal_drop(keypoint)
    return keypoint

def keypoint_normalization(keypoint):
    keypoint = abs(keypoint)

    mask_keypoints = ~((keypoint[...,0] < 5) & (keypoint[...,1] < 5))
    
    valid_points = keypoint[mask_keypoints].view(-1, 2)
    if valid_points.numel() > 0:
        global_mins, _ = valid_points.min(dim=0)
        global_maxs, _ = valid_points.max(dim=0)
    else:
        global_mins, _ = torch.zeros(2, device=keypoint.device)
        global_maxs, _ = torch.ones(2, device=keypoint.device)
                    
    global_ranges = global_maxs - global_mins
    global_ranges[global_ranges == 0] = 1.0

    normalized = (keypoint - global_mins) / global_ranges

    normalized[~mask_keypoints] = keypoint[~mask_keypoints] 

    return normalized

def filter_unstable_keypoints_to_num(keypoints, keep_n):
    """
    Conserva los 'keep_n' keypoints más estables (con menor varianza temporal).
    """
    if keep_n > keypoints.size(1):
        raise ValueError(f"keep_n ({keep_n}) mayor a cantidad de keypoints ({keypoints.size(1)})")

    T, N, _ = keypoints.shape

    # Calcular varianza temporal por keypoint
    var = keypoints.var(dim=0).mean(dim=1)  # (N,)

    # Obtener los índices de los keypoints más estables
    _, indices = torch.topk(-var, k=keep_n)  # usamos -var para orden ascendente
    stable_mask = torch.zeros(N, dtype=torch.bool)
    stable_mask[indices] = True

    # Aplicar la máscara
    filtered = keypoints.clone()
    for i in range(N):
        if not stable_mask[i]:
            filtered[:, i] = 0

    return filtered, stable_mask

def normalize_augment_data(keypoint, augmentation_type, n_keypoints = 133):
    #Keypoints a Tensor
    if not isinstance(keypoint, torch.Tensor):
        keypoint = torch.as_tensor(keypoint)    

    # Clean noise 
    keypoint, _ = filter_unstable_keypoints_to_num(keypoint, n_keypoints)

    if augmentation_type != "Original":
        keypoint = apply_augmentation(keypoint, augmentation_type)
    
    #Keypoints a Tensor
    if not isinstance(keypoint, torch.Tensor):
        keypoint = torch.as_tensor(keypoint)    
    # Keypoint Normalization
    keypoint = keypoint_normalization(keypoint)

    return keypoint

def remove_keypoints(keypoint):
    pose_keep = torch.tensor([i for i in range(18) if i not in {7, 8, 9, 10, 11 ,12, 13, 14, 15, 16, 17}], dtype=torch.long)

    pose    = torch.tensor(keypoint[:,pose_keep,:])   # 7 pose
    face    = torch.tensor(keypoint[:,28:92,:])       # 64 cara
    left_h  = torch.tensor(keypoint[:,93:113,:])      # 20 mano izq
    right_h = torch.tensor(keypoint[:,113:133,:])     # 20 mano der (era 113:134=21 -> off-by-one)

    keypoints = torch.cat([pose, face, left_h, right_h], dim=1)
    return keypoints


def resample_temporal(keypoint, target_frames):
    """Interpola una secuencia [T,J,2] a una longitud fija."""
    keypoint = torch.as_tensor(keypoint, dtype=torch.float32)
    if keypoint.shape[0] == target_frames:
        return keypoint
    x = keypoint.permute(1, 2, 0).reshape(1, -1, keypoint.shape[0])
    x = F.interpolate(x, size=int(target_frames), mode="linear", align_corners=True)
    return x.reshape(keypoint.shape[1], keypoint.shape[2], target_frames).permute(2, 0, 1)


def trim_active_interval(keypoint, margin=8, smooth_window=5):
    """Recorta al intervalo activo estimado por velocidad de ambas muñecas.

    Espera el layout reducido de 111 puntos: pose(7), cara(64), manos(20+20).
    Si no hay señal temporal útil, devuelve el clip completo.
    """
    x = torch.as_tensor(keypoint, dtype=torch.float32)
    if x.shape[0] < 3:
        return x
    wrist_idx = [71, 91]
    speed = torch.linalg.vector_norm(x[1:, wrist_idx] - x[:-1, wrist_idx], dim=-1).mean(dim=1)
    speed = torch.cat([speed[:1], speed])
    if smooth_window > 1 and speed.numel() >= smooth_window:
        pad = smooth_window // 2
        speed = F.avg_pool1d(
            F.pad(speed[None, None], (pad, pad), mode="replicate"),
            kernel_size=smooth_window,
            stride=1,
        ).flatten()[: x.shape[0]]
    p60 = torch.quantile(speed, 0.60)
    p95 = torch.quantile(speed, 0.95)
    threshold = torch.maximum(p60, 0.15 * p95)
    active = torch.where(speed > threshold)[0]
    if active.numel() == 0 or not torch.isfinite(threshold):
        return x
    start = max(0, int(active[0]) - int(margin))
    end = min(x.shape[0], int(active[-1]) + int(margin) + 1)
    return x[start:end] if end > start else x


def normalize_keypoints_torso(keypoint, eps=1e-6):
    """Centra en hombros y escala por su distancia, preservando movimiento."""
    x = torch.as_tensor(keypoint, dtype=torch.float32).clone()
    left_shoulder, right_shoulder = x[:, 2], x[:, 5]
    center = (left_shoulder + right_shoulder) / 2
    scale = torch.linalg.vector_norm(left_shoulder - right_shoulder, dim=-1)
    valid_scale = scale[torch.isfinite(scale) & (scale > eps)]
    if valid_scale.numel():
        denom = valid_scale.median()
    else:
        span = x.amax(dim=(0, 1)) - x.amin(dim=(0, 1))
        denom = torch.linalg.vector_norm(span).clamp_min(1.0)
    return (x - center[:, None, :]) / denom.clamp_min(eps)


def _rotate_about_origin(keypoint, max_degrees=10.0):
    angle = random.uniform(-max_degrees, max_degrees) * np.pi / 180.0
    c, s = np.cos(angle), np.sin(angle)
    matrix = torch.tensor([[c, -s], [s, c]], dtype=keypoint.dtype, device=keypoint.device)
    return torch.matmul(keypoint, matrix)


def _shear_squeeze(keypoint, magnitude=0.08):
    sx = random.uniform(1.0 - magnitude, 1.0 + magnitude)
    sy = random.uniform(1.0 - magnitude, 1.0 + magnitude)
    shear = random.uniform(-magnitude, magnitude)
    matrix = torch.tensor([[sx, shear], [0.0, sy]], dtype=keypoint.dtype, device=keypoint.device)
    return torch.matmul(keypoint, matrix)


def augment_isolated_keypoints(
    keypoint,
    jitter_sigma=0.01,
    temporal_drop_prob=0.15,
    probabilities=None,
):
    """Compone augmentations independientes y restaura la longitud original."""
    probs = {
        "jitter": 0.5,
        "rotation": 0.3,
        "temporal_rescale": 0.5,
        "temporal_drop": 0.5,
        "shear": 0.3,
    }
    if probabilities:
        probs.update(probabilities)
    x = torch.as_tensor(keypoint, dtype=torch.float32).clone()
    original_frames = x.shape[0]
    if random.random() < probs["jitter"]:
        x = x + torch.randn_like(x) * float(jitter_sigma)
    if random.random() < probs["rotation"]:
        x = _rotate_about_origin(x)
    if random.random() < probs["temporal_rescale"]:
        x = resample_temporal(x, max(2, round(x.shape[0] * random.uniform(0.8, 1.2))))
    if random.random() < probs["temporal_drop"] and x.shape[0] > 2:
        keep = max(2, round(x.shape[0] * (1.0 - temporal_drop_prob)))
        indices = sorted(random.sample(range(x.shape[0]), keep))
        x = x[indices]
    if random.random() < probs["shear"]:
        x = _shear_squeeze(x)
    return resample_temporal(x, original_frames)
