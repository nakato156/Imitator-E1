"""Tratamiento de anisotropía del espacio de embeddings de Gemma (v125).

PCA + whitening ajustado SOLO con embeddings de tokens vistos en train
(nunca con val/test, para no filtrar información de la partición). La
transformación inversa es fija (misma matriz que el forward) y se usa para
reconstruir el vector de 2048d antes de pasarlo a Gemma.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class EmbeddingTransform:
    mean: np.ndarray          # [dim]
    components: np.ndarray    # [n_components, dim], filas ortonormales (V^T de SVD)
    singular_values: np.ndarray  # [n_components]
    n_samples: int
    epsilon: float

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def n_components(self) -> int:
        return self.components.shape[0]

    def _scale(self) -> np.ndarray:
        # desviación estándar de cada componente principal sobre el train set
        return np.sqrt(self.singular_values**2 / max(self.n_samples - 1, 1) + self.epsilon)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] != self.dim:
            raise ValueError(f"esperaba dim={self.dim}, recibido {x.shape[-1]}")
        centered = x - self.mean
        projected = centered @ self.components.T          # [N, n_components]
        return projected / self._scale()

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if z.shape[-1] != self.n_components:
            raise ValueError(f"esperaba n_components={self.n_components}, recibido {z.shape[-1]}")
        unwhitened = z * self._scale()
        return unwhitened @ self.components + self.mean


@dataclass
class StandardizedEmbeddingTransform:
    """Transformación full-rank usada cuando PCA-512 no supera el gate."""

    mean: np.ndarray
    scale: np.ndarray
    epsilon: float

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def n_components(self) -> int:
        return self.dim

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] != self.dim:
            raise ValueError(f"esperaba dim={self.dim}, recibido {x.shape[-1]}")
        return (x - self.mean) / self.scale

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if z.shape[-1] != self.dim:
            raise ValueError(f"esperaba dim={self.dim}, recibido {z.shape[-1]}")
        return z * self.scale + self.mean


def fit_embedding_transform(
    train_embeddings: np.ndarray, n_components: int = 512, epsilon: float = 1e-5
) -> EmbeddingTransform:
    mean = train_embeddings.mean(axis=0)
    centered = train_embeddings - mean
    # SVD económico: centered = U @ diag(S) @ Vt, componentes = filas de Vt
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    return EmbeddingTransform(
        mean=mean.astype(np.float32),
        components=vt[:n_components].astype(np.float32),
        singular_values=s[:n_components].astype(np.float32),
        n_samples=train_embeddings.shape[0],
        epsilon=epsilon,
    )


def fit_standardized_embedding_transform(
    train_embeddings: np.ndarray, epsilon: float = 1e-5
) -> StandardizedEmbeddingTransform:
    mean = train_embeddings.mean(axis=0).astype(np.float32)
    variance = train_embeddings.var(axis=0).astype(np.float32)
    scale = np.sqrt(variance + epsilon).astype(np.float32)
    return StandardizedEmbeddingTransform(mean=mean, scale=scale, epsilon=epsilon)


def reconstruction_cosine(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    num = (original * reconstructed).sum(axis=-1)
    denom = np.linalg.norm(original, axis=-1) * np.linalg.norm(reconstructed, axis=-1)
    return num / np.clip(denom, 1e-12, None)
