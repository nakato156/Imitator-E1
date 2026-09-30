import numpy as np

from src.mslm.utils.embedding_space import (
    EmbeddingTransform,
    StandardizedEmbeddingTransform,
    fit_embedding_transform,
    fit_standardized_embedding_transform,
    reconstruction_cosine,
)


def _make_low_rank_data(n=2000, dim=2048, rank=512, seed=0):
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(rank, dim))
    coeffs = rng.normal(size=(n, rank))
    data = coeffs @ basis
    data += rng.normal(scale=1e-4, size=data.shape)  # ruido chico, no exactamente rank 512
    return data.astype(np.float32)


def test_full_rank_reconstruction_is_near_perfect():
    data = _make_low_rank_data()
    t = fit_embedding_transform(data, n_components=512, epsilon=1e-5)
    z = t.transform(data)
    assert z.shape == (data.shape[0], 512)
    recon = t.inverse_transform(z)
    cos = reconstruction_cosine(data, recon)
    assert cos.mean() >= 0.99
    assert np.percentile(cos, 5) >= 0.97


def test_transform_is_whitened_on_train_set():
    data = _make_low_rank_data()
    t = fit_embedding_transform(data, n_components=512, epsilon=1e-5)
    z = t.transform(data)
    cov = np.cov(z, rowvar=False)
    # whitening: covarianza de las componentes retenidas ~ identidad
    off_diag = cov - np.diag(np.diag(cov))
    assert np.abs(np.diag(cov) - 1.0).mean() < 0.05
    assert np.abs(off_diag).mean() < 0.05


def test_low_n_components_loses_information_relative_to_full():
    data = _make_low_rank_data(rank=2048)  # full-rank: PCA-512 SÍ debe perder señal
    t_full = fit_embedding_transform(data, n_components=2048, epsilon=1e-5)
    t_512 = fit_embedding_transform(data, n_components=512, epsilon=1e-5)
    cos_full = reconstruction_cosine(data, t_full.inverse_transform(t_full.transform(data))).mean()
    cos_512 = reconstruction_cosine(data, t_512.inverse_transform(t_512.transform(data))).mean()
    assert cos_full > cos_512


def test_reconstruction_cosine_identical_vectors_is_one():
    a = np.random.default_rng(0).normal(size=(10, 16)).astype(np.float32)
    assert np.allclose(reconstruction_cosine(a, a), 1.0, atol=1e-5)


def test_transform_rejects_wrong_dim():
    data = _make_low_rank_data(dim=2048)
    t = fit_embedding_transform(data, n_components=512)
    import pytest

    with pytest.raises(ValueError):
        t.transform(np.zeros((3, 100), dtype=np.float32))


def test_full_standardization_round_trip_is_exact():
    data = np.random.default_rng(4).normal(size=(50, 32)).astype(np.float32)
    t = fit_standardized_embedding_transform(data)
    assert isinstance(t, StandardizedEmbeddingTransform)
    z = t.transform(data)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(t.inverse_transform(z), data, atol=1e-5)


def test_full_standardization_handles_constant_dimensions():
    data = np.ones((10, 4), dtype=np.float32)
    t = fit_standardized_embedding_transform(data)
    z = t.transform(data)
    assert np.isfinite(z).all()
    assert np.allclose(t.inverse_transform(z), data)
