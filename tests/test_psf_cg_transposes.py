"""
tests/test_psf_cg_transposes.py

`PSFResampler.resample` runs the Conjugate Gradient products ``(v @ MT) @ M``
with the transposes of ``M`` and ``MT`` stored as CSR (`_rmatmul`): a CSR x
dense product instead of torch converting ``M`` on every step. Same result as
the plain ``v @ A`` products, for the unconstrained, bounded and conservative
fits, one row and a batch.

CPU-only, following `tests/test_psf.py`.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from healpix_resample import PSFResampler
from healpix_resample.psf import _csr_transpose, _rmatmul

NDATA = 60
LEVEL = 12


@pytest.fixture(scope="module")
def field():
    lon, lat = np.meshgrid(0.3 * np.arange(NDATA) / NDATA, 0.3 * np.arange(NDATA) / NDATA)
    lon, lat = lon.ravel(), lat.ravel()
    val = np.stack([0.5 + 0.3 * np.sin(lon * 40.0 + k) * np.cos(lat * 30.0) for k in range(3)])
    val[1, ::7] = np.nan
    return lon, lat, val


@pytest.fixture(scope="module", params=[torch.float32, torch.float64])
def op(field, request):
    lon, lat, _ = field
    return PSFResampler(lon, lat, level=LEVEL, Npt=9, threshold=0.1, dtype=request.param)


def _plain(op, monkeypatch):
    """The same resampler, with the plain ``v @ A`` products."""
    monkeypatch.setattr(op, "_cg_transposes", lambda: (None, None))


def test_rmatmul_is_the_product(op):
    for A in (op.MT, op.M):
        n = A.shape[0]
        v = torch.rand(3, n, dtype=op.dtype)
        np.testing.assert_allclose(_rmatmul(v, A, _csr_transpose(A)).numpy(), (v @ A).numpy(),
                                   rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kwargs", [
    {"lam": 0.1},
    {"lam": 0.1, "clip": (0.0, 1.0)},
    {"lam": 0.1, "conservative": True},
])
@pytest.mark.parametrize("rows", [slice(0, 1), slice(None)])
def test_resample_same_as_plain_products(op, field, kwargs, rows, monkeypatch):
    _, _, val = field
    v = val[rows] if val[rows].shape[0] > 1 else val[rows][0]
    fast = op.resample(v, max_iter=100, tol=1e-8, **kwargs)
    with monkeypatch.context() as m:
        _plain(op, m)
        ref = op.resample(v, max_iter=100, tol=1e-8, **kwargs)
    tol = 1e-4 if op.dtype == torch.float32 else 1e-10
    np.testing.assert_allclose(np.asarray(fast.cell_data), np.asarray(ref.cell_data), rtol=tol, atol=tol)


def test_transposes_built_once(op, field):
    _, _, val = field
    op.resample(val[0], lam=0.1)
    first = op._cg_transposes()
    op.resample(val[2], lam=0.1)
    assert op._cg_transposes() is first
