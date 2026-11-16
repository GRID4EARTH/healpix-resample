"""PSFResampler(kernel=...): a user-supplied raw link weight.

The default Gaussian ``exp(-2 d^2 / sigma_m^2)`` can be replaced by any
non-negative weight of the sample and cell positions (an anisotropic
response, a Gaussian integrated over a square detector aperture, ...). The
operators are then built by the same dual normalization, so the relations
the solvers rely on must still hold:

- M (N, K) is normalized per cell: its columns sum to 1;
- MT (K, N) is normalized per sample: its columns sum to 1;
- MT = Dx @ M.T @ Dy^-1, with Dx = ``cell_weight``.
"""

import math

import numpy as np
import pytest
import torch

import healpix_resample as hr
from healpix_resample.knn import _arc_distance, _lonlat_to_xyz

LEVEL = 19
FWHM_M = 25.0
LAT0, LON0 = 48.85, 2.35
M_PER_DEG_LAT = 111_000.0
M_PER_DEG_LON = 111_000.0 * math.cos(math.radians(LAT0))


def _samples(n=24, step_m=20.0):
    off = (np.arange(n) - (n - 1) / 2) * step_m
    xx, yy = np.meshgrid(off, off)
    return LON0 + xx.ravel() / M_PER_DEG_LON, LAT0 + yy.ravel() / M_PER_DEG_LAT


def _build(lon, lat, dtype, kernel=None):
    scale = hr.fwhm_to_scale(FWHM_M)
    npt = int(hr.recommend_npt(scale, LEVEL)["npt"])
    return hr.PSFResampler(lon, lat, LEVEL, sigma_m=scale, Npt=npt, dtype=dtype,
                           kernel=kernel, verbose=False)


def _gaussian_kernel(sigma_m, radius=6371000.0):
    """The default weight, recomputed from the coordinates the kernel gets."""
    def kernel(lon_s, lat_s, lon_c, lat_c):
        xyz_s = _lonlat_to_xyz(torch.deg2rad(lon_s), torch.deg2rad(lat_s))
        xyz_c = _lonlat_to_xyz(torch.deg2rad(lon_c), torch.deg2rad(lat_c))
        d = _arc_distance(xyz_c, xyz_s, radius)
        return torch.exp((-2.0) * (d * d) / (sigma_m * sigma_m))
    return kernel


def _anisotropic_kernel(sx_m, sy_m):
    """Gaussian elongated east-west, on the local tangent plane."""
    def kernel(lon_s, lat_s, lon_c, lat_c):
        dx = (lon_c - lon_s) * M_PER_DEG_LON
        dy = (lat_c - lat_s) * M_PER_DEG_LAT
        return torch.exp(-2.0 * (dx * dx / (sx_m * sx_m) + dy * dy / (sy_m * sy_m)))
    return kernel


def _dense(rs):
    return rs.M.to_dense().double(), rs.MT.to_dense().double()


@pytest.mark.parametrize("dtype, rtol", [(torch.float64, 1e-10), (torch.float32, 1e-5)])
def test_gaussian_kernel_reproduces_default(dtype, rtol):
    lon, lat = _samples()
    ref = _build(lon, lat, dtype)
    alt = _build(lon, lat, dtype, kernel=_gaussian_kernel(ref.sigma_m))
    assert torch.equal(ref.cell_ids, alt.cell_ids)
    M0, MT0 = _dense(ref)
    M1, MT1 = _dense(alt)
    torch.testing.assert_close(M1, M0, rtol=rtol, atol=rtol * 1e-2)
    torch.testing.assert_close(MT1, MT0, rtol=rtol, atol=rtol * 1e-2)
    torch.testing.assert_close(alt.cell_weight.double(), ref.cell_weight.double(),
                               rtol=rtol, atol=0.0)
    val = np.sin(lon * 50.0) + np.cos(lat * 70.0)
    out0 = ref.resample(val, lam=1e-2, max_iter=14).cell_data
    out1 = alt.resample(val, lam=1e-2, max_iter=14).cell_data
    torch.testing.assert_close(torch.as_tensor(out1).double(), torch.as_tensor(out0).double(),
                               rtol=100 * rtol, atol=100 * rtol)


def test_custom_kernel_keeps_dual_normalization():
    lon, lat = _samples()
    ref = _build(lon, lat, torch.float64)
    rs = _build(lon, lat, torch.float64, kernel=_anisotropic_kernel(18.0, 9.0))
    M, MT = _dense(rs)
    M0, _ = _dense(ref)
    assert (M >= 0).all() and (MT >= 0).all()
    assert not torch.allclose(M, M0), "the kernel did not change the operator"

    linked = M.sum(dim=0) > 0
    torch.testing.assert_close(M.sum(dim=0)[linked], torch.ones(int(linked.sum()), dtype=M.dtype))
    torch.testing.assert_close(MT.sum(dim=0), torch.ones(MT.shape[1], dtype=MT.dtype))

    # MT = Dx M^T Dy^-1: for each sample i, MT[k, i] / (Dx[k] M[i, k]) is the
    # same (1 / Dy[i]) over all its cells k.
    dx = rs.cell_weight.double()
    rhs = dx[:, None] * M.T
    nz = rhs > 0
    assert torch.equal(nz, MT > 0)
    ratio = torch.where(nz, MT / torch.where(nz, rhs, torch.ones_like(rhs)), torch.nan)
    mean = torch.nanmean(ratio, dim=0, keepdim=True)
    rel = torch.abs(ratio / mean - 1.0)
    assert float(rel[nz].max()) < 1e-12


def test_kernel_with_wrong_shape_is_rejected():
    lon, lat = _samples(n=8)

    def bad(lon_s, lat_s, lon_c, lat_c):
        return torch.ones(lon_s.numel() + 1, dtype=torch.float64)

    with pytest.raises(ValueError, match="kernel returned shape"):
        _build(lon, lat, torch.float64, kernel=bad)
