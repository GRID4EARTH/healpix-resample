"""Conservation-property tests for OverlapConservativeResampler.

Each test corresponds to a mathematical property of first-order
overlap-area conservative remapping (see the module docstring of
healpix_resample/overlap_conservative.py). The construction is exact on
the sphere, so the tolerances below are floating-point tolerances, not
approximation budgets.
"""
from __future__ import annotations

import numpy as np
import pytest

from healpix_resample.overlap_conservative import (
    OverlapConservativeResampler,
    _project,
    _unproject,
)

FOUR_PI = 4.0 * np.pi


def _global_grid(dlon=10.0, dlat=10.0):
    return np.arange(0.0, 360.0 + 1e-9, dlon), np.arange(-90.0, 90.0 + 1e-9, dlat)


# ---------------------------------------------------------------- geometry
def test_projection_roundtrip():
    rng = np.random.default_rng(0)
    lon = rng.uniform(0, 2 * np.pi, 50_000)
    z = rng.uniform(-1, 1, 50_000)
    x, y = _project(lon, z)
    lon2, z2, ok = _unproject(x, y)
    assert ok.all()
    np.testing.assert_allclose(lon2, lon, atol=5e-14)
    np.testing.assert_allclose(z2, z, atol=5e-15)


@pytest.mark.parametrize("level", [0, 1, 3])
def test_cell_centres_on_lattice(level):
    """Every HEALPix cell centre must sit on the (p, q) * s lattice with
    parity (nside + 1) mod 2 — the structural fact candidate search relies on."""
    healpix_geo = pytest.importorskip("healpix_geo")
    n = 2 ** level
    s = np.pi / (4 * n)
    ids = np.arange(12 * n * n, dtype=np.uint64)
    lonc, latc = healpix_geo.nested.healpix_to_lonlat(ids, level, ellipsoid="sphere")
    x, y = _project(np.radians(lonc), np.sin(np.radians(latc)))
    p = (x + y) / s
    q = (x - y) / s
    assert np.abs(p - np.round(p)).max() < 1e-10
    assert np.abs(q - np.round(q)).max() < 1e-10
    parity = (n + 1) % 2
    assert (np.round(p).astype(int) % 2 == parity).all()
    assert (np.round(q).astype(int) % 2 == parity).all()


def test_total_overlap_area_is_sphere():
    """Complete coverage: overlaps tile the sphere exactly (sum = 4 pi and
    every cell fully covered)."""
    lonb, latb = _global_grid()
    r = OverlapConservativeResampler(lonb, latb, level=3)
    assert r.cell_ids.size == 12 * 4 ** 3
    assert abs(r.covered_area.sum() / FOUR_PI - 1) < 1e-12
    np.testing.assert_allclose(r.covered_area / r.target_area, 1.0, atol=1e-11)


# --------------------------------------------------------- A: conservation
def test_A_global_conservation_intensive():
    lonb, latb = _global_grid()
    r = OverlapConservativeResampler(lonb, latb, level=4)
    rng = np.random.default_rng(1)
    y = rng.uniform(-3, 7, (latb.size - 1, lonb.size - 1))
    x = r.resample(y).cell_data
    i_src = float(np.sum(r.source_area * y.ravel()))
    i_dst = float(np.sum(r.target_area * x))
    assert abs(i_dst - i_src) / abs(i_src) < 1e-13


# ------------------------------------------------- B: constant preservation
@pytest.mark.parametrize("normalization", ["destination", "covered"])
def test_B_constant_field(normalization):
    lonb, latb = _global_grid()
    r = OverlapConservativeResampler(lonb, latb, level=3,
                                     normalization=normalization)
    x = r.resample(np.full((latb.size - 1, lonb.size - 1), 2.5)).cell_data
    np.testing.assert_allclose(x, 2.5, atol=1e-11)


# ------------------------------------------- C: one source cell is split
def test_C_source_cell_split_by_overlap():
    """A single coarse source cell overlapping many HEALPix cells must be
    distributed by area, not assigned to one cell (the hard-binning
    behaviour this class exists to complement)."""
    lonb = np.array([10.0, 40.0])
    latb = np.array([20.0, 50.0])
    r = OverlapConservativeResampler(lonb, latb, level=4,
                                     normalization="covered")
    assert r.cell_ids.size > 10  # genuinely split
    x = r.resample(np.array([[3.0]]), quantity="extensive").cell_data
    # extensive: total redistributed exactly, each share proportional to overlap
    assert abs(x.sum() - 3.0) < 1e-12
    frac = np.asarray(r.overlap.sum(axis=1)).ravel() / r.source_area[0]
    np.testing.assert_allclose(x, 3.0 * frac, rtol=1e-12)


# ------------------------------------------------------- D: non-negativity
def test_D_nonnegativity():
    lonb, latb = _global_grid(5, 5)
    r = OverlapConservativeResampler(lonb, latb, level=4)
    rng = np.random.default_rng(2)
    y = rng.uniform(0, 1, (latb.size - 1, lonb.size - 1))
    assert r.resample(y).cell_data.min() >= -1e-14
    assert r.weights.data.min() >= 0.0


# ----------------------------------------------------------- E: boundedness
def test_E_bounded_by_extrema_covered_norm():
    lonb, latb = _global_grid(15, 10)
    r = OverlapConservativeResampler(lonb, latb, level=3,
                                     normalization="covered")
    rng = np.random.default_rng(3)
    y = rng.uniform(2, 5, (latb.size - 1, lonb.size - 1))
    x = r.resample(y).cell_data
    assert x.min() >= y.min() - 1e-12
    assert x.max() <= y.max() + 1e-12


# ------------------------------------------------------ F: partial coverage
def test_F_partial_coverage_normalizations_differ():
    """A regional grid: partially covered boundary cells must read the
    constant field as < C under destination normalization and exactly C
    under covered normalization."""
    lonb = np.arange(10.0, 51.0, 5.0)
    latb = np.arange(-20.0, 21.0, 5.0)
    y = np.ones((latb.size - 1, lonb.size - 1))
    r_dst = OverlapConservativeResampler(lonb, latb, level=4,
                                         normalization="destination")
    r_cov = OverlapConservativeResampler(lonb, latb, level=4,
                                         normalization="covered")
    frac = r_dst.covered_area / r_dst.target_area
    assert frac.min() < 0.99 and frac.max() > 0.999  # both kinds present
    x_dst = r_dst.resample(y).cell_data
    x_cov = r_cov.resample(y).cell_data
    np.testing.assert_allclose(x_dst, frac, rtol=1e-10)   # = covered fraction
    np.testing.assert_allclose(x_cov, 1.0, atol=1e-11)    # constant preserved
    # destination norm still conserves the integral over the region
    i_src = float(np.sum(r_dst.source_area))
    i_dst = float(np.sum(r_dst.target_area * x_dst))
    assert abs(i_dst - i_src) / i_src < 1e-12


# ------------------------------------------------------- G: longitude wrap
def test_G_antimeridian():
    lonb = np.arange(120.0, 241.0, 10.0)  # crosses 180
    latb = np.arange(-30.0, 31.0, 10.0)
    r = OverlapConservativeResampler(lonb, latb, level=4)
    i_src = float(np.sum(r.source_area))
    i_dst = float(np.sum(r.target_area * r.resample(
        np.ones((latb.size - 1, lonb.size - 1))).cell_data))
    assert abs(i_dst - i_src) / i_src < 1e-12
    # same for a grid expressed in [-180, 180] crossing the antimeridian
    lonb2 = np.arange(-60.0, 61.0, 10.0) + 180.0  # 120..300 equivalent
    r2 = OverlapConservativeResampler(lonb2 - 360.0, latb, level=4)
    assert abs(np.sum(r2.covered_area) - i_src) / i_src < 1e-12


# --------------------------------------------------------------- H: poles
def test_H_polar_cells():
    lonb = np.arange(0.0, 361.0, 30.0)
    latb = np.array([80.0, 85.0, 90.0])  # touches the pole
    r = OverlapConservativeResampler(lonb, latb, level=4,
                                     normalization="covered")
    x = r.resample(np.ones((2, 12))).cell_data
    np.testing.assert_allclose(x, 1.0, atol=1e-11)
    cap_area = 2 * np.pi * (1 - np.sin(np.radians(80.0)))
    assert abs(r.covered_area.sum() - cap_area) / cap_area < 1e-12


# --------------------------------------------------- I: extensive quantity
def test_I_extensive_total_conserved():
    lonb, latb = _global_grid(20, 15)
    r = OverlapConservativeResampler(lonb, latb, level=3)
    rng = np.random.default_rng(4)
    q = rng.uniform(0, 10, (latb.size - 1, lonb.size - 1))
    out = r.resample(q, quantity="extensive").cell_data
    assert abs(out.sum() - q.sum()) / q.sum() < 1e-13


# --------------------------------------- contrast with hard binning (doc)
def test_overlap_differs_from_hard_binning_locally():
    """Same global total, different local redistribution: the constant
    field through overlap remapping is flat, while a coarse cell's worth
    of quantity through hard binning lands in a single cell."""
    lonb = np.array([10.0, 40.0])
    latb = np.array([20.0, 50.0])
    r = OverlapConservativeResampler(lonb, latb, level=4)
    shares = np.asarray(r.overlap.sum(axis=1)).ravel()
    assert (shares > 0).sum() > 1
    assert shares.max() / shares.sum() < 0.9  # no cell hoards the quantity


def test_ring_indexing_consistent():
    lonb, latb = _global_grid()
    rn = OverlapConservativeResampler(lonb, latb, level=2, nest=True)
    rr = OverlapConservativeResampler(lonb, latb, level=2, nest=False)
    y = np.cos(np.radians((latb[:-1] + latb[1:]) / 2))[:, None] * np.ones(
        (1, lonb.size - 1))
    xn = rn.resample(y)
    xr = rr.resample(y)
    # identical values at physically identical cells: map nested ids to the
    # ring numbering through cell-centre coordinates
    healpix_geo = pytest.importorskip("healpix_geo")
    lonc, latc = healpix_geo.nested.healpix_to_lonlat(
        xn.cell_ids.astype(np.uint64), 2, ellipsoid="sphere")
    ring_of_nested = healpix_geo.ring.lonlat_to_healpix(
        lonc, latc, 2, ellipsoid="sphere").astype(np.int64)
    order = np.argsort(xr.cell_ids)
    pos = order[np.searchsorted(xr.cell_ids[order], ring_of_nested)]
    np.testing.assert_allclose(xn.cell_data, xr.cell_data[pos], atol=1e-12)
