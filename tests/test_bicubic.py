"""
tests/test_bicubic.py

Test suite for `BicubicResampler` (`healpix_resample.bicubic`).

This is part of the package's first real test suite (see
`planning/00_init.md`, "Known gaps" -- no `tests/` directory existed before
the four `planning/0*_*.md` follow-up tasks). Shared layout convention
adopted here, for consistency with whichever other task files add their own
`tests/test_<module>.py`:

- One file per resampler module.
- Small, fast, synthetic fixtures (no real datasets).
- CPU-only by default (`device` left at its default, which falls back to CPU
  when CUDA is unavailable). Anything that specifically needs a GPU should
  be marked with `@pytest.mark.skipif(not torch.cuda.is_available())` --
  none of the tests below need one.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from healpix_resample import BicubicResampler, BilinearResampler
from healpix_resample.bicubic import _floor_signed


# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic datasets
# ─────────────────────────────────────────────────────────────────────────────

# Small structured grid near the origin -- same style as
# docs/tutorials/4resamplers.md, but kept small so the suite runs fast on CPU.
NDATA = 40
LEVEL = 10


def _small_grid(ndata: int = NDATA):
    """A small, densely-sampled patch -- good for basic shape/roundtrip checks
    and for the "linear field" comparison (curvature doesn't matter there)."""
    lon_grid, lat_grid = np.meshgrid(
        0.3 * np.arange(ndata) / ndata,
        0.3 * np.arange(ndata) / ndata,
    )
    return lon_grid.ravel(), lat_grid.ravel()


def _curved_grid(ndata: int = 60):
    """A wider patch (tens of degrees) so `sin(lon)*cos(lat)` has real
    curvature relative to the sampling scale -- a purely linear field, or a
    domain too small for the field to depart from its own tangent plane,
    cannot distinguish a cubic interpolator from a linear/bilinear one (see
    `planning/01_bicubic_resampler.md`)."""
    lon_grid, lat_grid = np.meshgrid(
        40.0 * np.arange(ndata) / ndata,
        40.0 * np.arange(ndata) / ndata,
    )
    return lon_grid.ravel(), lat_grid.ravel()


@pytest.fixture(scope="module")
def small_grid():
    return _small_grid()


@pytest.fixture(scope="module")
def curved_grid():
    return _curved_grid()


def _roundtrip_rmse(op, val):
    hval = op.resample(val).cell_data
    rval = op.invert(hval)
    return float(np.sqrt(np.mean((rval - val) ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# Basic shape / roundtrip smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_shape_roundtrip(small_grid):
    lon, lat = small_grid
    val = lon  # simple field: value = longitude

    op = BicubicResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)
    res = op.resample(val)

    assert res.cell_data.shape[0] == len(res.cell_ids)

    rval = op.invert(res.cell_data)
    mse = float(np.mean((rval - val) ** 2))
    assert np.isfinite(mse)


def test_default_npt_and_ring_search_max(small_grid):
    lon, lat = small_grid
    op = BicubicResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    assert op.Npt == 16
    # ring_search_max should have been auto-corrected (see nearest.py's
    # identical logic for Npt >= 16) so that, on this reasonably dense
    # synthetic grid, every sample finds all 16 neighbours (no -1 sentinels
    # left over from an under-sized ring search).
    assert bool((op.hi >= 0).all())


# ─────────────────────────────────────────────────────────────────────────────
# Recovery quality vs. BilinearResampler
# ─────────────────────────────────────────────────────────────────────────────

def test_linear_field_no_worse_than_bilinear(small_grid):
    lon, lat = small_grid
    val = lon  # purely linear -- bilinear and bicubic should both do well

    bicubic = BicubicResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)
    bilinear = BilinearResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    rmse_bicubic = _roundtrip_rmse(bicubic, val)
    rmse_bilinear = _roundtrip_rmse(bilinear, val)

    # Generous margin: a linear field can't showcase bicubic's advantage
    # (see test_curved_field_better_than_bilinear below for that), this just
    # guards against a gross regression.
    assert rmse_bicubic <= rmse_bilinear * 1.5 + 1e-8


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#54: at level 4 on a curved field the bicubic round-trip RMSE "
        "(0.008503) is worse than bilinear (0.007594); possibly related to #46. "
        "strict=True means CI fails if this starts passing -- remove the marker "
        "then."
    ),
)
def test_curved_field_better_than_bilinear(curved_grid):
    lon, lat = curved_grid
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    val = np.sin(lon_rad) * np.cos(lat_rad)  # has curvature over this domain

    # A coarser level than the shape/roundtrip tests: pixels need to be a few
    # multiples of the sample spacing (oversampled) for the smoothing-bias
    # difference between a 4-point linear kernel and a 16-point cubic kernel
    # to dominate over sampling noise.
    level = 4

    bicubic = BicubicResampler(lon_deg=lon, lat_deg=lat, level=level, verbose=False)
    bilinear = BilinearResampler(lon_deg=lon, lat_deg=lat, level=level, verbose=False)

    rmse_bicubic = _roundtrip_rmse(bicubic, val)
    rmse_bilinear = _roundtrip_rmse(bilinear, val)

    assert rmse_bicubic < rmse_bilinear


# ─────────────────────────────────────────────────────────────────────────────
# Batched (B, N) vs. plain (N,)
# ─────────────────────────────────────────────────────────────────────────────

def test_batched_and_unbatched(small_grid):
    lon, lat = small_grid
    op = BicubicResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    val_1d = lon
    val_2d = np.stack([lon, lon * 2.0], axis=0)  # (2, N)

    res_1d = op.resample(val_1d)
    res_2d = op.resample(val_2d)

    assert res_1d.cell_data.ndim == 1
    assert res_2d.cell_data.ndim == 2
    assert res_2d.cell_data.shape[0] == 2
    assert res_2d.cell_data.shape[1] == res_1d.cell_data.shape[0]

    rval_1d = op.invert(res_1d.cell_data)
    rval_2d = op.invert(res_2d.cell_data)
    assert rval_1d.ndim == 1
    assert rval_2d.shape == (2, len(lon))

    # M/MT are linear operators -- the second batch row (2x the first
    # sample values) must resample/invert to exactly 2x the first row.
    np.testing.assert_allclose(
        res_2d.cell_data[1], res_2d.cell_data[0] * 2.0, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(rval_2d[1], rval_2d[0] * 2.0, rtol=1e-5, atol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# NumPy / Torch in-out symmetry (the `T_Array` convention in base.py)
# ─────────────────────────────────────────────────────────────────────────────

def test_numpy_in_numpy_out(small_grid):
    lon, lat = small_grid
    op = BicubicResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)
    val = lon.astype(np.float64)

    res = op.resample(val)
    assert isinstance(res.cell_data, np.ndarray)
    assert isinstance(res.cell_ids, np.ndarray)

    rval = op.invert(res.cell_data)
    assert isinstance(rval, np.ndarray)


def test_torch_in_torch_out(small_grid):
    lon, lat = small_grid
    op = BicubicResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)
    val = torch.as_tensor(lon, dtype=torch.float64)

    res = op.resample(val)
    assert isinstance(res.cell_data, torch.Tensor)
    assert isinstance(res.cell_ids, torch.Tensor)

    rval = op.invert(res.cell_data)
    assert isinstance(rval, torch.Tensor)


# ─────────────────────────────────────────────────────────────────────────────
# Negative-weight cancellation guard
# ─────────────────────────────────────────────────────────────────────────────
# BicubicResampler.comp_matrix() floors per-cell/per-sample weight sums
# (norm_col / norm_row) relative to their *unsigned* counterparts before
# dividing by them, to avoid blow-up or sign flips from cancellation between
# Keys' kernel's positive central lobe and negative outer lobe. Test the
# guard helper directly, since engineering a real geometric cancellation
# case through the full resampler would be brittle and indirect.

def test_floor_signed_guards_near_cancellation():
    # index 0: near-total cancellation (norm ~ 0) despite large raw support
    # index 1, 2: well-conditioned, no cancellation -- must pass through as-is
    # index 3: exact zero from an empty/degenerate row -- must not become NaN
    norm = torch.tensor([1e-8, 5.0, -3.0, 0.0])
    norm_raw = torch.tensor([10.0, 5.0, 3.0, 2.0])

    safe = _floor_signed(norm, norm_raw)

    assert safe[1] == norm[1]
    assert safe[2] == norm[2]
    assert safe[0] > norm[0]
    assert safe[0] == pytest.approx(1e-3 * 10.0)
    assert safe[3] > 0.0
    assert torch.isfinite(safe).all()


def test_no_nan_or_inf_in_operators(small_grid):
    lon, lat = small_grid
    op = BicubicResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    assert torch.isfinite(op.M.values()).all()
    assert torch.isfinite(op.MT.values()).all()


# ─────────────────────────────────────────────────────────────────────────────
# conservative=True (issue #44: "conservative bi-linear is missing", applied
# to bicubic too) -- same guarantee and construction as
# BilinearResampler.resample(conservative=True): sum_k hval[k] == sum_i
# (valid i) val[i] * area[i]. Tolerances here are non-zero (unlike an exact
# equality check) because Keys' kernel is signed: `_floor_signed` can, for a
# rare pathologically-cancelled sample, make that sample's row in M_cons sum
# to only approximately (not bit-exactly) 1 -- see resample()'s docstring.
# On this suite's dense, well-conditioned grid that shouldn't actually bite,
# but the assertions are written to tolerate it rather than assume it away.
# ─────────────────────────────────────────────────────────────────────────────

def test_conservative_false_is_default_and_unchanged(small_grid):
    lon, lat = small_grid
    val = lon
    op = BicubicResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    res_default = op.resample(val)
    res_explicit = op.resample(val, conservative=False)

    np.testing.assert_array_equal(res_default.cell_data, res_explicit.cell_data)


def test_conservative_conserves_sum_uniform_area(small_grid):
    lon, lat = small_grid
    lon_rad, lat_rad = np.deg2rad(lon), np.deg2rad(lat)
    val = np.sin(lon_rad) * np.cos(lat_rad)
    op = BicubicResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    res = op.resample(val, conservative=True)

    np.testing.assert_allclose(float(np.sum(res.cell_data)), float(np.sum(val)), rtol=1e-3, atol=1e-6)


def test_conservative_conserves_sum_with_area(small_grid):
    lon, lat = small_grid
    lon_rad, lat_rad = np.deg2rad(lon), np.deg2rad(lat)
    val = np.sin(lon_rad) * np.cos(lat_rad)
    rng = np.random.default_rng(0)
    area = rng.uniform(0.5, 2.0, size=lon.shape)

    op = BicubicResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False, area=area)
    res = op.resample(val, conservative=True)

    expected = float(np.sum(val * area))
    np.testing.assert_allclose(float(np.sum(res.cell_data)), expected, rtol=1e-3, atol=1e-6)


def test_conservative_excludes_nan_sample_and_stays_finite(small_grid):
    lon, lat = small_grid
    lon_rad, lat_rad = np.deg2rad(lon), np.deg2rad(lat)
    val = np.sin(lon_rad) * np.cos(lat_rad)
    bad_idx = len(lon) // 2
    val_nan = val.copy()
    val_nan[bad_idx] = np.nan

    op = BicubicResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)
    res = op.resample(val_nan, conservative=True)

    assert np.all(np.isfinite(res.cell_data))

    valid = np.ones(lon.shape, dtype=bool)
    valid[bad_idx] = False
    expected = float(np.sum(val[valid]))
    np.testing.assert_allclose(float(np.sum(res.cell_data)), expected, rtol=1e-3, atol=1e-6)


def test_conservative_all_nan_row_is_all_nan(small_grid):
    lon, lat = small_grid
    op = BicubicResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    val_nan = np.full(lon.shape, np.nan)
    res = op.resample(val_nan, conservative=True)

    assert np.all(np.isnan(res.cell_data))
