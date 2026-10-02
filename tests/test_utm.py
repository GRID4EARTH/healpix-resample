"""
tests/test_utm.py

Test suite for the HEALPix -> UTM module (`healpix_resample.utm`): the
target-grid helper (`UTMGrid`), the operator (`HealpixToUTM`, built on each
resampler's `invert()`), and the one-call wrapper (`healpix_to_utm`).

Small/fast synthetic fixtures, CPU-only, like the other test modules.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import healpix_geo

pyproj = pytest.importorskip("pyproj")

from healpix_resample import (  # noqa: E402
    BilinearResampler,
    HealpixToUTM,
    UTMGrid,
    healpix_to_utm,
    utm_crs_from_lonlat,
)

LEVEL = 14  # ~400 m cells
EPSG = 32630  # UTM 30N
METHODS = ["nearest", "bilinear", "bicubic", "clough_tocher"]


def _wrap(lon):
    return (np.asarray(lon) + 180.0) % 360.0 - 180.0


def _affine(lon, lat):
    return 3.0 + 2.0 * (lon + 3.0) + 5.0 * (lat - 48.3)


def _curved(lon, lat):
    return np.sin(40.0 * (lon + 3.0)) * np.cos(60.0 * (lat - 48.3))


@pytest.fixture(scope="module")
def grid():
    # 12 km x 10 km at 100 m -> (100, 120)
    return UTMGrid.from_bounds(EPSG, (500000.0, 5340000.0, 512000.0, 5350000.0), 100.0)


@pytest.fixture(scope="module")
def cells():
    """HEALPix cells covering the grid with a ~5 km margin, and their centers."""
    big = UTMGrid.from_bounds(EPSG, (495000.0, 5335000.0, 517000.0, 5355000.0), 50.0)
    lon, lat = big.lonlat()
    ids = np.unique(
        healpix_geo.nested.lonlat_to_healpix(lon.ravel(), lat.ravel(), LEVEL, ellipsoid="WGS84")
    ).astype(np.int64)
    clon, clat = healpix_geo.nested.healpix_to_lonlat(ids.astype(np.uint64), LEVEL, ellipsoid="WGS84")
    return ids, _wrap(clon), np.asarray(clat)


# ─────────────────────────────────────────────────────────────────────────────
# UTMGrid
# ─────────────────────────────────────────────────────────────────────────────

def test_grid_from_bounds_pixel_centers(grid):
    assert grid.shape == (100, 120)
    assert grid.crs.to_epsg() == EPSG
    np.testing.assert_allclose(grid.x[[0, -1]], [500050.0, 511950.0])
    np.testing.assert_allclose(grid.y[[0, -1]], [5349950.0, 5340050.0])  # north-up

    lon, lat = grid.lonlat()
    assert lon.shape == lat.shape == grid.shape
    fwd = pyproj.Transformer.from_crs(4326, EPSG, always_xy=True)
    x, y = fwd.transform(lon, lat)
    np.testing.assert_allclose(x, np.broadcast_to(grid.x, grid.shape), atol=1e-3)
    np.testing.assert_allclose(y, np.broadcast_to(grid.y[:, None], grid.shape), atol=1e-3)


def test_grid_padded(grid):
    p = grid.padded(3)
    assert p.shape == (106, 126)
    np.testing.assert_allclose(p.x[3:-3], grid.x)
    np.testing.assert_allclose(p.y[3:-3], grid.y)
    np.testing.assert_allclose(np.diff(p.x), 100.0)
    np.testing.assert_allclose(np.diff(p.y), -100.0)
    assert grid.padded(0) is grid


def test_utm_crs_from_lonlat():
    assert utm_crs_from_lonlat([-4.5], [48.4]).to_epsg() == 32630  # Brest
    assert utm_crs_from_lonlat([2.3, 2.4], [48.8, 48.9]).to_epsg() == 32631
    assert utm_crs_from_lonlat([151.2], [-33.9]).to_epsg() == 32756  # southern hemisphere
    # antimeridian: centroid of 179.5E / 179.9E-ish wrapped points stays in zone 60
    assert utm_crs_from_lonlat([179.2, 179.8], [10.0, 10.0]).to_epsg() == 32660
    assert utm_crs_from_lonlat([179.9, -179.7], [10.0, 10.0]).to_epsg() == 32601


def test_grid_from_cell_ids(cells):
    ids, clon, clat = cells
    g = UTMGrid.from_cell_ids(ids, LEVEL, 200.0)
    assert g.crs.to_epsg() == EPSG
    np.testing.assert_allclose(np.diff(g.x), 200.0)
    fwd = pyproj.Transformer.from_crs(4326, g.crs, always_xy=True)
    x, y = fwd.transform(clon, clat)
    assert g.x[0] - 100.0 <= x.min() and x.max() <= g.x[-1] + 100.0
    assert g.y[-1] - 100.0 <= y.min() and y.max() <= g.y[0] + 100.0

    default = UTMGrid.from_cell_ids(ids, LEVEL)  # resolution = cell size
    assert 350.0 < default.x[1] - default.x[0] < 450.0


# ─────────────────────────────────────────────────────────────────────────────
# HealpixToUTM
# ─────────────────────────────────────────────────────────────────────────────

# max abs error budget on the affine / curved test fields, per method. The
# KNN-based invert() operators are weighted averages (not affine-exact);
# Clough-Tocher is a genuine C1 cubic interpolant.
_TOL = {
    "nearest": (0.03, 0.25),
    "bilinear": (0.01, 0.10),
    "bicubic": (0.01, 0.10),
    "clough_tocher": (1e-6, 2e-3),
}


@pytest.mark.parametrize("method", METHODS)
def test_full_coverage_accuracy(grid, cells, method):
    ids, clon, clat = cells
    lon, lat = grid.lonlat()
    op = HealpixToUTM(ids, LEVEL, grid, method=method)

    assert op.valid.shape == grid.shape
    assert op.valid.all()

    for field, tol in zip((_affine, _curved), _TOL[method]):
        out = op.resample(field(clon, clat))
        assert out.shape == grid.shape
        assert np.isfinite(out).all()
        assert np.abs(out - field(lon, lat)).max() < tol


def test_clough_tocher_is_most_accurate(grid, cells):
    ids, clon, clat = cells
    lon, lat = grid.lonlat()
    truth = _curved(lon, lat)
    rms = {
        m: np.sqrt(np.mean((HealpixToUTM(ids, LEVEL, grid, method=m).resample(_curved(clon, clat)) - truth) ** 2))
        for m in METHODS
    }
    assert rms["clough_tocher"] < 0.1 * min(rms["bilinear"], rms["bicubic"])
    assert rms["bilinear"] < rms["nearest"]


def test_nearest_is_containing_cell_lookup(grid, cells):
    ids, _, _ = cells
    lon, lat = grid.lonlat()
    out = HealpixToUTM(ids, LEVEL, grid, method="nearest").resample(ids.astype(np.float64) % 1000.0)
    pix = np.asarray(
        healpix_geo.nested.lonlat_to_healpix(lon.ravel(), lat.ravel(), LEVEL, ellipsoid="WGS84")
    ).astype(np.int64)
    np.testing.assert_array_equal(out.ravel(), pix % 1000)


@pytest.mark.parametrize("method", METHODS)
def test_cell_order_does_not_matter(grid, cells, method):
    ids, clon, clat = cells
    hval = _curved(clon, clat)
    ref = HealpixToUTM(ids, LEVEL, grid, method=method).resample(hval)
    perm = np.random.default_rng(0).permutation(ids.size)
    out = HealpixToUTM(ids[perm], LEVEL, grid, method=method).resample(hval[perm])
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_partial_coverage_is_nan_outside_footprint(grid, cells, method):
    """Cells only on the western half: no value may be produced for a pixel
    whose center is not in a data cell, and what is produced stays accurate."""
    ids, clon, clat = cells
    keep = clon < np.median(clon)
    lon, lat = grid.lonlat()
    op = HealpixToUTM(ids[keep], LEVEL, grid, method=method)
    out = op.resample(_affine(clon, clat)[keep])

    pix = np.asarray(
        healpix_geo.nested.lonlat_to_healpix(lon.ravel(), lat.ravel(), LEVEL, ellipsoid="WGS84")
    ).astype(np.int64).reshape(grid.shape)
    in_footprint = np.isin(pix, ids[keep])

    np.testing.assert_array_equal(np.isfinite(out), op.valid)
    assert not (op.valid & ~in_footprint).any()
    assert 0.4 < op.valid.mean() < 0.6
    # one-sided neighbourhoods at the footprint edge are less accurate for
    # the KNN-based operators (weighted averages, not affine-exact)
    assert np.nanmax(np.abs(out - _affine(lon, lat))) < 3.0 * _TOL[method][0]


def test_batch_and_array_types(grid, cells):
    ids, clon, clat = cells
    op = HealpixToUTM(ids, LEVEL, grid, method="bilinear")
    a, c = _affine(clon, clat), _curved(clon, clat)

    single = op.resample(a)
    batch = op.resample(np.stack([a, c]))
    assert isinstance(batch, np.ndarray) and batch.shape == (2,) + grid.shape
    np.testing.assert_allclose(batch[0], single)

    t = op.resample(torch.as_tensor(np.stack([a, c])))
    assert isinstance(t, torch.Tensor)
    np.testing.assert_allclose(t.cpu().numpy(), batch)

    f32 = op.resample(a.astype(np.float32))
    np.testing.assert_allclose(f32, single, rtol=1e-5, atol=1e-5)


def test_nan_cell_stays_local(grid, cells):
    ids, clon, clat = cells
    hval = _affine(clon, clat)
    fwd = pyproj.Transformer.from_crs(4326, EPSG, always_xy=True)
    cx, cy = fwd.transform(clon, clat)
    k = np.argmin((cx - 506000.0) ** 2 + (cy - 5345000.0) ** 2)
    hval[k] = np.nan

    out = HealpixToUTM(ids, LEVEL, grid, method="bilinear").resample(hval)
    bad = np.isnan(out)
    assert bad.any()
    X, Y = np.meshgrid(grid.x, grid.y)
    assert np.hypot(X[bad] - cx[k], Y[bad] - cy[k]).max() < 2000.0


def test_resampler_class_and_kwargs(grid, cells):
    ids, clon, clat = cells
    hval = _curved(clon, clat)
    by_name = HealpixToUTM(ids, LEVEL, grid, method="bilinear").resample(hval)
    by_class = HealpixToUTM(ids, LEVEL, grid, method=BilinearResampler, dtype=torch.float64).resample(hval)
    np.testing.assert_allclose(by_class, by_name)


def test_input_validation(grid, cells):
    ids, clon, clat = cells
    with pytest.raises(ValueError, match="unknown method"):
        HealpixToUTM(ids, LEVEL, grid, method="spline")
    with pytest.raises(ValueError, match="unique"):
        HealpixToUTM(np.concatenate([ids, ids[:1]]), LEVEL, grid)
    with pytest.raises(TypeError):
        HealpixToUTM(ids, LEVEL, (grid.x, grid.y))
    op = HealpixToUTM(ids, LEVEL, grid, method="nearest")
    with pytest.raises(ValueError, match="K="):
        op.resample(np.zeros(ids.size + 1))


# ─────────────────────────────────────────────────────────────────────────────
# healpix_to_utm
# ─────────────────────────────────────────────────────────────────────────────

def test_healpix_to_utm_explicit_grid(grid, cells):
    ids, clon, clat = cells
    hval = _curved(clon, clat)
    out, g = healpix_to_utm(hval, ids, LEVEL, grid=grid, method="clough_tocher")
    assert g is grid
    np.testing.assert_allclose(out, HealpixToUTM(ids, LEVEL, grid, method="clough_tocher").resample(hval))
    with pytest.raises(ValueError):
        healpix_to_utm(hval, ids, LEVEL, grid=grid, resolution=100.0)


def test_healpix_to_utm_auto_grid(cells):
    ids, clon, clat = cells
    out, g = healpix_to_utm(_affine(clon, clat), ids, LEVEL, resolution=200.0)
    assert g.crs.to_epsg() == EPSG
    assert out.shape == g.shape
    lon, lat = g.lonlat()
    ok = np.isfinite(out)
    assert ok.mean() > 0.9
    assert np.abs(out - _affine(lon, lat))[ok].max() < 3.0 * _TOL["bilinear"][0]  # footprint edge
