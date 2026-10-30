"""Tests of GridBilinearResampler (bilinear interpolation of lon/lat grids at
HEALPix cell centres)."""
from __future__ import annotations

import healpix_geo
import numpy as np
import pytest
import torch
from scipy.interpolate import RegularGridInterpolator

from healpix_resample import GridBilinearResampler

LEVEL = 14


def _cells(lon, lat, area_level=8, level=LEVEL):
    cell = healpix_geo.nested.lonlat_to_healpix(np.array([lon]), np.array([lat]), area_level, ellipsoid="WGS84")
    cells = healpix_geo.nested.zoom_to(cell, area_level, level)[0]
    clon, clat = healpix_geo.nested.healpix_to_lonlat(cells, level, ellipsoid="WGS84")
    return cells, clon, clat


def _axes(clon, clat, step=0.01, pad=0.05):
    lat = np.arange(clat.min() - pad, clat.max() + pad, step)
    lon = np.arange(clon.min() - pad, clon.max() + pad, step)
    return lat, lon


def test_matches_scipy_regular_grid_interpolator():
    cells, clon, clat = _cells(6.8, 45.8)
    lat, lon = _axes(clon, clat)
    val = np.random.default_rng(0).normal(size=(lat.size, lon.size))
    r = GridBilinearResampler(lon, lat, LEVEL, out_cell_ids=cells)
    out = r.resample(val.ravel())
    ref = RegularGridInterpolator((lat, lon), val)(np.c_[clat, clon])
    np.testing.assert_array_equal(out.cell_ids, cells)
    np.testing.assert_allclose(out.cell_data, ref, rtol=0, atol=1e-12)


def test_axis_order_and_batch():
    cells, clon, clat = _cells(6.8, 45.8)
    lat, lon = _axes(clon, clat)
    val = np.random.default_rng(1).normal(size=(2, lat.size, lon.size))
    ref = GridBilinearResampler(lon, lat, LEVEL, out_cell_ids=cells).resample(val.reshape(2, -1)).cell_data
    flipped = val[:, ::-1, ::-1]
    out = GridBilinearResampler(lon[::-1], lat[::-1], LEVEL, out_cell_ids=cells).resample(
        torch.as_tensor(flipped.reshape(2, -1).copy())
    )
    assert isinstance(out.cell_data, torch.Tensor) and out.cell_data.shape == (2, cells.size)
    np.testing.assert_allclose(out.cell_data.numpy(), ref, atol=1e-12)


def test_antimeridian_window():
    parts = [_cells(x, 0.15, area_level=9) for x in (179.99, -179.99)]
    cells = np.concatenate([p[0] for p in parts])
    clon = np.concatenate([p[1] for p in parts])
    clat = np.concatenate([p[2] for p in parts])
    lat = np.arange(-0.3, 0.6, 0.01)
    lon = np.r_[np.arange(-180.0, -179.6, 0.01), np.arange(179.6, 180.0, 0.01)]  # source order
    f = lambda la, lo: np.sin(np.radians(lo) * 500) + la
    r = GridBilinearResampler(lon, lat, LEVEL, out_cell_ids=cells)
    out = r.resample(f(lat[:, None], lon[None]).ravel()).cell_data
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, f(clat, clon), atol=2e-3)


def test_global_axis_wraps():
    cells, clon, clat = _cells(179.999, 10.0, area_level=9)
    lat = np.arange(9.0, 11.0, 0.1)
    lon = np.arange(-180.0, 180.0, 0.1)
    out = GridBilinearResampler(lon, lat, LEVEL, out_cell_ids=cells).resample(
        np.broadcast_to(lat[:, None] * 2.0, (lat.size, lon.size)).ravel()
    ).cell_data
    np.testing.assert_allclose(out, clat * 2.0, atol=1e-9)


def test_bands_with_different_longitude_steps():
    # 1" south of 50°N, 1.5" north of it (Copernicus DEM tiles): cells between
    # the two bands get a value, a linear field is exact
    cells, clon, clat = _cells(10.0, 50.0, area_level=12, level=18)
    latS = np.arange(int(49.98 * 3600), 50 * 3600) / 3600
    latN = np.arange(50 * 3600, int(50.02 * 3600)) / 3600
    lonS = np.arange(int(9.95 * 3600), int(10.05 * 3600)) / 3600
    lonN = np.arange(int(9.95 * 2400), int(10.05 * 2400)) / 2400
    f = lambda la, lo: 1000 * la + 300 * lo
    val = np.r_[f(latS[:, None], lonS[None]).ravel(), f(latN[:, None], lonN[None]).ravel()]
    r = GridBilinearResampler([lonS, lonN], [latS, latN], 18, out_cell_ids=cells)
    out = r.resample(val).cell_data
    gap = (clat > latS[-1]) & (clat < latN[0])
    assert gap.any() and np.isfinite(out).all()
    np.testing.assert_allclose(out, f(clat, clon), atol=1e-6)


def test_nan_corners():
    cells, clon, clat = _cells(6.8, 45.8)
    lat, lon = _axes(clon, clat)
    val = np.broadcast_to(lat[:, None] + 0 * lon[None], (lat.size, lon.size)).copy()
    val[:, lon < 6.8] = np.nan
    lenient = GridBilinearResampler(lon, lat, LEVEL, out_cell_ids=cells).resample(val.ravel()).cell_data
    strict = GridBilinearResampler(lon, lat, LEVEL, out_cell_ids=cells, min_weight=1.0).resample(val.ravel()).cell_data
    j = np.searchsorted(lon, 6.8)  # first valid column
    east = clon >= lon[j]
    np.testing.assert_allclose(strict[east], clat[east], atol=1e-12)
    assert np.isnan(strict[clon < lon[j]]).all()
    # renormalised over the valid corners: also valid up to half a pixel west of the edge
    half = (clon < lon[j]) & (clon > lon[j] - 0.5 * (lon[1] - lon[0]))
    assert half.any() and np.isfinite(lenient[half]).all()
    assert np.isnan(lenient[clon < lon[j] - 0.5 * (lon[1] - lon[0]) - 1e-9]).all()


def test_default_cells_cover_the_grid():
    lat = np.linspace(45.0, 45.2, 21)
    lon = np.linspace(6.0, 6.2, 21)
    r = GridBilinearResampler(lon, lat, 12)
    out = r.resample(np.ones(lat.size * lon.size))
    clon, clat = healpix_geo.nested.healpix_to_lonlat(out.cell_ids.astype(np.uint64), 12, ellipsoid="WGS84")
    inner = (clat > 45.0) & (clat < 45.2) & (clon > 6.0) & (clon < 6.2)
    assert np.all(out.cell_data[inner] == 1.0)
    outer = (clat < 45.0) | (clat > 45.2)
    assert outer.any() and np.isnan(out.cell_data[outer]).all()
