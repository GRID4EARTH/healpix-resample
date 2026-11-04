"""
tests/test_neighbour_links.py

A resampler built with the links of an earlier one on the same geometry
(``neighbours=``) skips the neighbour search and computes their distances
again: it is the same resampler, bit for bit (distances, ties, operators,
results), for the fine and the coarse searches, with and without
``out_cell_ids``, and with tied links.

CPU-only.
"""
from __future__ import annotations

import healpix_geo
import numpy as np
import pytest
import torch

from healpix_resample import NearestResampler, NeighbourLinks, PSFResampler


def _grid(n=70, lat0=45.0):
    lon, lat = np.meshgrid(0.3 * np.arange(n) / n, lat0 + 0.3 * np.arange(n) / n)
    rng = np.random.default_rng(3)
    return lon.ravel() + rng.uniform(0, 1e-4, n * n), lat.ravel()


def _ties_grid():
    # 1 arc-second samples at 45.65 N, level 18: whole columns of samples
    # have their Npt-th and (Npt+1)-th nearest cells at the same distance
    lon, lat = np.meshgrid(2.0 + np.arange(60) / 3600.0, 45.65 + np.arange(60) / 3600.0)
    return lon.ravel(), lat.ravel()


def _same(a, b):
    if isinstance(a, torch.Tensor):
        if a.layout != torch.strided:
            a, b = a.to_dense(), b.to_dense()
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(torch.nan_to_num(a, nan=-7.0), torch.nan_to_num(b, nan=-7.0))
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("batch_size", [None, 1000])
@pytest.mark.parametrize("delta", [None, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_psf_from_links_is_the_same(delta, dtype, batch_size):
    lon, lat = _grid()
    kw = dict(level=12, Npt=9, threshold=0.1, dtype=dtype, coarse_level_delta=delta,
              batch_size=batch_size)
    a = PSFResampler(lon, lat, **kw)
    assert isinstance(a.neighbours, NeighbourLinks)
    b = PSFResampler(lon, lat, neighbours=a.neighbours, **kw)
    for name in ("cell_ids", "hi", "d_m", "tie_rows", "tie_hi", "tie_d", "M", "MT"):
        _same(getattr(a, name), getattr(b, name))
    val = np.sin(lon * 40.0) + lat
    _same(a.resample(val, lam=0.1).cell_data, b.resample(val, lam=0.1).cell_data)


def test_ties_from_links_are_the_same():
    lon, lat = _ties_grid()
    a = PSFResampler(lon, lat, level=18, Npt=9, threshold=0.1, dtype=torch.float64)
    assert a.tie_rows.numel() > 0
    b = PSFResampler(lon, lat, level=18, Npt=9, threshold=0.1, dtype=torch.float64, neighbours=a.neighbours)
    for name in ("tie_rows", "tie_hi", "tie_d", "d_m", "M"):
        _same(getattr(a, name), getattr(b, name))


def test_nearest_with_out_cell_ids_from_links_is_the_same():
    lon, lat = _grid()
    chunk = healpix_geo.nested.lonlat_to_healpix(np.array([0.15]), np.array([45.15]), 8)
    out = healpix_geo.nested.zoom_to(chunk.astype(np.uint64), 8, 12)[0]
    kw = dict(level=12, out_cell_ids=out, coarse_level_delta=2)
    a = NearestResampler(lon, lat, **kw)
    b = NearestResampler(lon, lat, neighbours=a.neighbours, **kw)
    _same(a.cell_ids, b.cell_ids)
    _same(a.hi_k, b.hi_k)
    val = np.arange(lon.size, dtype=np.float64)
    _same(a.resample(val).cell_data, b.resample(val).cell_data)


def test_links_of_other_samples_are_refused():
    lon, lat = _grid()
    a = PSFResampler(lon, lat, level=12, Npt=9, threshold=0.1)
    with pytest.raises(ValueError, match="neighbours"):
        PSFResampler(lon[:-5], lat[:-5], level=12, Npt=9, threshold=0.1, neighbours=a.neighbours)
