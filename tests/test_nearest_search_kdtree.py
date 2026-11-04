"""
tests/test_nearest_search_kdtree.py

`NearestResampler(search="kdtree")` gives every requested cell its nearest
sample from a KD-tree of the samples, without the KNN links: the same cells
and samples as the default search, on a full chunk, on a partial one and with
``max_distance_m``.

On the CPU; the CUDA cases of the bucket search are skipped without a GPU.
"""
from __future__ import annotations

import healpix_geo
import numpy as np
import pytest

from healpix_resample import NearestResampler

LEVEL = 14


@pytest.fixture(scope="module")
def chunk():
    """The level-14 cells of a level-8 cell, and samples on a regular grid
    slightly coarser than the cells over it."""
    cell = healpix_geo.nested.lonlat_to_healpix(np.array([0.5]), np.array([45.0]), 8)
    out = healpix_geo.nested.zoom_to(cell.astype(np.uint64), 8, LEVEL)[0]
    clon, clat = map(np.asarray, healpix_geo.nested.healpix_to_lonlat(out, LEVEL))
    lon, lat = np.meshgrid(
        np.linspace(clon.min() - 0.01, clon.max() + 0.01, 70),
        np.linspace(clat.min() - 0.01, clat.max() + 0.01, 70),
    )
    rng = np.random.default_rng(0)
    lon = lon.ravel() + rng.uniform(-1e-4, 1e-4, lon.size)
    lat = lat.ravel() + rng.uniform(-1e-4, 1e-4, lat.size)
    return lon, lat, out


@pytest.mark.parametrize("part", ["full", "partial"])
@pytest.mark.parametrize("max_distance_m", [None, 200.0])
def test_same_samples_as_the_knn_search(chunk, part, max_distance_m):
    lon, lat, out = chunk
    if part == "partial":
        keep = lon < np.quantile(lon, 0.3)
        lon, lat = lon[keep], lat[keep]
    kw = dict(level=LEVEL, out_cell_ids=out, max_distance_m=max_distance_m, verbose=False, device="cpu")
    knn = NearestResampler(lon, lat, **kw)
    kd = NearestResampler(lon, lat, search="kdtree", **kw)
    if max_distance_m is not None:
        # the KD-tree drops every cell farther than max_distance_m from its
        # nearest sample, the KNN path only the cells its rings did not reach:
        # the cells both keep have the same sample
        common, i_kd, i_knn = np.intersect1d(
            np.asarray(kd.cell_ids), np.asarray(knn.cell_ids), return_indices=True
        )
        assert common.size == kd.K > 0
        np.testing.assert_array_equal(np.asarray(kd.hi_k)[i_kd], np.asarray(knn.hi_k)[i_knn])
        return
    np.testing.assert_array_equal(np.asarray(kd.cell_ids), np.asarray(knn.cell_ids))
    np.testing.assert_array_equal(np.asarray(kd.hi_k), np.asarray(knn.hi_k))
    val = np.arange(lon.size, dtype=np.float64)
    a, b = knn.resample(val), kd.resample(val)
    np.testing.assert_array_equal(np.asarray(a.cell_ids), np.asarray(b.cell_ids))
    np.testing.assert_array_equal(np.asarray(a.cell_data), np.asarray(b.cell_data))


def test_needs_out_cell_ids(chunk):
    lon, lat, _ = chunk
    with pytest.raises(ValueError, match="out_cell_ids"):
        NearestResampler(lon, lat, level=LEVEL, search="kdtree")


def test_unknown_search(chunk):
    lon, lat, out = chunk
    with pytest.raises(ValueError, match="search"):
        NearestResampler(lon, lat, level=LEVEL, out_cell_ids=out, search="brute")


def _bucket_search(lon, lat, out, device, **kw):
    import torch

    from healpix_resample.knn import _lonlat_to_xyz

    cells = torch.unique(torch.as_tensor(out.astype(np.int64)))
    clon, clat = healpix_geo.nested.healpix_to_lonlat(cells.numpy().astype(np.uint64), LEVEL)
    xyz_cells = _lonlat_to_xyz(torch.deg2rad(torch.as_tensor(np.asarray(clon), dtype=torch.float64)),
                               torch.deg2rad(torch.as_tensor(np.asarray(clat), dtype=torch.float64)))
    lon_t = torch.as_tensor(lon, dtype=torch.float64)
    lat_t = torch.as_tensor(lat, dtype=torch.float64)
    xyz_src = _lonlat_to_xyz(torch.deg2rad(lon_t), torch.deg2rad(lat_t))
    kd = NearestResampler._nearest_source(xyz_cells, xyz_src)
    bk = NearestResampler._nearest_source_buckets(
        cells, LEVEL, lon_t, lat_t, xyz_cells, xyz_src, ellipsoid="WGS84", device=device, **kw)
    return kd, bk


@pytest.mark.parametrize("part", ["full", "partial"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bucket_search_same_samples_as_the_kdtree(chunk, part, device):
    """The bucket search (used on CUDA devices) picks the KD-tree's samples,
    also for the cells outside the samples' extent (they go to the KD-tree)."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    lon, lat, out = chunk
    if part == "partial":
        keep = lon < np.quantile(lon, 0.3)
        lon, lat = lon[keep], lat[keep]
    (hi_kd, dot_kd), (hi_bk, dot_bk) = _bucket_search(lon, lat, out, device)
    np.testing.assert_array_equal(hi_bk.numpy(), hi_kd.numpy())
    np.testing.assert_allclose(dot_bk.numpy(), dot_kd.numpy(), rtol=0, atol=1e-15)


def test_bucket_search_too_many_samples_per_bucket_goes_to_the_kdtree(chunk, monkeypatch):
    """Above ``max_pairs`` for a batch of 1024 cells, the bucket search hands
    the whole search to the KD-tree instead of comparing the buckets."""
    calls = []
    kdtree = NearestResampler._nearest_source

    def spy(*args):
        calls.append(args[0].shape[0])
        return kdtree(*args)

    lon, lat, out = chunk
    (hi_kd, dot_kd), _ = _bucket_search(lon, lat, out, "cpu")
    monkeypatch.setattr(NearestResampler, "_nearest_source", staticmethod(spy))
    _, (hi_bk, dot_bk) = _bucket_search(lon, lat, out, "cpu", max_pairs=1024)
    assert calls[-1] == np.unique(out).size        # every cell, in one KD-tree query
    np.testing.assert_array_equal(hi_bk.numpy(), hi_kd.numpy())
    np.testing.assert_array_equal(dot_bk.numpy(), dot_kd.numpy())
