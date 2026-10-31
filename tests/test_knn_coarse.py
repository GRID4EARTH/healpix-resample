"""`coarse_level_delta`: the candidate cells of the neighbour search taken from
coarser nested cells give the cells, neighbours and distances of the fine-level
search (GRID4EARTH/healpix-resample#95)."""

import numpy as np
import pytest
import torch

import healpix_resample as hr
from healpix_resample.knn import healpix_weighted_nearest

KW = {"nest": True, "ellipsoid": "WGS84", "verbose": False}
SEARCH = {"threshold": 0.05, "ring_weight": 2, "ring_search_max": 2}


@pytest.fixture(scope="module")
def dense_grid():
    """A regular lon/lat grid finer than the level-13 cells (~7 m vs ~800 m):
    the healpix-convert case, many samples per cell."""
    lon, lat = np.meshgrid(np.linspace(6.0, 6.3, 120), np.linspace(45.0, 45.2, 90))
    return lon.ravel(), lat.ravel()


@pytest.fixture(scope="module")
def s2_like_grid():
    """About one sample per cell (10 m samples, level-20 cells ~6 m): the
    Sentinel-2 case, Npt = 9 with the default rings."""
    lon, lat = np.meshgrid(
        np.linspace(-5.990, -5.975, 130), np.linspace(37.290, 37.300, 110)
    )
    return lon.ravel(), lat.ravel()


@pytest.fixture(scope="module")
def sparse_points():
    """Random samples much coarser than the level-17 cells: with a high
    threshold only the sample's own cell is kept (partial rows)."""
    rng = np.random.default_rng(1)
    return rng.uniform(6.0, 6.3, 1500), rng.uniform(45.0, 45.2, 1500)


def _search(lon, lat, level, npt, **kw):
    return healpix_weighted_nearest(
        torch.as_tensor(lon), torch.as_tensor(lat), level, npt, return_ties=True, **kw
    )


def _neighbours(cells, idx, dist):
    """Per sample: its neighbour cell ids and distances, sorted by distance
    then id (the order of equal distances is not part of the result)."""
    ids = torch.where(
        idx >= 0, cells[torch.clamp(idx, min=0)], torch.full_like(idx, -1)
    ).numpy()
    d = dist.numpy()
    order = (
        np.stack([np.lexsort((i, x)) for i, x in zip(ids, d)])
        if len(ids)
        else np.zeros_like(ids)
    )
    return (
        torch.from_numpy(np.take_along_axis(ids, order, 1)),
        torch.from_numpy(np.take_along_axis(d, order, 1)),
    )


def _ties(cells, ties):
    rows, idx, dist = ties
    return sorted(zip(rows.tolist(), cells[idx].tolist(), dist.tolist()))


def _assert_same(ref, out):
    cells_r, idx_r, dist_r, ties_r = ref
    cells_o, idx_o, dist_o, ties_o = out
    assert torch.equal(cells_r, cells_o)
    ids_r, d_r = _neighbours(cells_r, idx_r, dist_r)
    ids_o, d_o = _neighbours(cells_o, idx_o, dist_o)
    assert torch.equal(ids_r, ids_o)
    assert torch.equal(d_r, d_o)
    assert _ties(cells_r, ties_r) == _ties(cells_o, ties_o)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_dense_samples_same_as_fine_search(dense_grid, k):
    lon, lat = dense_grid
    ref = _search(lon, lat, 13, 4, **SEARCH)
    out = _search(lon, lat, 13, 4, coarse_level_delta=k, **SEARCH)
    assert ref[0].numel() > 0
    _assert_same(ref, out)


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("batch_size", [None, 997])
def test_s2_like_samples_same_as_fine_search(s2_like_grid, k, batch_size):
    lon, lat = s2_like_grid
    ref = _search(lon, lat, 20, 9, ring_search_max=4)
    out = _search(
        lon, lat, 20, 9, ring_search_max=4, coarse_level_delta=k, batch_size=batch_size
    )
    _assert_same(ref, out)


def test_gap_in_the_samples(s2_like_grid):
    lon, lat = s2_like_grid
    sel = np.abs(lon - lon.mean()) > 0.002
    ref = _search(lon[sel], lat[sel], 20, 9, ring_search_max=4)
    out = _search(lon[sel], lat[sel], 20, 9, ring_search_max=4, coarse_level_delta=1)
    _assert_same(ref, out)


def test_out_cell_ids_same_as_fine_search(dense_grid):
    lon, lat = dense_grid
    cells = _search(lon, lat, 13, 4, **SEARCH)[0][::3]
    ref = _search(lon, lat, 13, 4, out_cell_ids=cells, **SEARCH)
    out = _search(lon, lat, 13, 4, out_cell_ids=cells, coarse_level_delta=1, **SEARCH)
    _assert_same(ref, out)


def test_partial_rows_same_as_fine_search(sparse_points):
    lon, lat = sparse_points
    kw = {"threshold": 0.5, "ring_search_max": 1}
    ref = _search(lon, lat, 17, 4, **kw)
    out = _search(lon, lat, 17, 4, coarse_level_delta=1, **kw)
    assert bool((ref[1] < 0).any()), "the fixture should exercise the partial rows"
    _assert_same(ref, out)


@pytest.mark.parametrize("k", [1, 2])
def test_samples_across_base_faces(k):
    # a grid across the edge between the north polar base faces 0 and 1 (the
    # 90 E meridian): the rings of the samples next to the edge reach the
    # other face and go through the fine search
    import healpix_geo.nested as hn

    lon, lat = np.meshgrid(np.linspace(89.9, 90.1, 70), np.linspace(59.97, 60.03, 60))
    lon, lat = lon.ravel(), lat.ravel()
    faces = hn.healpix_to_base_cell_coordinates(
        hn.lonlat_to_healpix(lon, lat, 15, ellipsoid="WGS84"), 15
    )[0]
    assert np.unique(faces).size > 1, "the fixture should straddle two base faces"
    ref = _search(lon, lat, 15, 9, ring_search_max=4)
    out = _search(lon, lat, 15, 9, ring_search_max=4, coarse_level_delta=k)
    _assert_same(ref, out)


def test_ties_same_as_fine_search():
    # 1 arc-second grid next to the level-18 ring i = 243000, where whole
    # columns of samples have their 4th and 5th nearest cells at the same
    # distance (test_knn_ties.py)
    k = np.arange(int(45.6528 * 3600), int(45.6568 * 3600) + 1) / 3600.0
    j = np.arange(int(6.850 * 3600), int(6.860 * 3600) + 1) / 3600.0
    lat2, lon2 = np.meshgrid(k, j, indexing="ij")
    kw = {"threshold": 0.05, "ring_weight": 2, "ring_search_max": 2}
    ref = _search(lon2.ravel(), lat2.ravel(), 18, 4, **kw)
    out = _search(lon2.ravel(), lat2.ravel(), 18, 4, coarse_level_delta=2, **kw)
    assert ref[3][0].numel() > 0
    _assert_same(ref, out)


@pytest.mark.parametrize(
    "cls",
    [
        pytest.param(hr.PSFResampler, id="psf"),
        pytest.param(hr.NearestResampler, id="nearest"),
        pytest.param(hr.BilinearResampler, id="bilinear"),
    ],
)
def test_resampler_keyword(s2_like_grid, cls):
    lon, lat = s2_like_grid
    val = np.sin(lon * 3000.0) + np.cos(lat * 2000.0)
    ref = cls(lon_deg=lon, lat_deg=lat, level=20, **KW)
    out = cls(lon_deg=lon, lat_deg=lat, level=20, coarse_level_delta=1, **KW)
    assert torch.equal(ref.cell_ids, out.cell_ids)
    a = ref.resample(val).cell_data
    b = out.resample(val).cell_data
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-9)


def test_coarse_level_delta_needs_nested(dense_grid):
    lon, lat = dense_grid
    with pytest.raises(ValueError, match="nested"):
        _search(lon, lat, 13, 4, nest=False, coarse_level_delta=1, **SEARCH)


@pytest.mark.parametrize("k", [0, 14])
def test_coarse_level_delta_range(dense_grid, k):
    lon, lat = dense_grid
    with pytest.raises(ValueError, match="coarse_level_delta"):
        _search(lon, lat, 13, 4, coarse_level_delta=k, **SEARCH)
