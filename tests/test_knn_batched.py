"""`batch_size`: the neighbour search in slices of samples gives the result of
the unbatched search (cells, neighbours, distances, operators, values)."""

import numpy as np
import pytest
import torch

import healpix_resample as hr
from healpix_resample.knn import healpix_weighted_nearest

KW = {"nest": True, "ellipsoid": "WGS84", "verbose": False}
SEARCH = {"threshold": 0.05, "ring_weight": 2, "ring_search_max": 2}


@pytest.fixture(scope="module")
def dense_grid():
    """A regular lon/lat grid finer than the level-13 cells (~7 m vs ~800 m)
    is the healpix-convert case: many samples per cell."""
    lon, lat = np.meshgrid(np.linspace(6.0, 6.3, 120), np.linspace(45.0, 45.2, 90))
    return lon.ravel(), lat.ravel()


@pytest.fixture(scope="module")
def sparse_points():
    """Random samples much coarser than the level-17 cells: with a high
    threshold only the sample's own cell is kept, so a sample finds one kept
    cell within one ring, not 4 (partial rows, #74)."""
    rng = np.random.default_rng(1)
    return rng.uniform(6.0, 6.3, 1500), rng.uniform(45.0, 45.2, 1500)


def _search(lon, lat, level, **kw):
    return healpix_weighted_nearest(torch.as_tensor(lon), torch.as_tensor(lat), level, 4, **{**SEARCH, **kw})


@pytest.mark.parametrize("batch_size", [97, 1000, 10**7])
def test_search_identical_to_unbatched(dense_grid, batch_size):
    lon, lat = dense_grid
    ref = _search(lon, lat, 13)
    out = _search(lon, lat, 13, batch_size=batch_size)
    for a, b in zip(ref, out):
        assert torch.equal(a, b)
    assert ref[0].numel() > 0


def test_search_identical_with_partial_rows(sparse_points):
    lon, lat = sparse_points
    kw = {"threshold": 0.5, "ring_search_max": 1}
    ref = _search(lon, lat, 17, **kw)
    out = _search(lon, lat, 17, batch_size=211, **kw)
    assert bool((ref[1] < 0).any()), "the fixture should exercise the partial rows"
    for a, b in zip(ref, out):
        assert torch.equal(a, b)


def test_search_identical_with_out_cell_ids(dense_grid):
    lon, lat = dense_grid
    cells = _search(lon, lat, 13)[0][::3]
    ref = _search(lon, lat, 13, out_cell_ids=cells)
    out = _search(lon, lat, 13, out_cell_ids=cells, batch_size=500)
    for a, b in zip(ref, out):
        assert torch.equal(a, b)


def test_search_batch_size_must_be_positive(dense_grid):
    lon, lat = dense_grid
    with pytest.raises(ValueError):
        _search(lon, lat, 13, batch_size=0)


@pytest.mark.parametrize(
    "cls",
    [
        pytest.param(hr.BilinearResampler, id="bilinear"),
        pytest.param(hr.KNeighborsResampler, id="knn"),
        pytest.param(hr.BicubicResampler, id="bicubic"),
    ],
)
def test_resampler_identical_to_unbatched(dense_grid, cls):
    lon, lat = dense_grid
    val = (np.sin(lon * 40) + np.cos(lat * 60)).astype(np.float32)
    extra = {"Npt": 4} if cls is hr.KNeighborsResampler else {}
    ref = cls(lon_deg=lon, lat_deg=lat, level=13, threshold=0.05, ring_weight=2, **extra, **KW)
    out = cls(lon_deg=lon, lat_deg=lat, level=13, threshold=0.05, ring_weight=2, batch_size=1000, **extra, **KW)
    assert out.batch_size == 1000 and ref.batch_size is None
    assert torch.equal(ref.cell_ids, out.cell_ids)
    assert torch.equal(ref.hi, out.hi)
    assert torch.equal(ref.d_m, out.d_m)
    assert torch.equal(ref.M.values(), out.M.values())
    assert torch.equal(ref.M.col_indices(), out.M.col_indices())
    r0, r1 = ref.resample(val), out.resample(val)
    np.testing.assert_array_equal(r0.cell_ids, r1.cell_ids)
    np.testing.assert_array_equal(r0.cell_data, r1.cell_data)
