"""Cells at the same distance as the Npt-th nearest one are all linked.

At level 18 the north-cap ring i = 243000 (lat 45.6548) has 4 * 243000 cells,
1/2700 degree apart: commensurate with a 1 arc-second lon/lat grid, so every
4th input column of the rows next to the ring has its 4th and 5th nearest
cells (two cells of the ring, symmetric about the sample's meridian) at
exactly the same distance. Keeping only one of them left the choice to the
last bits of the distances or to the order of `torch.topk` for equal values:
1-2 ULP on the coordinates, or another platform, moved heights by metres
along the ring.
"""

import numpy as np
import pytest
import torch

import healpix_resample as hr
from healpix_resample.knn import healpix_weighted_nearest

KW = {"level": 18, "nest": True, "ellipsoid": "WGS84", "verbose": False, "threshold": 0.05, "ring_weight": 2}
R = 6371000.0


@pytest.fixture(scope="module")
def tie_grid():
    """1 arc-second grid around ring i = 243000 and a terrain-like field (m)."""
    k = np.arange(int(45.6528 * 3600), int(45.6568 * 3600) + 1)
    j = np.arange(int(6.850 * 3600), int(6.860 * 3600) + 1)
    lat, lon = k / 3600.0, j / 3600.0
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    x = (lon2 - 6.855) * np.pi / 180 * R * np.cos(np.radians(45.6548))
    y = (lat2 - 45.6548) * np.pi / 180 * R
    val = 0.8 * y + 0.4 * x + 60 * np.sin(x / 150) * np.cos(y / 170)
    return lat, lon, val.ravel()


def _ulp_noise(a, seed):
    rng = np.random.default_rng(seed)
    return a + rng.integers(-2, 3, a.size) * np.spacing(a)


def _resample(lat, lon, val, **kw):
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    op = hr.BilinearResampler(lon_deg=lon2.ravel(), lat_deg=lat2.ravel(), **{**KW, **kw})
    out = op.resample(val)
    return op, np.asarray(out.cell_ids), np.asarray(out.cell_data)


def test_tied_cells_are_all_linked(tie_grid):
    lat, lon, val = tie_grid
    op, _, _ = _resample(lat, lon, val)
    assert op.Npt == 4
    assert op.hi.shape[1] > 4, "the grid should have samples with a tie"
    tied = op.hi[:, 4] >= 0
    # the extra cell is at the distance of the 4th one
    assert torch.allclose(op.d_m[tied, 4], op.d_m[tied, 3], rtol=0, atol=1e-6)
    # samples without a tie keep exactly 4 links
    assert bool((op.hi[~tied, 4:] < 0).all())


@pytest.mark.parametrize("batch_size", [None, 1000])
def test_result_does_not_depend_on_the_last_bits(tie_grid, batch_size):
    lat, lon, val = tie_grid
    _, c0, v0 = _resample(lat, lon, val, batch_size=batch_size)
    for seed in range(3):
        _, c1, v1 = _resample(_ulp_noise(lat, seed), _ulp_noise(lon, seed + 10), val, batch_size=batch_size)
        np.testing.assert_array_equal(c1, c0)
        # the weights move with the last bits of the distances (~1e-9 m), not the links
        np.testing.assert_allclose(v1, v0, rtol=0, atol=1e-6)


def test_batched_search_identical_with_ties(tie_grid):
    lat, lon, _ = tie_grid
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    args = (torch.as_tensor(lon2.ravel()), torch.as_tensor(lat2.ravel()), 18, 4)
    kw = {"threshold": 0.05, "ring_weight": 2, "ring_search_max": 2}
    ref = healpix_weighted_nearest(*args, **kw)
    out = healpix_weighted_nearest(*args, batch_size=777, **kw)
    assert ref[1].shape[1] > 4
    for a, b in zip(ref, out):
        assert torch.equal(a, b)
