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


def _resample(lat, lon, val, cls=hr.BilinearResampler, **kw):
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    op = cls(lon_deg=lon2.ravel(), lat_deg=lat2.ravel(), **{**KW, **kw})
    out = op.resample(val)
    return op, np.asarray(out.cell_ids), np.asarray(out.cell_data)


def _eager_M(op):
    """M from all the links (``hi`` and the tied ones) through COO + coalesce."""
    valid = op.hi >= 0
    rows = torch.arange(op.N)[:, None].expand(op.N, op.Npt)[valid]
    w, w_tie = op._all_link_weights()
    rows = torch.cat([rows, op.tie_rows])
    cols = torch.cat([op.hi[valid], op.tie_hi])
    w = torch.cat([w[valid], w_tie])
    w = w / op._norm_col(cols, w)[cols]
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), w.to(op.dtype), size=(op.N, op.K)).coalesce()


def test_tied_cells_are_all_linked(tie_grid):
    lat, lon, val = tie_grid
    op, _, _ = _resample(lat, lon, val)
    # the tied links are kept apart: `hi` / `d_m` stay (N, Npt)
    assert op.hi.shape == (op.N, 4) and op.d_m.shape == (op.N, 4)
    assert op.tie_rows.numel() > 0, "the grid should have samples with a tie"
    # the extra cell is at the distance of the 4th one
    torch.testing.assert_close(op.tie_d, op.d_m[op.tie_rows, 3], rtol=0, atol=1e-6)
    # and it is linked in M
    M = op.M.to_dense()
    assert bool((M[op.tie_rows, op.tie_hi] > 0).all())


@pytest.mark.parametrize("cls", [hr.BilinearResampler, hr.BicubicResampler], ids=["bilinear", "bicubic"])
def test_operator_with_ties_equals_coalesced_build(tie_grid, cls):
    """The tied links inserted into the CSC arrays of M give the same matrix as
    one COO build of all the links, with sorted rows in every column."""
    lat, lon, val = tie_grid
    op, _, _ = _resample(lat, lon, val, cls=cls)
    assert op.tie_rows.numel() > 0
    actual = op.M.to_sparse_coo().coalesce()
    expected = _eager_M(op)
    torch.testing.assert_close(actual.indices(), expected.indices(), rtol=0, atol=0)
    torch.testing.assert_close(actual.values(), expected.values())
    rows, ccol = op.M.row_indices(), op.M.ccol_indices()
    for a, b in zip(ccol[:-1].tolist(), ccol[1:].tolist()):
        assert bool((rows[a + 1 : b] > rows[a : b - 1]).all())


@pytest.mark.parametrize("batch_size", [None, 1000])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    ("cls", "atol"),
    [
        pytest.param(hr.BilinearResampler, 1e-6, id="bilinear"),
        pytest.param(hr.BicubicResampler, 1e-6, id="bicubic"),
        pytest.param(hr.NearestResampler, 1e-6, id="nearest"),
        # the conjugate gradient stops at its own tolerance: up to 1 cm here
        # (3 mm on macOS arm64, 1.0 cm on Linux x86_64; on main the PSF
        # result moved by 2.4 m on this grid)
        pytest.param(hr.PSFResampler, 5e-2, id="psf"),
    ],
)
def test_result_does_not_depend_on_the_last_bits(tie_grid, batch_size, seed, cls, atol):
    lat, lon, val = tie_grid
    _, c0, v0 = _resample(lat, lon, val, cls=cls, batch_size=batch_size)
    _, c1, v1 = _resample(_ulp_noise(lat, seed), _ulp_noise(lon, seed + 10), val, cls=cls, batch_size=batch_size)
    np.testing.assert_array_equal(c1, c0)
    # the weights move with the last bits of the distances (~1e-9 m), not the links
    np.testing.assert_allclose(v1, v0, rtol=0, atol=atol)


def test_batched_search_identical_with_ties(tie_grid):
    lat, lon, _ = tie_grid
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    args = (torch.as_tensor(lon2.ravel()), torch.as_tensor(lat2.ravel()), 18, 4)
    kw = {"threshold": 0.05, "ring_weight": 2, "ring_search_max": 2, "return_ties": True}
    *ref, ref_ties = healpix_weighted_nearest(*args, **kw)
    *out, out_ties = healpix_weighted_nearest(*args, batch_size=777, **kw)
    assert ref_ties[0].numel() > 0
    for a, b in zip([*ref, *ref_ties], [*out, *out_ties]):
        assert torch.equal(a, b)


def test_search_without_return_ties_has_extra_columns(tie_grid):
    """Called directly, the search returns the tied cells as extra columns."""
    lat, lon, _ = tie_grid
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    args = (torch.as_tensor(lon2.ravel()), torch.as_tensor(lat2.ravel()), 18, 4)
    kw = {"threshold": 0.05, "ring_weight": 2}
    _, idx, dist = healpix_weighted_nearest(*args, **kw)
    _, idx4, dist4, (rows, tie_idx, tie_dist) = healpix_weighted_nearest(*args, return_ties=True, **kw)
    assert idx.shape[1] > 4
    torch.testing.assert_close(idx[:, :4], idx4, rtol=0, atol=0)
    torch.testing.assert_close(idx[rows, 4], tie_idx, rtol=0, atol=0)
    torch.testing.assert_close(dist[rows, 4], tie_dist, rtol=0, atol=0)
    assert bool((idx[:, 4:].ge(0).sum(dim=1) == torch.bincount(rows, minlength=idx.shape[0])).all())
