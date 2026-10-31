"""Neighbour search: input dtype (#73), sparse samples (#74), nearest fill distance (#75)."""

import healpix_geo.nested as hn
import numpy as np
import pytest
import torch

import healpix_resample as hr
import healpix_resample.knn as knn

KW = {"nest": True, "ellipsoid": "WGS84", "verbose": False}

INTERPOLATING = [
    pytest.param(hr.BilinearResampler, id="bilinear"),
    pytest.param(hr.PSFResampler, id="psf"),
    pytest.param(hr.BicubicResampler, id="bicubic"),
]
ALL = INTERPOLATING + [pytest.param(hr.NearestResampler, id="nearest")]


@pytest.fixture(scope="module")
def sparse_grid():
    """81 samples on a 0.25 deg grid: coarse input for a level-11 target."""
    lon, lat = np.meshgrid(np.linspace(1.0, 3.0, 9), np.linspace(48.0, 50.0, 9))
    return lon.ravel(), lat.ravel()


# ── #73: float32 coordinates ─────────────────────────────────────────────────


@pytest.mark.parametrize("cls", ALL)
def test_float32_coordinates_give_the_float64_result(sparse_grid, cls):
    lon, lat = sparse_grid
    op64 = cls(lon_deg=lon, lat_deg=lat, level=11, **KW)
    op32 = cls(lon_deg=lon.astype(np.float32), lat_deg=lat.astype(np.float32), level=11, **KW)

    # float32 rounding of the coordinates (~1e-7 deg) may move a cell at the
    # margin, not the bulk of the result
    common = np.intersect1d(op32.cell_ids.cpu().numpy(), op64.cell_ids.cpu().numpy())
    assert common.size >= 0.95 * max(op32.K, op64.K)
    assert op32.K > 0


def test_float32_torch_coordinates():
    lon = torch.linspace(1.0, 3.0, 50, dtype=torch.float32)
    lat = torch.linspace(48.0, 50.0, 50, dtype=torch.float32)
    op = hr.BilinearResampler(lon_deg=lon, lat_deg=lat, level=9, **KW)
    assert op.K > 0


# ── #74: samples much sparser than the cells ─────────────────────────────────


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("level", [11, 13])
def test_sparse_samples_still_reach_cells(sparse_grid, cls, level):
    lon, lat = sparse_grid
    op = cls(lon_deg=lon, lat_deg=lat, level=level, **KW)
    assert op.K > 0

    # every sample keeps at least one link, even with fewer than Npt cells found
    assert (op.hi >= 0).any(dim=1).all()

    res = op.resample(np.cos(np.radians(lat)))
    assert np.isfinite(np.asarray(res.cell_data)).any()


@pytest.mark.parametrize("cls", INTERPOLATING)
def test_partial_neighbours_are_used_by_the_interpolating_resamplers(sparse_grid, cls):
    # A constant field must come back constant on every cell that has a link,
    # whether its samples found Npt cells or fewer.
    lon, lat = sparse_grid
    op = cls(lon_deg=lon, lat_deg=lat, level=11, **KW)
    res = op.resample(np.full(lon.size, 3.0))
    data = np.asarray(res.cell_data)
    finite = np.isfinite(data)
    assert finite.any()
    np.testing.assert_allclose(data[finite], 3.0, rtol=1e-4)


def test_no_sample_reaching_any_kept_cell_raises(sparse_grid, monkeypatch):
    # Cells pass the threshold but no sample links to any of them: this used to
    # return an empty operator silently.
    lon, lat = sparse_grid
    real = knn.healpix_weighted_nearest

    def no_links(*args, **kwargs):
        cell_ids, hi, d, ties = real(*args, **kwargs)
        return cell_ids, torch.full_like(hi, -1), torch.full_like(d, float("inf")), tuple(t[:0] for t in ties)

    monkeypatch.setattr(knn, "healpix_weighted_nearest", no_links)
    with pytest.raises(RuntimeError, match="no sample found any of them"):
        hr.PSFResampler(lon_deg=lon, lat_deg=lat, level=11, **KW)


# ── #75: NearestResampler fill distance ──────────────────────────────────────


@pytest.fixture(scope="module")
def chunk_outside():
    """The level-18 cells of one level-11 chunk, and three samples ~1 km east of it."""
    level, chunk_level = 18, 11
    n = 4 ** (level - chunk_level)
    chunk = int(hn.lonlat_to_healpix(np.array([2.35]), np.array([48.85]), chunk_level, ellipsoid="WGS84")[0])
    out_cells = np.arange(chunk * n, (chunk + 1) * n, dtype=np.uint64)
    vlon, vlat = hn.vertices(np.array([chunk], dtype=np.uint64), chunk_level, ellipsoid="WGS84")
    east = int(np.argmax(vlon[0]))
    lon = np.asarray(vlon[0])[east] + 0.015 + np.array([0.0, 0.0002, 0.0004])
    lat = np.asarray(vlat[0])[east] + np.array([0.0, 0.0002, -0.0002])
    return lon, lat, out_cells, level


def _distance_to_nearest_sample_m(cell_ids, lon, lat, level, radius=6371000.0):
    clon, clat = hn.healpix_to_lonlat(np.asarray(cell_ids, dtype=np.uint64), level, ellipsoid="WGS84")

    def xyz(lo, la):
        lo, la = np.radians(lo), np.radians(la)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)

    dots = xyz(np.asarray(clon), np.asarray(clat)) @ xyz(lon, lat).T
    return radius * np.arccos(np.clip(dots.max(axis=1), -1.0, 1.0))


def test_nearest_without_max_distance_fills_every_requested_cell(chunk_outside):
    lon, lat, out_cells, level = chunk_outside
    op = hr.NearestResampler(lon_deg=lon, lat_deg=lat, level=level, out_cell_ids=out_cells, **KW)
    assert op.K == out_cells.size


def test_nearest_max_distance_leaves_far_cells_out(chunk_outside):
    lon, lat, out_cells, level = chunk_outside
    op = hr.NearestResampler(
        lon_deg=lon, lat_deg=lat, level=level, out_cell_ids=out_cells, max_distance_m=2000.0, **KW
    )
    assert 0 < op.K < out_cells.size
    cell_ids = op.cell_ids.cpu().numpy()
    assert np.isin(cell_ids, out_cells.astype(np.int64)).all()
    assert _distance_to_nearest_sample_m(cell_ids, lon, lat, level).max() <= 2000.0 * (1 + 1e-6)

    res = op.resample(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert np.asarray(res.cell_data).shape == (2, op.K)


def test_nearest_max_distance_can_leave_every_cell_out(chunk_outside):
    lon, lat, out_cells, level = chunk_outside
    op = hr.NearestResampler(
        lon_deg=lon, lat_deg=lat, level=level, out_cell_ids=out_cells, max_distance_m=500.0, **KW
    )
    assert op.K == 0
    res = op.resample(np.array([1.0, 2.0, 3.0]))
    assert np.asarray(res.cell_ids).size == 0


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_nearest_max_distance_must_be_positive(chunk_outside, bad):
    lon, lat, out_cells, level = chunk_outside
    with pytest.raises(ValueError, match="max_distance_m"):
        hr.NearestResampler(
            lon_deg=lon, lat_deg=lat, level=level, out_cell_ids=out_cells, max_distance_m=bad, **KW
        )
