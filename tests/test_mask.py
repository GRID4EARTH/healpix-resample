"""
tests/test_mask.py

Test suite for `healpix_resample.mask` (`BitmaskResampler`,
`CategoricalResampler` -- issue #43, "make specific resampler for mask-like
data").

CPU-only by default, following the shared layout convention from the other
test files in this package (see `tests/test_bilinear.py`).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from healpix_resample import (
    BitmaskResampler,
    CategoricalResampler,
    CategoricalResampleResults,
    BicubicResampler,
)


NDATA = 60
LEVEL = 10
SPAN = 0.3  # degrees


def _grid(ndata: int = NDATA, span: float = SPAN):
    lon_grid, lat_grid = np.meshgrid(
        span * np.arange(ndata) / ndata,
        span * np.arange(ndata) / ndata,
    )
    return lon_grid.ravel(), lat_grid.ravel()


@pytest.fixture(scope="module")
def grid():
    return _grid()


# ─────────────────────────────────────────────────────────────────────────────
# BitmaskResampler -- independent flags, "OR"
# ─────────────────────────────────────────────────────────────────────────────

def test_bitmask_requires_positive_n_bits(grid):
    lon, lat = grid
    with pytest.raises(ValueError):
        BitmaskResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, n_bits=0)


def test_bitmask_rejects_bad_threshold(grid):
    lon, lat = grid
    with pytest.raises(ValueError):
        BitmaskResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, n_bits=2, bit_threshold=1.5)


def test_bitmask_rejects_nan(grid):
    lon, lat = grid
    op = BitmaskResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, n_bits=2)
    bad_mask = np.zeros(lon.shape, dtype=np.float64)
    bad_mask[0] = np.nan
    with pytest.raises(ValueError):
        op.resample(bad_mask)


def test_bitmask_rejects_non_1d(grid):
    lon, lat = grid
    op = BitmaskResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, n_bits=2)
    with pytest.raises(ValueError):
        op.resample(np.zeros((2, len(lon))))


def test_bitmask_recovers_independent_flags_deep_inside_regions(grid):
    """Two independent flags (bit0 by longitude, bit1 by latitude, so all
    4 combinations occur) must be recovered exactly for output cells whose
    center sits comfortably inside one quadrant, away from either boundary
    (where every one of BilinearResampler's 4 nearest samples agrees on
    both bits)."""
    lon, lat = grid
    lon_mid, lat_mid = SPAN / 2, SPAN / 2

    bit0 = (lon > lon_mid).astype(np.int64)
    bit1 = (lat > lat_mid).astype(np.int64)
    mask = bit0 | (bit1 << 1)

    op = BitmaskResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, n_bits=2)
    res = op.resample(mask)

    assert res.cell_data.dtype.kind in "iu"
    assert np.all(res.cell_data >= 0) and np.all(res.cell_data <= 3)

    # Check a handful of cells deep inside each quadrant via their centers.
    import healpix_geo
    cell_lon, cell_lat = healpix_geo.nested.healpix_to_lonlat(
        res.cell_ids.astype(np.uint64), LEVEL, ellipsoid="WGS84"
    )
    cell_lon, cell_lat = np.asarray(cell_lon), np.asarray(cell_lat)
    margin = SPAN * 0.15  # comfortably away from the mid-point boundary

    by_id = dict(zip(res.cell_ids.tolist(), res.cell_data.tolist()))
    for want_lon_hi, want_lat_hi, expected in [
        (False, False, 0),
        (True, False, 1),
        (False, True, 2),
        (True, True, 3),
    ]:
        lon_ok = (cell_lon > lon_mid + margin) if want_lon_hi else (cell_lon < lon_mid - margin)
        lat_ok = (cell_lat > lat_mid + margin) if want_lat_hi else (cell_lat < lat_mid - margin)
        deep_mask = lon_ok & lat_ok
        assert np.any(deep_mask), "test geometry needs adjusting -- no deep-interior cell found"
        deep_ids = res.cell_ids[deep_mask]
        for cid in deep_ids[: min(5, len(deep_ids))].tolist():
            assert by_id[cid] == expected


def test_bitmask_numpy_torch_symmetry(grid):
    lon, lat = grid
    op = BitmaskResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, n_bits=2)
    mask_np = ((lon > SPAN / 2).astype(np.int64)) | ((lat > SPAN / 2).astype(np.int64) << 1)

    res_np = op.resample(mask_np)
    assert isinstance(res_np.cell_data, np.ndarray)
    assert isinstance(res_np.cell_ids, np.ndarray)

    res_t = op.resample(torch.as_tensor(mask_np))
    assert isinstance(res_t.cell_data, torch.Tensor)
    assert isinstance(res_t.cell_ids, torch.Tensor)

    np.testing.assert_array_equal(res_np.cell_data, res_t.cell_data.numpy())


# ─────────────────────────────────────────────────────────────────────────────
# CategoricalResampler -- mutually-exclusive classes, argmax
# ─────────────────────────────────────────────────────────────────────────────

def test_categorical_rejects_nan(grid):
    lon, lat = grid
    op = CategoricalResampler(lon_deg=lon, lat_deg=lat, level=LEVEL)
    bad_mask = np.zeros(lon.shape, dtype=np.float64)
    bad_mask[0] = np.nan
    with pytest.raises(ValueError):
        op.resample(bad_mask)


def test_categorical_rejects_non_1d(grid):
    lon, lat = grid
    op = CategoricalResampler(lon_deg=lon, lat_deg=lat, level=LEVEL)
    with pytest.raises(ValueError):
        op.resample(np.zeros((2, len(lon))))


def test_categorical_single_class_is_trivial(grid):
    lon, lat = grid
    op = CategoricalResampler(lon_deg=lon, lat_deg=lat, level=LEVEL)
    mask = np.full(lon.shape, 7, dtype=np.int64)

    res = op.resample(mask)
    assert np.all(res.cell_data == 7)


def test_categorical_recovers_dominant_class_deep_inside_regions(grid):
    """Three classes split by longitude tercile: cells whose center sits
    comfortably inside one tercile (away from either boundary) must recover
    that tercile's class exactly."""
    lon, lat = grid
    edges = np.quantile(lon, [1 / 3, 2 / 3])
    land_cover = np.digitize(lon, edges).astype(np.int64)  # 0, 1, or 2

    op = CategoricalResampler(lon_deg=lon, lat_deg=lat, level=LEVEL)
    res = op.resample(land_cover)

    import healpix_geo
    cell_lon, _ = healpix_geo.nested.healpix_to_lonlat(
        res.cell_ids.astype(np.uint64), LEVEL, ellipsoid="WGS84"
    )
    cell_lon = np.asarray(cell_lon)
    by_id = dict(zip(res.cell_ids.tolist(), res.cell_data.tolist()))

    margin = SPAN * 0.03
    deep_class0 = res.cell_ids[cell_lon < edges[0] - margin]
    deep_class2 = res.cell_ids[cell_lon > edges[1] + margin]

    assert len(deep_class0) > 0 and len(deep_class2) > 0
    for cid in deep_class0[: min(5, len(deep_class0))].tolist():
        assert by_id[cid] == 0
    for cid in deep_class2[: min(5, len(deep_class2))].tolist():
        assert by_id[cid] == 2


def test_categorical_return_scores_shape_and_consistency(grid):
    lon, lat = grid
    edges = np.quantile(lon, [1 / 3, 2 / 3])
    land_cover = np.digitize(lon, edges).astype(np.int64)

    op = CategoricalResampler(lon_deg=lon, lat_deg=lat, level=LEVEL)
    res = op.resample(land_cover, return_scores=True)

    assert isinstance(res, CategoricalResampleResults)
    n_classes = len(res.classes)
    K = len(res.cell_ids)
    assert res.scores.shape == (n_classes, K)

    # softmax must sum to 1 across classes for every cell
    np.testing.assert_allclose(res.scores.sum(axis=0), np.ones(K), rtol=1e-5, atol=1e-6)

    # the hard argmax decision (cell_data) must match the highest-softmax-score class
    winner_idx = np.argmax(res.scores, axis=0)
    expected_winner_class = res.classes[winner_idx]
    np.testing.assert_array_equal(res.cell_data, expected_winner_class)


def test_categorical_without_return_scores_has_no_extra_fields(grid):
    lon, lat = grid
    op = CategoricalResampler(lon_deg=lon, lat_deg=lat, level=LEVEL)
    mask = np.full(lon.shape, 1, dtype=np.int64)

    res = op.resample(mask)
    assert not isinstance(res, CategoricalResampleResults)


def test_categorical_numpy_torch_symmetry(grid):
    lon, lat = grid
    op = CategoricalResampler(lon_deg=lon, lat_deg=lat, level=LEVEL)
    edges = np.quantile(lon, [0.5])
    mask_np = np.digitize(lon, edges).astype(np.int64)

    res_np = op.resample(mask_np)
    assert isinstance(res_np.cell_data, np.ndarray)

    res_t = op.resample(torch.as_tensor(mask_np))
    assert isinstance(res_t.cell_data, torch.Tensor)

    np.testing.assert_array_equal(res_np.cell_data, res_t.cell_data.numpy())


# ─────────────────────────────────────────────────────────────────────────────
# kernel= swappable
# ─────────────────────────────────────────────────────────────────────────────

def test_kernel_swappable_to_bicubic(grid):
    lon, lat = grid
    mask = ((lon > SPAN / 2).astype(np.int64)) | ((lat > SPAN / 2).astype(np.int64) << 1)

    op_bitmask = BitmaskResampler(
        lon_deg=lon, lat_deg=lat, level=LEVEL, n_bits=2, kernel=BicubicResampler, verbose=False
    )
    res = op_bitmask.resample(mask)
    assert len(res.cell_ids) == op_bitmask.cell_ids.numel()

    edges = np.quantile(lon, [1 / 3, 2 / 3])
    land_cover = np.digitize(lon, edges).astype(np.int64)
    op_cat = CategoricalResampler(
        lon_deg=lon, lat_deg=lat, level=LEVEL, kernel=BicubicResampler, verbose=False
    )
    res_cat = op_cat.resample(land_cover)
    assert len(res_cat.cell_ids) == op_cat.cell_ids.numel()
