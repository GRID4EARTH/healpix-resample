"""
tests/test_psf.py

Test suite for `PSFResampler`'s NaN-filtering behaviour
(`healpix_resample.psf.PSFResampler.resample`).

See `planning/02_psf_nan_filtering.md`: unlike `BilinearResampler` /
`NearestResampler`, a single unfiltered NaN sample can poison the *entire*
reconstructed field for a PSFResampler batch row (not just cells near the bad
sample), because the CG solve's `_wdot` reduction sums over all K cells per
row. These tests exercise the fix: NaN samples are zeroed out of the CG solve
per batch row, and rows that are entirely NaN are excluded from the shared CG
loop and returned as all-NaN directly.

CPU-only by default (`PSFResampler`'s own default `device="cpu"`), following
the shared layout convention established in `tests/test_bicubic.py`.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from healpix_resample import PSFResampler
from healpix_resample.knn import healpix_weighted_nearest


NDATA = 40
LEVEL = 8


def _grid(ndata: int = NDATA):
    lon_grid, lat_grid = np.meshgrid(
        0.3 * np.arange(ndata) / ndata,
        0.3 * np.arange(ndata) / ndata,
    )
    return lon_grid.ravel(), lat_grid.ravel()


def _field(lon, lat):
    # smooth but non-linear, so a single missing sample has a real (if
    # small) local effect rather than being perfectly interpolated anyway.
    return np.sin(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))


@pytest.fixture(scope="module")
def grid():
    return _grid()


# ─────────────────────────────────────────────────────────────────────────────
# Single NaN sample: rest of the field must stay finite (and, far from the
# NaN sample, close to what dropping that sample entirely would give).
# ─────────────────────────────────────────────────────────────────────────────

def test_single_nan_sample_keeps_field_finite(grid):
    lon, lat = grid
    val = _field(lon, lat)
    bad_idx = len(lon) // 2  # an interior sample

    # Fixed output cell set shared by every resampler built below, so their
    # results can be compared cell-for-cell regardless of how each one's own
    # internal cell discovery happens to order things.
    baseline = PSFResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, threshold=0.3, verbose=False)
    out_ids = baseline.cell_ids

    val_nan = val.copy()
    val_nan[bad_idx] = np.nan

    op_nan = PSFResampler(
        lon_deg=lon, lat_deg=lat, level=LEVEL, threshold=0.3, verbose=False, out_cell_ids=out_ids
    )
    res_nan = op_nan.resample(val_nan, lam=0.0, tol=1e-10, max_iter=300)

    # This is the assertion that would have failed before the fix: without
    # filtering, the single NaN sample poisons every cell in the CG solve,
    # not just cells near it.
    assert np.all(np.isfinite(res_nan.cell_data))

    # Sanity check: cells far from the excluded sample should be close to
    # the result you'd get with that sample entirely absent from the
    # geometry (not just zeroed) -- the fix's normalization-dilution caveat
    # (see resample()'s docstring) is a local effect near the bad sample.
    lon_wo = np.delete(lon, bad_idx)
    lat_wo = np.delete(lat, bad_idx)
    val_wo = np.delete(val, bad_idx)
    op_wo = PSFResampler(
        lon_deg=lon_wo, lat_deg=lat_wo, level=LEVEL, threshold=0.3, verbose=False, out_cell_ids=out_ids
    )
    res_wo = op_wo.resample(val_wo, lam=0.0, tol=1e-10, max_iter=300)

    ids_common = np.intersect1d(res_nan.cell_ids, res_wo.cell_ids)
    assert len(ids_common) > 0.9 * len(out_ids)  # almost everything should survive in both

    order_nan = np.argsort(res_nan.cell_ids)
    order_wo = np.argsort(res_wo.cell_ids)
    ids_nan_sorted = res_nan.cell_ids[order_nan]
    ids_wo_sorted = res_wo.cell_ids[order_wo]
    data_nan_sorted = res_nan.cell_data[order_nan]
    data_wo_sorted = res_wo.cell_data[order_wo]

    common_mask_nan = np.isin(ids_nan_sorted, ids_common)
    common_mask_wo = np.isin(ids_wo_sorted, ids_common)
    ids_nan_common = ids_nan_sorted[common_mask_nan]
    ids_wo_common = ids_wo_sorted[common_mask_wo]
    assert np.array_equal(ids_nan_common, ids_wo_common)

    data_nan_common = data_nan_sorted[common_mask_nan]
    data_wo_common = data_wo_sorted[common_mask_wo]

    import healpix_geo
    lon_c, lat_c = healpix_geo.nested.healpix_to_lonlat(
        ids_nan_common.astype(np.uint64), LEVEL, ellipsoid="WGS84"
    )
    dist_deg = np.sqrt((np.asarray(lon_c) - lon[bad_idx]) ** 2 + (np.asarray(lat_c) - lat[bad_idx]) ** 2)
    far_mask = dist_deg > np.percentile(dist_deg, 75)

    np.testing.assert_allclose(
        data_nan_common[far_mask], data_wo_common[far_mask], rtol=0.05, atol=0.05
    )


# ─────────────────────────────────────────────────────────────────────────────
# All-NaN input
# ─────────────────────────────────────────────────────────────────────────────

def test_all_nan_1d_input(grid):
    lon, lat = grid
    op = PSFResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, threshold=0.3, verbose=False)

    val_nan = np.full_like(lon, np.nan)
    res = op.resample(val_nan, lam=0.0)

    assert np.all(np.isnan(res.cell_data))
    assert int(res.cg_niters) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Batched: one all-NaN row must not contaminate a fully-valid row
# ─────────────────────────────────────────────────────────────────────────────

def test_batched_all_nan_row_does_not_contaminate_valid_row(grid):
    lon, lat = grid
    val = _field(lon, lat)
    op = PSFResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, threshold=0.3, verbose=False)

    val_batch = np.stack([val, np.full_like(val, np.nan)], axis=0)  # (2, N)

    res_batch = op.resample(val_batch, lam=0.0, tol=1e-10, max_iter=300)
    res_solo = op.resample(val, lam=0.0, tol=1e-10, max_iter=300)

    assert np.all(np.isnan(res_batch.cell_data[1]))
    np.testing.assert_allclose(res_batch.cell_data[0], res_solo.cell_data, rtol=1e-5, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Partially-NaN + conservative=True: target_mean must exclude NaN samples
# (and their area) from the area-weighted mean, not just from the numerator.
# ─────────────────────────────────────────────────────────────────────────────

def test_conservative_excludes_nan_from_area_weighted_mean(grid):
    lon, lat = grid
    val = _field(lon, lat)
    rng = np.random.default_rng(0)
    area = rng.uniform(0.5, 2.0, size=lon.shape)

    op = PSFResampler(
        lon_deg=lon, lat_deg=lat, level=LEVEL, threshold=0.3, verbose=False, area=area
    )

    val_nan = val.copy()
    nan_idx = np.arange(0, len(val), 7)  # scattered NaNs
    val_nan[nan_idx] = np.nan

    res = op.resample(val_nan, lam=0.0, conservative=True, tol=1e-10, max_iter=300)

    valid_mask = ~np.isnan(val_nan)
    expected_mean = np.sum(val_nan[valid_mask] * area[valid_mask]) / np.sum(area[valid_mask])
    actual_mean = float(np.mean(res.cell_data))

    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-3, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# fill_missing_out_cells: opt-in fallback for weakly-supported out_cell_ids
#
# See planning/04_parent_cell_subsetting.md's follow-up discussion: cells that
# pass healpix_weighted_nearest's wide-neighbourhood weight-sum threshold
# (so they're retained in cell_ids when out_cell_ids is set -- that path
# skips KNeighborsResampler's own orphaned-cell pruning, see knn.py) but were
# never actually selected by any sample's own Npt-nearest search end up with
# an empty (K,N)/(N,K) column -- the same "orphaned cell" phenomenon fixed by
# pruning in the auto-discovery (out_cell_ids=None) path, left deliberately
# unpruned here so PSFResampler's own out_cell_ids fallback can handle it.
# We locate a genuine orphaned cell for the shared `grid` fixture directly via
# healpix_weighted_nearest (the same helper KNeighborsResampler itself calls),
# rather than guessing geometry blind, and skip gracefully if this particular
# grid/threshold combination doesn't happen to produce one.
# ─────────────────────────────────────────────────────────────────────────────

def _find_orphan_and_good_cells(lon, lat, level=LEVEL, threshold=0.3, npt=9):
    lon_t = torch.as_tensor(lon, dtype=torch.float64)
    lat_t = torch.as_tensor(lat, dtype=torch.float64)
    cell_ids, idx_k, _dist_k = healpix_weighted_nearest(
        lon_t, lat_t, level=level, Npt=npt, threshold=threshold,
    )
    reachable = torch.zeros(cell_ids.numel(), dtype=torch.bool)
    valid_hi = idx_k[idx_k >= 0]
    if valid_hi.numel() > 0:
        reachable[valid_hi] = True
    orphan_ids = cell_ids[~reachable].numpy()
    good_ids = cell_ids[reachable].numpy()
    return orphan_ids, good_ids


def test_fill_missing_out_cells_default_false_produces_nan(grid):
    lon, lat = grid
    val = _field(lon, lat)

    orphan_ids, good_ids = _find_orphan_and_good_cells(lon, lat)
    if len(orphan_ids) == 0 or len(good_ids) < 5:
        pytest.skip(
            "no weakly-supported ('orphaned') cell found for this grid/threshold "
            "combination -- nothing here to exercise fill_missing_out_cells against"
        )

    bad_id = int(orphan_ids[0])
    out_ids = np.concatenate([good_ids[:5], [bad_id]]).astype(np.int64)

    op = PSFResampler(
        lon_deg=lon, lat_deg=lat, level=LEVEL, threshold=0.3, verbose=False, out_cell_ids=out_ids
    )
    # Construction records the weak cell instead of running the expensive
    # per-cell fallback loop for it (the whole point of the flag defaulting
    # to False).
    assert op.uncomputable_out_cells is not None
    assert op.uncomputable_out_cells.numel() >= 1

    res = op.resample(val, lam=0.0, tol=1e-8, max_iter=100)
    by_id = dict(zip(res.cell_ids.tolist(), res.cell_data.tolist()))

    assert bad_id in by_id
    assert np.isnan(by_id[bad_id])
    for gid in good_ids[:5].tolist():
        assert np.isfinite(by_id[gid])


def test_fill_missing_out_cells_true_restores_fallback(grid):
    lon, lat = grid
    val = _field(lon, lat)

    orphan_ids, good_ids = _find_orphan_and_good_cells(lon, lat)
    if len(orphan_ids) == 0 or len(good_ids) < 5:
        pytest.skip(
            "no weakly-supported ('orphaned') cell found for this grid/threshold "
            "combination -- nothing here to exercise fill_missing_out_cells against"
        )

    bad_id = int(orphan_ids[0])
    out_ids = np.concatenate([good_ids[:5], [bad_id]]).astype(np.int64)

    op = PSFResampler(
        lon_deg=lon, lat_deg=lat, level=LEVEL, threshold=0.3, verbose=False,
        out_cell_ids=out_ids, fill_missing_out_cells=True,
    )
    # Opting back in restores the pre-flag behaviour exactly: the weak
    # column gets patched in via the nearest-sample fallback instead of
    # being left for resample() to nan out.
    assert op.uncomputable_out_cells is None or op.uncomputable_out_cells.numel() == 0

    res = op.resample(val, lam=0.0, tol=1e-8, max_iter=100)
    by_id = dict(zip(res.cell_ids.tolist(), res.cell_data.tolist()))

    assert bad_id in by_id
    assert np.isfinite(by_id[bad_id])
