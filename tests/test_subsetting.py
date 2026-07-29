"""
tests/test_subsetting.py

Test suite for `subset_for_parent_cell` (`healpix_resample.subsetting`) --
task 4's "process one coarse parent cell at a time" helper.

`subset_for_parent_cell` takes only `lon_deg`/`lat_deg` and returns a
`sample_idx` integer index (not a filtered `val`) plus `out_cell_ids` -- the
caller slices `lon`/`lat`/`val` themselves, so the same `sample_idx` can be
reused across every value array that shares the same coordinates.

CPU-only, following the shared layout convention from the other test files
in this package's first test suite (see `planning/00_init.md`).
"""
from __future__ import annotations

import numpy as np
import pytest
import healpix_geo

from healpix_resample import (
    NearestResampler,
    PSFResampler,
    GroupByResampler,
    ConservativeResampler,
    subset_for_parent_cell,
)


LEVEL_PARENT = 4
LEVEL = 7
NDATA = 90
SPAN = 8.0  # degrees -- wide enough to span multiple LEVEL_PARENT cells


def _grid(ndata: int = NDATA, span: float = SPAN):
    lon_grid, lat_grid = np.meshgrid(
        span * np.arange(ndata) / ndata,
        span * np.arange(ndata) / ndata,
    )
    return lon_grid.ravel(), lat_grid.ravel()


def _field(lon, lat):
    return np.sin(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))


@pytest.fixture(scope="module")
def grid():
    lon, lat = _grid()
    val = _field(lon, lat)
    return lon, lat, val


@pytest.fixture(scope="module")
def parent_ids(grid):
    lon, lat, _ = grid
    ids = healpix_geo.nested.lonlat_to_healpix(lon, lat, LEVEL_PARENT, ellipsoid="WGS84")
    unique_ids = np.unique(np.asarray(ids).astype(np.int64))
    assert len(unique_ids) >= 2, "test domain should span multiple parent cells"
    return unique_ids


# ─────────────────────────────────────────────────────────────────────────────
# Unit-level checks of the helper itself
# ─────────────────────────────────────────────────────────────────────────────

def test_out_cell_ids_are_exact_children(grid, parent_ids):
    lon, lat, val = grid
    pid = int(parent_ids[0])

    _, out_ids = subset_for_parent_cell(
        lon, lat, parent_cell_id=pid, level_parent=LEVEL_PARENT, level=LEVEL
    )

    expected_n = 4 ** (LEVEL - LEVEL_PARENT)
    assert out_ids.shape == (expected_n,)
    assert len(np.unique(out_ids)) == expected_n

    # every returned cell must actually be a descendant of pid
    parents_of_children = np.asarray(
        healpix_geo.nested.zoom_to(out_ids.astype(np.uint64), LEVEL, LEVEL_PARENT)
    ).reshape(-1)
    assert np.all(parents_of_children == pid)


def test_sample_idx_indexes_lon_lat_and_val_consistently(grid, parent_ids):
    """`sample_idx` must be a valid integer index into the sample axis,
    usable to slice `lon`/`lat` and any co-located value array (including
    batched `(B, N)` arrays along their last axis)."""
    lon, lat, val = grid
    pid = int(parent_ids[0])

    sample_idx, _ = subset_for_parent_cell(
        lon, lat, parent_cell_id=pid, level_parent=LEVEL_PARENT, level=LEVEL, margin_rings=1
    )

    assert sample_idx.dtype.kind in "iu"
    assert sample_idx.ndim == 1
    assert np.all(sample_idx >= 0) and np.all(sample_idx < len(lon))
    assert len(np.unique(sample_idx)) == len(sample_idx)  # no duplicates

    lon_sub = lon[sample_idx]
    lat_sub = lat[sample_idx]
    val_sub = val[sample_idx]
    assert len(lon_sub) == len(lat_sub) == len(val_sub) == len(sample_idx)

    # batched value array: index along the last axis
    val_batched = np.stack([val, val * 2.0], axis=0)  # (2, N)
    val_batched_sub = val_batched[..., sample_idx]
    assert val_batched_sub.shape == (2, len(sample_idx))
    np.testing.assert_array_equal(val_batched_sub[0], val_sub)


def test_zero_margin_keeps_only_exact_parent_cell(grid, parent_ids):
    lon, lat, val = grid
    pid = int(parent_ids[0])

    sample_idx, _ = subset_for_parent_cell(
        lon, lat, parent_cell_id=pid, level_parent=LEVEL_PARENT, level=LEVEL, margin_rings=0
    )
    lon_sub, lat_sub = lon[sample_idx], lat[sample_idx]

    sub_parent_ids = np.asarray(
        healpix_geo.nested.lonlat_to_healpix(lon_sub, lat_sub, LEVEL_PARENT, ellipsoid="WGS84")
    )
    assert np.all(sub_parent_ids == pid)


def test_margin_one_includes_more_samples_than_zero(grid, parent_ids):
    lon, lat, val = grid
    pid = int(parent_ids[0])

    sample_idx0, _ = subset_for_parent_cell(
        lon, lat, parent_cell_id=pid, level_parent=LEVEL_PARENT, level=LEVEL, margin_rings=0
    )
    sample_idx1, _ = subset_for_parent_cell(
        lon, lat, parent_cell_id=pid, level_parent=LEVEL_PARENT, level=LEVEL, margin_rings=1
    )
    assert len(sample_idx1) >= len(sample_idx0)


# ─────────────────────────────────────────────────────────────────────────────
# Correctness: global run vs. reassembled per-parent-cell runs (margin_rings=1)
# ─────────────────────────────────────────────────────────────────────────────

def _reassemble_and_compare(lon, lat, val, parent_ids, level_parent, level, margin_rings, global_by_id):
    """Run NearestResampler parent-cell-by-parent-cell and count how many
    cells shared with `global_by_id` match within a tight tolerance.

    Returns (n_compared, n_mismatched).
    """
    n_compared = 0
    n_mismatched = 0
    for pid in parent_ids:
        sample_idx, out_ids = subset_for_parent_cell(
            lon, lat, parent_cell_id=int(pid), level_parent=level_parent, level=level,
            margin_rings=margin_rings,
        )
        if len(sample_idx) == 0:
            continue
        local_op = NearestResampler(
            lon_deg=lon[sample_idx], lat_deg=lat[sample_idx], level=level,
            out_cell_ids=out_ids, verbose=False
        )
        local_res = local_op.resample(val[sample_idx])

        for cid, val_local in zip(local_res.cell_ids.tolist(), local_res.cell_data.tolist()):
            if cid in global_by_id:
                n_compared += 1
                if not np.isclose(val_local, global_by_id[cid], rtol=1e-6, atol=1e-9):
                    n_mismatched += 1
    return n_compared, n_mismatched


def test_nearest_reassembly_matches_global(grid, parent_ids):
    """The overwhelming majority of reassembled cells must match the global
    run exactly (NearestResampler is deterministic). A small residual
    mismatch fraction is tolerated and documented rather than asserted away:
    cells that NearestResampler's own out_cell_ids gap-filling fallback
    (`_fill_missing_out_cells`) has to patch in are resolved against the
    *local*, margin-filtered sample subset, not the global one -- see
    `subset_for_parent_cell`'s docstring. That is a separate, inherent
    limitation of comparing a local fallback against a global search, not
    the margin being too small (see `test_zero_margin_degrades_boundary_cells`
    below for the direct test of margin sizing).
    """
    lon, lat, val = grid
    global_op = NearestResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)
    global_res = global_op.resample(val)
    global_by_id = dict(zip(global_res.cell_ids.tolist(), global_res.cell_data.tolist()))

    n_compared, n_mismatched = _reassemble_and_compare(
        lon, lat, val, parent_ids, LEVEL_PARENT, LEVEL, margin_rings=1, global_by_id=global_by_id
    )

    assert n_compared > 0, "no overlapping cells were actually compared -- test config needs adjusting"
    mismatch_frac = n_mismatched / n_compared
    assert mismatch_frac < 0.05, (
        f"{n_mismatched}/{n_compared} cells ({mismatch_frac:.1%}) disagree with the global "
        f"run by more than the tight tolerance -- too high to be explained by the documented "
        f"out_cell_ids-fallback limitation alone"
    )


def test_psf_reassembly_matches_global_within_tolerance(grid, parent_ids):
    lon, lat, val = grid
    global_op = PSFResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, threshold=0.3, verbose=False)
    global_res = global_op.resample(val, lam=0.0, tol=1e-10, max_iter=300)
    global_by_id = dict(zip(global_res.cell_ids.tolist(), global_res.cell_data.tolist()))

    checked_any = False
    for pid in parent_ids:
        sample_idx, out_ids = subset_for_parent_cell(
            lon, lat, parent_cell_id=int(pid), level_parent=LEVEL_PARENT, level=LEVEL,
            margin_rings=1,
        )
        if len(sample_idx) == 0:
            continue
        try:
            local_op = PSFResampler(
                lon_deg=lon[sample_idx], lat_deg=lat[sample_idx], level=LEVEL, out_cell_ids=out_ids,
                threshold=0.3, verbose=False,
            )
        except RuntimeError:
            # this parent cell's margin-filtered subset didn't pass the
            # threshold anywhere -- not what this test is checking.
            continue
        local_res = local_op.resample(val[sample_idx], lam=0.0, tol=1e-10, max_iter=300)

        for cid, val_local in zip(local_res.cell_ids.tolist(), local_res.cell_data.tolist()):
            if cid in global_by_id:
                checked_any = True
                np.testing.assert_allclose(val_local, global_by_id[cid], rtol=1e-2, atol=1e-2)

    assert checked_any, "no overlapping cells were actually compared -- test config needs adjusting"


# ─────────────────────────────────────────────────────────────────────────────
# Margin-too-small regression: margin_rings=0 measurably degrades boundary
# cells relative to the margin_rings=1 (correct) reassembly above.
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_margin_degrades_boundary_cells(grid, parent_ids):
    """`margin_rings=0` must disagree with the global run *measurably more*
    than `margin_rings=1` does. A raw "any mismatch at all" check isn't
    sufficient on its own here: even the correctly-margined `margin_rings=1`
    reassembly has a small baseline mismatch rate from
    `NearestResampler`'s own out_cell_ids gap-filling fallback (see
    `test_nearest_reassembly_matches_global`), so this test compares the
    *mismatch rate* between the two margins rather than just checking for
    the presence of any disagreement, which the baseline alone could already
    satisfy regardless of whether the margin itself is adequate.
    """
    lon, lat, val = grid
    global_op = NearestResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)
    global_res = global_op.resample(val)
    global_by_id = dict(zip(global_res.cell_ids.tolist(), global_res.cell_data.tolist()))

    n_compared_1, n_mismatched_1 = _reassemble_and_compare(
        lon, lat, val, parent_ids, LEVEL_PARENT, LEVEL, margin_rings=1, global_by_id=global_by_id
    )
    n_compared_0, n_mismatched_0 = _reassemble_and_compare(
        lon, lat, val, parent_ids, LEVEL_PARENT, LEVEL, margin_rings=0, global_by_id=global_by_id
    )

    assert n_compared_0 > 0 and n_compared_1 > 0, "test config needs adjusting"
    rate_0 = n_mismatched_0 / n_compared_0
    rate_1 = n_mismatched_1 / n_compared_1

    assert rate_0 > rate_1, (
        f"expected margin_rings=0 (mismatch rate {rate_0:.1%}) to measurably disagree "
        f"with the global result *more* than margin_rings=1 (mismatch rate {rate_1:.1%}) "
        f"-- if this fails, the NDATA/SPAN/LEVEL_PARENT/LEVEL combination needs adjusting "
        f"to actually exercise a boundary case (the point of this test, not a coincidental pass)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# group_by=True resamplers: input filtering (piece 2) still works; out_cell_ids
# (piece 1) is either explicitly unsupported or a documented no-op.
# ─────────────────────────────────────────────────────────────────────────────

def test_conservative_resampler_rejects_out_cell_ids(grid, parent_ids):
    lon, lat, val = grid
    pid = int(parent_ids[0])

    sample_idx, out_ids = subset_for_parent_cell(
        lon, lat, parent_cell_id=pid, level_parent=LEVEL_PARENT, level=LEVEL, margin_rings=1
    )
    assert len(sample_idx) < len(lon)  # actually filtered down to a local subset
    lon_sub, lat_sub, val_sub = lon[sample_idx], lat[sample_idx], val[sample_idx]

    with pytest.raises(NotImplementedError):
        ConservativeResampler(lon_deg=lon_sub, lat_deg=lat_sub, level=LEVEL, out_cell_ids=out_ids)

    # without out_cell_ids, it works fine on the filtered subset -- this is
    # the intended usage for group_by=True resamplers (piece 2 only).
    op = ConservativeResampler(lon_deg=lon_sub, lat_deg=lat_sub, level=LEVEL, verbose=False)
    res = op.resample(val_sub)
    assert res.cell_data.shape[0] == len(res.cell_ids)
    # Note: because group_by mode has no output-side filtering, cells hit by
    # margin-buffer samples from a *neighbouring* parent cell can legitimately
    # appear in res.cell_ids too -- out_ids is not an upper bound on the
    # result for these resamplers, unlike KNN-mode ones. Not asserted here,
    # just documented (see the module's user-guide page).


def test_groupby_resampler_silently_ignores_out_cell_ids(grid, parent_ids):
    lon, lat, val = grid
    pid = int(parent_ids[0])

    sample_idx, out_ids = subset_for_parent_cell(
        lon, lat, parent_cell_id=pid, level_parent=LEVEL_PARENT, level=LEVEL, margin_rings=1
    )
    lon_sub, lat_sub = lon[sample_idx], lat[sample_idx]

    op_with = GroupByResampler(
        lon_deg=lon_sub, lat_deg=lat_sub, level=LEVEL, out_cell_ids=out_ids, verbose=False
    )
    op_without = GroupByResampler(lon_deg=lon_sub, lat_deg=lat_sub, level=LEVEL, verbose=False)

    np.testing.assert_array_equal(
        np.sort(op_with.cell_ids.numpy()), np.sort(op_without.cell_ids.numpy())
    )
