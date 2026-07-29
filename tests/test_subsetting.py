"""
tests/test_subsetting.py

Test suite for `subset_for_parent_cell` (`healpix_resample.subsetting`) --
task 4's "process one coarse parent cell at a time" helper.

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

    _, _, _, out_ids = subset_for_parent_cell(
        lon, lat, val, parent_cell_id=pid, level_parent=LEVEL_PARENT, level=LEVEL
    )

    expected_n = 4 ** (LEVEL - LEVEL_PARENT)
    assert out_ids.shape == (expected_n,)
    assert len(np.unique(out_ids)) == expected_n

    # every returned cell must actually be a descendant of pid
    parents_of_children = np.asarray(
        healpix_geo.nested.zoom_to(out_ids.astype(np.uint64), LEVEL, LEVEL_PARENT)
    ).reshape(-1)
    assert np.all(parents_of_children == pid)


def test_zero_margin_keeps_only_exact_parent_cell(grid, parent_ids):
    lon, lat, val = grid
    pid = int(parent_ids[0])

    lon_sub, lat_sub, val_sub, _ = subset_for_parent_cell(
        lon, lat, val, parent_cell_id=pid, level_parent=LEVEL_PARENT, level=LEVEL, margin_rings=0
    )

    sub_parent_ids = np.asarray(
        healpix_geo.nested.lonlat_to_healpix(lon_sub, lat_sub, LEVEL_PARENT, ellipsoid="WGS84")
    )
    assert np.all(sub_parent_ids == pid)
    assert len(lon_sub) == len(lat_sub) == val_sub.shape[-1]


def test_margin_one_includes_more_samples_than_zero(grid, parent_ids):
    lon, lat, val = grid
    pid = int(parent_ids[0])

    lon_sub0, _, _, _ = subset_for_parent_cell(
        lon, lat, val, parent_cell_id=pid, level_parent=LEVEL_PARENT, level=LEVEL, margin_rings=0
    )
    lon_sub1, _, _, _ = subset_for_parent_cell(
        lon, lat, val, parent_cell_id=pid, level_parent=LEVEL_PARENT, level=LEVEL, margin_rings=1
    )
    assert len(lon_sub1) >= len(lon_sub0)


# ─────────────────────────────────────────────────────────────────────────────
# Correctness: global run vs. reassembled per-parent-cell runs (margin_rings=1)
# ─────────────────────────────────────────────────────────────────────────────

def test_nearest_reassembly_matches_global(grid, parent_ids):
    lon, lat, val = grid
    global_op = NearestResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)
    global_res = global_op.resample(val)
    global_by_id = dict(zip(global_res.cell_ids.tolist(), global_res.cell_data.tolist()))

    checked_any = False
    for pid in parent_ids:
        lon_sub, lat_sub, val_sub, out_ids = subset_for_parent_cell(
            lon, lat, val, parent_cell_id=int(pid), level_parent=LEVEL_PARENT, level=LEVEL,
            margin_rings=1,
        )
        if len(lon_sub) == 0:
            continue
        local_op = NearestResampler(
            lon_deg=lon_sub, lat_deg=lat_sub, level=LEVEL, out_cell_ids=out_ids, verbose=False
        )
        local_res = local_op.resample(val_sub)

        for cid, val_local in zip(local_res.cell_ids.tolist(), local_res.cell_data.tolist()):
            if cid in global_by_id:
                checked_any = True
                np.testing.assert_allclose(val_local, global_by_id[cid], rtol=1e-6, atol=1e-9)

    assert checked_any, "no overlapping cells were actually compared -- test config needs adjusting"


def test_psf_reassembly_matches_global_within_tolerance(grid, parent_ids):
    lon, lat, val = grid
    global_op = PSFResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, threshold=0.3, verbose=False)
    global_res = global_op.resample(val, lam=0.0, tol=1e-10, max_iter=300)
    global_by_id = dict(zip(global_res.cell_ids.tolist(), global_res.cell_data.tolist()))

    checked_any = False
    for pid in parent_ids:
        lon_sub, lat_sub, val_sub, out_ids = subset_for_parent_cell(
            lon, lat, val, parent_cell_id=int(pid), level_parent=LEVEL_PARENT, level=LEVEL,
            margin_rings=1,
        )
        if len(lon_sub) == 0:
            continue
        try:
            local_op = PSFResampler(
                lon_deg=lon_sub, lat_deg=lat_sub, level=LEVEL, out_cell_ids=out_ids,
                threshold=0.3, verbose=False,
            )
        except RuntimeError:
            # this parent cell's margin-filtered subset didn't pass the
            # threshold anywhere -- not what this test is checking.
            continue
        local_res = local_op.resample(val_sub, lam=0.0, tol=1e-10, max_iter=300)

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
    lon, lat, val = grid
    global_op = NearestResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)
    global_res = global_op.resample(val)
    global_by_id = dict(zip(global_res.cell_ids.tolist(), global_res.cell_data.tolist()))

    any_mismatch = False
    for pid in parent_ids:
        lon_sub0, lat_sub0, val_sub0, out_ids0 = subset_for_parent_cell(
            lon, lat, val, parent_cell_id=int(pid), level_parent=LEVEL_PARENT, level=LEVEL,
            margin_rings=0,
        )
        if len(lon_sub0) == 0:
            continue
        local_op0 = NearestResampler(
            lon_deg=lon_sub0, lat_deg=lat_sub0, level=LEVEL, out_cell_ids=out_ids0, verbose=False
        )
        local_res0 = local_op0.resample(val_sub0)

        for cid, val_local in zip(local_res0.cell_ids.tolist(), local_res0.cell_data.tolist()):
            if cid in global_by_id and not np.isclose(val_local, global_by_id[cid], rtol=1e-6, atol=1e-9):
                any_mismatch = True
                break
        if any_mismatch:
            break

    assert any_mismatch, (
        "expected margin_rings=0 to measurably disagree with the global result on at "
        "least one boundary cell somewhere in this domain -- if this fails, the "
        "NDATA/SPAN/LEVEL_PARENT/LEVEL combination needs adjusting to actually exercise "
        "a boundary case (the point of this test, not a coincidental pass)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# group_by=True resamplers: input filtering (piece 2) still works; out_cell_ids
# (piece 1) is either explicitly unsupported or a documented no-op.
# ─────────────────────────────────────────────────────────────────────────────

def test_conservative_resampler_rejects_out_cell_ids(grid, parent_ids):
    lon, lat, val = grid
    pid = int(parent_ids[0])

    lon_sub, lat_sub, val_sub, out_ids = subset_for_parent_cell(
        lon, lat, val, parent_cell_id=pid, level_parent=LEVEL_PARENT, level=LEVEL, margin_rings=1
    )
    assert len(lon_sub) < len(lon)  # actually filtered down to a local subset

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

    lon_sub, lat_sub, val_sub, out_ids = subset_for_parent_cell(
        lon, lat, val, parent_cell_id=pid, level_parent=LEVEL_PARENT, level=LEVEL, margin_rings=1
    )

    op_with = GroupByResampler(
        lon_deg=lon_sub, lat_deg=lat_sub, level=LEVEL, out_cell_ids=out_ids, verbose=False
    )
    op_without = GroupByResampler(lon_deg=lon_sub, lat_deg=lat_sub, level=LEVEL, verbose=False)

    np.testing.assert_array_equal(
        np.sort(op_with.cell_ids.numpy()), np.sort(op_without.cell_ids.numpy())
    )
