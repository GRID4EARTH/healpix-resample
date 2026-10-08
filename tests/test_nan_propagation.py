"""
tests/test_nan_propagation.py

Shared regression tests for the fix in `KNeighborsResampler.__init__`
(`knn.py`), which is a base-class fix affecting every resampler that goes
through the default (non-`group_by`) KNN construction path -- not just
`BilinearResampler`, where the symptom was originally reported. See
`planning/03_bilinear_nan_investigation.md` and `tests/test_bilinear.py`'s
module docstring for the full investigation trace.

Before the fix, a HEALPix cell could pass `healpix_weighted_nearest`'s wide
-radius retention threshold without ever being selected by any sample's own
narrower Npt-nearest-cell search, leaving it with zero real links in
`self.hi`/`M`. This affected `BilinearResampler` and the default (no
`out_cell_ids`) path of `PSFResampler` identically (both read back a
silently-wrong `0.0`), and `NearestResampler` via a related but differently
-shaped symptom (its own `-1` sentinel for exactly this case getting
`clamp(min=0)`'d into sample 0's value instead of a real reconstruction).

These tests exercise the same two properties -- "no orphaned/zero-link
cells" and "all-NaN input gives all-NaN output" -- across all three affected
resamplers with a shared synthetic grid.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from healpix_resample import BilinearResampler, NearestResampler, PSFResampler


NDATA = 60
LEVEL = 10


def _grid(ndata: int = NDATA):
    lon_grid, lat_grid = np.meshgrid(
        0.3 * np.arange(ndata) / ndata,
        0.3 * np.arange(ndata) / ndata,
    )
    return lon_grid.ravel(), lat_grid.ravel()


@pytest.fixture(scope="module")
def grid():
    return _grid()


RESAMPLER_FACTORIES = {
    "bilinear": lambda lon, lat: BilinearResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False),
    "nearest": lambda lon, lat: NearestResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False),
    "psf": lambda lon, lat: PSFResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, threshold=0.3, verbose=False),
}


@pytest.mark.parametrize("name", list(RESAMPLER_FACTORIES.keys()))
def test_no_orphaned_cells(grid, name):
    """Every retained cell must be reachable by at least one sample's own
    KNN link (`self.hi`) -- the root cause fixed in `KNeighborsResampler.__init__`."""
    lon, lat = grid
    op = RESAMPLER_FACTORIES[name](lon, lat)

    reachable = torch.zeros(op.K, dtype=torch.bool)
    valid_hi = op.hi[op.hi >= 0]
    if valid_hi.numel() > 0:
        reachable[valid_hi] = True

    n_orphaned = int((~reachable).sum())
    assert n_orphaned == 0, f"{name}: {n_orphaned}/{op.K} cells have zero real links"


@pytest.mark.parametrize("name", list(RESAMPLER_FACTORIES.keys()))
def test_all_nan_input_is_fully_nan_output(grid, name):
    lon, lat = grid
    op = RESAMPLER_FACTORIES[name](lon, lat)

    val_nan = np.full(lon.shape, np.nan)
    res = op.resample(val_nan)

    assert np.all(np.isnan(res.cell_data)), f"{name}: not fully NaN on all-NaN input"


@pytest.mark.parametrize("name", ["bilinear", "nearest"])
def test_finite_input_never_silently_zero(grid, name):
    """On a smooth, essentially-never-exactly-zero finite field, no cell
    should read back as exactly 0.0 -- that was the symptom of an orphaned
    cell, independent of NaN. (Excluded for PSFResampler: its CG solve can
    legitimately produce values arbitrarily close to 0 even for well
    -supported cells, so this specific check isn't a meaningful signal there
    -- `test_no_orphaned_cells` above is the direct, unambiguous check for
    all three.)"""
    lon, lat = grid
    op = RESAMPLER_FACTORIES[name](lon, lat)

    # `lon` itself is exactly 0.0 along the whole first grid column (40 of the
    # 1600 samples), and NearestResampler copies its nearest sample's value
    # verbatim -- so cells near that edge read back an entirely legitimate
    # 0.0. Offset the field so it is bounded away from zero, which is what
    # "essentially-never-exactly-zero" above intends; then any exact 0.0 really
    # does indicate an orphaned cell.
    res = op.resample(lon + 1.0)
    assert not np.any(res.cell_data == 0.0), f"{name}: found a suspicious exact-0.0 cell"
