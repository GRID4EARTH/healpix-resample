"""
tests/test_bilinear.py

Test suite for `BilinearResampler` NaN behaviour, following up on the
investigation in `planning/03_bilinear_nan_investigation.md`.

Summary of the investigation (see conversation/PR for the full trace): the
reported symptom -- an all-NaN input to `BilinearResampler.resample()`
producing some non-NaN output cells -- was **not** a CSR-vs-COO sparse-matmul
NaN-propagation bug (the leading hypothesis in the planning doc). A minimal
toy repro (`test_coo_and_csr_agree_on_nan_propagation` below) showed both
sparse formats propagate NaN identically and correctly.

The real cause: `healpix_weighted_nearest` (`knn.py`) decides which cells to
*retain* via a wide-radius accumulated-weight threshold test, independent of
the narrower per-sample search that actually links each sample to its `Npt`
nearest retained cells (`self.hi`). A cell can pass the wide-radius retention
test while never being selected by any sample's own nearest-cell search,
leaving it an empty column in `M` -- which correctly (per sparse semantics)
returns exactly `0.0` for *any* input, not just NaN. Fixed in
`KNeighborsResampler.__init__` by pruning such "orphaned" cells before any
subclass's `comp_matrix()` runs (scoped to the default `out_cell_ids=None`
case). See the fix's own comment block in `knn.py` for the full reasoning.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from healpix_resample import BilinearResampler


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


# ─────────────────────────────────────────────────────────────────────────────
# The minimal repro that refuted the CSR-vs-COO hypothesis -- kept as a
# regression test in case a future PyTorch version reintroduces a real
# format-dependent discrepancy.
# ─────────────────────────────────────────────────────────────────────────────

def test_coo_and_csr_agree_on_nan_propagation():
    N, K = 5, 3
    rows = torch.tensor([0, 1, 2, 3, 4])
    cols = torch.tensor([0, 0, 1, 1, 2])
    vals = torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0])
    coo = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (N, K)).coalesce()
    csr = coo.to_sparse_csr()

    y = torch.tensor([[float("nan"), 1.0, 1.0, 1.0, 1.0]])
    out_coo = y @ coo
    out_csr = y @ csr

    # column 0 (touches the NaN sample) -> NaN in both; columns 1, 2 -> finite
    torch.testing.assert_close(out_coo.isnan(), out_csr.isnan())
    assert bool(out_coo[0, 0].isnan())
    assert bool(out_csr[0, 0].isnan())
    assert torch.isfinite(out_coo[0, 1:]).all()
    assert torch.isfinite(out_csr[0, 1:]).all()


# ─────────────────────────────────────────────────────────────────────────────
# The actual bug: no retained cell should be "orphaned" (zero real links)
# ─────────────────────────────────────────────────────────────────────────────

def test_no_orphaned_cells_in_M(grid):
    """Every cell in `cell_ids` must have at least one real (i, k) link in M
    -- i.e. no column of M is entirely empty. Before the `knn.py` fix, cells
    that passed the wide-radius retention threshold without being selected by
    any sample's own Npt-nearest search would silently read back as 0.0."""
    lon, lat = grid
    op = BilinearResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    M_coo = op.M.to_sparse_coo().coalesce()
    cols_with_data = torch.unique(M_coo.indices()[1])
    assert cols_with_data.numel() == op.K, (
        f"{op.K - cols_with_data.numel()} of {op.K} cells have zero entries in M"
    )


def test_all_nan_input_is_fully_nan_output(grid):
    lon, lat = grid
    op = BilinearResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    val_nan = np.full(lon.shape, np.nan)
    res = op.resample(val_nan)

    assert np.all(np.isnan(res.cell_data))


def test_all_nan_input_on_finite_data_is_not_silently_zero(grid):
    """Regression guard for the actual bug: on a perfectly finite, non-zero
    input, no cell should come back as exactly 0.0 -- that was the symptom of
    an orphaned (zero-link) cell, unrelated to NaN at all."""
    lon, lat = grid
    op = BilinearResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    res = op.resample(lon)  # smooth, essentially never exactly 0 over this domain
    assert not np.any(res.cell_data == 0.0)


def test_single_nan_sample_only_affects_its_own_cells(grid):
    lon, lat = grid
    op = BilinearResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    bad_idx = len(lon) // 2
    val = np.zeros(lon.shape)
    val[bad_idx] = np.nan

    res = op.resample(val)

    touched = np.unique(op.hi[bad_idx].numpy())
    touched = touched[touched >= 0]

    assert np.all(np.isnan(res.cell_data[touched]))
    n_nan_total = int(np.isnan(res.cell_data).sum())
    assert n_nan_total == len(touched)


# ─────────────────────────────────────────────────────────────────────────────
# conservative=True (issue #44: "conservative bi-linear is missing")
#
# Each sample's own (area-weighted) value is redistributed across its 4
# nearest cells via self.M_cons -- the same inverse-distance weights as the
# default interpolation path, but normalized so each *sample's* own weights
# sum to 1 instead of being normalized per output cell. Guarantee under test:
# sum_k hval[k] == sum_i (valid i) val[i] * area[i] exactly.
# ─────────────────────────────────────────────────────────────────────────────

def _field(lon, lat):
    return np.sin(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))


def test_conservative_false_is_default_and_unchanged(grid):
    lon, lat = grid
    val = _field(lon, lat)
    op = BilinearResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    res_default = op.resample(val)
    res_explicit = op.resample(val, conservative=False)

    np.testing.assert_array_equal(res_default.cell_data, res_explicit.cell_data)


def test_conservative_conserves_exact_sum_uniform_area(grid):
    lon, lat = grid
    val = _field(lon, lat)
    op = BilinearResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    res = op.resample(val, conservative=True)

    np.testing.assert_allclose(float(np.sum(res.cell_data)), float(np.sum(val)), rtol=1e-5, atol=1e-6)


def test_conservative_conserves_exact_sum_with_area(grid):
    lon, lat = grid
    val = _field(lon, lat)
    rng = np.random.default_rng(0)
    area = rng.uniform(0.5, 2.0, size=lon.shape)

    op = BilinearResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False, area=area)
    res = op.resample(val, conservative=True)

    expected = float(np.sum(val * area))
    np.testing.assert_allclose(float(np.sum(res.cell_data)), expected, rtol=1e-5, atol=1e-6)


def test_conservative_excludes_nan_sample_and_stays_finite(grid):
    """A single NaN sample must not propagate to any cell (unlike the
    interpolation path, where cells the NaN sample links to go nan): its
    value AND area are simply excluded from the conserved total."""
    lon, lat = grid
    val = _field(lon, lat)
    rng = np.random.default_rng(1)
    area = rng.uniform(0.5, 2.0, size=lon.shape)

    bad_idx = len(lon) // 2
    val_nan = val.copy()
    val_nan[bad_idx] = np.nan

    op = BilinearResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False, area=area)
    res = op.resample(val_nan, conservative=True)

    assert np.all(np.isfinite(res.cell_data))

    valid = np.ones(lon.shape, dtype=bool)
    valid[bad_idx] = False
    expected = float(np.sum(val[valid] * area[valid]))
    np.testing.assert_allclose(float(np.sum(res.cell_data)), expected, rtol=1e-5, atol=1e-6)


def test_conservative_all_nan_row_is_all_nan(grid):
    lon, lat = grid
    op = BilinearResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    val_nan = np.full(lon.shape, np.nan)
    res = op.resample(val_nan, conservative=True)

    assert np.all(np.isnan(res.cell_data))


def test_conservative_batched_scales_linearly(grid):
    lon, lat = grid
    val = _field(lon, lat)
    op = BilinearResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    val_batch = np.stack([val, val * 2.0], axis=0)  # (2, N)
    res = op.resample(val_batch, conservative=True)

    assert res.cell_data.shape[0] == 2
    np.testing.assert_allclose(res.cell_data[1], res.cell_data[0] * 2.0, rtol=1e-5, atol=1e-8)
