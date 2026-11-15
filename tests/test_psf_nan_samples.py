"""
tests/test_psf_nan_samples.py

NaN samples carry no weight in the fit of `PSFResampler.resample`: the cells
next to them are those of a resampler that never saw them (before, the NaN
samples were fitted as 0 and pulled the neighbouring cells towards 0).

CPU-only, following `tests/test_psf.py`.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import healpix_geo.nested as hgn
import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from healpix_resample import PSFResampler

SP, N, R = 500.0, 50, 6371000.0          # sample spacing [m], N x N samples
LEVEL = 14
C = 300.0                                  # field far from 0 (kelvin-like)


def _lonlat(x, y):
    return 3 + np.degrees(x / (R * np.cos(np.radians(45)))), 45 + np.degrees(y / R)


def _truth(x, y):
    return C * (1 + 0.1 * x / (N * SP))


_KW = dict(level=LEVEL, sigma_m=1.3 * SP / 1.1774, Npt=14, threshold=0.1)


@pytest.fixture(scope="module")
def samples():
    x, y = (a.ravel() * SP for a in np.meshgrid(np.arange(N), np.arange(N)))
    return x, y


@pytest.fixture(scope="module")
def op(samples):
    return PSFResampler(*_lonlat(*samples), **_KW)


@pytest.fixture(scope="module")
def cells(op):
    ids = np.asarray(op.cell_ids)
    clon, clat = hgn.healpix_to_lonlat(ids.astype("uint64"), LEVEL, "WGS84")
    cx = np.radians(clon - 3) * R * np.cos(np.radians(45))
    cy = np.radians(clat - 45) * R
    inner = (cx > 2 * SP) & (cx < (N - 3) * SP) & (cy > 2 * SP) & (cy < (N - 3) * SP)
    return ids, cx, cy, inner


def _bad_east(samples):
    return samples[0] > N * SP / 2 - SP / 2


def _bad_isolated(samples):
    return np.isin(np.arange(N * N), [b * N + a for a in (12, 25, 38) for b in (12, 25, 38)])


def _field(samples, bad):
    v = _truth(*samples).astype("f4")
    v[bad] = np.nan
    return v


def _share(op, bad):
    valid = torch.tensor(~bad, dtype=torch.float32)[None]
    ones = torch.ones_like(valid)
    return ((valid @ op.M) / (ones @ op.M))[0].numpy()


def _near_nan(samples, cells, bad, op, lo=0.0, hi=3.0):
    """Inner cells kept by min_valid_weight, within [lo, hi) spacings of a NaN sample."""
    _, cx, cy, inner = cells
    d = cKDTree(np.c_[samples[0][bad], samples[1][bad]] / SP).query(np.c_[cx, cy] / SP)[0]
    return inner & (_share(op, bad) >= 0.5) & (d >= lo) & (d < hi)


def _removed_first(samples, cells, bad, **resample_kw):
    """The cells of a resampler built from the valid samples only."""
    ids = cells[0]
    x, y = samples[0][~bad], samples[1][~bad]
    ref_op = PSFResampler(*_lonlat(x, y), out_cell_ids=ids, **_KW)
    out = np.full(ids.size, np.nan)
    v = _truth(x, y).astype("f4")
    out[np.isin(ids, ref_op.cell_ids)] = ref_op.resample(v, **resample_kw).cell_data
    return out


def _err(out, cells, sel):
    """Max |out - known field| on the cells ``sel``, in units of the field."""
    return np.nanmax(np.abs(out - _truth(cells[1], cells[2]))[sel]) / C


# --- (i) accuracy next to the NaN samples ------------------------------------

@pytest.mark.parametrize("pattern", [_bad_east, _bad_isolated])
def test_cells_next_to_nan_samples_match_the_nan_removed_result(op, samples, cells, pattern):
    bad = pattern(samples)
    out = np.asarray(op.resample(_field(samples, bad), lam=0.1).cell_data, dtype="f8")
    ref = _removed_first(samples, cells, bad, lam=0.1)
    sel = _near_nan(samples, cells, bad, op)
    assert sel.sum() > 50
    # before the change: 4 to 170 % of the field in this selection
    assert np.nanmax(np.abs(out - ref)[sel]) / C < 1e-3
    assert _err(out, cells, sel) < 1e-3


# --- (ii) no NaN: unchanged --------------------------------------------------

# Last commit of main before `min_valid_weight` (the one `test_psf_nan_support.py`
# uses): a fixed commit that stays in the history, whatever happens to the branches.
# Input without NaN samples goes the same way in it as here. It needs the history
# (CI checks out with fetch-depth 0).
_REFERENCE_COMMIT = "2be7c6270aa8b2345b231b26f5aeabaa71555276"


@pytest.fixture(scope="module")
def base_cls():
    root = Path(__file__).resolve().parents[1]
    try:
        src = subprocess.run(
            ["git", "show", f"{_REFERENCE_COMMIT}:healpix_resample/psf.py"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        if os.environ.get("CI"):
            pytest.fail(f"commit {_REFERENCE_COMMIT} not available: CI must fetch the history")
        pytest.skip(f"commit {_REFERENCE_COMMIT} not available (shallow clone)")
    spec = importlib.util.spec_from_loader("_psf_base_reference", loader=None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    exec(compile(src, "psf_base.py", "exec"), mod.__dict__)
    return mod.PSFResampler


@pytest.mark.parametrize(
    "kwargs",
    [dict(lam=0.1), dict(lam=0.1, jacobi=True), dict(lam=0.1, conservative=True),
     dict(lam=0.1, clip=(0.0, None)), dict(lam=0.0)],
)
def test_input_without_nan_is_bit_for_bit_the_reference(op, base_cls, samples, kwargs):
    base_op = base_cls(*_lonlat(*samples), **_KW)
    v = _field(samples, np.zeros(N * N, dtype=bool))
    for val in (v, np.stack([v, v[::-1].copy()])):    # one row, and a batch
        new = op.resample(val, **kwargs)
        old = base_op.resample(val, **kwargs)
        np.testing.assert_array_equal(new.cell_data, old.cell_data)
        assert int(new.cg_niters) == int(old.cg_niters)


# --- (iii) rows with different NaN patterns ----------------------------------

def test_rows_with_different_nan_patterns(op, samples, cells):
    # Each row is the single-row result to the solver tolerance, not bit for
    # bit: the CG stopping test is shared by the rows of a call, so the number
    # of iterations (and the last digits) depend on the other rows.
    bads = [_bad_east(samples), _bad_isolated(samples), np.zeros(N * N, dtype=bool)]
    batch = np.stack([_field(samples, b) for b in bads])
    out = np.asarray(op.resample(batch, lam=0.1).cell_data, dtype="f8")
    for row, bad in zip(out, bads):
        single = np.asarray(op.resample(_field(samples, bad), lam=0.1).cell_data, dtype="f8")
        assert np.array_equal(np.isnan(row), np.isnan(single))
        assert np.nanmax(np.abs(row - single)) / C < 1e-5        # solver tolerance
        if bad.any():
            assert _err(row, cells, _near_nan(samples, cells, bad, op)) < 1e-3


# --- (iv) bounded fit, (v) conservative and Jacobi ---------------------------

@pytest.mark.parametrize("kwargs", [
    dict(clip=(0.0, None)),
    dict(clip=("input", "input")),
    dict(conservative=True),
    dict(jacobi=True),
    dict(clip=(0.0, None), conservative=True, jacobi=True),
])
@pytest.mark.parametrize("pattern", [_bad_east, _bad_isolated])
def test_paths_with_nan_samples(op, samples, cells, kwargs, pattern):
    bad = pattern(samples)
    out = np.asarray(op.resample(_field(samples, bad), lam=0.1, **kwargs).cell_data, dtype="f8")
    sel = _near_nan(samples, cells, bad, op)
    assert np.isfinite(out[sel]).all()
    # conservative also shifts every cell to carry the valid mean (1.4e-3 observed)
    assert _err(out, cells, sel) < (2e-3 if kwargs.get("conservative") else 1e-3)


def test_jacobi_agrees_with_plain_solve_with_nan_samples(op, samples):
    # one diagonal per row: the same solution to the tolerance
    batch = np.stack([_field(samples, _bad_east(samples)), _field(samples, _bad_isolated(samples))])
    kw = dict(lam=0.1, tol=1e-10, max_iter=300)
    a = np.asarray(op.resample(batch, **kw).cell_data, dtype="f8")
    b = np.asarray(op.resample(batch, jacobi=True, **kw).cell_data, dtype="f8")
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.nanmax(np.abs(a - b)) / C < 1e-4


def test_masked_cg_diagonal_matches_dense(op, samples):
    bad = _bad_isolated(samples)
    mask = torch.tensor(~bad, dtype=torch.float32)[None]
    M, MT = op.M.to_dense().double().numpy(), op.MT.to_dense().double().numpy()
    expected = np.einsum("kn,nk,n->k", MT, M, (~bad).astype("f8"))
    got = op._cg_diagonal(mask)[0].double().numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-7)
    assert op._cg_diagonal().shape == (op.K,)


# --- (vi) a tiny problem against dense linear algebra -------------------------
# A smaller grid (K = 1026 cells), float64, a rough noisy field: the CG solution
# is compared with the solution of the normal equations built with dense
# matrices. This is what pins the operator: on a smooth field the fit is so
# small that a wrong operator is hidden.

NT = 24


@pytest.fixture(scope="module")
def tiny():
    x, y = (a.ravel() * SP for a in np.meshgrid(np.arange(NT), np.arange(NT)))
    op = PSFResampler(*_lonlat(x, y), dtype=torch.float64, **_KW)
    rng = np.random.default_rng(0)
    field = C * (1 + 0.2 * np.sin(2 * np.pi * x / (5 * SP)) * np.sin(2 * np.pi * y / (7 * SP)))
    field = field + 20 * rng.normal(size=x.size)
    bads = {
        "east": x > NT * SP / 2 - SP / 2,
        "random": rng.random(x.size) < 0.3,
    }
    dense = dict(M=op.M.to_dense().numpy(), MT=op.MT.to_dense().numpy(), area=op.area.numpy(),
                 cw=op.cell_weight.numpy())
    return op, field, bads, dense


def _dense_fit(dense, y, bad, lam):
    """x_ref, delta of the masked fit with dense matrices (row-vector convention),
    and the operator ``A`` (K, K) with ``d @ A = b``."""
    M, MT = dense["M"], dense["MT"]
    v = (~bad).astype("f8")
    share = (v @ M) / M.sum(0)
    xref = (np.where(bad, 0.0, y) @ M) / np.where(share > 0, share, 1.0)
    A = (MT * v) @ M + lam * np.eye(M.shape[1])
    b = ((np.where(bad, 0.0, y) - xref @ MT) * v) @ M
    return share, xref, A, b


@pytest.mark.parametrize("lam", [0.1, 1.0])
@pytest.mark.parametrize("pattern", ["east", "random"])
def test_masked_fit_equals_the_dense_solution(tiny, pattern, lam):
    op, field, bads, dense = tiny
    bad = bads[pattern]
    val = np.where(bad, np.nan, field)
    out = np.asarray(op.resample(val, lam=lam, tol=1e-13, max_iter=20000,
                                 min_valid_weight=0.0).cell_data)
    share, xref, A, b = _dense_fit(dense, field, bad, lam)
    live = share > 0
    # d @ A = b, with the cells that carry no valid weight left out (b = 0, A = lam I there)
    d = np.linalg.solve(A.T, b)
    np.testing.assert_allclose(out[live], (xref + d)[live], rtol=0, atol=1e-7 * C)
    assert np.isnan(out[~live]).all()


def _dense_conservative(dense, y, bad, lam, live):
    """The conservative fit with dense matrices: ``x_ref + d + mu w`` with
    ``w @ A = Dx^-1 live`` and ``mu`` such that the live cells sum to the target."""
    share, xref, A, b = _dense_fit(dense, y, bad, lam)
    h = xref + np.linalg.solve(A.T, b)
    w = np.linalg.solve(A.T, live / dense["cw"] * live)
    area = dense["area"]
    target = (np.where(bad, 0.0, y) * area).sum() / (area * ~bad).sum() * live.sum()
    mu = (target - (h * live).sum()) / (w * live).sum()
    return h + mu * w, target


@pytest.mark.parametrize("pattern", ["east", "random"])
def test_conservative_with_nan_samples_equals_the_dense_solution(tiny, pattern):
    # a batch of two rows with different NaN patterns, one call
    op, field, bads, dense = tiny
    rows = [pattern, "random" if pattern == "east" else "east"]
    batch = np.stack([np.where(bads[r], np.nan, field) for r in rows])
    out = np.asarray(op.resample(batch, lam=0.1, conservative=True, tol=1e-13, max_iter=20000,
                                 min_valid_weight=0.0).cell_data)
    for row, r in zip(out, rows):
        share = _dense_fit(dense, field, bads[r], 0.1)[0]
        live = (share > 0).astype("f8")
        expected, target = _dense_conservative(dense, field, bads[r], 0.1, live)
        np.testing.assert_allclose(row[live > 0], expected[live > 0], rtol=0, atol=1e-6 * C)
        assert np.nansum(row) == pytest.approx(target, rel=1e-9)


# --- (vii) conservative: the returned cells meet the target ------------------

def _target_mean(op, bad, val):
    area = op.area.numpy().astype("f8")
    return np.sum(np.where(bad, 0.0, val) * area) / np.sum(area * ~bad)


@pytest.mark.parametrize("min_valid_weight", [0.5, 0.0])
@pytest.mark.parametrize("lam", [0.1, 0.0])
def test_conservative_returned_cells_carry_the_target(op, samples, lam, min_valid_weight):
    # the sum of the returned cells is (area-weighted valid mean) x (number of
    # returned cells), also with the default min_valid_weight: the cells that
    # are set to NaN do not take part in the constraint
    bads = [_bad_east(samples), _bad_isolated(samples)]
    vals = [_field(samples, b) for b in bads]
    out = np.asarray(op.resample(np.stack(vals), lam=lam, conservative=True,
                                 min_valid_weight=min_valid_weight).cell_data, dtype="f8")
    for row, bad, val in zip(out, bads, vals):
        n = np.isfinite(row).sum()
        assert n > 0
        assert row[np.isfinite(row)].sum() == pytest.approx(_target_mean(op, bad, val) * n, rel=1e-5)
    assert np.isnan(out[0]).any()      # the half plane leaves cells without data


def test_conservative_target_is_the_valid_mean_for_a_single_row(op, samples):
    bad = _bad_east(samples)
    val = _field(samples, bad)
    row = np.asarray(op.resample(val, lam=0.1, conservative=True).cell_data, dtype="f8")
    n = np.isfinite(row).sum()
    assert np.nanmean(row) == pytest.approx(_target_mean(op, bad, val), rel=1e-5)
    assert n < op.K


# --- (viii) an all-NaN row -----------------------------------------------------

@pytest.mark.parametrize("kwargs", [dict(), dict(clip=(0.0, None)), dict(conservative=True), dict(jacobi=True)])
def test_all_nan_row_is_nan_and_does_not_raise(op, samples, kwargs):
    batch = np.stack([np.full(N * N, np.nan, dtype="f4"), _field(samples, _bad_isolated(samples))])
    out = np.asarray(op.resample(batch, lam=0.1, **kwargs).cell_data)
    assert np.isnan(out[0]).all()
    assert np.isfinite(out[1]).any()
