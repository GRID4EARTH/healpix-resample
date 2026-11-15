"""
tests/test_psf_nan_support.py

Output cells whose kernel support holds only NaN samples (or mostly NaN
samples) are returned as NaN instead of being solved from zero-filled
samples (#119): `PSFResampler.resample(..., min_valid_weight=...)`.

CPU-only, following `tests/test_psf.py`.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from healpix_resample import PSFResampler


LEVEL = 12


@pytest.fixture(scope="module")
def grid():
    lon, lat = np.meshgrid(np.linspace(0.0, 0.5, 120), np.linspace(45.0, 45.5, 120))
    return lon.ravel(), lat.ravel()


@pytest.fixture(scope="module")
def op(grid):
    lon, lat = grid
    return PSFResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, threshold=0.3)


@pytest.fixture(scope="module")
def half_nan(grid):
    lon, _ = grid
    val = np.full(lon.shape, 0.15, dtype="f4")
    val[lon > 0.25] = np.nan
    return val


def _cell_lon(op):
    import healpy as hp

    lon, _ = hp.pix2ang(2**LEVEL, np.asarray(op.cell_ids), nest=True, lonlat=True)
    return (lon + 180.0) % 360.0 - 180.0


def _cell_lon_any(op):
    try:
        return _cell_lon(op)
    except ImportError:  # pragma: no cover
        pytest.skip("healpy not available")


@pytest.mark.parametrize("clip", [None, ("input", "input"), (-0.1, None)])
def test_cells_over_nan_block_are_nan(op, half_nan, clip):
    out = np.asarray(op.resample(half_nan, lam=0.1, clip=clip).cell_data)
    lon = _cell_lon_any(op)
    assert np.isnan(out[lon > 0.30]).all()
    inside = lon < 0.15
    assert np.isfinite(out[inside]).all()
    np.testing.assert_allclose(out[inside], 0.15, atol=0.02)


def test_min_valid_weight_gives_nested_nan_sets(op, half_nan):
    nan_sets = {
        m: np.isnan(np.asarray(op.resample(half_nan, lam=0.1, min_valid_weight=m).cell_data))
        for m in (0.0, 0.5, 1.0)
    }
    assert nan_sets[0.0].any()
    assert (nan_sets[0.0] <= nan_sets[0.5]).all()
    assert (nan_sets[0.5] <= nan_sets[1.0]).all()
    assert nan_sets[0.0].sum() < nan_sets[0.5].sum() < nan_sets[1.0].sum()


def test_mixed_batch_masks_per_row(op, half_nan):
    full = np.full_like(half_nan, 0.15)
    allnan = np.full_like(half_nan, np.nan)
    out = np.asarray(op.resample(np.stack([full, half_nan, allnan]), lam=0.1).cell_data)
    single = np.asarray(op.resample(half_nan, lam=0.1).cell_data)
    assert not np.isnan(out[0]).any()
    np.testing.assert_array_equal(np.isnan(out[1]), np.isnan(single))
    assert np.isnan(out[1]).any()
    assert np.isnan(out[2]).all()


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), "0.5", None, True])
def test_invalid_min_valid_weight_raises(op, half_nan, bad):
    with pytest.raises(ValueError, match="min_valid_weight"):
        op.resample(half_nan, min_valid_weight=bad)


# --- comparison with the behaviour before the masking -----------------------

# Last commit before `min_valid_weight`: the reference for "only NaN is added".
# A fixed commit, not a branch, so that the comparison keeps its meaning once the
# change is merged. It needs the history (CI checks out with fetch-depth 0).
_REFERENCE_COMMIT = "2be7c6270aa8b2345b231b26f5aeabaa71555276"


@pytest.fixture(scope="module")
def main_cls():
    """`PSFResampler` as before the masking, loaded from git."""
    root = Path(__file__).resolve().parents[1]
    try:
        src = subprocess.run(
            ["git", "show", f"{_REFERENCE_COMMIT}:healpix_resample/psf.py"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"commit {_REFERENCE_COMMIT} not available (shallow clone)")
    spec = importlib.util.spec_from_loader("_psf_main_reference", loader=None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    exec(compile(src, "psf_main.py", "exec"), mod.__dict__)
    return mod.PSFResampler


@pytest.fixture(scope="module")
def scattered_nan(grid):
    rng = np.random.default_rng(0)
    val = (0.5 + np.sin(grid[0] * 20) * np.cos(grid[1] * 30)).astype("f4")
    val[rng.random(val.shape) < 0.1] = np.nan
    return val


@pytest.fixture(scope="module")
def ramp_nan(grid):
    lon, lat = grid
    val = (1.0 + 4.0 * lon + 0.1 * np.sin(lat * 200)).astype("f4")
    val[(lon > 0.25) & (lat > 45.25)] = np.nan
    return val


def _same_as_main(new, old):
    a, b = np.asarray(new.cell_data), np.asarray(old.cell_data)
    keep = ~np.isnan(a)
    assert keep.any() and np.isnan(a).any()
    assert np.max(np.abs(a[keep] - b[keep])) == 0.0
    assert int(new.cg_niters) == int(old.cg_niters)


@pytest.mark.parametrize("which", ["half_nan", "scattered_nan"])
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(lam=0.1),
        dict(lam=0.1, tol=1e-10, max_iter=300),
        dict(lam=0.1, jacobi=True),
        dict(lam=0.1, conservative=True),
        dict(lam=0.1, clip=("input", "input")),
        dict(lam=0.1, clip=(0.0, None), conservative=True),
    ],
)
def test_unmasked_cells_equal_main(op, main_cls, grid, request, which, kwargs):
    val = request.getfixturevalue(which)
    old_op = main_cls(lon_deg=grid[0], lat_deg=grid[1], level=LEVEL, threshold=0.3)
    _same_as_main(op.resample(val, **kwargs), old_op.resample(val, **kwargs))


def test_unmasked_cells_equal_main_with_x0_and_1d(op, main_cls, grid, half_nan):
    old_op = main_cls(lon_deg=grid[0], lat_deg=grid[1], level=LEVEL, threshold=0.3)
    x0 = torch.full((1, len(op.cell_ids)), 0.01, dtype=torch.float32)
    new = op.resample(half_nan, lam=0.1, x0=x0)
    assert np.asarray(new.cell_data).ndim == 1  # 1-D in, 1-D out
    _same_as_main(new, old_op.resample(half_nan, lam=0.1, x0=x0))


def test_unmasked_cells_equal_main_float64(main_cls, grid, half_nan):
    kw = dict(lon_deg=grid[0], lat_deg=grid[1], level=LEVEL, threshold=0.3, dtype=torch.float64)
    new = PSFResampler(**kw).resample(half_nan, lam=0.1)
    old = main_cls(**kw).resample(half_nan, lam=0.1)
    _same_as_main(new, old)


def test_fully_valid_row_equals_main(op, main_cls, grid):
    old_op = main_cls(lon_deg=grid[0], lat_deg=grid[1], level=LEVEL, threshold=0.3)
    val = (np.sin(grid[0] * 20) + np.cos(grid[1] * 30)).astype("f4")
    new = np.asarray(op.resample(val, lam=0.1).cell_data)
    old = np.asarray(old_op.resample(val, lam=0.1).cell_data)
    assert not np.isnan(new).any()
    np.testing.assert_array_equal(new, old)


def test_cells_over_nan_block_were_wrong_on_main(op, main_cls, grid, half_nan):
    old_op = main_cls(lon_deg=grid[0], lat_deg=grid[1], level=LEVEL, threshold=0.3)
    old = np.asarray(old_op.resample(half_nan, lam=0.1).cell_data)
    new = np.asarray(op.resample(half_nan, lam=0.1).cell_data)
    assert not np.isnan(old).any() and (np.abs(old) < 0.01).sum() > 100
    assert np.isnan(new).sum() >= (np.abs(old) < 0.01).sum()
    assert not (np.abs(new) < 0.01).any()


# --- exact share, brute force ------------------------------------------------

def _dense_share(op, val):
    M = op.M.to_dense().double().numpy()
    valid = (~np.isnan(val)).astype(np.float64)
    return (valid @ M) / (np.ones(M.shape[0]) @ M)


@pytest.mark.parametrize("which", ["half_nan", "scattered_nan"])
@pytest.mark.parametrize("mvw", [0.0, 0.3, 0.5, 1.0])
def test_nan_set_is_the_dense_share_threshold(op, request, which, mvw):
    val = request.getfixturevalue(which)
    share = _dense_share(op, val)
    tol = 100 * np.finfo(np.float32).eps
    expected = (share <= 0) | (share < mvw - tol)
    out = np.asarray(op.resample(val, lam=0.1, min_valid_weight=mvw).cell_data)
    np.testing.assert_array_equal(np.isnan(out), expected)
    if mvw == 0.0:
        assert (share[~expected] > 0).all()


def test_scattered_nan_one_dimensional_and_batch_agree(op, scattered_nan, half_nan):
    one = np.asarray(op.resample(scattered_nan, lam=0.1).cell_data)
    batch = np.asarray(op.resample(np.stack([half_nan, scattered_nan]), lam=0.1).cell_data)
    np.testing.assert_array_equal(np.isnan(batch[1]), np.isnan(one))


# --- clip counters and conservative ------------------------------------------

def test_counters_count_the_unmasked_cells_at_a_bound(op, main_cls, grid, ramp_nan):
    res = op.resample(ramp_nan, lam=0.1, clip=("input", "input"))
    out = np.asarray(res.cell_data)
    lo, hi = np.nanmin(ramp_nan), np.nanmax(ramp_nan)
    keep = ~np.isnan(out)
    assert np.isnan(out).any()
    assert int(res.n_clipped_low) == int((out[keep] == np.float32(lo)).sum())
    assert int(res.n_clipped_high) == int((out[keep] == np.float32(hi)).sum())
    # on main the masked cells were counted as well
    old_op = main_cls(lon_deg=grid[0], lat_deg=grid[1], level=LEVEL, threshold=0.3)
    old = old_op.resample(ramp_nan, lam=0.1, clip=("input", "input"))
    assert int(old.n_clipped_low) + int(old.n_clipped_high) >= int(res.n_clipped_low) + int(res.n_clipped_high)
    assert int(old.n_clipped_low) > int(res.n_clipped_low)


def test_conservative_is_computed_before_the_masking(op, main_cls, grid, half_nan):
    # documented: the correction still sums over the cells masked afterwards,
    # so the unmasked cells are those of main and the mean of the unmasked
    # cells is not the input mean.
    old_op = main_cls(lon_deg=grid[0], lat_deg=grid[1], level=LEVEL, threshold=0.3)
    new = op.resample(half_nan, lam=0.1, conservative=True)
    _same_as_main(new, old_op.resample(half_nan, lam=0.1, conservative=True))


# --- out_cell_ids ------------------------------------------------------------

@pytest.mark.parametrize("fill", [False, True])
def test_out_cell_ids(grid, half_nan, fill):
    lon, lat = grid
    base = PSFResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, threshold=0.3)
    ids = base.cell_ids
    op = PSFResampler(
        lon_deg=lon, lat_deg=lat, level=LEVEL, threshold=0.3,
        out_cell_ids=ids, fill_missing_out_cells=fill,
    )
    m0 = np.asarray(op.resample(half_nan, lam=0.1, min_valid_weight=0.0).cell_data)
    m5 = np.asarray(op.resample(half_nan, lam=0.1).cell_data)
    assert np.isnan(m5).sum() > np.isnan(m0).sum() > 0
    assert (np.isnan(m0) <= np.isnan(m5)).all()
    assert np.isfinite(m5[_cell_lon_any(op) < 0.15]).any()
