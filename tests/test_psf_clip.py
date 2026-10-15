"""
tests/test_psf_clip.py

Bounded fit of `PSFResampler.resample(..., clip=(lo, hi))`.

The deconvolution of the PSF fit over- and undershoots across sharp edges:
negative values for a non-negative field (#65), values outside the input
range on high-contrast Sentinel-2 edges (#68). `clip` imposes hard bounds
inside the fit (`bounded_least_squares_cg`): cells pushed out of the box are
held at the bound and the free neighbours take up the residual they leave.

CPU-only, following `tests/test_psf.py`.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.optimize
import torch

from healpix_resample import PSFResampler
from healpix_resample.psf import bounded_least_squares_cg


NDATA = 60
LEVEL = 12


@pytest.fixture(scope="module")
def step():
    """A dark/bright step (0.01 | 1.0) on a regular grid finer than the cells."""
    lon, lat = np.meshgrid(
        0.3 * np.arange(NDATA) / NDATA,
        0.3 * np.arange(NDATA) / NDATA,
    )
    lon, lat = lon.ravel(), lat.ravel()
    val = np.where(lon < 0.15, 0.01, 1.0)
    return lon, lat, val


@pytest.fixture(scope="module")
def op(step):
    lon, lat, _ = step
    return PSFResampler(lon, lat, level=LEVEL, Npt=9, threshold=0.1)


@pytest.fixture(scope="module")
def op64(step):
    lon, lat, _ = step
    return PSFResampler(lon, lat, level=LEVEL, Npt=9, threshold=0.1, dtype=torch.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Default unchanged, and the issue reproduced
# ─────────────────────────────────────────────────────────────────────────────

def test_clip_none_is_the_unconstrained_fit(op, step):
    _, _, val = step
    ref = op.resample(val, lam=0.1)
    res = op.resample(val, lam=0.1, clip=None)
    np.testing.assert_array_equal(res.cell_data, ref.cell_data)
    assert res.n_clipped_low is None and res.n_clipped_high is None


def test_unconstrained_fit_leaves_the_input_range(op, step):
    # the behaviour reported in #65 / #68, kept as the reason for `clip`
    _, _, val = step
    h = op.resample(val, lam=0.1).cell_data
    assert h.min() < 0.0
    assert h.max() > val.max()


# ─────────────────────────────────────────────────────────────────────────────
# Bounds, one side or both
# ─────────────────────────────────────────────────────────────────────────────

def test_lower_bound_only(op, step):
    _, _, val = step
    res = op.resample(val, lam=0.1, clip=(0.0, None))
    assert res.cell_data.min() >= 0.0
    assert res.n_clipped_low > 0
    assert res.n_clipped_high == 0
    # the upper side is not constrained
    assert res.cell_data.max() > val.max()


def test_upper_bound_only(op, step):
    _, _, val = step
    res = op.resample(val, lam=0.1, clip=(None, 1.0))
    assert res.cell_data.max() <= 1.0
    assert res.n_clipped_high > 0
    assert res.n_clipped_low == 0


def test_input_range_per_batch_row(op, step):
    _, _, val = step
    vals = np.stack([val, 3.0 * val - 1.0])
    res = op.resample(vals, lam=0.1, clip=("input", "input"))
    for row, h in zip(vals.astype(res.cell_data.dtype), res.cell_data):
        assert h.min() >= row.min()
        assert h.max() <= row.max()
    assert res.n_clipped_low.shape == (2,)
    assert (res.n_clipped_low > 0).all() and (res.n_clipped_high > 0).all()


def test_torch_input_returns_torch_counts(op, step):
    _, _, val = step
    res = op.resample(torch.as_tensor(val), lam=0.1, clip=(0.0, None))
    assert isinstance(res.cell_data, torch.Tensor)
    assert isinstance(res.n_clipped_low, torch.Tensor)
    assert int(res.n_clipped_low) > 0


# ─────────────────────────────────────────────────────────────────────────────
# The bounds are in the fit, not a clip of its result
# ─────────────────────────────────────────────────────────────────────────────

def _objective(op, y, x_ref, lam):
    """J(delta) = 1/2 <delta, A delta>_Dx - <delta, b>_Dx, whose stationarity
    condition A delta = b is the system `least_squares_cg` solves."""
    M, MT, dx = op.M, op.MT, op.cell_weight
    b = (y - x_ref @ MT) @ M

    def J(delta):
        Ad = (delta @ MT) @ M + lam * delta
        return 0.5 * torch.sum(delta * Ad * dx) - torch.sum(delta * b * dx)

    def grad(delta):
        Ad = (delta @ MT) @ M + lam * delta
        return dx * (Ad - b)

    return J, grad


@pytest.mark.parametrize("clip", [(0.0, None), (None, 1.0), (0.01, 1.0)])
def test_bounded_fit_reaches_the_box_constrained_minimum(op64, step, clip):
    _, _, val = step
    lam = 0.1
    y = torch.as_tensor(val, dtype=torch.float64)[None, :]
    x_ref = y @ op64.M
    J, grad = _objective(op64, y, x_ref, lam)

    h = torch.as_tensor(
        op64.resample(val, lam=lam, max_iter=500, tol=1e-12, clip=clip).cell_data
    )[None, :]
    j_ours = float(J(h - x_ref))

    xr = x_ref[0].numpy()
    lb = [None] * xr.size if clip[0] is None else list(clip[0] - xr)
    ub = [None] * xr.size if clip[1] is None else list(clip[1] - xr)
    bounds = list(zip(lb, ub))
    ref = scipy.optimize.minimize(
        lambda d: float(J(torch.as_tensor(d)[None, :])),
        np.zeros_like(xr),
        jac=lambda d: grad(torch.as_tensor(d)[None, :])[0].numpy(),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12},
    )
    j_ref = float(ref.fun)

    # as good as a generic box-constrained optimiser
    assert j_ours <= j_ref + 1e-6 * abs(j_ref)

    # and strictly better than clipping the unconstrained solution afterwards
    free = torch.as_tensor(op64.resample(val, lam=lam, max_iter=500, tol=1e-12).cell_data)
    clipped = torch.clamp(free, min=None if clip[0] is None else clip[0],
                          max=None if clip[1] is None else clip[1])[None, :]
    assert j_ours < float(J(clipped - x_ref))


def test_held_cells_constrain_their_neighbours(op, step):
    # A post-hoc clip only changes the cells out of the box. The bounded fit
    # also moves free cells next to the held ones.
    _, _, val = step
    free_fit = op.resample(val, lam=0.1).cell_data
    bounded = op.resample(val, lam=0.1, clip=(0.0, None)).cell_data
    inside = free_fit > 0.0
    assert np.abs(bounded[inside] - free_fit[inside]).max() > 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# With conservative=True
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("clip", [(0.0, None), (None, 1.0), ("input", "input")])
def test_conservative_with_clip_keeps_bounds_and_mean(op, step, clip):
    _, _, val = step
    res = op.resample(val, lam=0.1, clip=clip, conservative=True)
    h = res.cell_data
    val32 = val.astype(h.dtype)  # the bounds are taken in the operator's dtype
    lo = val32.min() if clip[0] == "input" else clip[0]
    hi = val32.max() if clip[1] == "input" else clip[1]
    if lo is not None:
        assert h.min() >= lo
    if hi is not None:
        assert h.max() <= hi
    # uniform grid: the area-weighted mean is the plain mean
    np.testing.assert_allclose(h.mean(), val.mean(), rtol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# NaN rows, empty operator, argument checks
# ─────────────────────────────────────────────────────────────────────────────

def test_all_nan_row_with_clip(op, step):
    _, _, val = step
    vals = np.stack([val, np.full_like(val, np.nan)])
    res = op.resample(vals, lam=0.1, clip=(0.0, None))
    assert np.isnan(res.cell_data[1]).all()
    assert res.n_clipped_low[1] == 0
    assert res.cell_data[0].min() >= 0.0


def test_bounded_solver_with_no_cell():
    # K = 0: legitimate with out_cell_ids when no requested cell has support
    N, B = 5, 2
    M = torch.zeros((N, 0), dtype=torch.float64)
    MT = torch.zeros((0, N), dtype=torch.float64)
    y = torch.ones((B, N), dtype=torch.float64)
    x_ref = y @ M
    delta, info = bounded_least_squares_cg(
        M, MT, y, x_ref, torch.zeros_like(x_ref),
        lower=torch.zeros_like(x_ref), upper=torch.full_like(x_ref, np.inf),
        verbose=False, weight=torch.ones(0, dtype=torch.float64),
    )
    assert delta.shape == (B, 0)
    assert info["at_lower"].shape == (B, 0)


@pytest.mark.parametrize(
    "clip", [0.0, (0.0,), (0.0, None, None), ("min", None), (1.0, 0.0), (np.nan, None)]
)
def test_invalid_clip_raises(op, step, clip):
    _, _, val = step
    with pytest.raises(ValueError):
        op.resample(val, clip=clip)


def test_invalid_max_outer_raises(op, step):
    _, _, val = step
    with pytest.raises(ValueError):
        op.resample(val, clip=(0.0, None), max_outer=0)
