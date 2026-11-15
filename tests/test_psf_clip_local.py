"""
tests/test_psf_clip_local.py

``PSFResampler.resample(..., clip=("local", "local"))``: each output cell is
bounded by the min / max of the valid samples linked to it, so the result does
not depend on the other samples of the call.

CPU-only, following `tests/test_psf_clip.py`.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from healpix_resample import PSFResampler

LEVEL = 12
N = 70
SPAN = 0.35


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(3)
    lon, lat = np.meshgrid(SPAN * np.arange(N) / N, SPAN * np.arange(N) / N)
    lon, lat = lon.ravel(), lat.ravel()
    val = 0.1 + 0.05 * np.sin(lon * 30)
    val = np.where(lon + lat > 0.3, val + 0.4, val)          # a sharp edge
    val = np.where(np.hypot(lon - 0.1, lat - 0.2) < 0.012, 0.95, val)  # a bright spot
    val = (val + rng.normal(0, 0.01, val.size)).astype(np.float32)
    return lon, lat, val


@pytest.fixture(scope="module")
def op(scene):
    lon, lat, _ = scene
    return PSFResampler(lon, lat, level=LEVEL, Npt=9, threshold=0.1)


def _brute_local_range(op, val, valid, support=0.0, margin=0.0):
    """Reference: loop over the cells and their links."""
    A = op._M_t
    crow, col, w = A.crow_indices().numpy(), A.col_indices().numpy(), A.values().numpy()
    lo = np.full(op.K, -np.inf)
    hi = np.full(op.K, np.inf)
    for k in range(op.K):
        s, e = crow[k], crow[k + 1]
        c, ww = col[s:e], w[s:e]
        if ww.size == 0:
            continue
        keep = (ww > 0) & (ww >= support * ww.max()) if support > 0 else np.ones(ww.size, bool)
        c = c[keep]
        c = c[valid[c]]
        if c.size == 0:
            continue
        a, b = val[c].min(), val[c].max()
        lo[k], hi[k] = a - margin * (b - a), b + margin * (b - a)
    return lo, hi


@pytest.mark.parametrize("support,margin", [(0.0, 0.0), (0.2, 0.0), (0.0, 0.25), (0.5, 0.1)])
def test_local_range_matches_the_brute_force(op, scene, support, margin):
    _, _, val = scene
    v = val.copy()
    v[100:140] = np.nan
    valid = ~np.isnan(v)
    lo, hi = op._local_range(
        torch.as_tensor(np.nan_to_num(v))[None], torch.as_tensor(valid)[None], support, margin
    )
    rlo, rhi = _brute_local_range(op, np.nan_to_num(v), valid, support, margin)
    np.testing.assert_allclose(lo[0].numpy(), rlo, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(hi[0].numpy(), rhi, rtol=1e-6, atol=1e-6)


def test_output_stays_inside_the_local_bounds(op, scene):
    _, _, val = scene
    res = op.resample(val, lam=0.1, clip=("local", "local"))
    lo, hi = op._local_range(
        torch.as_tensor(val)[None], torch.ones((1, val.size), dtype=torch.bool)
    )
    h = res.cell_data
    assert np.all(h >= lo[0].numpy() - 1e-6) and np.all(h <= hi[0].numpy() + 1e-6)
    assert res.n_clipped_low is not None and res.n_clipped_high is not None


def test_local_is_not_looser_than_input(op, scene):
    _, _, val = scene
    res = op.resample(val, lam=0.1, clip=("local", "local"))
    assert res.cell_data.max() <= val.max() + 1e-6
    assert res.cell_data.min() >= val.min() - 1e-6


def test_batch_rows_match_single_rows(op, scene):
    _, _, val = scene
    both = np.stack([val, val[::-1].copy()])
    res = op.resample(both, lam=0.1, clip=("local", "local")).cell_data
    for b in range(2):
        one = op.resample(both[b], lam=0.1, clip=("local", "local")).cell_data
        np.testing.assert_allclose(res[b], one, atol=1e-5)


def test_one_side_local_other_side_fixed(op, scene):
    _, _, val = scene
    res = op.resample(val, lam=0.1, clip=(0.0, "local")).cell_data
    assert res.min() >= -1e-9
    hi = op._local_range(torch.as_tensor(val)[None], torch.ones((1, val.size), dtype=torch.bool))[1]
    assert np.all(res <= hi[0].numpy() + 1e-6)
    # a fixed side beyond a local bound keeps the box non-empty
    res = op.resample(val, lam=0.1, clip=("local", 0.05)).cell_data
    assert np.isfinite(res).all()


def test_result_does_not_depend_on_the_chunk(scene):
    """The same cells from two windows of different size. The bounds of
    "input" change with the window (a far outlier only the large one holds
    lifts the upper bound where the fit overshoots a bright spot); local
    bounds do not. The buffers cover the coupling of the fit itself."""
    lon, lat, val = scene
    val = val.copy()
    cx = cy = 0.14
    half = 0.06                                   # evaluation region (degrees)
    val[np.hypot(lon - cx, lat - cy) < 0.008] = 0.95   # a bright spot in it
    val[np.argmin(np.hypot(lon - 0.345, lat - 0.345))] = 5.0   # far outlier

    def run(buf, clip):
        m = (np.abs(lon - cx) <= half + buf) & (np.abs(lat - cy) <= half + buf)
        o = PSFResampler(lon[m], lat[m], level=LEVEL, Npt=9, threshold=0.1)
        r = o.resample(val[m], lam=0.1, clip=clip)
        return dict(zip(r.cell_ids.tolist(), r.cell_data.tolist()))

    def share(clip):
        import healpix_geo as hg

        small, large = run(0.12, clip), run(0.21, clip)
        ids = np.array(list(small))
        lo, la = hg.nested.healpix_to_lonlat(ids, LEVEL, ellipsoid="WGS84")
        ids = ids[(np.abs(lo - cx) <= half) & (np.abs(la - cy) <= half)]
        a = np.array([small[i] for i in ids])
        b = np.array([large[i] for i in ids])
        return (np.abs(a - b) > 1e-4).mean()

    assert share(("local", "local")) == 0.0
    assert share(("input", "input")) > 0.0


def test_nan_only_support_stays_nan(op, scene):
    _, _, val = scene
    v = val.copy()
    v[: N * 15] = np.nan     # the lower strip has no data
    res = op.resample(v, lam=0.1, clip=("local", "local")).cell_data
    assert np.isnan(res).any()
    ref = op.resample(v, lam=0.1, clip=("input", "input")).cell_data
    np.testing.assert_array_equal(np.isnan(res), np.isnan(ref))
    assert np.isfinite(res[~np.isnan(res)]).all()


def test_all_nan_row_is_nan(op, scene):
    _, _, val = scene
    both = np.stack([val, np.full_like(val, np.nan)])
    res = op.resample(both, lam=0.1, clip=("local", "local")).cell_data
    assert np.isnan(res[1]).all() and np.isfinite(res[0]).all()


def test_margin_loosens_the_bounds(op, scene):
    _, _, val = scene
    r0 = op.resample(val, lam=0.1, clip=("local", "local")).cell_data
    r1 = op.resample(val, lam=0.1, clip=("local", "local"), local_margin=0.5).cell_data
    rn = op.resample(val, lam=0.1, clip=None).cell_data
    assert np.abs(r1 - rn).max() <= np.abs(r0 - rn).max() + 1e-6


@pytest.mark.parametrize("kw", [dict(local_support=-0.1), dict(local_support=1.5),
                                dict(local_margin=-1.0), dict(local_support=True)])
def test_invalid_local_arguments(op, scene, kw):
    _, _, val = scene
    with pytest.raises(ValueError):
        op.resample(val, clip=("local", "local"), **kw)


def test_unknown_clip_word_still_raises(op, scene):
    _, _, val = scene
    with pytest.raises(ValueError, match="local"):
        op.resample(val, clip=("neighbours", None))
