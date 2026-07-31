"""
tests/test_clough_tocher.py

Test suite for `CloughTocherResampler` (`healpix_resample.clough_tocher`).

Same shared conventions as `tests/test_bicubic.py` (see `planning/00_init.md`,
"Known gaps"): one file per resampler module, small/fast synthetic fixtures,
CPU-only by default.

IMPORTANT -- read before trusting a green run of this file: this test suite
was written in the same sandbox-less session as `clough_tocher.py` itself
(see that module's "NOTE on implementation risk" docstring) and has **not**
actually been executed. `test_affine_field_is_exact` and
`test_edge_value_continuity_from_both_sides` are the two checks
`planning/05_clough_tocher_resampler.md`'s validation plan says must pass
before anything built on this module is trustworthy -- run those first, and
look hard at failures there before trusting anything else in this file.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from healpix_resample import BicubicResampler, CloughTocherResampler
from healpix_resample.clough_tocher import (
    _barycentric_2d,
    _gnomonic_project,
    _triangle_ct_coefficients,
)

scipy_interpolate = pytest.importorskip("scipy.interpolate")


# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic datasets
# ─────────────────────────────────────────────────────────────────────────────

NDATA = 24
LEVEL = 10


def _small_grid(ndata: int = NDATA):
    """A small, densely-sampled patch -- good for the affine-exactness check
    (a plane is degree-1, so a dense-vs-sparse grid doesn't matter) and for
    basic shape/roundtrip-style smoke tests. Deliberately irregular (not a
    perfect meshgrid) so the Delaunay triangulation isn't a trivially uniform
    pattern -- jitter the grid slightly."""
    rng = np.random.default_rng(0)
    lon_grid, lat_grid = np.meshgrid(
        0.3 * np.arange(ndata) / ndata,
        0.3 * np.arange(ndata) / ndata,
    )
    lon = lon_grid.ravel() + rng.uniform(-1e-4, 1e-4, size=lon_grid.size)
    lat = lat_grid.ravel() + rng.uniform(-1e-4, 1e-4, size=lat_grid.size)
    return lon, lat


def _curved_grid(ndata: int = 40):
    """A wider patch (tens of degrees) so `sin(lon)*cos(lat)` has real
    curvature relative to the sampling scale -- see `test_bicubic.py`'s
    identical rationale for why this needs to be wide, not just dense."""
    rng = np.random.default_rng(1)
    lon_grid, lat_grid = np.meshgrid(
        30.0 * np.arange(ndata) / ndata,
        30.0 * np.arange(ndata) / ndata,
    )
    lon = lon_grid.ravel() + rng.uniform(-1e-3, 1e-3, size=lon_grid.size)
    lat = lat_grid.ravel() + rng.uniform(-1e-3, 1e-3, size=lat_grid.size)
    return lon, lat


@pytest.fixture(scope="module")
def small_grid():
    return _small_grid()


@pytest.fixture(scope="module")
def curved_grid():
    return _curved_grid()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Affine exactness -- the strongest, most discriminating check (see
#    planning/05's validation plan item 1): a correctly-built CT patch must
#    reproduce an affine field exactly (to float precision) everywhere inside
#    the convex hull.
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_operator_affine_exact_and_row_sums_zero(small_grid):
    """Direct check of Gx/Gy (planning/05 step 3's own correctness check,
    "run this numerically before writing a single line of the CT
    macro-element code"): for f affine in the projected plane, grad_x = Gx@f
    must reproduce the true constant slope exactly at every vertex, and
    (structural invariant, see clough_tocher.py's docstring) every row of
    Gx/Gy must sum to exactly zero, since a constant field has zero gradient.
    """
    lon, lat = small_grid
    op = CloughTocherResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    x = op.points2d[:, 0]
    y = op.points2d[:, 1]
    b, c = 2.3, -1.7  # f = a + b*x + c*y
    f = torch.as_tensor(3.0 + b * x + c * y, dtype=op.dtype, device=op.device)

    gx = (op.Gx @ f).cpu().numpy()
    gy = (op.Gy @ f).cpu().numpy()

    np.testing.assert_allclose(gx, b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gy, c, rtol=1e-6, atol=1e-6)

    ones = torch.ones(op.N, dtype=op.dtype, device=op.device)
    row_sum_x = (op.Gx @ ones).cpu().numpy()
    row_sum_y = (op.Gy @ ones).cpu().numpy()
    np.testing.assert_allclose(row_sum_x, 0.0, atol=1e-8)
    np.testing.assert_allclose(row_sum_y, 0.0, atol=1e-8)


def test_affine_field_is_exact(small_grid):
    """The full pipeline (gradient estimation -> CT macro-element -> sparse
    M) must reproduce an affine field exactly everywhere retained -- corner
    values and gradients are exact (previous test), and any correctly-built
    cubic patch that matches affine data and affine gradients at all three
    corners *is* that affine function (a plane is a degenerate cubic).
    """
    lon, lat = small_grid
    op = CloughTocherResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    x = op.points2d[:, 0]
    y = op.points2d[:, 1]
    a, b, c = 5.0, 2.3, -1.7
    val = a + b * x + c * y

    res = op.resample(val)

    # Ground truth: evaluate the same affine function at each retained
    # output cell's own projected position.
    hp_lon, hp_lat = _cell_lonlat(op)
    qx, qy, _ = _gnomonic_project(hp_lon, hp_lat, op._tangent_point, op._east, op._north, radius=op.radius)
    expected = a + b * qx + c * qy

    np.testing.assert_allclose(res.cell_data, expected, rtol=1e-6, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 2. C1/C0 continuity across a shared internal (macro-triangle) edge,
#    evaluated from both adjacent triangles' own formulas independently --
#    planning/05's validation plan item 2.
# ─────────────────────────────────────────────────────────────────────────────

def test_edge_value_continuity_from_both_sides(small_grid):
    """Pick two Delaunay triangles that share an edge, pick points along that
    edge, and evaluate the CT patch value from each triangle's own
    micro-triangle decomposition independently (bypassing `find_simplex`,
    which would otherwise just pick one) -- they must agree, since the two
    triangles' control nets are built (by the C1-compatibility formulas in
    `_triangle_ct_coefficients`) to describe the same surface there.
    """
    lon, lat = small_grid
    op = CloughTocherResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    rng = np.random.default_rng(2)
    f = torch.as_tensor(rng.normal(size=op.N), dtype=op.dtype, device=op.device)

    tri = op.tri
    indptr, indices = tri.vertex_neighbor_vertices
    simplices = tri.simplices

    # Find a shared edge (p, q) between two distinct simplices.
    found = None
    for s0 in range(simplices.shape[0]):
        v = simplices[s0]
        for e in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])]:
            p, q = e
            for s1 in range(simplices.shape[0]):
                if s1 == s0:
                    continue
                if p in simplices[s1] and q in simplices[s1]:
                    found = (s0, s1, p, q)
                    break
            if found:
                break
        if found:
            break
    assert found is not None, "expected at least one interior shared edge in this triangulation"
    s0, s1, p, q = found

    P = op.points2d[p]
    Q = op.points2d[q]

    def eval_from_simplex(simplex_id, px, py):
        verts = simplices[simplex_id].astype(np.int64)
        U = op.points2d[verts][None, :, :]  # (1, 3, 2)
        ct = _triangle_ct_coefficients(U)
        beta = _barycentric_2d(np.array([px]), np.array([py]), U[:, 0, :], U[:, 1, :], U[:, 2, :])
        c = int(np.argmin(beta[0]))
        a_, b_ = (c + 1) % 3, (c + 2) % 3
        Z = U.mean(axis=1)
        gamma = _barycentric_2d(
            np.array([px]), np.array([py]), U[:, a_, :], U[:, b_, :], Z
        )[0]
        ga, gb, gz = gamma

        def coef_to_value(coef_row):
            fvals = f.cpu().numpy()[verts]
            gxvals = (op.Gx @ f).cpu().numpy()[verts]
            gyvals = (op.Gy @ f).cpu().numpy()[verts]
            local = np.concatenate(
                [fvals, np.stack([gxvals, gyvals], axis=-1).reshape(-1)]
            )
            # column order [f0,f1,f2,gx0,gy0,gx1,gy1,gx2,gy2] matches
            # concatenation of fvals (3) then interleaved (gx,gy) per vertex.
            return float(coef_row[0] @ local)

        val = (
            ga ** 3 * coef_to_value(ct["V"][a_])
            + gb ** 3 * coef_to_value(ct["V"][b_])
            + gz ** 3 * coef_to_value(ct["S"])
            + 3 * ga * ga * gb * coef_to_value(ct["T"][(a_, b_)])
            + 3 * ga * gb * gb * coef_to_value(ct["T"][(b_, a_)])
            + 3 * ga * ga * gz * coef_to_value(ct["I1"][a_])
            + 3 * ga * gz * gz * coef_to_value(ct["I2"][a_])
            + 3 * gb * gb * gz * coef_to_value(ct["I1"][b_])
            + 3 * gb * gz * gz * coef_to_value(ct["I2"][b_])
            + 6 * ga * gb * gz * coef_to_value(ct["C"][c])
        )
        return val

    for t in [0.2, 0.5, 0.8]:
        px = P[0] + t * (Q[0] - P[0])
        py = P[1] + t * (Q[1] - P[1])
        v0 = eval_from_simplex(s0, px, py)
        v1 = eval_from_simplex(s1, px, py)
        assert v0 == pytest.approx(v1, abs=1e-6), (
            f"CT patch value disagrees across shared edge at t={t}: "
            f"triangle {s0} gives {v0}, triangle {s1} gives {v1}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Comparison against scipy.interpolate.CloughTocher2DInterpolator
# ─────────────────────────────────────────────────────────────────────────────

def _cell_lonlat(op):
    import healpix_geo

    hp = healpix_geo.nested if op.nest else healpix_geo.ring
    cell_np = op.cell_ids.detach().cpu().numpy().astype(np.uint64)
    lon, lat = hp.healpix_to_lonlat(cell_np, op.level, ellipsoid=op.ellipsoid)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def test_close_to_scipy_clough_tocher_on_smooth_field(small_grid):
    """Not expected to match bit-for-bit (different gradient estimator --
    see the module docstring), but should be quantitatively close on a
    smooth field, reusing *this* resampler's own triangulation to isolate
    the comparison to "gradient estimator + evaluation", not
    "triangulation differences"."""
    lon, lat = small_grid
    op = CloughTocherResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    x = op.points2d[:, 0]
    y = op.points2d[:, 1]
    scale = 1.0 / max(op.points2d.std(), 1e-9)
    val = np.sin(x * scale) * np.cos(y * scale)

    res = op.resample(val)

    scipy_interp = scipy_interpolate.CloughTocher2DInterpolator(op.tri, val)
    hp_lon, hp_lat = _cell_lonlat(op)
    qx, qy, _ = _gnomonic_project(hp_lon, hp_lat, op._tangent_point, op._east, op._north, radius=op.radius)
    scipy_vals = scipy_interp(np.stack([qx, qy], axis=-1))

    finite = np.isfinite(scipy_vals)
    assert finite.sum() > 0.5 * finite.size  # most retained cells should be inside scipy's own hull too

    diff = res.cell_data[finite] - scipy_vals[finite]
    rms = float(np.sqrt(np.mean(diff ** 2)))
    field_scale = float(np.std(val))
    # "reasonably low" relative to the field's own scale -- generous (20%)
    # because the two gradient estimators are genuinely different, not a
    # tight numerical-agreement bound.
    assert rms < 0.2 * field_scale


# ─────────────────────────────────────────────────────────────────────────────
# 4. Outside-the-convex-hull exclusion
# ─────────────────────────────────────────────────────────────────────────────

def test_outside_hull_cells_excluded(small_grid):
    """A HEALPix cell whose center falls outside the sample convex hull must
    not appear in `cell_ids` at all (not silently given a garbage value)."""
    lon, lat = small_grid
    op = CloughTocherResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    outside_lon = lon.max() + 5.0  # well beyond the sample footprint
    outside_lat = lat.max() + 5.0
    import healpix_geo

    hp = healpix_geo.nested if op.nest else healpix_geo.ring
    outside_cell = int(
        np.asarray(hp.lonlat_to_healpix(np.array([outside_lon]), np.array([outside_lat]), op.level)).astype(np.int64)[0]
    )

    assert outside_cell not in set(op.cell_ids.cpu().numpy().tolist())


# ─────────────────────────────────────────────────────────────────────────────
# 5. Batched (B, N) vs. plain (N,); NumPy/Torch in-out symmetry
# ─────────────────────────────────────────────────────────────────────────────

def test_batched_and_unbatched(small_grid):
    lon, lat = small_grid
    op = CloughTocherResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)

    val_1d = lon
    val_2d = np.stack([lon, lon * 2.0], axis=0)  # (2, N)

    res_1d = op.resample(val_1d)
    res_2d = op.resample(val_2d)

    assert res_1d.cell_data.ndim == 1
    assert res_2d.cell_data.ndim == 2
    assert res_2d.cell_data.shape[0] == 2
    assert res_2d.cell_data.shape[1] == res_1d.cell_data.shape[0]

    # M is a linear operator -- doubling the input row must double the output.
    np.testing.assert_allclose(res_2d.cell_data[1], res_2d.cell_data[0] * 2.0, rtol=1e-5, atol=1e-8)


def test_numpy_in_numpy_out(small_grid):
    lon, lat = small_grid
    op = CloughTocherResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)
    val = lon.astype(np.float64)

    res = op.resample(val)
    assert isinstance(res.cell_data, np.ndarray)
    assert isinstance(res.cell_ids, np.ndarray)


def test_torch_in_torch_out(small_grid):
    lon, lat = small_grid
    op = CloughTocherResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)
    val = torch.as_tensor(lon, dtype=torch.float64)

    res = op.resample(val)
    assert isinstance(res.cell_data, torch.Tensor)
    assert isinstance(res.cell_ids, torch.Tensor)


# ─────────────────────────────────────────────────────────────────────────────
# 6. invert() is deliberately unimplemented
# ─────────────────────────────────────────────────────────────────────────────

def test_invert_raises_not_implemented(small_grid):
    lon, lat = small_grid
    op = CloughTocherResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, verbose=False)
    with pytest.raises(NotImplementedError):
        op.invert(np.zeros(op.K))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Small-scale-artifact regression vs BicubicResampler -- the actual
#    motivating comparison from planning/05's "Why this resampler exists".
# ─────────────────────────────────────────────────────────────────────────────

def test_smoother_second_derivative_than_bicubic_on_curved_field(curved_grid):
    """Not a proof of general superiority, just the standing regression this
    resampler was built to satisfy: on a smooth curved field, a simple local
    roughness proxy (mean squared second finite difference along the
    retained-cell ordering) should be no worse for Clough-Tocher than for
    BicubicResampler, whose discrete KNN neighbour-set switching is the
    documented source of small-scale artifacts (see the module docstring).
    """
    lon, lat = curved_grid
    lon_rad, lat_rad = np.deg2rad(lon), np.deg2rad(lat)
    val = np.sin(lon_rad) * np.cos(lat_rad)

    level = 6

    ct = CloughTocherResampler(lon_deg=lon, lat_deg=lat, level=level, verbose=False)
    bicubic = BicubicResampler(lon_deg=lon, lat_deg=lat, level=level, verbose=False)

    res_ct = ct.resample(val)
    res_bicubic = bicubic.resample(val)

    def roughness(cell_ids, cell_data):
        order = np.argsort(cell_ids)
        v = np.asarray(cell_data)[order]
        if v.size < 3:
            return 0.0
        d2 = v[2:] - 2 * v[1:-1] + v[:-2]
        return float(np.mean(d2 ** 2))

    r_ct = roughness(res_ct.cell_ids, res_ct.cell_data)
    r_bicubic = roughness(res_bicubic.cell_ids, res_bicubic.cell_data)

    # Generous margin (not a tight bound): the point is "not dramatically
    # worse", since cell_ids ordering differs between the two resamplers
    # (different candidate-cell logic) so this is a coarse proxy, not an
    # apples-to-apples per-cell comparison.
    assert r_ct < r_bicubic * 3.0 + 1e-12
