"""Numerical self-tests of revision_tools (no data, no GPU needed).

    pixi run -e notebooks python notebooks/revision_run.py selftest

Covers the pieces written for the revision: the Richardson--Lucy snapshot
loop, the Fourier deconvolutions, the weighted/Euclidean CG and the E2
schemes on a toy dual-normalized operator pair, the stacked cubic readout,
the notebook-cell markers and the cross-validation/statistics code.
"""
from __future__ import annotations

import json
import math
import sys
import traceback

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator

import revision_tools as rt

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def smooth_image(shape=(48, 40), seed=0):
    rng = np.random.default_rng(seed)
    img = rng.random(shape)
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, 2.0, mode="wrap") + 0.2


def toy_operators(n_side=14, k_side=22, fwhm=1.6, seed=1):
    """Dual-normalized pair built from raw Gaussian weights, as in the paper:
    A = Dy^-1 G (rows sum to 1), B = Dx^-1 G^T (rows sum to 1)."""
    rng = np.random.default_rng(seed)
    ys, xs = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij")
    samples = np.column_stack([ys.ravel(), xs.ravel()]).astype(float)
    samples += rng.normal(0, 0.05, samples.shape)
    t = np.linspace(-0.5, n_side - 0.5, k_side)
    cy, cx = np.meshgrid(t, t, indexing="ij")
    cells = np.column_stack([cy.ravel(), cx.ravel()])
    s = fwhm / math.sqrt(2 * math.log(2))
    d2 = ((samples[:, None, :] - cells[None, :, :]) ** 2).sum(-1)
    G = np.exp(-2 * d2 / s ** 2)
    G[G < 1e-6] = 0.0
    keep = G.sum(0) > 0.1
    G = G[:, keep]
    dy, dx = G.sum(1), G.sum(0)
    A = sp.csr_matrix(G / dy[:, None])
    B = sp.csr_matrix((G / dx[None, :]).T)
    k = G.shape[1]
    ops = dict(A=A, B=B, dx=dx, bad=np.zeros(k, bool), cell_ids=np.arange(k))
    truth = np.sin(cells[keep, 0] / 2.0) + np.cos(cells[keep, 1] / 3.0) + 2.0
    y = A @ truth + rng.normal(0, 1e-3, A.shape[0])
    return ops, dy, y


def test_rl():
    img = smooth_image()
    k = rt.gaussian_kernel(15.0, 10.0)
    check("gaussian kernel normalized", abs(k.sum() - 1) < 1e-12)
    snaps = rt.rl_snapshots(img / img.max(), k, [3, 7])
    again = rt.rl_snapshots(img / img.max(), k, [7])
    check("RL snapshots equal separate runs",
          np.allclose(snaps[7], again[7], rtol=0, atol=0))
    try:
        diff = rt.check_rl_against_skimage(img / img.max(), k, num_iter=25)
        check("RL loop == skimage.richardson_lucy", diff < 1e-10, f"max diff {diff:.2e}")
    except ImportError:
        print("[SKIP] scikit-image not installed: RL equivalence not tested")
    for variant in ("clip", "minmax"):
        x01, lo, scale = rt.rl_normalize(img, variant)
        check(f"RL normalization {variant} in [0,1]", x01.min() >= 0 and x01.max() <= 1)


def test_fourier():
    img = smooth_image((32, 32))
    k = rt.gaussian_kernel(15.0, 10.0)
    H = rt._transfer(k, img.shape)
    blurred = np.fft.ifft2(H * np.fft.fft2(img)).real
    rec = rt.tikhonov_fields(blurred, k, [1e-12])[1e-12]
    err = np.abs(rec - img).max()
    check("Tikhonov(identity) inverts a circular blur as lam->0", err < 1e-4, f"max err {err:.1e}")
    rec_l = rt.tikhonov_fields(blurred, k, [1e-12], regularizer="laplacian")[1e-12]
    check("Tikhonov(Laplacian) inverts a circular blur as lam->0",
          np.abs(rec_l - img).max() < 1e-3)
    strong = rt.tikhonov_fields(blurred, k, [1e3], regularizer="laplacian")[1e3]
    check("Tikhonov(Laplacian) smooths for large lam", strong.std() < blurred.std())
    try:
        wh = rt.wiener_hunt_field(blurred, k)
        check("Wiener-Hunt returns a finite field", wh.shape == img.shape and np.isfinite(wh).all())
    except ImportError:
        print("[SKIP] scikit-image not installed: Wiener-Hunt not tested")
    fields, runtimes = rt.deconvolution_fields(blurred, k, rl_variants=("minmax",))
    n_expected = len(rt.RL_ITERS) + 2 * len(rt.TIK_LAMBDAS)
    check("deconvolution_fields grid size", len(fields) >= n_expected, f"{len(fields)} fields")


def test_operators_and_cg():
    ops, dy, y = toy_operators()
    A, B, dx = ops["A"], ops["B"], ops["dx"]
    # Weighted adjoint identity of the paper: B = Dx^-1 A^T Dy.
    lhs = B.toarray()
    rhs = (A.T.toarray() * dy[None, :]) / dx[:, None]
    check("weighted adjoint B = Dx^-1 A^T Dy", np.allclose(lhs, rhs, atol=1e-12))
    lam = 1e-2
    x_ref = B @ y
    b = B @ (y - A @ x_ref)
    H = (B @ A).toarray() + lam * np.eye(len(dx))
    direct = np.linalg.solve(H, b)
    snaps = rt.pcg_snapshots(lambda v: B @ (A @ v) + lam * v, b, [5, 400], weight=dx, tol=1e-14)
    err = np.abs(snaps[400] - direct).max() / np.abs(direct).max()
    check("weighted CG converges to the direct solve", err < 1e-8, f"rel err {err:.1e}")
    sep = rt.pcg_snapshots(lambda v: B @ (A @ v) + lam * v, b, [5], weight=dx, tol=1e-14)
    check("CG snapshot == separate run", np.allclose(snaps[5], sep[5], rtol=0, atol=0))
    # D_x H is symmetric: the reason for the weighted inner product.
    DH = dx[:, None] * H
    check("Dx (BA + lam I) symmetric", np.allclose(DH, DH.T, atol=1e-10))

    fields = rt.e2_fields(ops, y)
    fams = {k[0] for k in fields}
    expected = {"e2_xref", "e2_wcg", "e2_cg_euclid", "e2_jacobi", "e2_tik_x0",
                "e2_tik_dref", "e2_sirt", "e2_mlem"}
    check("E2 families present", expected <= fams, str(sorted(fams)))
    check("E2 fields finite", all(np.isfinite(v).all() for v in fields.values()))
    x100 = fields[("e2_wcg", "", 1e-2, 100, np.nan)]
    gap = np.abs(x100 - (x_ref + direct)).max() / np.abs(x_ref + direct).max()
    check("E2 weighted CG (100 it) close to the exact solution", gap < 1e-4, f"rel gap {gap:.1e}")
    # Euclidean Tikhonov from 0 converges to (A^T A + lam I)^-1 A^T y.
    AtA = (A.T @ A).toarray()
    tik = np.linalg.solve(AtA + lam * np.eye(len(dx)), A.T @ y)
    xt = fields[("e2_tik_x0", "", 1e-2, 100, np.nan)]
    gap_t = np.abs(xt - tik).max() / np.abs(tik).max()
    check("E2 Euclidean Tikhonov (100 it) close to its exact solution", gap_t < 1e-3,
          f"rel gap {gap_t:.1e}")
    # Residuals decrease for the iterative schemes.
    def res(x):
        return np.linalg.norm(y - A @ x)
    check("SIRT reduces the data residual",
          res(fields[("e2_sirt", "", np.nan, 100, 1.0)]) < res(x_ref))
    check("MLEM reduces the data residual",
          res(fields[("e2_mlem", "", np.nan, 200, np.nan)]) < res(x_ref))


def test_stacked_cubic():
    rng = np.random.default_rng(3)
    y_ax, x_ax = np.arange(20.0), np.arange(24.0)
    fields = [rng.random((20, 24)) for _ in range(3)]
    q = np.column_stack([rng.uniform(0, 19, 500), rng.uniform(0, 23, 500)])
    stacked = RegularGridInterpolator((y_ax, x_ax), np.stack(fields, -1), method="cubic")(q)
    single = np.column_stack([RegularGridInterpolator((y_ax, x_ax), f, method="cubic")(q)
                              for f in fields])
    check("stacked cubic readout == per-field readout",
          np.allclose(stacked, single, atol=1e-12))


def test_notebook_markers():
    nb = json.loads(rt.SYNTH_NOTEBOOK.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    counts = {m: sum(m in c for c in cells) for m in rt.SYNTH_CELL_MARKERS}
    check("synthetic notebook markers each hit one cell",
          all(v == 1 for v in counts.values()), str(counts))


def test_analysis():
    try:
        import pandas as pd
    except ImportError:
        print("[SKIP] pandas not installed: analysis not tested")
        return
    rng = np.random.default_rng(5)
    rows = []
    for cls in rt.CLASSES:
        for r in range(10):
            rid = f"{cls}_{r}"
            base = rng.uniform(0.01, 0.03)
            meta = dict(bench="real", scene=rid, scene_class=cls, region_id=rid, patch_id=rid)
            rows.append(dict(meta, family="geo_cubic", method="classical_cubic",
                             variant="", rmse=base))
            for lam in (1e-3, 1e-2):
                for f in (1.0, 1.2):
                    best = lam == 1e-2 and f == 1.0
                    rows.append(dict(meta, family="psf_aware", method="psf_aware", variant="",
                                     width_factor=f, lam=lam, n_iter=14,
                                     rmse=base * (0.8 if best else 0.9) * rng.uniform(0.99, 1.01)))
            for it in (10, 100):
                rows.append(dict(meta, family="rl2", method="rl2_clip", variant="clip",
                                 width_factor=1.0, n_iter=it,
                                 rmse=base * (0.85 if it == 100 else 0.95)))
    df = rt.prepare(pd.DataFrame(rows, columns=rt.RESULT_COLUMNS))
    folds = rt.assign_folds(df[["region_id", "scene_class"]].drop_duplicates())
    per = pd.Series(folds).groupby(lambda r: r.rsplit("_", 1)[0]).apply(
        lambda s: s.value_counts().tolist())
    check("folds: 2 regions per class per fold", all(v == [2] * 5 for v in per))
    sel = rt.select(df, "psf_aware", "B", "5fold")
    check("CV selects the uniformly best configuration",
          (sel["config"].str.contains(r"\|1\|0.01\|14\|")).all(), sel["config"].iloc[0])
    sel_a = rt.select(df, "rl2", "A", "loo_class")
    check("LOO-class selection picks 100 iterations", (sel_a["config"].str.contains(r"\|100\|")).all())
    oracle = rt.select(df, "psf_aware", "B", "oracle").set_index("region_id")["rmse"]
    cv = sel.set_index("region_id")["rmse"].reindex(oracle.index)
    check("oracle <= CV per region", (oracle <= cv + 1e-15).all())
    stats = rt.paired_stats(sel, sel_a)
    allrow = [r for r in stats if r["scene_class"] == "all"][0]
    check("paired stats: PSF-aware wins every region", allrow["wins"] == 40,
          f"p={allrow['sign_test_p']:.2e}")
    m, lo, hi = rt.bootstrap_mean_ci(np.ones(10))
    check("bootstrap of a constant", m == lo == hi == 1.0)


def test_e3_kernels():
    """E3: the Gaussian (x) box weight, its pixel kernel, the half-pixel
    Gaussian of arm S1 (protocol section E3)."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    sigma = rt._SIGMA_G
    t = np.linspace(-25.0, 25.0, 41)
    uu, vv = np.meshgrid(t, t)
    gauss = np.exp(-(uu ** 2 + vv ** 2) / (2.0 * sigma ** 2))
    w0 = rt.box_gaussian_weight(uu, vv, a=1e-3)
    check("box (x) Gaussian -> Gaussian as the aperture -> 0",
          np.max(np.abs(w0 - gauss)) < 1e-6, f"max diff {np.max(np.abs(w0 - gauss)):.1e}")
    w = rt.box_gaussian_weight(uu, vv)
    check("box (x) Gaussian: unit peak, symmetric, separable",
          np.isclose(rt.box_gaussian_weight(0.0, 0.0), 1.0) and np.allclose(w, w.T)
          and np.allclose(w, w[::-1, ::-1]))
    # integral on a fine grid (0.25 m) against the closed form a^2 / peak
    s = np.arange(-60.0, 60.0 + 1e-9, 0.25)
    su, sv = np.meshgrid(s, s)
    num = rt.box_gaussian_weight(su, sv).sum() * 0.25 ** 2
    check("box (x) Gaussian integral == a^2 / peak",
          abs(num / rt.box_gaussian_integral() - 1.0) < 1e-6,
          f"{num:.6f} vs {rt.box_gaussian_integral():.6f}")
    # second moment per axis: sigma^2 + a^2 / 12 (the equivalent Gaussian)
    var = (rt.box_gaussian_weight(su, sv) * su ** 2).sum() * 0.25 ** 2 / num
    check("box (x) Gaussian variance == sigma_G^2 + sigma_box^2",
          abs(var / (rt._SIGMA_G ** 2 + rt._SIGMA_BOX ** 2) - 1.0) < 1e-6)
    k = rt.box_gaussian_pixel_kernel(10.0)
    check("S3 pixel kernel: 7x7, sums to 1, symmetric",
          k.shape == (7, 7) and np.isclose(k.sum(), 1.0) and np.allclose(k, k.T)
          and np.allclose(k, k[::-1, ::-1]))

    sigma_px = rt._SIGMA_G / 2.5
    const = np.full((40, 44), 3.25)
    check("half-pixel Gaussian keeps a constant",
          np.allclose(rt.half_shift_gaussian(const, sigma_px), 3.25, atol=1e-12))
    # sampled analytic Gaussian bump: the result must follow the analytic
    # convolution (variances add) at the half-pixel points
    n, sb = 64, 2.0
    i = np.arange(n, dtype=float)
    yy, xx = np.meshgrid(i, i, indexing="ij")
    c0, c1 = 31.3, 30.6
    bump = np.exp(-((yy - c0) ** 2 + (xx - c1) ** 2) / (2 * sb ** 2))
    out = rt.half_shift_gaussian(bump, sigma_px)
    st2 = sb ** 2 + sigma_px ** 2
    exact = (sb ** 2 / st2) * np.exp(-((yy + 0.5 - c0) ** 2 + (xx + 0.5 - c1) ** 2) / (2 * st2))
    err = np.max(np.abs(out - exact)) / exact.max()
    check("half-pixel Gaussian == analytic convolution (<= 1e-3 of peak)", err <= 1e-3,
          f"max err {err:.1e}")
    # smooth texture: agrees with a cubic-spline reading of gaussian_filter
    tex = smooth_image((96, 96), seed=7)
    ri, ci = np.meshgrid(np.arange(96) + 0.5, np.arange(96) + 0.5, indexing="ij")
    ref = map_coordinates(gaussian_filter(tex, sigma_px, mode="reflect"),
                          [ri.ravel(), ci.ravel()], order=3, mode="reflect").reshape(96, 96)
    ours = rt.half_shift_gaussian(tex, sigma_px)
    inner = (slice(12, -12), slice(12, -12))
    rel = np.sqrt(np.mean((ours[inner] - ref[inner]) ** 2)) / np.sqrt(np.mean(ref[inner] ** 2))
    check("half-pixel Gaussian == spline reading of gaussian_filter (<= 1e-3 RMS)",
          rel <= 1e-3, f"rel RMS {rel:.1e}")


def test_e5_geometry():
    """E5: sign of the registration shift, reference read at the shifted
    positions, raster readout, evaluation cells (protocol section E5)."""
    try:
        import revision_e5 as e5
        from affine import Affine
    except ImportError as exc:
        print(f"[SKIP] E5 geometry not tested: {exc}")
        return
    rng = np.random.default_rng(11)
    kk = rng.uniform(2 * np.pi / 200.0, 2 * np.pi / 40.0, 12)
    th = rng.uniform(0.0, 2 * np.pi, 12)
    ph = rng.uniform(0.0, 2 * np.pi, 12)

    def field(x, y):
        return 0.1 + 0.02 * sum(np.cos(k * (np.cos(t) * x + np.sin(t) * y) + p)
                                for k, t, p in zip(kk, th, ph))

    x0, y1 = 500000.0, 5000000.0
    t10 = Affine(10.0, 0.0, x0, 0.0, -10.0, y1)
    t05 = Affine(5.0, 0.0, x0, 0.0, -5.0, y1)
    xs10, ys10 = e5.s2_centres(t10, (128, 128))
    xs05, ys05 = e5.s2_centres(t05, (256, 256))
    X10, Y10 = np.meshgrid(xs10, ys10)
    X05, Y05 = np.meshgrid(xs05, ys05)
    s2 = field(X10, Y10)
    dx, dy = 3.0, -2.0            # VENmuS shows the scene of p + (dx, dy) at p
    ven = field(X05 + dx, Y05 + dy)
    (drow, dcol), dev = e5.estimate_shift(s2, ven)
    check("E5 shift: sign and size (VENmuS read at x - 10 dcol, y + 10 drow)",
          abs(10 * dcol - dx) < 0.1 and abs(10 * drow + dy) < 0.1 and dev < 0.05,
          f"drow {drow:+.3f} dcol {dcol:+.3f} px, quadrant dev {dev:.3f}")
    ii, jj = rng.integers(60, 196, 50), rng.integers(60, 196, 50)
    px, py = xs05[jj] + dx, ys05[ii] + dy
    ref = e5.reference_values(ven, t05, px[:, None], py[:, None], (drow, dcol), (2.0, 0.5))
    check("E5 reference: registered and harmonized VENmuS == 2 F + 0.5",
          np.allclose(ref, 2.0 * field(px, py) + 0.5, atol=1e-12))
    lin = 2.0 * X10 + 3.0 * Y10
    qx = rng.uniform(x0 + 100.0, x0 + 1100.0, (40, 16))
    qy = rng.uniform(y1 - 1100.0, y1 - 100.0, (40, 16))
    v = e5.readout([lin], xs10, ys10, qx, qy, method="linear")[:, 0]
    check("E5 readout: north-up raster read at sub-cell points",
          np.allclose(v, (2.0 * qx + 3.0 * qy).mean(axis=1), rtol=0, atol=1e-6))
    crs = "EPSG:32631"
    cells = e5.evaluation_cells(t10, (128, 128), crs, 20)
    cx, cy = e5.child_points_utm(cells, 20, 23, crs)
    mx, my = cx.mean(axis=1), cy.mean(axis=1)
    inside = ((mx > x0 + 195) & (mx < x0 + 1085) & (my < y1 - 195) & (my > y1 - 1085)).all()
    expect = 880.0 ** 2 / (4 * np.pi * 6371000.0 ** 2 / (12 * 4 ** 20))
    check("E5 evaluation cells: inside the 200 m margin, count ~ area / cell area",
          inside and 0.9 < len(cells) / expect < 1.1, f"{len(cells)} cells, ~{expect:.0f}")


def main():
    for test in (test_rl, test_fourier, test_operators_and_cg, test_stacked_cubic,
                 test_notebook_markers, test_analysis, test_e3_kernels, test_e5_geometry):
        try:
            test()
        except Exception as exc:
            traceback.print_exc()
            check(test.__name__, False, f"raised {type(exc).__name__}: {exc}")
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
