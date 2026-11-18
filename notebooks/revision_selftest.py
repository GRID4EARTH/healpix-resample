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


def test_e5_spectral():
    """E5-S (exploratory): gain, registration phase, alias partners and the
    error-budget normalization on plane waves of known transfer."""
    try:
        import pandas as pd
        import revision_e5_spectral as e5s
    except ImportError as exc:
        print(f"[SKIP] E5 spectral not tested: {exc}")
        return
    u10 = 10.0 * (np.arange(128) + 0.5)
    u05 = 5.0 * (np.arange(256) + 0.5)
    R10, C10 = np.meshgrid(u10, u10, indexing="ij")
    R05, C05 = np.meshgrid(u05, u05, indexing="ij")

    def waves(ur, uc, specs, amps):
        return sum(a * np.cos(2 * np.pi * k * (np.cos(t) * ur + np.sin(t) * uc) + p)
                   for (k, t, p), a in zip(specs, amps))

    def frame(s2, ven, shift, gain):
        rings, scalars = e5s.patch_rings(s2, ven, shift, gain)
        return e5s._ring_frame(rings), pd.DataFrame([scalars])

    specs = [(0.012, 0.3, 0.1), (0.025, 1.2, 2.0), (0.040, 2.2, 4.0)]
    D = (4.0, -7.0)               # m, rows / columns: VENmuS shows the scene of u + D at u
    mtf = e5s.nominal_mtf(np.array([s[0] for s in specs]))
    s2 = 0.1 + waves(R10, C10, specs, 0.01 * mtf)
    ven = 0.3 + 2.0 * waves(R05 + D[0], C05 + D[1], specs, 0.01 * np.ones(3))
    rings, _ = frame(s2, ven, (D[0] / 10, D[1] / 10), 0.5)
    d = e5s.derive(rings).set_index("ring")
    errs = []
    for (k, _, _), m in zip(specs, mtf):
        r = int(k * e5s.PATCH_M // e5s.RING)
        errs.append((abs(d.H[r] * m - 1), 1 - d.gamma2[r]))
    errs = np.array(errs)
    check("E5-S: H == 1/MTF and gamma2 == 1 on registered plane waves",
          (errs[:, 0] < 0.02).all() and (errs[:, 1] < 0.01).all(),
          f"max |H MTF - 1| {errs[:, 0].max():.1e}, 1 - gamma2 {errs[:, 1].max():.1e}")
    wf, wp = frame(s2, ven, (-D[0] / 10, -D[1] / 10), 0.5)
    wrong = e5s.derive(wf).set_index("ring")
    r = int(0.040 * e5s.PATCH_M // e5s.RING)
    resid = np.array([wp.resid_shift_row_m[0], wp.resid_shift_col_m[0]])
    # registering by -D leaves VENmuS showing F(u + 2 D): residual +2 D
    check("E5-S: wrong-sign shift lowers gamma2; refinement finds the residual 2D",
          wrong.gamma2[r] < 0.8 and wrong.gamma2_refined[r] > 0.99
          and np.abs(resid - 2 * np.array(D)).max() < 0.3,
          f"gamma2 {wrong.gamma2[r]:.2f} -> {wrong.gamma2_refined[r]:.3f}, "
          f"residual {resid.round(2)} m")
    spec_a = [(0.07, 0.0, 0.7)]   # beyond the Sentinel-2 Nyquist: aliased to 0.03 m^-1
    s2a = 0.1 + waves(R10, C10, spec_a, [0.01])
    vena = 0.1 + waves(R05, C05, spec_a, [0.01])
    da = e5s.derive(frame(s2a, vena, (0.0, 0.0), 1.0)[0]).set_index("ring")
    r = int(0.03 * e5s.PATCH_M // e5s.RING)
    check("E5-S: alias partner k + 1/(10 m) carries the aliased wave",
          da.alias_row[r] > 0.95, f"alias coherence {da.alias_row[r]:.3f}")
    rng = np.random.default_rng(13)
    sigma = 0.004
    fine = waves(R05, C05, specs, [0.01] * 3)
    s2n = 0.1 + waves(R10, C10, specs, [0.01] * 3)
    venn = 0.1 + fine + sigma * rng.standard_normal(fine.shape)
    rn, pn = frame(s2n, venn, (0.0, 0.0), 1.0)
    b = e5s.budget(rn, pn)
    check("E5-S: budget RMSE of the identity == white-noise sigma",
          abs(b["rmse_identity_est"] / sigma - 1) < 0.03,
          f"{b['rmse_identity_est']:.5f} vs {sigma}, out-of-band share {b['share_out']:.2f}")


def _fake_gt():
    """The constants of the paper's real-data module that E6 changes, with
    their paper values (for hosts where the module cannot be imported)."""
    from types import SimpleNamespace
    return SimpleNamespace(
        REGION_DATA_SUBDIR="multi_patch_sentinel2", BAND="b04", PATCH_SIZE=256,
        CENTRAL_SIZE=256, PIXEL_SIZE_M=10.0, BLOCK=2, PIXEL_SIZE_COARSE_M=20.0,
        NATIVE_LEVEL=20, NATIVE_FWHM_M=12.5, REF_LEVEL=19, DEGRADE_FWHM_M=25.0,
        RECON_FWHM_M=25.0, EDGE_MARGIN_FACTOR=8.0, EDGE_MARGIN_M=200.0,
        MAX_BRIGHT_FRACTION=0.30, _PATCH_FINGERPRINT_CACHE={}, region_coordinates={})


def test_e6():
    """E6: arm constants, context restore, pixel windows, random draw,
    Landsat scaling, E6e region choice, reproduction criterion."""
    import revision_e6 as e6
    try:
        gt = rt.load_gt()
        real = True
    except Exception as exc:
        print(f"[SKIP] paper module not importable ({type(exc).__name__}); fake constants")
        gt, real = _fake_gt(), False
    if real:
        want = e6.arm_constants(gt, e6.E1_ARM)
        have = {n: getattr(gt, n) for n in e6.GT_NAMES}
        check("E6: the E1 arm reproduces the paper constants", want == have,
              str({k: (want[k], have[k]) for k in want if want[k] != have[k]}))
    before = {n: getattr(gt, n) for n in e6.GT_NAMES}
    gt._PATCH_FINGERPRINT_CACHE["x"] = "stale"
    with e6.gt_config(gt, e6.ARMS["b_4_19"]):
        inside = dict(block=gt.BLOCK, fwhm=gt.DEGRADE_FWHM_M, recon=gt.RECON_FWHM_M,
                      margin=gt.EDGE_MARGIN_M, central=gt.CENTRAL_SIZE,
                      sub=gt.REGION_DATA_SUBDIR, cache=dict(gt._PATCH_FINGERPRINT_CACHE))
        gt._PATCH_FINGERPRINT_CACHE["y"] = "arm"
    check("E6: b_4_19 sets 40 m, 50 m, 400 m, 512^2 and its own data directory",
          inside == dict(block=4, fwhm=50.0, recon=50.0, margin=400.0, central=512,
                         sub="revision_e6/s2_b04_512", cache={}), str(inside))
    try:
        with e6.gt_config(gt, e6.ARMS["d_l19"]):
            ls = (gt.PIXEL_SIZE_M, gt.DEGRADE_FWHM_M, gt.EDGE_MARGIN_M, gt.NATIVE_LEVEL,
                  gt.REF_LEVEL, gt.NATIVE_FWHM_M, gt.BAND)
            raise KeyError("inside")
    except KeyError:
        pass
    check("E6: d_l19 sets 30 m, 75 m, 600 m, levels 19/18, 37.5 m, band red",
          ls == (30.0, 75.0, 600.0, 19, 18, 37.5, "red"), str(ls))
    after = {n: getattr(gt, n) for n in e6.GT_NAMES}
    check("E6: constants and fingerprint cache restored (also after an exception)",
          after == before and gt._PATCH_FINGERPRINT_CACHE == {})

    from affine import Affine
    t = Affine(10.0, 0.0, 300000.0, 0.0, -10.0, 5000000.0)
    col0, row0 = 1234, 567
    xf, yf = t.c + (col0 + 0.5) * 10.0, t.f - (row0 + 0.5) * 10.0
    offs = {n: e6.window_offsets(xf, yf, t, new_size=n) for n in (256, 512, 1024)}
    centre = {n: (t.c + (c + n / 2) * 10.0, t.f - (r + n / 2) * 10.0)
              for n, (c, r) in offs.items()}
    check("E6: windows of 256/512/1024 share the frozen patch's centre",
          offs[256] == (col0, row0) and offs[512] == (col0 - 128, row0 - 128)
          and len(set(centre.values())) == 1, str(offs))

    lat, lon = e6.draw_points()
    lat2, _ = e6.draw_points()
    s0, s1 = np.sin(np.radians(e6.RANDOM_LAT))
    s = np.sin(np.radians(lat))
    sd = (s1 - s0) / math.sqrt(12.0 * lat.size)
    check("E6c: draw is reproducible, in the latitude band, uniform in area",
          np.array_equal(lat, lat2) and lat.min() >= -56.0 and lat.max() <= 84.0
          and abs(s.mean() - 0.5 * (s0 + s1)) < 4 * sd and abs(lon.mean()) < 4 * 360 / math.sqrt(12 * lon.size),
          f"mean sin(lat) {s.mean():.4f} vs {0.5 * (s0 + s1):.4f}")
    refl = e6.landsat_reflectance(np.array([0, 7273, 20000]))
    check("E6d: Landsat C2 L2 reflectance (DN 0 -> NaN, 2.75e-5 DN - 0.2)",
          np.isnan(refl[0]) and abs(refl[1] - 7.5e-6) < 1e-12 and abs(refl[2] - 0.35) < 1e-12,
          str(refl))

    fake = _fake_gt()
    scenes = []
    for cls in rt.CLASSES:
        for rid in ("zeta", "alpha", "mid"):
            s_ = f"region__{cls}__{rid}"
            fake.region_coordinates[s_] = dict(scene_class=cls, region_id=rid)
            scenes.append(s_)
    picked = e6.e6e_scenes(fake, scenes)
    check("E6e: 2 regions per class, first in alphabetical order",
          picked == [f"region__{c}__{r}" for c in rt.CLASSES for r in ("alpha", "mid")])

    import pandas as pd
    comp = pd.DataFrame([
        dict(arm="e1", scheme="5fold", mode="B", competitor="rl2", scene_class="all",
             mean_rel_margin=0.014, rel_ci_low=0.01, rel_ci_high=0.02, wins=39, n_regions=40),
        dict(arm="x", scheme="5fold", mode="B", competitor="rl2", scene_class="all",
             mean_rel_margin=0.01, rel_ci_low=0.002, rel_ci_high=0.02, wins=30, n_regions=40),
        dict(arm="y", scheme="5fold", mode="B", competitor="rl2", scene_class="all",
             mean_rel_margin=0.01, rel_ci_low=-0.002, rel_ci_high=0.02, wins=25, n_regions=40),
        dict(arm="z", scheme="5fold", mode="B", competitor="rl2", scene_class="all",
             mean_rel_margin=-0.01, rel_ci_low=-0.02, rel_ci_high=-0.002, wins=5, n_regions=40)])
    rep = e6.reproduction(comp).set_index("arm")["reproduced"]
    check("E6: reproduction = same sign as E1 and CI excluding 0",
          bool(rep["x"]) and not bool(rep["y"]) and not bool(rep["z"]), str(rep.to_dict()))
    check("E6-X: names unchanged by the E6-Y generalization",
          e6.FINE_FAMILIES == ("psf_aware", "psf_aware_l19", "psf_aware_l20", "rl2", "tik_id",
                               "tik_lap", "wiener_hunt", "geo_cubic")
          and e6.FINE_STUDY == "e6x_fine" and e6.FINE_ARM == "b_2_18"
          and e6.FINE_RUNS["e6y"].study == "e6y_fine"
          and e6.ARMS[e6.FINE_RUNS["e6y"].arm].ref_level == 17)
    try:
        from healpix_resample.psf_geometry import fwhm_to_scale, recommend_npt
    except ImportError as exc:
        print(f"[SKIP] E6-Y neighbours: {exc}")
    else:
        same_x, mass_y = True, []
        for key, fwhm0 in (("e6x", 25.0), ("e6y", 75.0)):
            for level in e6.FINE_RUNS[key].levels:
                for f in rt.WIDTH_FACTORS["real"]:
                    s = fwhm_to_scale(f * fwhm0)
                    free = recommend_npt(s, level, q_max=e6.FINE_NPT_MAX)
                    if key == "e6x":
                        same_x &= free["npt"] == recommend_npt(s, level)["npt"]
                    else:
                        mass_y.append(free["mass_at_npt"])
        check("E6-Y: lifting the Npt cap leaves E6-X unchanged and keeps 99 % of the "
              "E6-Y kernel mass", same_x and min(mass_y) >= 0.99, f"min mass {min(mass_y):.4f}")
    try:
        import healpix_geo
    except ImportError as exc:
        print(f"[SKIP] E6-X/E6-Y children: {exc}")
        return
    ok = True
    for run in e6.FINE_RUNS.values():
        parent_level = e6.ARMS[run.arm].ref_level
        parents = np.asarray(healpix_geo.nested.lonlat_to_healpix(
            np.array([2.35, -60.1, 140.0]), np.array([48.85, -33.4, 35.7]), parent_level,
            ellipsoid="WGS84"), dtype=np.int64)
        for level in run.levels:
            k = 4 ** (level - parent_level)
            kids = (parents[:, None] * k + np.arange(k)[None, :]).reshape(-1)
            lon, lat = healpix_geo.nested.healpix_to_lonlat(kids.astype(np.uint64), level,
                                                            ellipsoid="WGS84")
            back = np.asarray(healpix_geo.nested.lonlat_to_healpix(
                np.asarray(lon), np.asarray(lat), parent_level, ellipsoid="WGS84"))
            ok &= np.array_equal(back.reshape(len(parents), k),
                                 np.repeat(parents[:, None], k, 1))
    check("E6-X/E6-Y: nested children (L19, L20 of L18; L18, L19 of L17) lie in their cell", ok)


def test_e7():
    """E7: quadrature overlap weights (partition of unity, conservation,
    constant field, agreement with the exact spherical overlap), hard binning
    and constrained PSF-aware conservation."""
    import revision_e7 as e7
    rng = np.random.default_rng(5)
    nx, ny, pix = 23, 17, 20.0
    xs = 1000.0 + pix * np.arange(nx)
    ys = 5000.0 - pix * np.arange(ny)              # north-up, descending
    cell = 12.4

    def to_cells(px, py):
        i = np.floor((px - 987.0) / cell).astype(np.int64)
        j = np.floor((py - 4321.0) / cell).astype(np.int64)
        return i * 100000 + j
    cells, O = e7.overlap_weights(xs, ys, pix, to_cells, quad=e7.QUAD)
    colsum = np.asarray(O.sum(axis=0)).ravel()
    y = rng.random(nx * ny) + 0.5
    x, covered = e7.overlap_remap(O, y)
    xc, _ = e7.overlap_remap(O, np.full(nx * ny, 3.25))
    check("E7 overlap: each pixel's weights sum to 1; integral conserved; constant kept",
          np.allclose(colsum, 1.0, atol=1e-12)
          and abs(e7.conservation_error(covered, x, y.sum())) < 1e-13
          and np.allclose(xc, 3.25, atol=1e-12))
    # exact areas on this planar grid: pixel square x cell square intersections
    px0 = xs[:, None] - pix / 2
    ex = []
    for k, cid in enumerate(cells[:200]):
        i, j = divmod(int(cid), 100000)
        cx0, cy0 = 987.0 + i * cell, 4321.0 + j * cell
        row = O.getrow(k).toarray().ravel()
        for p in np.flatnonzero(row):
            r_, c_ = divmod(p, nx)
            ox = max(0.0, min(xs[c_] + pix / 2, cx0 + cell) - max(xs[c_] - pix / 2, cx0))
            oy = max(0.0, min(ys[r_] + pix / 2, cy0 + cell) - max(ys[r_] - pix / 2, cy0))
            ex.append(abs(row[p] - ox * oy / pix ** 2))
    del px0
    # one side of the overlap is off by at most one sub-square: 2/Q + 1/Q^2
    bound = 2.0 / e7.QUAD + 1.0 / e7.QUAD ** 2
    check("E7 overlap: quadrature areas within (2Q+1)/Q^2 of a pixel of the exact ones",
          max(ex) <= bound + 1e-12, f"max abs error {max(ex):.3f}, mean "
          f"{np.mean(ex):.4f} of a pixel")
    try:
        import healpix_geo
        from healpix_resample.overlap_conservative import OverlapConservativeResampler
    except Exception as exc:
        print(f"[SKIP] E7 package comparisons: {type(exc).__name__}: {exc}")
        return
    lonb = np.linspace(2.0, 2.2, 21)
    latb = np.linspace(45.0, 45.16, 17)
    level = 12
    ocr = OverlapConservativeResampler(lonb, latb, level, normalization="covered")
    lonc, latc = 0.5 * (lonb[1:] + lonb[:-1]), 0.5 * (latb[1:] + latb[:-1])
    f = 0.3 + 0.1 * np.sin(np.radians(lonc)[None, :] * 300) * np.cos(np.radians(latc)[:, None] * 200)
    exact = ocr.resample(f)
    mine_cells, Om = e7.overlap_weights(
        lonc, latc, 0.01,
        lambda a, b: healpix_geo.nested.lonlat_to_healpix(a, b, level, ellipsoid="sphere"))
    mine, _ = e7.overlap_remap(Om, f.reshape(-1))
    ex_map = dict(zip(np.asarray(exact.cell_ids).tolist(), np.asarray(exact.cell_data)))
    common = [k for k, c in enumerate(mine_cells) if int(c) in ex_map]
    d = np.array([mine[k] - ex_map[int(mine_cells[k])] for k in common])
    # cells the quadrature misses can only be slivers of the grid edge
    check("E7 overlap: quadrature remap == OverlapConservativeResampler (lon/lat grid)",
          len(common) >= 0.95 * len(ex_map) and np.max(np.abs(d)) < 2e-3,
          f"{len(common)}/{len(ex_map)} cells, max |diff| {np.max(np.abs(d)):.1e}")
    try:
        import torch  # noqa: F401
        from healpix_resample import ConservativeResampler, PSFResampler, fwhm_to_scale
    except Exception as exc:
        print(f"[SKIP] E7 PSF and binning checks: {type(exc).__name__}: {exc}")
        return
    lon2, lat2 = np.meshgrid(np.linspace(2.0, 2.02, 40), np.linspace(45.0, 45.015, 36))
    vals = 0.2 + 0.05 * rng.random(lon2.size)
    area = np.ones(lon2.size)
    cr = ConservativeResampler(lon_deg=lon2.ravel(), lat_deg=lat2.ravel(), level=18, area=area,
                               verbose=False)
    h = rt.to_numpy(cr.resample(vals).cell_data)
    check("E7 hard binning: sum of cell totals == sum of a_i y_i",
          abs(h.sum() - vals.sum()) < 1e-10 * vals.sum())
    rs = PSFResampler(lon_deg=lon2.ravel(), lat_deg=lat2.ravel(), level=18,
                      sigma_m=fwhm_to_scale(60.0), Npt=20, threshold=0.1, verbose=False,
                      area=area)
    errs = []
    for cons in (False, True):
        out = rt.to_numpy(rs.resample(vals, lam=1e-2, max_iter=30, conservative=cons).cell_data)
        errs.append(abs(e7.conservation_error(np.full(out.size, area.sum() / out.size),
                                              out.astype(np.float64), vals.sum())))
    check("E7 constrained PSF-aware conserves the mean (float32)", errs[1] < 1e-5,
          f"unconstrained {errs[0]:.1e}, constrained {errs[1]:.1e}")


def main():
    for test in (test_rl, test_fourier, test_operators_and_cg, test_stacked_cubic,
                 test_notebook_markers, test_analysis, test_e3_kernels, test_e5_geometry,
                 test_e5_spectral, test_e6, test_e7):
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
