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


def main():
    for test in (test_rl, test_fourier, test_operators_and_cg, test_stacked_cubic,
                 test_notebook_markers, test_analysis):
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
