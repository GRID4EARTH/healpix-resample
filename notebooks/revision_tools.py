"""Shared machinery for the JSTARS major revision: experiments E0 to E4.

Protocol (frozen before any production run): ``review/08_protocole_preenregistre.md``.
Command-line entry point: ``notebooks/revision_run.py``.

Every function here expects ``cwd == notebooks/``: the paper modules it reuses
resolve their caches and tables relative to that directory.  Nothing in this
module writes into a CSV the paper is built from; every output goes under
``notebooks/tables/revision/``.

Conventions
-----------
* Paper notation: A (N x K) maps a HEALPix field to the source samples, B
  (K x N) maps samples to cells, ``x_ref = B y``.  In ``healpix_resample`` the
  row-vector conventions are ``x_ref = y @ M`` and ``y_hat = x @ MT``, so
  ``B = M.T`` and ``A = MT.T``.
* Long-format result rows: one row per (region, family, configuration), with
  ``rmse`` against the benchmark's reference on the benchmark's own cells.
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve as signal_convolve

NOTEBOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NOTEBOOK_DIR.parent
OUT_DIR = NOTEBOOK_DIR / "tables" / "revision"

# =============================================================================
# Protocol constants -- frozen in review/08_protocole_preenregistre.md
# =============================================================================
CV_SEED = 20261116
BOOT_SEED = 20260818          # same seed as the paper's region bootstrap
N_BOOT = 5000
N_FOLDS = 5
CLASSES = ("agriculture", "forest", "urban", "water")

REAL_NOMINAL_FWHM_M = 25.0    # imposed degradation (gt.DEGRADE_FWHM_M)
SYNTH_NOMINAL_FWHM_M = 12.5   # synthetic Gaussian component
_SIGMA_G = SYNTH_NOMINAL_FWHM_M / (2.0 * math.sqrt(2.0 * math.log(2.0)))
_SIGMA_BOX = 10.0 / math.sqrt(12.0)
# Second-moment equivalent Gaussian of Gaussian (x) 10 m box: 14.23 / 12.5.
SYNTH_EQUIV_FACTOR = math.sqrt(_SIGMA_G ** 2 + _SIGMA_BOX ** 2) / _SIGMA_G
# Amendment 1 (2026-11-16, review/08_protocole_preenregistre.md): run 1 selected
# the edge of every open grid marked "A1" below; those grids were extended and
# E1/E2 re-run.  Run 1 is kept in tables/revision_run1/.  Natural bounds
# (lam = 0, SIRT omega < 2) are not open edges.
WIDTH_FACTORS = {
    # sqrt(1.25): combined nominal response sqrt(25^2 + 12.5^2) = 27.95 m.
    "real": (0.9, 1.0, math.sqrt(1.25), 1.2, 1.3, 1.4, 1.5),             # A1: + 1.3-1.5
    "synthetic": (0.9, 1.0, SYNTH_EQUIV_FACTOR, 1.2, 1.3, 1.4, 1.5),     # A1: + 1.3-1.5
}
NOMINAL_FWHM = {"real": REAL_NOMINAL_FWHM_M, "synthetic": SYNTH_NOMINAL_FWHM_M}

PSF_LAMBDAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
PSF_ITERS = (5, 14, 30, 100)          # package runs; run-1 choices interior or converged
PSF_TOL = 1e-8
E2_ITERS = (5, 14, 30, 100, 300, 1000)                     # A1: CG-type E2 arms, + 300, 1000
RL_ITERS = (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)  # A1: + 1000-5000
TIK_LAMBDAS = tuple(float(v) for v in np.logspace(-10, 0, 21))  # A1: down to 1e-10
SIRT_OMEGAS = (0.5, 1.0, 1.5, 1.9)                         # A1: + 1.9 (< 2)
SIRT_ITERS = (5, 14, 30, 100, 300, 1000)                   # A1: + 300, 1000
MLEM_ITERS = (5, 10, 20, 50, 100, 200, 500, 1000, 2000)    # A1: + 500-2000
# The package check compares e2_wcg with the package at every PSF_ITERS point.
assert set(PSF_ITERS) <= set(E2_ITERS)

# E4 (width x lambda sensitivity): the widths >= 0.9 come from the E1 run;
# this complement adds the under-estimation side planned for E4.
E4_EXTRA_WIDTHS = (0.6, 0.7, 0.8)
E4_ITERS = (14, 30)
E4_COMPETITORS = ("rl2", "tik_id", "tik_lap", "wiener_hunt")
E4_SAFE_WIN = 0.9

# E3 (response mismatch from the 10 m detector aperture, synthetic bench).
# S0 and S2 are the E1 run at f = 1 and f = SYNTH_EQUIV_FACTOR; S1 and S3 are
# computed by `run_synthetic_region(..., arm=...)`.
E3_APERTURE_M = 10.0
E3_ARMS = ("S0", "S1", "S2", "S3")
E3_NEW_ARMS = ("S1", "S3")
E3_KERNEL_RADIUS_PX = 3          # discrete kernel of the deconvolutions (S3)
E3_FAMILIES = ("psf_aware", "rl2", "tik_id", "tik_lap", "wiener_hunt", "geo_cubic")
E3_CONTRASTS = (("S0", "S3"), ("S0", "S2"), ("S2", "S3"), ("S0", "S1"))

# The configurations reported in the submitted paper.
PAPER_CONFIG = {
    "real": {
        "psf_aware": dict(width_factor=1.0, lam=1e-2, n_iter=14),
        "rl2": dict(width_factor=1.0, n_iter=100, variant="clip"),
    },
    "synthetic": {
        "psf_aware": dict(width_factor=1.0, lam=1e-3, n_iter=14),
        "rl2": dict(width_factor=1.0, n_iter=100, variant="minmax"),
        "tik_id": dict(width_factor=1.0, lam=1e-2),
    },
}
PAPER_TABLES = {
    "real": ("real_groundtruth_multiregion_metrics_native20_ref19_block2_"
             "fwhm25_it14_refpsf_aware_estcell_average.csv"),
    "synthetic": "multi_patch_two_stage_40_metrics.csv",
}
E0_TOL = 1e-4

RESULT_COLUMNS = [
    "bench", "scene", "scene_class", "region_id", "patch_id",
    "family", "method", "variant", "width_factor", "fwhm_m",
    "lam", "n_iter", "omega", "rmse", "n_cells", "corr", "runtime_s",
]


# =============================================================================
# Small helpers
# =============================================================================
def to_numpy(a):
    """numpy view of a numpy array or a (possibly CUDA) torch tensor."""
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def ensure_out_dir(*parts) -> Path:
    path = OUT_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def gaussian_kernel(fwhm_m: float, pixel_size_m: float, radius_sigma: float = 4.0):
    """Normalized isotropic Gaussian, identical to the paper's RL kernels
    (``gt.reconstruct_richardson_lucy`` and the notebook's
    ``gaussian_psf_kernel``)."""
    sigma_px = fwhm_m / (2.0 * math.sqrt(2.0 * math.log(2.0))) / pixel_size_m
    radius = max(1, int(math.ceil(radius_sigma * sigma_px)))
    axis = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_px ** 2))
    return kernel / kernel.sum()


def box_gaussian_weight(u, v, sigma=_SIGMA_G, a=E3_APERTURE_M):
    """Gaussian (std ``sigma``) convolved with an ``a`` x ``a`` square
    aperture, at offsets ``(u, v)`` (metres, along the aperture's sides),
    scaled to a peak of 1 like the package's Gaussian (E3, arm S3):

        w = prod_{t in (u, v)} [erf((t + a/2)/(sqrt2 s)) - erf((t - a/2)/(sqrt2 s))] / 2
    """
    from scipy.special import erf
    s = math.sqrt(2.0) * sigma

    def axis(t):
        t = np.asarray(t, dtype=np.float64)
        return 0.5 * (erf((t + a / 2.0) / s) - erf((t - a / 2.0) / s))

    return axis(u) * axis(v) / float(axis(0.0)) ** 2


def box_gaussian_integral(sigma=_SIGMA_G, a=E3_APERTURE_M):
    """Integral (m^2) of `box_gaussian_weight` over the plane: a^2 / peak."""
    from scipy.special import erf
    return a * a / erf(a / (2.0 * math.sqrt(2.0) * sigma)) ** 2


def box_gaussian_pixel_kernel(pixel_m, radius_px=E3_KERNEL_RADIUS_PX, **kw):
    """`box_gaussian_weight` at the integer pixel offsets, normalized to sum
    1: the deconvolution kernel of arm S3 on the 10 m detector grid."""
    t = np.arange(-radius_px, radius_px + 1) * float(pixel_m)
    uu, vv = np.meshgrid(t, t)
    k = box_gaussian_weight(uu, vv, **kw)
    return k / k.sum()


def utm_box_kernel(crs_utm, **kw):
    """`PSFResampler(kernel=...)` for arm S3: `box_gaussian_weight` of the
    offset of the cell centre from the sample in the patch's UTM frame -- the
    frame the synthetic generation blurs and averages in (both points are
    projected with the patch's own transformer, as the generation does)."""
    import pyproj
    import torch
    to_utm = pyproj.Transformer.from_crs(4326, crs_utm, always_xy=True)

    def kernel(lon_s, lat_s, lon_c, lat_c):
        xs, ys = to_utm.transform(to_numpy(lon_s), to_numpy(lat_s))
        xc, yc = to_utm.transform(to_numpy(lon_c), to_numpy(lat_c))
        w = box_gaussian_weight(np.asarray(xc) - np.asarray(xs),
                                np.asarray(yc) - np.asarray(ys), **kw)
        return torch.as_tensor(w, dtype=torch.float64)
    return kernel


def half_shift_gaussian(image, sigma_px, truncate=4.0):
    """Gaussian convolution of ``image`` evaluated half a pixel down and right
    of each pixel centre: ``out[i, j]`` is the blurred value at (i + 1/2,
    j + 1/2).  The Gaussian is sampled at the half-integer offsets with the
    truncation and the boundary (``reflect``) of ``scipy.ndimage.gaussian_filter``;
    nothing is interpolated (E3, arm S1)."""
    radius = int(truncate * float(sigma_px) + 0.5)
    m = np.arange(-radius + 1, radius + 1)
    h = np.exp(-((m - 0.5) ** 2) / (2.0 * float(sigma_px) ** 2))
    h /= h.sum()
    out = np.asarray(image, dtype=np.float64)
    for axis in (0, 1):
        n = out.shape[axis]
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="symmetric")   # == ndimage "reflect"
        acc = np.zeros_like(out)
        for k, w in zip(m, h):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(radius + k, radius + k + n)
            acc += w * padded[tuple(sl)]
        out = acc
    return out


def point_sampled_detector(ns, patch_row, products):
    """Arm S1: the synthetic latent texture blurred by the 12.5 m Gaussian and
    sampled at the 10 m detector centres (no aperture average).  The detector
    centre (r, c) sits at fine-grid position (4r + 3/2, 4c + 3/2)."""
    from scipy.ndimage import gaussian_filter
    grids = ns["patch_coordinate_grids"](patch_row)
    latent_hr, _ = ns["load_or_sample_esri_patch"](patch_row, grids)
    sigma_m = ns["TRUE_FWHM_M"] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_px = sigma_m / (ns["DETECTOR_GSD_M"] / ns["OVERSAMPLE"])
    # Same latent and blur as build_synthetic_patch: refuse to go on otherwise.
    blurred_hr = gaussian_filter(latent_hr, sigma=sigma_px, mode="reflect")
    if not np.array_equal(ns["block_mean"](blurred_hr), products["blurred_detector"]):
        raise RuntimeError(f"{patch_row.patch_id}: blurred field differs from the paper's")
    os_ = int(ns["OVERSAMPLE"])
    if os_ != 4:
        raise RuntimeError("the half-pixel offset below assumes OVERSAMPLE = 4")
    shifted = half_shift_gaussian(latent_hr, sigma_px)
    return shifted[os_ // 2 - 1::os_, os_ // 2 - 1::os_], grids


def provenance() -> dict:
    """Code version and environment, written next to every run's outputs."""
    info = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0]}
    try:
        import subprocess
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        info["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT, text=True).strip())
    except Exception as exc:  # pragma: no cover - informational only
        info["git_error"] = repr(exc)
    for mod in ("numpy", "scipy", "torch", "skimage", "healpix_resample", "pandas"):
        try:
            info[mod] = getattr(importlib.import_module(mod), "__version__", "?")
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def write_provenance(tag: str, extra: dict | None = None) -> Path:
    path = ensure_out_dir() / f"provenance_{tag}.json"
    payload = {**provenance(), **(extra or {})}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# =============================================================================
# Deconvolution baselines (two-stage: deconvolve on the source grid, then read
# out on the common cell-average estimand)
# =============================================================================
def rl_snapshots(image, psf, iters, eps=1e-12):
    """Richardson--Lucy iterates at the requested iteration counts.

    Same recursion as ``skimage.restoration.richardson_lucy`` (0.26): constant
    0.5 start, ``scipy.signal.convolve(mode='same')``, ``eps`` added to the
    blurred estimate, no clipping.  One run to ``max(iters)`` replaces one run
    per grid point; ``check_rl_against_skimage`` verifies the equivalence.
    """
    image = np.asarray(image, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)
    psf_mirror = np.flip(psf)
    estimate = np.full(image.shape, 0.5, dtype=np.float64)
    targets = sorted({int(k) for k in iters})
    out = {}
    for k in range(1, targets[-1] + 1):
        conv = signal_convolve(estimate, psf, mode="same") + eps
        estimate *= signal_convolve(image / conv, psf_mirror, mode="same")
        if k in targets:
            out[k] = estimate.copy()
    return out


def check_rl_against_skimage(image, psf, num_iter=100):
    """Max abs difference between ``rl_snapshots`` and scikit-image."""
    from skimage.restoration import richardson_lucy
    ours = rl_snapshots(image, psf, [num_iter])[num_iter]
    ref = richardson_lucy(np.asarray(image, dtype=np.float64), psf,
                          num_iter=num_iter, clip=False)
    return float(np.max(np.abs(ours - ref)))


def rl_normalize(img, variant):
    """The two normalizations used by the paper's two RL baselines."""
    img = np.asarray(img, dtype=np.float64)
    if variant == "clip":        # real-data paper baseline
        lo, hi = np.percentile(img, [0.5, 99.5])
        hi = max(hi, lo + 1e-6)
        return np.clip((img - lo) / (hi - lo), 0.0, 1.0), lo, hi - lo
    if variant == "minmax":      # synthetic paper baseline
        lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
        scale = hi - lo if hi > lo else 1.0
        return np.clip((img - lo) / scale, 0.0, 1.0), lo, scale
    raise ValueError(f"unknown RL normalization {variant!r}")


def _transfer(kernel, shape):
    """Circular transfer function of a small centred kernel (as the paper's
    Fourier Tikhonov baseline builds it)."""
    kh, kw = kernel.shape
    full = np.zeros(shape, dtype=np.float64)
    full[:kh, :kw] = kernel
    full = np.roll(full, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.fft2(full)


_LAPLACIAN = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])


def tikhonov_fields(image, kernel, lambdas, regularizer="identity"):
    """Fourier Tikhonov deconvolution for every lambda.

    ``identity``: H* Y / (|H|^2 + lam), the paper's baseline.
    ``laplacian``: H* Y / (|H|^2 + lam |L|^2), a smoothness prior.
    """
    image = np.asarray(image, dtype=np.float64)
    H = _transfer(kernel, image.shape)
    Y = np.fft.fft2(image)
    if regularizer == "identity":
        R = np.ones(image.shape)
    elif regularizer == "laplacian":
        R = np.abs(_transfer(_LAPLACIAN, image.shape)) ** 2
    else:
        raise ValueError(regularizer)
    num = np.conj(H) * Y
    den_h = np.abs(H) ** 2
    return {float(lam): np.fft.ifft2(num / (den_h + lam * R)).real for lam in lambdas}


def wiener_hunt_field(image, kernel, seed=CV_SEED):
    """Self-tuned Wiener--Hunt deconvolution (Laplacian prior, noise and
    prior precisions sampled by Gibbs): no hyperparameter to tune."""
    from skimage.restoration import unsupervised_wiener
    image = np.asarray(image, dtype=np.float64)
    try:
        field, _ = unsupervised_wiener(image, kernel, clip=False,
                                       rng=np.random.default_rng(seed))
    except TypeError:  # older scikit-image: random_state
        field, _ = unsupervised_wiener(image, kernel, clip=False,
                                       random_state=seed)
    return np.asarray(field, dtype=np.float64)


def deconvolution_fields(image, kernel, *, rl_variants):
    """Every two-stage field for one width: {(family, variant, lam, n_iter): field}."""
    fields, runtimes = {}, {}
    for variant in rl_variants:
        t0 = time.perf_counter()
        img01, lo, scale = rl_normalize(image, variant)
        snaps = rl_snapshots(img01, kernel, RL_ITERS)
        dt = (time.perf_counter() - t0) / len(snaps)
        for k, est in snaps.items():
            key = ("rl2", variant, np.nan, k)
            fields[key] = est * scale + lo
            runtimes[key] = dt
    for reg, family in (("identity", "tik_id"), ("laplacian", "tik_lap")):
        t0 = time.perf_counter()
        res = tikhonov_fields(image, kernel, TIK_LAMBDAS, regularizer=reg)
        dt = (time.perf_counter() - t0) / len(res)
        for lam, est in res.items():
            key = (family, "", lam, np.nan)
            fields[key] = est
            runtimes[key] = dt
    t0 = time.perf_counter()
    try:
        fields[("wiener_hunt", "", np.nan, np.nan)] = wiener_hunt_field(image, kernel)
        runtimes[("wiener_hunt", "", np.nan, np.nan)] = time.perf_counter() - t0
    except Exception as exc:
        print(f"    [WARN] Wiener-Hunt failed: {type(exc).__name__}: {exc}")
    return fields, runtimes


# =============================================================================
# E2: alternative inversion schemes on the package's own operators
# =============================================================================
def extract_operators(resampler):
    """A (N x K), B (K x N), Dx (K,), cell ids (K,), uncomputable mask (K,)
    of a built ``PSFResampler``, as float64 scipy matrices."""
    import torch

    M = resampler.M.coalesce().cpu()                  # (N, K) COO
    n, k = M.shape
    idx = M.indices().numpy()
    B = sp.csr_matrix((M.values().double().numpy(), (idx[1], idx[0])), shape=(k, n))

    MT = resampler.MT.cpu()                           # (K, N), usually CSR
    if MT.layout == torch.sparse_csr:
        MT_sp = sp.csr_matrix(
            (MT.values().double().numpy(),
             MT.col_indices().numpy().astype(np.int64),
             MT.crow_indices().numpy().astype(np.int64)),
            shape=(k, n))
    else:
        MT = MT.coalesce()
        ij = MT.indices().numpy()
        MT_sp = sp.csr_matrix((MT.values().double().numpy(), (ij[0], ij[1])), shape=(k, n))
    A = MT_sp.T.tocsr()

    dx = to_numpy(resampler.cell_weight).astype(np.float64)
    bad = np.zeros(k, dtype=bool)
    unc = getattr(resampler, "uncomputable_out_cells", None)
    if unc is not None and unc.numel() > 0:
        bad[to_numpy(unc).astype(np.int64)] = True
    cell_ids = to_numpy(resampler.cell_ids).astype(np.int64)
    return dict(A=A, B=B, dx=dx, bad=bad, cell_ids=cell_ids)


def pcg_snapshots(H, b, iters, weight=None, precond=None, tol=PSF_TOL):
    """(Preconditioned) Conjugate Gradient from x0 = 0, iterates kept at the
    requested counts.  ``weight`` selects the inner product <u, v>_w; the
    recursion and stopping rule mirror ``healpix_resample.psf.conjugate_gradient``.
    Once converged, later snapshots repeat the converged iterate, exactly as a
    separate run with a larger ``max_iter`` would return it."""
    if weight is None:
        def dot(u, v):
            return float(np.dot(u, v))
    else:
        def dot(u, v):
            return float(np.dot(u * weight, v))
    targets = sorted({int(k) for k in iters})
    x = np.zeros_like(b)
    r = b.copy()
    z = r if precond is None else r * precond
    p = z.copy()
    rz = dot(r, z)
    b_norm = math.sqrt(dot(b, b))
    done = b_norm == 0.0
    out = {}
    for k in range(1, targets[-1] + 1):
        if not done:
            Ap = H(p)
            den = dot(p, Ap)
            if abs(den) < 1e-30:
                done = True
            else:
                alpha = rz / den
                x = x + alpha * p
                r = r - alpha * Ap
                if math.sqrt(dot(r, r)) <= tol * b_norm:
                    done = True
                else:
                    z = r if precond is None else r * precond
                    rz_new = dot(r, z)
                    p = z + (rz_new / rz) * p
                    rz = rz_new
        if k in targets:
            out[k] = x.copy()
    return out


def e2_fields(ops, y):
    """Every E2 estimate: {(family, variant, lam, n_iter, omega): x (K,)}."""
    A, B, dx = ops["A"], ops["B"], ops["dx"]
    At = A.T.tocsr()
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x_ref = B @ y
    r_ref = y - A @ x_ref
    out = {("e2_xref", "", np.nan, 0, np.nan): x_ref}

    diag_ba = np.asarray(B.multiply(A.T).sum(axis=1)).reshape(-1)
    b_w = B @ r_ref
    b_tik0 = At @ y
    b_tikd = At @ r_ref
    for lam in PSF_LAMBDAS:
        def H_w(v, lam=lam):
            return B @ (A @ v) + lam * v

        def H_e(v, lam=lam):
            return At @ (A @ v) + lam * v

        d = diag_ba + lam
        jac = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 1.0)
        runs = {
            "e2_wcg": (x_ref, pcg_snapshots(H_w, b_w, E2_ITERS, weight=dx)),
            "e2_cg_euclid": (x_ref, pcg_snapshots(H_w, b_w, E2_ITERS, weight=None)),
            "e2_jacobi": (x_ref, pcg_snapshots(H_w, b_w, E2_ITERS, weight=dx, precond=jac)),
            "e2_tik_x0": (0.0, pcg_snapshots(H_e, b_tik0, E2_ITERS, weight=None)),
            "e2_tik_dref": (x_ref, pcg_snapshots(H_e, b_tikd, E2_ITERS, weight=None)),
        }
        for family, (base, snaps) in runs.items():
            for k, delta in snaps.items():
                out[(family, "", float(lam), k, np.nan)] = base + delta

    for omega in SIRT_OMEGAS:
        x = x_ref.copy()
        targets = set(SIRT_ITERS)
        for k in range(1, max(SIRT_ITERS) + 1):
            x = x + omega * (B @ (y - A @ x))
            if k in targets:
                out[("e2_sirt", "", np.nan, k, float(omega))] = x.copy()

    # MLEM on HEALPix.  Rows of A sum to one, so a constant shift c of the data
    # is a constant shift of the field: run on y + c > 0, subtract c after.
    span = float(np.ptp(y)) if y.size else 1.0
    c = max(0.0, -float(y.min())) + 1e-3 * max(span, 1e-12)
    ys = y + c
    at1 = At @ np.ones_like(y)
    x = np.maximum(B @ ys, 1e-12)
    targets = set(MLEM_ITERS)
    for k in range(1, max(MLEM_ITERS) + 1):
        ratio = ys / np.maximum(A @ x, 1e-12)
        x = np.where(at1 > 0, x * (At @ ratio) / np.where(at1 > 0, at1, 1.0), x)
        if k in targets:
            out[("e2_mlem", "", np.nan, k, np.nan)] = x - c

    for key, val in out.items():
        val = np.asarray(val, dtype=np.float64)
        val[ops["bad"]] = np.nan
        out[key] = val
    return out


# =============================================================================
# Loaders for the two paper benchmarks
# =============================================================================
def load_gt():
    """The paper's real-data module, imported exactly as its notebooks do."""
    path = REPO_ROOT / "tests" / "real_groundtruth_common_tools.py"
    spec = importlib.util.spec_from_file_location("real_groundtruth_common_tools", path)
    gt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gt)
    # The protocol is written against these values; refuse to run otherwise.
    assert abs(gt.DEGRADE_FWHM_M - REAL_NOMINAL_FWHM_M) < 1e-9, gt.DEGRADE_FWHM_M
    assert gt.RECON_FWHM_M == gt.DEGRADE_FWHM_M
    assert abs(gt.EDGE_MARGIN_M - 200.0) < 1e-9, gt.EDGE_MARGIN_M
    assert gt.REFERENCE_MODE == "psf_aware" and gt.BASELINE_ESTIMAND == "cell_average"
    return gt


SYNTH_NOTEBOOK = NOTEBOOK_DIR / "multi_patch_latitude_validation.ipynb"
# Code cells executed, identified by a string unique to each (cell order kept).
SYNTH_CELL_MARKERS = (
    "PAPER_DATA_OFFLINE_GUARD",
    "RANDOM_SEED = 20260818",
    "REGION_RECORDS = [",
    "def build_patch_manifest",
    "class EsriTileCache",
    "def build_synthetic_patch",
    "def run_psf_aware",
    "def radial_psd",
    "def patch_cache_paths",
    "TWO_STAGE_40_REGION_MARKER",
)
# Top-level lines skipped: the manifest CSV is a versioned paper table.
SYNTH_SKIP_LINES = (re.compile(r"^patch_manifest\.to_csv\("),)


def load_synthetic_namespace(offline: bool = True) -> dict:
    """Execute the synthetic notebook's definition cells, unchanged, in a
    namespace.  This reuses the paper's exact generation code (Esri sampling,
    blur, aperture, child-cell reference, readout) instead of a copy."""
    nb = json.loads(SYNTH_NOTEBOOK.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    if str(NOTEBOOK_DIR) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK_DIR))
    ns = {"__name__": "multi_patch_latitude_validation", "display": lambda *a, **k: None}
    for marker in SYNTH_CELL_MARKERS:
        hits = [c for c in cells if marker in c]
        if len(hits) != 1:
            raise RuntimeError(f"marker {marker!r} matches {len(hits)} code cells")
        src = "\n".join(line for line in hits[0].splitlines()
                        if not any(p.match(line) for p in SYNTH_SKIP_LINES))
        exec(compile(src, f"{SYNTH_NOTEBOOK.name}<{marker}>", "exec"), ns)
    ns["OFFLINE"] = bool(offline)
    return ns


def synthetic_selected_patches(ns):
    """The 40 patches of the paper's one-patch-per-region two-stage table."""
    import pandas as pd
    table = pd.read_csv(NOTEBOOK_DIR / "tables" / PAPER_TABLES["synthetic"])
    wanted = table[["patch_id", "scene_class", "region_id"]].drop_duplicates()
    manifest = ns["patch_manifest"]
    sub = manifest[manifest["patch_id"].isin(set(wanted["patch_id"]))]
    sub = sub.sort_values(["scene_class", "region_id"]).reset_index(drop=True)
    if len(sub) != 40 or sub["region_id"].nunique() != 40:
        raise RuntimeError(f"expected 40 selected patches, found {len(sub)}")
    return list(sub.itertuples(index=False))


def real_usable_scenes(gt):
    scenes = gt.load_region_sites()
    ensure_out_dir()
    usable, _ = gt.screen_region_patches(
        scenes, csv_name="revision/real_quality_screen.csv", verbose=False)
    print(f"{len(usable)}/{len(scenes)} region patches usable.")
    return sorted(usable)


# =============================================================================
# Per-region runners (E1 + E2)
# =============================================================================
def _row(bench, meta, family, method, *, variant="", width_factor=np.nan,
         fwhm_m=np.nan, lam=np.nan, n_iter=np.nan, omega=np.nan,
         rmse=np.nan, n_cells=0, corr=np.nan, runtime_s=np.nan):
    return dict(bench=bench, **meta, family=family, method=method, variant=variant,
                width_factor=width_factor, fwhm_m=fwhm_m, lam=lam, n_iter=n_iter,
                omega=omega, rmse=rmse, n_cells=n_cells, corr=corr, runtime_s=runtime_s)


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 1 else np.nan


def _family_method(key):
    family, variant = key[0], key[1]
    return family, family if not variant else f"{family}_{variant}"


def region_csv_path(bench, scene, study="e1e2"):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scene))
    return ensure_out_dir(f"{study}_{bench}") / f"{safe}.csv"


def _write_rows(rows, path):
    import pandas as pd
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return frame


def run_real_region(gt, scene, *, do_e2=True, force=False, widths=None, study="e1e2"):
    """E1 (+ E2) on one real region; returns the region's result frame.
    ``widths``/``study`` let E4 reuse it on other width factors."""
    import pandas as pd
    out_path = region_csv_path("real", scene, study)
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    meta_src = gt.region_coordinates[scene]
    meta = dict(scene=scene, scene_class=meta_src["scene_class"],
                region_id=meta_src["region_id"], patch_id=meta_src.get("patch_id", ""))
    ref = gt.build_reference(scene, force=False)
    g = gt.build_coarse_grid(scene)
    rows = []

    def score(cell_ids, cell_data, family, method, **params):
        e, t, _ = gt.align_and_mask(np.asarray(cell_ids), np.asarray(cell_data), ref, g)
        m = gt.compute_metrics(e, t, scene, method)
        rows.append(_row("real", meta, family, method, rmse=m["rmse"],
                         n_cells=m["n_cells"], corr=m["corr"], **params))

    # Geometric baselines: the paper's own functions and caches.
    for method in gt.CLASSICAL_METHODS:
        d = gt.reconstruct_classical(scene, method=method, force=force)
        score(d["cell_ids"], d["cell_data"], f"geo_{method}", f"classical_{method}")

    # Shared readout of raster fields at the reference cells (cell average).
    lon_q, lat_q, n_per_cell = gt._readout_points(ref["cell_ids_ref"])
    x_q, y_q = gt.lonlat_to_utm(lon_q, lat_q, g["da"])
    query = np.stack([y_q, x_q], axis=-1)

    def readout(fields):
        stack = np.stack(fields, axis=-1)
        y_ax, x_ax, img = gt._axes_ascending(g["y_coarse"], g["x_coarse"], stack)
        interp = RegularGridInterpolator((y_ax, x_ax), img, method="cubic",
                                         bounds_error=False, fill_value=None)
        vals = interp(query)
        return vals.reshape(-1, n_per_cell, stack.shape[-1]).mean(axis=1)

    y_obs = g["img_coarse"].reshape(-1)
    lon_c = g["lon_coarse"].reshape(-1)
    lat_c = g["lat_coarse"].reshape(-1)
    for f in (widths or WIDTH_FACTORS["real"]):
        fwhm = f * REAL_NOMINAL_FWHM_M
        common = dict(width_factor=f, fwhm_m=fwhm)

        # --- PSF-aware grid (operator built once per width) ---------------
        scale = gt.fwhm_to_scale(fwhm)
        npt = gt.recommend_npt(scale, gt.REF_LEVEL)["npt"]
        t0 = time.perf_counter()
        rs = gt.PSFResampler(lon_deg=lon_c, lat_deg=lat_c, level=gt.REF_LEVEL,
                             sigma_m=scale, Npt=npt, out_cell_ids=ref["cell_ids_ref"],
                             threshold=gt.THRESHOLD, device=gt.DEVICE, verbose=False)
        build_s = time.perf_counter() - t0
        pkg = {}
        for lam in PSF_LAMBDAS:
            for it in PSF_ITERS:
                t1 = time.perf_counter()
                res = rs.resample(y_obs, lam=lam, max_iter=it, tol=PSF_TOL)
                pkg[(lam, it)] = res
                score(to_numpy(res.cell_ids), to_numpy(res.cell_data), "psf_aware",
                      "psf_aware", lam=lam, n_iter=it,
                      runtime_s=build_s + time.perf_counter() - t1, **common)

        # --- E2 on the same operators, nominal width only -------------------
        if do_e2 and f == 1.0:
            _run_e2(rs, y_obs, pkg, score, common)

        # --- two-stage deconvolution baselines ------------------------------
        kernel = gaussian_kernel(fwhm, gt.PIXEL_SIZE_COARSE_M)
        fields, runtimes = deconvolution_fields(g["img_coarse"], kernel,
                                                rl_variants=("clip", "minmax"))
        keys = list(fields)
        vals = readout([fields[k] for k in keys])
        for j, key in enumerate(keys):
            family, method = _family_method(key)
            score(ref["cell_ids_ref"], vals[:, j], family, method, variant=key[1],
                  lam=key[2], n_iter=key[3], runtime_s=runtimes[key], **common)
        del rs

    return _write_rows(rows, out_path)


def _run_e2(rs, y_obs, pkg, score, common):
    """E2 arms plus the equivalence check against the package's own solve."""
    ops = extract_operators(rs)
    ids_pkg = to_numpy(pkg[(PSF_LAMBDAS[0], PSF_ITERS[0])].cell_ids).astype(np.int64)
    if not np.array_equal(ids_pkg, ops["cell_ids"]):
        raise RuntimeError("operator cell ids differ from the package output ordering")
    t0 = time.perf_counter()
    fields = e2_fields(ops, y_obs)
    dt = (time.perf_counter() - t0) / len(fields)
    for (family, variant, lam, n_iter, omega), x in fields.items():
        score(ops["cell_ids"], x, family, family, variant=variant, lam=lam,
              n_iter=n_iter, omega=omega, runtime_s=dt, **common)
    # Diagnostic: the float64 re-implementation must reproduce the package.
    for (lam, it), res in pkg.items():
        ours = fields[("e2_wcg", "", float(lam), it, np.nan)]
        theirs = to_numpy(res.cell_data).astype(np.float64)
        m = np.isfinite(ours) & np.isfinite(theirs)
        diff = float(np.max(np.abs(ours[m] - theirs[m]))) if m.any() else np.nan
        scale = float(np.max(np.abs(theirs[m]))) if m.any() else np.nan
        score(ops["cell_ids"], theirs, "check_wcg_vs_package", "check_wcg_vs_package",
              lam=lam, n_iter=it, omega=diff / scale if scale else np.nan, **common)


def run_synthetic_region(ns, patch_row, *, do_e2=True, force=False, widths=None,
                         study="e1e2", arm="S0"):
    """E1 (+ E2) on one synthetic region (its paper-selected patch).
    ``widths``/``study`` let E4 reuse it on other width factors.  ``arm``
    (E3): "S0" is the paper's set-up; "S1" point-samples the blurred texture
    at the detector centres (no aperture); "S3" models the aperture exactly,
    in A (`utm_box_kernel`) and in the deconvolution kernel."""
    import pandas as pd
    if arm not in ("S0",) + E3_NEW_ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if arm != "S0" and (do_e2 or tuple(widths or ()) != (1.0,)):
        raise ValueError("E3 arms run without E2, at widths=(1.0,)")
    out_path = region_csv_path("synthetic", patch_row.patch_id, study)
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    meta = dict(scene=patch_row.patch_id, scene_class=patch_row.scene_class,
                region_id=patch_row.region_id, patch_id=patch_row.patch_id)
    products = ns["build_synthetic_patch"](patch_row)
    ref_ids, ref_vals = products["parent_ids"], products["hpx_latent"]
    rows = []
    crs_utm = None
    if arm == "S1":
        detector, _ = point_sampled_detector(ns, patch_row, products)
        # the geometric baselines read products["blurred_detector"]
        products = dict(products, blurred_detector=detector)
    elif arm == "S3":
        crs_utm = ns["patch_coordinate_grids"](patch_row)["crs_utm"]

    def score(cell_ids, cell_data, family, method, **params):
        aligned = ns["align_estimate"](ref_ids, ref_vals, cell_ids, cell_data)
        r = aligned["reference"].to_numpy()
        e = aligned["estimate"].to_numpy()
        m = np.isfinite(r) & np.isfinite(e)
        rows.append(_row("synthetic", meta, family, method, rmse=ns["rmse"](r, e),
                         n_cells=int(m.sum()), corr=_corr(r, e), **params))

    for method in ("nearest", "linear", "cubic"):
        est = ns["geometric_cell_average"](products, method)
        score(ref_ids, est, f"geo_{method}", f"classical_{method}")
    if arm != "S0":
        # denominator of the paper's recovery fraction (Gaussian-blurred
        # texture, no aperture): the same for every arm of a region
        score(ref_ids, products["hpx_blurred"], "ref_blurred", "ref_blurred")

    query = np.column_stack([products["child_y"].ravel(), products["child_x"].ravel()])
    n_cells, n_child = products["child_y"].shape

    def readout(fields, chunk=8):
        out = []
        for i in range(0, len(fields), chunk):
            stack = np.stack(fields[i:i + chunk], axis=-1)
            vals = ns["regular_interpolator"](products["y"], products["x"], stack,
                                              method="cubic")(query)
            if not np.all(np.isfinite(vals)):
                near = ns["regular_interpolator"](products["y"], products["x"], stack,
                                                  method="nearest")(query)
                vals = np.where(np.isfinite(vals), vals, near)
            out.append(vals.reshape(n_cells, n_child, -1).mean(axis=1))
        return np.concatenate(out, axis=1)

    observed = products["blurred_detector"].astype(np.float64)
    lon = products["lon"].ravel().astype(np.float64)
    lat = products["lat"].ravel().astype(np.float64)
    for f in (widths or WIDTH_FACTORS["synthetic"]):
        fwhm = f * SYNTH_NOMINAL_FWHM_M
        kernel = None
        if arm == "S3":
            # exact Gaussian (x) box in A; the neighbour search, Npt and the
            # recorded width are those of the second-moment equivalent Gaussian
            fwhm = SYNTH_EQUIV_FACTOR * SYNTH_NOMINAL_FWHM_M
            kernel = utm_box_kernel(crs_utm)
        common = dict(width_factor=f, fwhm_m=fwhm)

        scale = ns["fwhm_to_scale"](fwhm)
        npt = int(ns["recommend_npt"](scale, ns["HEALPIX_LEVEL"],
                                      target_mass=ns["TARGET_KERNEL_MASS"])["npt"])
        t0 = time.perf_counter()
        extra = {} if kernel is None else {"kernel": kernel}
        rs = ns["PSFResampler"](lon_deg=lon, lat_deg=lat, level=ns["HEALPIX_LEVEL"],
                                threshold=ns["THRESHOLD"], verbose=False,
                                ellipsoid="sphere", sigma_m=scale, Npt=npt,
                                device=ns["DEVICE"], **extra)
        build_s = time.perf_counter() - t0
        if arm != "S0":
            # Diagnostic: kernel mass captured by each sample's links, in
            # units of the kernel's integral (cell area x sum of weights);
            # carried in `omega` like the package check.
            w = to_numpy(rs._link_weights()).astype(np.float64)
            w[to_numpy(rs.hi) < 0] = 0.0
            cell_area = 4.0 * math.pi * rs.radius ** 2 / (12.0 * 4.0 ** ns["HEALPIX_LEVEL"])
            if arm == "S3":
                integral = box_gaussian_integral()
            else:
                integral = math.pi * scale ** 2 / 2.0      # of exp(-2 d^2 / s^2)
            captured = w.sum(axis=1) * cell_area / integral
            rows.append(_row("synthetic", meta, "check_kernel_mass", "check_kernel_mass",
                             omega=float(np.median(captured)), n_iter=int(npt),
                             n_cells=int(rs.K), **common))
        pkg = {}
        for lam in PSF_LAMBDAS:
            for it in PSF_ITERS:
                t1 = time.perf_counter()
                res = rs.resample(observed.ravel(), lam=lam, max_iter=it, tol=PSF_TOL)
                pkg[(lam, it)] = res
                score(to_numpy(res.cell_ids), to_numpy(res.cell_data), "psf_aware",
                      "psf_aware", lam=lam, n_iter=it,
                      runtime_s=build_s + time.perf_counter() - t1, **common)
        if do_e2 and f == 1.0:
            _run_e2(rs, observed.ravel(), pkg, score, common)

        if arm == "S3":
            kernel = box_gaussian_pixel_kernel(ns["DETECTOR_GSD_M"])
        else:
            kernel = gaussian_kernel(fwhm, ns["DETECTOR_GSD_M"])
        fields, runtimes = deconvolution_fields(observed, kernel, rl_variants=("minmax",))
        keys = list(fields)
        vals = readout([fields[k] for k in keys])
        for j, key in enumerate(keys):
            family, method = _family_method(key)
            score(ref_ids, vals[:, j], family, method, variant=key[1],
                  lam=key[2], n_iter=key[3], runtime_s=runtimes[key], **common)
        del rs
        if "release_patch_memory" in ns:
            ns["release_patch_memory"]()

    return _write_rows(rows, out_path)


def collect_results(bench, study="e1e2"):
    import pandas as pd
    files = sorted(ensure_out_dir(f"{study}_{bench}").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no per-region results under {OUT_DIR / f'{study}_{bench}'}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


# =============================================================================
# E0: regression of the submitted numbers at the current commit
# =============================================================================
def e0_real(gt, scenes):
    """Re-run the paper's 40-region protocol (force=True) into
    tables/revision/ and compare with the submitted CSV."""
    import pandas as pd
    ensure_out_dir()
    metrics, failures = gt.run_multiregion(force=True, scenes=scenes,
                                           csv_prefix="revision/e0_real_multiregion")
    paper = pd.read_csv(NOTEBOOK_DIR / "tables" / PAPER_TABLES["real"])
    return _e0_compare(metrics[["scene", "method", "rmse"]],
                       paper[["scene", "method", "rmse"]], "real")


def e0_synthetic(ns, patches):
    """Recompute the three paper arms of the one-patch-per-region synthetic
    ablation with the notebook's own code, and compare with the submitted CSV."""
    import pandas as pd
    rows = []
    for i, row in enumerate(patches, 1):
        products = ns["build_synthetic_patch"](row)
        psf = ns["run_psf_aware"](products, ns["TRUE_FWHM_M"])
        rows.append(dict(scene=row.patch_id, method="psf_aware_matched", rmse=psf["rmse"]))
        kernel = ns["gaussian_psf_kernel"](ns["TRUE_FWHM_M"], ns["DETECTOR_GSD_M"])
        observed = products["blurred_detector"].astype(np.float64)
        tik = ns["tikhonov_deconvolve"](observed, kernel, lam=ns["TIKHONOV_LAMBDA"])
        rows.append(dict(scene=row.patch_id, method="tikhonov_bicubic",
                         rmse=ns["rmse"](products["hpx_latent"],
                                         ns["field_to_healpix_cell_average"](products, tik))))
        lo, hi = float(np.nanmin(observed)), float(np.nanmax(observed))
        scale = hi - lo if hi > lo else 1.0
        rl01 = ns["richardson_lucy"](np.clip((observed - lo) / scale, 0, 1), kernel,
                                     num_iter=ns["RICHARDSON_LUCY_ITER"], clip=False)
        rows.append(dict(scene=row.patch_id, method="richardson_lucy",
                         rmse=ns["rmse"](products["hpx_latent"],
                                         ns["field_to_healpix_cell_average"](
                                             products, rl01 * scale + lo))))
        print(f"[{i:02d}/{len(patches)}] {row.patch_id}", flush=True)
        ns["release_patch_memory"]()
    ours = pd.DataFrame(rows)
    paper = pd.read_csv(NOTEBOOK_DIR / "tables" / PAPER_TABLES["synthetic"])
    paper = paper.rename(columns={"patch_id": "scene", "rmse_vs_latent": "rmse"})
    return _e0_compare(ours, paper[["scene", "method", "rmse"]], "synthetic")


def _e0_compare(ours, paper, bench):
    merged = ours.merge(paper, on=["scene", "method"], how="outer",
                        suffixes=("_now", "_paper"), indicator=True)
    merged["rel_diff"] = (merged["rmse_now"] - merged["rmse_paper"]).abs() / merged["rmse_paper"]
    merged["ok"] = merged["rel_diff"] <= E0_TOL
    out = ensure_out_dir() / f"e0_{bench}_comparison.csv"
    merged.to_csv(out, index=False)
    n_bad = int((~merged["ok"]).sum())
    print(f"E0 {bench}: {len(merged)} (scene, method) pairs, max rel. diff "
          f"{merged['rel_diff'].max():.3g}, {n_bad} above {E0_TOL:g} -> {out}")
    return merged


# =============================================================================
# Analysis: cross-validated selection, oracle, paired statistics
# =============================================================================
CONFIG_COLS = ["variant", "width_factor", "lam", "n_iter", "omega"]
E1_FAMILIES = ("psf_aware", "rl2", "tik_id", "tik_lap", "wiener_hunt",
               "geo_nearest", "geo_linear", "geo_cubic")
E2_FAMILIES = ("psf_aware", "e2_xref", "e2_wcg", "e2_cg_euclid", "e2_jacobi",
               "e2_sirt", "e2_tik_x0", "e2_tik_dref", "e2_mlem")


def _config_label(frame):
    def fmt(v):
        if isinstance(v, float):
            return "-" if np.isnan(v) else f"{v:.6g}"
        return str(v) if v not in ("", None) else "-"
    return frame[CONFIG_COLS].apply(lambda r: "|".join(fmt(v) for v in r), axis=1)


def prepare(results):
    """Add the configuration label and the bicubic-relative score."""
    df = results.copy()
    df["variant"] = df["variant"].fillna("")
    df = df[~df["family"].str.startswith("check_")].copy()
    df["config"] = _config_label(df)
    bic = df[df["family"] == "geo_cubic"].set_index("region_id")["rmse"]
    df["rel"] = df["rmse"] / df["region_id"].map(bic)
    return df


def assign_folds(regions):
    """Stratified 5-fold assignment: within each class the region ids are
    sorted, permuted with CV_SEED (classes in fixed order), fold = rank mod 5."""
    rng = np.random.default_rng(CV_SEED)
    folds = {}
    for cls in CLASSES:
        ids = sorted(regions.loc[regions["scene_class"] == cls, "region_id"].unique())
        for rank, rid in enumerate(rng.permutation(ids)):
            folds[rid] = rank % N_FOLDS
    return folds


def _width_filter(df, mode):
    if mode == "A":
        return df[df["width_factor"].isna() | np.isclose(df["width_factor"], 1.0)]
    return df


def select(df, family, mode, scheme):
    """Per-region score of `family` under cross-validated selection
    (scheme '5fold' or 'loo_class'), or the per-region 'oracle'."""
    import pandas as pd
    sub = _width_filter(df[df["family"] == family], mode)
    if sub.empty:
        return pd.DataFrame()
    rel = sub.pivot_table(index="region_id", columns="config", values="rel", aggfunc="first")
    rmse = sub.pivot_table(index="region_id", columns="config", values="rmse", aggfunc="first")
    classes = sub.groupby("region_id")["scene_class"].first()
    out = []
    if scheme == "oracle":
        for rid in rmse.index:
            cfg = rmse.loc[rid].idxmin()
            out.append((rid, cfg, rmse.loc[rid, cfg]))
    else:
        if scheme == "5fold":
            folds = assign_folds(sub[["region_id", "scene_class"]].drop_duplicates())
            groups = [[r for r in rel.index if folds.get(r) == k] for k in range(N_FOLDS)]
        elif scheme == "loo_class":
            groups = [[r] for r in rel.index]
        else:
            raise ValueError(scheme)
        for test in groups:
            if not test:
                continue
            pool = [r for r in rel.index if r not in test]
            if scheme == "loo_class":
                pool = [r for r in pool if classes[r] == classes[test[0]]]
            cfg = rel.loc[pool].mean(axis=0, skipna=True).idxmin()
            out.extend((rid, cfg, rmse.loc[rid, cfg]) for rid in test)
    frame = pd.DataFrame(out, columns=["region_id", "config", "rmse"])
    frame["family"], frame["mode"], frame["scheme"] = family, mode, scheme
    frame["scene_class"] = frame["region_id"].map(classes)
    return frame


def paper_config(df, bench, family):
    import pandas as pd
    cfg = PAPER_CONFIG[bench].get(family)
    if cfg is None:
        return pd.DataFrame()
    sub = df[df["family"] == family]
    for key, val in cfg.items():
        sub = sub[sub[key] == val] if isinstance(val, str) else sub[np.isclose(sub[key], val)]
    frame = sub[["region_id", "scene_class", "config", "rmse"]].copy()
    frame["family"], frame["mode"], frame["scheme"] = family, "A", "paper"
    return frame


def bootstrap_mean_ci(values, n_rep=N_BOOT, seed=BOOT_SEED, alpha=0.05):
    """Percentile bootstrap over regions, identical to the paper's."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_rep, v.size))].mean(axis=1)
    return (float(v.mean()), float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def paired_stats(ours, theirs):
    """Delta = competitor - ours per region (positive favours ours)."""
    from scipy.stats import binomtest
    joined = ours.set_index("region_id")[["rmse", "scene_class"]].join(
        theirs.set_index("region_id")[["rmse"]], rsuffix="_comp", how="inner")
    rows = []
    for cls, grp in [("all", joined)] + list(joined.groupby("scene_class")):
        d = (grp["rmse_comp"] - grp["rmse"]).to_numpy()
        d = d[np.isfinite(d)]
        n, wins = int(d.size), int((d > 0).sum())
        mean, lo, hi = bootstrap_mean_ci(d)
        rel = ((grp["rmse_comp"] - grp["rmse"]) / grp["rmse_comp"]).mean()
        rows.append(dict(scene_class=cls, n_regions=n, wins=wins,
                         sign_test_p=binomtest(wins, n).pvalue if n else np.nan,
                         mean_delta=mean, ci_low=lo, ci_high=hi,
                         mean_rel_margin=float(rel)))
    return rows


def analyze(bench):
    """Write the E1/E2 selection and comparison tables for one benchmark."""
    import pandas as pd
    df = prepare(collect_results(bench))
    out = ensure_out_dir(f"analysis_{bench}")

    selections = []
    for family in sorted(set(E1_FAMILIES) | set(E2_FAMILIES)):
        if family not in set(df["family"]):
            continue
        for mode in ("A", "B"):
            for scheme in ("5fold", "loo_class", "oracle"):
                selections.append(select(df, family, mode, scheme))
        selections.append(paper_config(df, bench, family))
    sel = pd.concat([s for s in selections if len(s)], ignore_index=True)
    sel.to_csv(out / "selected_configs.csv", index=False)

    def get(family, mode, scheme):
        return sel[(sel.family == family) & (sel["mode"] == mode) & (sel.scheme == scheme)]

    comparisons = []
    ours_variants = [("A", "paper"), ("A", "5fold"), ("B", "5fold")]
    for study, families in (("E1", E1_FAMILIES), ("E2", E2_FAMILIES)):
        for comp in families:
            if comp == "psf_aware" or comp not in set(df["family"]):
                continue
            for o_mode, o_scheme in ours_variants:
                ours = get("psf_aware", o_mode, o_scheme)
                if ours.empty:
                    continue
                for c_scheme in ("paper", "5fold", "loo_class", "oracle"):
                    c_mode = "A" if c_scheme == "paper" else o_mode
                    theirs = get(comp, c_mode, c_scheme)
                    if theirs.empty:
                        continue
                    for r in paired_stats(ours, theirs):
                        comparisons.append(dict(study=study, competitor=comp,
                                                ours=f"{o_mode}/{o_scheme}",
                                                theirs=f"{c_mode}/{c_scheme}", **r))
    comp = pd.DataFrame(comparisons)
    comp.to_csv(out / "comparisons.csv", index=False)

    # Median relative RMSE of every configuration (curves for the figures).
    curves = (df.groupby(["family", "config", "variant", "width_factor", "lam",
                          "n_iter", "omega"], dropna=False)
              .agg(n_regions=("region_id", "nunique"), median_rel=("rel", "median"),
                   q25_rel=("rel", lambda s: s.quantile(0.25)),
                   q75_rel=("rel", lambda s: s.quantile(0.75)),
                   median_rmse=("rmse", "median"), median_runtime_s=("runtime_s", "median"))
              .reset_index())
    curves.to_csv(out / "config_curves.csv", index=False)

    checks = collect_results(bench)
    checks = checks[checks["family"] == "check_wcg_vs_package"]
    if len(checks):
        # `omega` carries the max relative field difference for this row type.
        print(f"[{bench}] float64 re-implementation vs package: max rel. field "
              f"difference {checks['omega'].max():.3g}")
    head = comp[(comp.scene_class == "all") & (comp.theirs.str.endswith(("5fold", "oracle")))]
    print(head[["study", "competitor", "ours", "theirs", "wins", "n_regions",
                "mean_rel_margin", "ci_low", "ci_high"]].to_string(index=False))
    print(f"-> {out}")
    return sel, comp, curves


# =============================================================================
# E4: assumed-width x damping sensitivity (R2.3)
# =============================================================================
def e4_frame(bench):
    """PSF-modelling families at every assumed width: the E1 run (f >= 0.9)
    plus the E4 complement (E4_EXTRA_WIDTHS); geo_cubic for the score."""
    import pandas as pd
    fams = set(E4_COMPETITORS) | {"psf_aware"}
    base = collect_results(bench, "e1e2")
    parts = [base[base["family"].isin(fams | {"geo_cubic"})]]
    try:
        extra = collect_results(bench, "e4")
        parts.append(extra[extra["family"].isin(fams)])
    except FileNotFoundError:
        print(f"[{bench}] no E4 complement yet: widths >= 0.9 only")
    df = prepare(pd.concat(parts, ignore_index=True))
    df["f"] = df["width_factor"].round(6)
    return df


def e4_analyze(bench):
    """E4 tables.  At each assumed width f, every competitor is tuned by the
    E1 5-fold CV restricted to that f; the strongest competitor of a region is
    the best of those tuned competitors *for that region* (chosen with the
    reference: favourable to the competitors).  PSF-aware is evaluated at every
    (lam, n_iter in E4_ITERS) without tuning, which gives the f x lam map."""
    import pandas as pd
    df = e4_frame(bench)
    out = ensure_out_dir(f"analysis_{bench}")
    bic = df[df["family"] == "geo_cubic"].set_index("region_id")["rmse"]
    classes = df.groupby("region_id")["scene_class"].first()
    widths = sorted(df["f"].dropna().unique())
    curve_rows, map_rows = [], []
    for f in widths:
        at_f = df[np.isclose(df["f"], f)]
        tuned = {}
        for fam in E4_COMPETITORS + ("psf_aware",):
            s = select(at_f, fam, "B", "5fold")       # one width in the frame
            if len(s):
                tuned[fam] = s.set_index("region_id")["rmse"]
        for fam, r in tuned.items():
            rel = (r / bic.reindex(r.index)).dropna()
            curve_rows.append(dict(f=f, family=fam, tuning="5fold@f", n_regions=len(rel),
                                   median_rel=rel.median(), q25_rel=rel.quantile(0.25),
                                   q75_rel=rel.quantile(0.75)))
        comp = pd.concat({k: v for k, v in tuned.items() if k != "psf_aware"}, axis=1)
        best = comp.min(axis=1)
        ours_all = at_f[(at_f["family"] == "psf_aware") & at_f["n_iter"].isin(E4_ITERS)]
        for (lam, it), g in ours_all.groupby(["lam", "n_iter"]):
            ours = g.set_index("region_id")["rmse"]
            idx = ours.index.intersection(best.index)
            margin = (best[idx] - ours[idx]) / best[idx]        # > 0 favours PSF-aware
            win = ours[idx] < best[idx]
            for cls, sub in [("all", idx)] + [(c, idx[classes[idx] == c]) for c in CLASSES]:
                row = dict(f=f, lam=lam, n_iter=int(it), scene_class=cls,
                           n_regions=len(sub), wins=int(win[sub].sum()),
                           win_frac=float(win[sub].mean()) if len(sub) else np.nan,
                           median_margin=float(margin[sub].median()) if len(sub) else np.nan,
                           mean_margin=float(margin[sub].mean()) if len(sub) else np.nan)
                for fam in comp.columns:
                    row[f"win_frac_vs_{fam}"] = float((ours[sub] < comp.loc[sub, fam]).mean())
                map_rows.append(row)
    curves = pd.DataFrame(curve_rows)
    emap = pd.DataFrame(map_rows)
    curves.to_csv(out / "e4_width_curves.csv", index=False)
    emap.to_csv(out / "e4_width_lambda_map.csv", index=False)
    safe = e4_safe_zone(emap)
    safe.to_csv(out / "e4_safe_zone.csv", index=False)
    print(f"[{bench}] E4 safe zone (win fraction >= {E4_SAFE_WIN:g} against the "
          f"strongest tuned competitor, all regions):")
    print(safe.to_string(index=False))
    print(f"-> {out}")
    return curves, emap, safe


def e4_safe_zone(emap):
    """For each (lam, n_iter): the contiguous range of f around 1 where
    PSF-aware wins at least E4_SAFE_WIN of the regions."""
    import pandas as pd
    rows = []
    allc = emap[emap["scene_class"] == "all"]
    for (lam, it), g in allc.groupby(["lam", "n_iter"]):
        g = g.sort_values("f").reset_index(drop=True)
        ok = (g["win_frac"] >= E4_SAFE_WIN).to_numpy()
        i0 = int(np.argmin(np.abs(g["f"].to_numpy() - 1.0)))
        if not ok[i0]:
            rows.append(dict(lam=lam, n_iter=it, f_min=np.nan, f_max=np.nan,
                             min_win_frac=np.nan))
            continue
        lo = hi = i0
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        while hi < len(ok) - 1 and ok[hi + 1]:
            hi += 1
        rows.append(dict(lam=lam, n_iter=it, f_min=g["f"][lo], f_max=g["f"][hi],
                         min_win_frac=float(g["win_frac"][lo:hi + 1].min()),
                         f_min_is_grid_edge=lo == 0, f_max_is_grid_edge=hi == len(ok) - 1))
    return pd.DataFrame(rows)


# Reference data-viz palette (light, print): categorical slots 1-4 in their
# validated order, blue/red diverging pair with a grey midpoint, blue ramp.
_E4_SERIES = {"psf_aware": ("PSF-aware", "#2a78d6", "o", "-"),
              "rl2": ("Richardson-Lucy", "#eb6834", "s", "--"),
              "tik_id": ("Tikhonov", "#1baf7a", "^", "-."),
              "wiener_hunt": ("Wiener-Hunt", "#eda100", "D", ":")}
_E4_DIVERGING = ("#e34948", "#f0efec", "#2a78d6")
_E4_SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
                  "#0d366b")
_E4_INK = "#0b0b0b"
_E4_INK2 = "#52514e"
_E4_EQUIV = {"real": math.sqrt(1.25), "synthetic": SYNTH_EQUIV_FACTOR}
_E4_MARGIN_PCT = 15.0       # colour range of the margin map (saturates beyond)
_E4_DIVERGED_PCT = 1000.0   # |median margin| above this: diverged solve


def _e4_f_label(f):
    if np.isclose(f, math.sqrt(1.25)):
        return "1.118"
    if np.isclose(f, SYNTH_EQUIV_FACTOR):
        return "1.138"
    return f"{f:g}"


def e4_figures(benches=("real", "synthetic"), n_iter=30):
    """Fig. E4a: f x lam maps (median margin, wins) for PSF-aware against the
    strongest tuned competitor.  Fig. E4b: relative RMSE versus f."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.edgecolor": _E4_INK2, "axes.labelcolor": _E4_INK,
                         "xtick.color": _E4_INK2, "ytick.color": _E4_INK2,
                         "pdf.fonttype": 42})
    figdir = ensure_out_dir("figures")
    div = LinearSegmentedColormap.from_list("e4div", _E4_DIVERGING)
    seq = LinearSegmentedColormap.from_list("e4seq", _E4_SEQUENTIAL)

    # ---- E4a: maps -------------------------------------------------------
    fig, axes = plt.subplots(len(benches), 2, figsize=(7.16, 2.3 * len(benches)),
                             constrained_layout=True, squeeze=False)
    for r, bench in enumerate(benches):
        emap = pd.read_csv(OUT_DIR / f"analysis_{bench}" / "e4_width_lambda_map.csv")
        g = emap[(emap["scene_class"] == "all") & (emap["n_iter"] == n_iter)]
        fs = sorted(g["f"].unique())
        lams = sorted(g["lam"].unique())
        margin = g.pivot(index="lam", columns="f", values="median_margin").loc[lams, fs] * 100
        wins = g.pivot(index="lam", columns="f", values="wins").loc[lams, fs]
        n = int(g["n_regions"].max())
        # Undamped (lam = 0) package solves diverge at small f: those cells
        # are labelled "div." and left uncoloured instead of driving the scale.
        diverged = np.abs(margin.to_numpy()) > _E4_DIVERGED_PCT
        panels = ((margin, div, TwoSlopeNorm(0.0, -_E4_MARGIN_PCT, _E4_MARGIN_PCT),
                   "median margin vs strongest tuned competitor (%)"),
                  (wins, seq, plt.Normalize(0, n), f"regions won (of {n})"))
        for c, (data, cmap, norm, title) in enumerate(panels):
            ax = axes[r, c]
            shown = data.to_numpy().astype(float)
            if c == 0:
                shown = np.where(diverged, np.nan, shown)
            im = ax.imshow(np.ma.masked_invalid(shown), cmap=cmap, norm=norm, aspect="auto")
            for i in range(len(lams)):
                for j in range(len(fs)):
                    v = data.to_numpy()[i, j]
                    if not np.isfinite(v):
                        continue
                    if c == 0 and diverged[i, j]:
                        ax.text(j, i, "div.", ha="center", va="center", fontsize=6,
                                color=_E4_INK2)
                        continue
                    rgba = cmap(norm(v))
                    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    label = f"{v:.0f}" if (c == 1 or abs(v) >= 10) else f"{v:.1f}"
                    ax.text(j, i, label, ha="center", va="center", fontsize=6,
                            color="white" if lum < 0.5 else _E4_INK)
                    if c == 1 and v >= E4_SAFE_WIN * n:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                                   lw=1.0, ec=_E4_INK))
            ax.set_xticks(range(len(fs)), [_e4_f_label(f) for f in fs])
            ax.set_yticks(range(len(lams)), [f"{v:g}" for v in lams])
            j_eq = int(np.argmin(np.abs(np.asarray(fs) - _E4_EQUIV[bench])))
            ax.get_xticklabels()[j_eq].set_fontweight("bold")
            ax.set_xlabel("assumed width factor f (bold: equivalent width)")
            ax.set_ylabel("damping λ")
            ax.set_title(f"{bench}: {title}", loc="left")
            for s in ax.spines.values():
                s.set_visible(False)
            fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01,
                         extend="both" if c == 0 else "neither").outline.set_visible(False)
    fig.savefig(figdir / f"e4_width_lambda_maps_it{n_iter}.pdf")
    fig.savefig(figdir / f"e4_width_lambda_maps_it{n_iter}.png", dpi=200)
    plt.close(fig)

    # ---- E4b: curves -----------------------------------------------------
    fig, axes = plt.subplots(1, len(benches), figsize=(7.16, 2.5), constrained_layout=True,
                             squeeze=False)
    for c, bench in enumerate(benches):
        cur = pd.read_csv(OUT_DIR / f"analysis_{bench}" / "e4_width_curves.csv")
        ax = axes[0, c]
        for fam, (label, color, marker, ls) in _E4_SERIES.items():
            g = cur[cur["family"] == fam].sort_values("f")
            if g.empty:
                continue
            ax.fill_between(g["f"], g["q25_rel"], g["q75_rel"], color=color, alpha=0.15, lw=0)
            ax.plot(g["f"], g["median_rel"], color=color, lw=1.6, ls=ls, marker=marker,
                    ms=4, mec="white", mew=0.6, label=label)
        ax.axvline(_E4_EQUIV[bench], color=_E4_INK2, lw=0.8, ls=(0, (2, 2)))
        ax.axhline(1.0, color=_E4_INK2, lw=0.6)
        ax.text(_E4_EQUIV[bench], 0.98, " equivalent width", transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=6.5, color=_E4_INK2)
        ax.set_xlabel("assumed width factor f")
        ax.set_ylabel("RMSE / bicubic RMSE (median, IQR)")
        ax.set_title(f"{bench} (each method tuned by 5-fold CV at each f)", loc="left")
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.savefig(figdir / "e4_width_curves.pdf")
    fig.savefig(figdir / "e4_width_curves.png", dpi=200)
    plt.close(fig)
    print(f"-> {figdir}")


# =============================================================================
# E3: response mismatch from the 10 m detector aperture (R1.5, R2.2)
# =============================================================================
# arm -> (study directory, width factor of A and of the deconvolution kernels)
_E3_ARM_SOURCE = {"S0": ("e1e2", 1.0), "S1": ("e3s1", 1.0),
                  "S2": ("e1e2", SYNTH_EQUIV_FACTOR), "S3": ("e3s3", 1.0)}
_E3_ARM_LABEL = {"S0": "S0\npaper set-up", "S1": "S1\npoint-sampled",
                 "S2": "S2\nequivalent width", "S3": "S3\nexact kernel"}


def e3_study(arm):
    return _E3_ARM_SOURCE[arm][0]


def e3_arm_frame(arm, cache):
    """One arm's rows: geometric baselines, `ref_blurred` and the
    PSF-modelling families at the arm's width (E2 arms left out)."""
    study, f = _E3_ARM_SOURCE[arm]
    if study not in cache:
        cache[study] = collect_results("synthetic", study)
    raw = cache[study]
    keep = (raw["width_factor"].isna() | np.isclose(raw["width_factor"], f)) \
        & ~raw["family"].str.startswith("e2_")
    df = prepare(raw[keep])
    df["arm"] = arm
    return df


def e3_analyze():
    """E3 tables (protocol section E3): recovery per arm and family, PSF-aware
    against each competitor within an arm, and arm-to-arm contrasts; every
    family tuned by the E1 5-fold CV inside its arm."""
    import pandas as pd
    out = ensure_out_dir("analysis_synthetic")
    cache = {}
    frames = {arm: e3_arm_frame(arm, cache) for arm in E3_ARMS}

    # Recovery denominator RMSE(blurred, latent): no aperture, so identical
    # in S1 and S3 (and valid for S0, S2: same texture and Gaussian).
    blurred = {arm: frames[arm][frames[arm]["family"] == "ref_blurred"]
               .set_index("region_id")["rmse"] for arm in E3_NEW_ARMS}
    b1, b3 = blurred["S1"], blurred["S3"].reindex(blurred["S1"].index)
    if not np.allclose(b1, b3, rtol=1e-12, atol=0):
        raise RuntimeError("ref_blurred differs between S1 and S3")
    blurred = b1

    for arm in E3_NEW_ARMS:
        chk = cache[e3_study(arm)]
        chk = chk[chk["family"] == "check_kernel_mass"]
        print(f"[E3 {arm}] Npt {int(chk['n_iter'].median())}, captured kernel mass: "
              f"median {chk['omega'].median():.4f}, min {chk['omega'].min():.4f}")

    tuned = {}
    for arm, df in frames.items():
        for fam in E3_FAMILIES:
            s = select(df, fam, "B", "5fold")      # one width in each arm's frame
            if len(s):
                tuned[(arm, fam)] = s

    rec_rows = []
    for (arm, fam), s in tuned.items():
        r = s.set_index("region_id")
        rec = 1.0 - r["rmse"] / blurred.reindex(r.index)
        bic = frames[arm][frames[arm]["family"] == "geo_cubic"].set_index("region_id")["rmse"]
        rel = r["rmse"] / bic.reindex(r.index)
        for cls, idx in [("all", r.index)] + [(c, r.index[r["scene_class"] == c])
                                               for c in CLASSES]:
            mean, lo, hi = bootstrap_mean_ci(rec[idx])
            rec_rows.append(dict(arm=arm, family=fam, scene_class=cls, n_regions=len(idx),
                                 mean_recovery=mean, ci_low=lo, ci_high=hi,
                                 median_rel_bicubic=float(rel[idx].median()),
                                 mean_rmse=float(r.loc[idx, "rmse"].mean()),
                                 top_config=r.loc[idx, "config"].mode().iat[0]))
    recovery = pd.DataFrame(rec_rows)

    win_rows = []
    for arm in E3_ARMS:
        ours = tuned[(arm, "psf_aware")]
        comps = {fam: tuned[(arm, fam)] for fam in E3_FAMILIES
                 if fam != "psf_aware" and (arm, fam) in tuned}
        for fam, theirs in comps.items():
            for r in paired_stats(ours, theirs):
                win_rows.append(dict(arm=arm, competitor=fam, **r))
        # strongest deconvolution competitor of each region, chosen with the
        # reference (as in E4: favourable to the competitors)
        best = pd.concat({f: comps[f].set_index("region_id")["rmse"]
                          for f in E4_COMPETITORS if f in comps}, axis=1).min(axis=1)
        theirs = pd.DataFrame({"region_id": best.index, "rmse": best.to_numpy()})
        o = ours.set_index("region_id")
        margin = (best - o["rmse"].reindex(best.index)) / best
        for r in paired_stats(ours, theirs):
            if r["scene_class"] == "all":
                idx = margin.index
            else:
                idx = margin.index[(o["scene_class"].reindex(margin.index)
                                    == r["scene_class"]).to_numpy()]
            _, mlo, mhi = bootstrap_mean_ci(margin[idx])
            win_rows.append(dict(arm=arm, competitor="strongest", **r,
                                 rel_margin_ci_low=mlo, rel_margin_ci_high=mhi))
    within = pd.DataFrame(win_rows)

    x_rows = []
    for fam in E3_FAMILIES:
        for a, b in E3_CONTRASTS:
            if (a, fam) in tuned and (b, fam) in tuned:
                # paired_stats: Delta = theirs - ours = RMSE_a - RMSE_b (> 0: b better)
                for r in paired_stats(tuned[(b, fam)], tuned[(a, fam)]):
                    x_rows.append(dict(family=fam, contrast=f"{a}-{b}", **r))
    across = pd.DataFrame(x_rows)

    recovery.to_csv(out / "e3_recovery.csv", index=False)
    within.to_csv(out / "e3_within_arm.csv", index=False)
    across.to_csv(out / "e3_across_arms.csv", index=False)
    pd.concat([s.assign(arm=a) for (a, _), s in tuned.items()]).to_csv(
        out / "e3_selected_configs.csv", index=False)

    pd.set_option("display.width", 200)
    allc = recovery[recovery["scene_class"] == "all"]
    print(allc.pivot(index="family", columns="arm", values="mean_recovery").round(4))
    w = within[within["scene_class"] == "all"]
    print(w.pivot(index="competitor", columns="arm", values="mean_rel_margin").round(4))
    print(w.pivot(index="competitor", columns="arm", values="wins"))
    x = across[(across["scene_class"] == "all") & across["family"].isin(["psf_aware", "rl2"])]
    print(x[["family", "contrast", "wins", "n_regions", "mean_delta", "ci_low", "ci_high",
             "mean_rel_margin"]].to_string(index=False))
    e3_real()
    print(f"-> {out}")
    return recovery, within, across


def e3_real():
    """Real bench, no new run: nominal 25 m (R0) and combined nominal 27.95 m
    (R2) widths, PSF-aware against each competitor tuned at the same width."""
    import pandas as pd
    df = e4_frame("real")
    rows = []
    for label, f in (("R0", 1.0), ("R2", math.sqrt(1.25))):
        at_f = df[np.isclose(df["f"], round(f, 6))]
        tuned = {fam: select(at_f, fam, "B", "5fold")
                 for fam in E4_COMPETITORS + ("psf_aware",)}
        ours = tuned.pop("psf_aware")
        best = pd.concat({k: v.set_index("region_id")["rmse"] for k, v in tuned.items()},
                         axis=1).min(axis=1)
        tuned["strongest"] = pd.DataFrame({"region_id": best.index, "rmse": best.to_numpy()})
        for comp, theirs in tuned.items():
            for r in paired_stats(ours, theirs):
                rows.append(dict(arm=label, width_factor=f, competitor=comp, **r))
    table = pd.DataFrame(rows)
    out = ensure_out_dir("analysis_real")
    table.to_csv(out / "e3_real_widths.csv", index=False)
    print(table[table["scene_class"] == "all"][["arm", "competitor", "wins", "n_regions",
                                                "mean_rel_margin"]].to_string(index=False))
    return table


def e3_figure():
    """Fig. E3: (a) recovery per arm for each family, (b) PSF-aware margin over
    the strongest tuned competitor per arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.edgecolor": _E4_INK2, "axes.labelcolor": _E4_INK,
                         "xtick.color": _E4_INK2, "ytick.color": _E4_INK2,
                         "pdf.fonttype": 42})
    src = OUT_DIR / "analysis_synthetic"
    rec = pd.read_csv(src / "e3_recovery.csv")
    rec = rec[rec["scene_class"] == "all"]
    win = pd.read_csv(src / "e3_within_arm.csv")
    win = win[(win["scene_class"] == "all") & (win["competitor"] == "strongest")]
    x0 = np.arange(len(E3_ARMS))

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6), constrained_layout=True)
    ax = axes[0]
    offsets = np.linspace(-0.24, 0.24, len(_E4_SERIES))
    for off, (fam, (label, color, marker, _)) in zip(offsets, _E4_SERIES.items()):
        g = rec[rec["family"] == fam].set_index("arm").reindex(list(E3_ARMS))
        y = g["mean_recovery"].to_numpy() * 100
        err = np.vstack([y - g["ci_low"].to_numpy() * 100, g["ci_high"].to_numpy() * 100 - y])
        ax.errorbar(x0 + off, y, yerr=err, fmt=marker, color=color, ms=4.5, mec="white",
                    mew=0.6, elinewidth=1.0, capsize=0, label=label)
    ax.axhline(0.0, color=_E4_INK2, lw=0.6)
    ax.set_xticks(x0, [_E3_ARM_LABEL[a] for a in E3_ARMS])
    ax.set_ylabel("recovery vs blurred texture (%, mean, 95% CI)")
    ax.set_title("(a) recovery, each method tuned by 5-fold CV in its arm", loc="left")
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    g = win.set_index("arm").reindex(list(E3_ARMS))
    y = g["mean_rel_margin"].to_numpy() * 100
    err = np.vstack([y - g["rel_margin_ci_low"].to_numpy() * 100,
                     g["rel_margin_ci_high"].to_numpy() * 100 - y])
    color = _E4_SERIES["psf_aware"][1]
    ax.errorbar(x0, y, yerr=err, fmt="o", color=color, ms=5, mec="white", mew=0.6,
                elinewidth=1.0, capsize=0)
    for xi, yi, (_, r) in zip(x0, y, g.iterrows()):
        ax.annotate(f"{int(r['wins'])}/{int(r['n_regions'])}", (xi, yi),
                    xytext=(7, 0), textcoords="offset points", va="center",
                    fontsize=6.5, color=_E4_INK2)
    ax.axhline(0.0, color=_E4_INK2, lw=0.6)
    ax.set_xticks(x0, [_E3_ARM_LABEL[a] for a in E3_ARMS])
    ax.set_xlim(-0.5, len(E3_ARMS) - 0.3)
    ax.set_ylabel("PSF-aware margin (%, mean, 95% CI)")
    ax.set_title("(b) vs strongest tuned competitor (regions won)", loc="left")
    for a in axes:
        a.grid(axis="y", color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    figdir = ensure_out_dir("figures")
    fig.savefig(figdir / "e3_arms.pdf")
    fig.savefig(figdir / "e3_arms.png", dpi=200)
    plt.close(fig)
    print(f"-> {figdir / 'e3_arms.pdf'}")
