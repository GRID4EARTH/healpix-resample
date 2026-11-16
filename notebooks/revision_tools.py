"""Shared machinery for the JSTARS major revision: experiments E0, E1 and E2.

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
WIDTH_FACTORS = {
    # sqrt(1.25): combined nominal response sqrt(25^2 + 12.5^2) = 27.95 m.
    "real": (0.9, 1.0, math.sqrt(1.25), 1.2),
    "synthetic": (0.9, 1.0, SYNTH_EQUIV_FACTOR, 1.2),
}
NOMINAL_FWHM = {"real": REAL_NOMINAL_FWHM_M, "synthetic": SYNTH_NOMINAL_FWHM_M}

PSF_LAMBDAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
PSF_ITERS = (5, 14, 30, 100)
PSF_TOL = 1e-8
RL_ITERS = (5, 10, 20, 50, 100, 200, 500)
TIK_LAMBDAS = tuple(float(v) for v in np.logspace(-5, 0, 11))
SIRT_OMEGAS = (0.5, 1.0, 1.5)
SIRT_ITERS = (5, 14, 30, 100)
MLEM_ITERS = (5, 10, 20, 50, 100, 200)

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
            "e2_wcg": (x_ref, pcg_snapshots(H_w, b_w, PSF_ITERS, weight=dx)),
            "e2_cg_euclid": (x_ref, pcg_snapshots(H_w, b_w, PSF_ITERS, weight=None)),
            "e2_jacobi": (x_ref, pcg_snapshots(H_w, b_w, PSF_ITERS, weight=dx, precond=jac)),
            "e2_tik_x0": (0.0, pcg_snapshots(H_e, b_tik0, PSF_ITERS, weight=None)),
            "e2_tik_dref": (x_ref, pcg_snapshots(H_e, b_tikd, PSF_ITERS, weight=None)),
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


def region_csv_path(bench, scene):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scene))
    return ensure_out_dir(f"e1e2_{bench}") / f"{safe}.csv"


def _write_rows(rows, path):
    import pandas as pd
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return frame


def run_real_region(gt, scene, *, do_e2=True, force=False):
    """E1 (+ E2) on one real region; returns the region's result frame."""
    import pandas as pd
    out_path = region_csv_path("real", scene)
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
    for f in WIDTH_FACTORS["real"]:
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


def run_synthetic_region(ns, patch_row, *, do_e2=True, force=False):
    """E1 (+ E2) on one synthetic region (its paper-selected patch)."""
    import pandas as pd
    out_path = region_csv_path("synthetic", patch_row.patch_id)
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    meta = dict(scene=patch_row.patch_id, scene_class=patch_row.scene_class,
                region_id=patch_row.region_id, patch_id=patch_row.patch_id)
    products = ns["build_synthetic_patch"](patch_row)
    ref_ids, ref_vals = products["parent_ids"], products["hpx_latent"]
    rows = []

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
    for f in WIDTH_FACTORS["synthetic"]:
        fwhm = f * SYNTH_NOMINAL_FWHM_M
        common = dict(width_factor=f, fwhm_m=fwhm)

        scale = ns["fwhm_to_scale"](fwhm)
        npt = int(ns["recommend_npt"](scale, ns["HEALPIX_LEVEL"],
                                      target_mass=ns["TARGET_KERNEL_MASS"])["npt"])
        t0 = time.perf_counter()
        rs = ns["PSFResampler"](lon_deg=lon, lat_deg=lat, level=ns["HEALPIX_LEVEL"],
                                threshold=ns["THRESHOLD"], verbose=False,
                                ellipsoid="sphere", sigma_m=scale, Npt=npt,
                                device=ns["DEVICE"])
        build_s = time.perf_counter() - t0
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


def collect_results(bench):
    import pandas as pd
    files = sorted(ensure_out_dir(f"e1e2_{bench}").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no per-region results under {OUT_DIR / f'e1e2_{bench}'}")
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
