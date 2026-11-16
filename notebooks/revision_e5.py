"""E5 of the JSTARS revision: cross-sensor reference from SEN2VENmuS.

Sentinel-2 B4 at 10 m is resampled to HEALPix by every method and scored
against VENmuS B4 at 5 m acquired the same day (Michel et al., Data 2022, 7,
96; SEN2VENmuS v2.0.0, doi:10.5281/zenodo.14603764).  Protocol: section E5 of
``review/08_protocole_preenregistre.md``.

The VENmuS files are CC BY-NC 4.0 and are not redistributed: ``fetch`` downloads
the 13 protocol sites from Zenodo into ``notebooks/data/sen2venus/`` (only the
B2-B3-B4-B8 archives at 5 m and 10 m are kept), and only metrics leave this
module.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
import zipfile

import numpy as np

import revision_tools as rt

ZENODO_RECORD = "14603764"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
DATA_DIR = rt.NOTEBOOK_DIR / "data" / "sen2venus"
# VENmuS viewing zenith angle <= 25 deg (table 1 of the Zenodo notice).
E5_SITES = ("FR-LQ1", "NARYN", "FGMANAUS", "MAD-AMBO", "ARM", "BAMBENW2", "ES-IC3XG",
            "ANJI", "ATTO", "ESGISB-3", "ESGISB-1", "FR-BIL", "K34-AMAZ")
KEPT_ARCHIVES = ("_b2b3b4b8_10m.zip", "_b2b3b4b8_05m.zip")
B4 = 3                      # 1-based band index of B4 in the b2b3b4b8 files
DN_SCALE = 1e4              # int16 reflectance x 10 000

# Filters (protocol, section E5)
INVALID_LOW, INVALID_HIGH = -1000, 15000        # DN
MIN_RELATIVE_STD = 1e-3                         # real-bench "flat" criterion
BRIGHT_THRESHOLD, MAX_BRIGHT_FRACTION = 0.30, 0.30
COARSE_FWHM_M = 40.0
MIN_COARSE_CORR = 0.9
REG_FWHM_M = 10.0           # VENmuS smoothing before 2x2 averaging to 10 m
REG_UPSAMPLE = 100
MAX_SHIFT_PX = 1.0
MAX_QUADRANT_DEV_PX = 0.2
PATCHES_PER_SITE = 24
SAMPLE_SEED = rt.CV_SEED

# Geometry and methods
S2_NOMINAL_FWHM_M = 12.5
E5_WIDTHS = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
E5_LEVELS = (20, 19)
REF_SUBLEVEL = 23           # reference quadrature: L23 sub-cell centres
READOUT_DEPTH = 2           # baselines read at level + 2 (paper estimand)
EDGE_MARGIN_M = 200.0
ELLIPSOID = "WGS84"
THRESHOLD = 0.1
TARGET_KERNEL_MASS = 0.99
S2_PIXEL_M = 10.0
VEN_PIXEL_M = 5.0

# Amendment 3 (rule of amendment 1): grids extended where the E5 run retained
# an open edge.  Only the added configurations are computed (`part="ext"`).
E5_EXT_WIDTHS = (0.2, 0.3, 0.4, 0.5)          # deconvolutions: their f choice sat at 0.6
E5_EXT_PSF_LAMBDAS = (0.3, 1.0, 3.0, 10.0)    # psf_aware: lambda choice sat at 0.1
E5_EXT_PSF_ITERS = (0, 1, 2)                  # psf_aware: 5 iterations; 0 = x_ref = B y
E5_EXT_RL_ITERS = (1, 2)                      # rl2: 5 iterations (RL starts from a constant)


# =============================================================================
# Download
# =============================================================================
def _md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(sites=E5_SITES, data_dir=DATA_DIR):
    """Download each site archive, check its MD5, keep index.csv and the
    B2-B3-B4-B8 archives at 5 m and 10 m, delete the rest (login node)."""
    import requests
    record = requests.get(ZENODO_API, timeout=60).json()
    files = {f["key"]: f for f in record["files"]}
    data_dir.mkdir(parents=True, exist_ok=True)
    for site in sites:
        site_dir = data_dir / site
        done = site_dir / ".complete"
        if done.exists():
            print(f"[{site}] already fetched")
            continue
        meta = files[f"{site}.zip"]
        url = meta["links"]["self"]
        md5 = meta["checksum"].split(":", 1)[1]
        tmp = data_dir / f"{site}.zip.part"
        t0 = time.time()
        for attempt in range(3):
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for block in r.iter_content(chunk_size=1 << 22):
                        fh.write(block)
            if _md5(tmp) == md5:
                break
            print(f"[{site}] MD5 mismatch, retry {attempt + 1}")
        else:
            raise RuntimeError(f"{site}: MD5 mismatch after 3 attempts")
        size = tmp.stat().st_size / 1e9
        site_dir.mkdir(exist_ok=True)
        kept = 0
        with zipfile.ZipFile(tmp) as z:
            for name in z.namelist():
                base = os.path.basename(name)
                if base == "index.csv" or base.endswith(KEPT_ARCHIVES):
                    with z.open(name) as src, open(site_dir / base, "wb") as dst:
                        while True:
                            block = src.read(1 << 22)
                            if not block:
                                break
                            dst.write(block)
                    kept += 1
        tmp.unlink()
        done.write_text(f"md5 {md5}\n", encoding="utf-8")
        print(f"[{site}] {size:.2f} GB in {time.time() - t0:.0f} s, kept {kept} files",
              flush=True)


# =============================================================================
# Reading
# =============================================================================
def _index(site, data_dir=DATA_DIR):
    import pandas as pd
    idx = pd.read_csv(data_dir / site / "index.csv", sep=None, engine="python")
    rows = []
    for p10, p05 in zip(idx["b2b3b4b8_10m"], idx["b2b3b4b8_05m"]):
        z10, m10 = p10.split("/", 1)
        z05, m05 = p05.split("/", 1)
        stem = os.path.basename(m10)[: -len("_b2b3b4b8_10m.tif")]   # site_idx_date_tile
        rest = stem[len(site) + 1:]
        num, date, tile = rest.split("_")
        rows.append(dict(site=site, patch_id=stem, pair_id=f"{site}_{date}_{tile}",
                         patch_index=int(num), zip10=z10, member10=m10,
                         zip05=z05, member05=m05))
    return pd.DataFrame(rows)


def _read_tif(blob):
    import rasterio
    with rasterio.MemoryFile(blob) as mem, mem.open() as r:
        return r.read(B4).astype(np.float64), r.transform, r.crs


def read_patch(row, data_dir=DATA_DIR, archives=None):
    """(S2 B4 DN 128x128, VENmuS B4 DN 256x256, S2 transform, VEN transform, CRS).
    ``archives``: optional dict of open ZipFiles reused across patches."""
    site_dir = data_dir / row.site
    out = []
    for zname, member in ((row.zip10, row.member10), (row.zip05, row.member05)):
        if archives is not None and zname in archives:
            z = archives[zname]
        else:
            z = zipfile.ZipFile(site_dir / zname)
            if archives is not None:
                archives[zname] = z
        out.append(_read_tif(z.read(member)))
    (s2, t10, crs10), (ven, t05, crs05) = out
    if crs10 != crs05 or t10.c != t05.c or t10.f != t05.f:
        raise RuntimeError(f"{row.patch_id}: 10 m and 5 m footprints differ")
    return s2, ven, t10, t05, crs10


# =============================================================================
# Filters, harmonization and co-registration (input only)
# =============================================================================
def _block2(img):
    n, m = img.shape
    return img.reshape(n // 2, 2, m // 2, 2).mean(axis=(1, 3))


def _gauss(img, fwhm_m, pixel_m):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, fwhm_m / (2.0 * math.sqrt(2.0 * math.log(2.0))) / pixel_m,
                           mode="reflect")


def _inner(img, fwhm_m, pixel_m):
    sigma = fwhm_m / (2.0 * math.sqrt(2.0 * math.log(2.0))) / pixel_m
    b = int(math.ceil(4.0 * sigma))
    return img[b:-b, b:-b] if b > 0 else img


def _xcorr_shift(a, b):
    """Sub-pixel shift registering ``b`` to ``a``: cross-correlation of the
    Hann-windowed, mean-removed images, refined by upsampled DFT.  Without the
    window the coincident patch edges pull the estimate to zero; the phase
    normalization biases it to zero as well (selftest, protocol amendment 2)."""
    from skimage.filters import window
    from skimage.registration import phase_cross_correlation
    w = window("hann", a.shape)
    a = (a - a.mean()) * w
    b = (b - b.mean()) * w
    return np.asarray(phase_cross_correlation(a, b, upsample_factor=REG_UPSAMPLE,
                                              normalization=None)[0])


def estimate_shift(s2, ven):
    """Shift (drow, dcol), in Sentinel-2 pixels, registering VENmuS to
    Sentinel-2: S2[r, c] ~ VEN10[r - drow, c - dcol].  Also returns the
    largest deviation of the four quadrant estimates from it."""
    ven10 = _block2(_gauss(ven, REG_FWHM_M, VEN_PIXEL_M))
    full = _xcorr_shift(s2, ven10)
    h, w = s2.shape
    dev = 0.0
    for r0 in (0, h // 2):
        for c0 in (0, w // 2):
            q = _xcorr_shift(s2[r0:r0 + h // 2, c0:c0 + w // 2],
                             ven10[r0:r0 + h // 2, c0:c0 + w // 2])
            dev = max(dev, float(np.hypot(*(q - full))))
    return full, dev


def patch_quality(row, s2_dn, ven_dn):
    """Filter descriptors and decisions of one patch (protocol, E5 filters)."""
    out = dict(site=row.site, patch_id=row.patch_id, pair_id=row.pair_id)
    flags = []
    invalid = ((s2_dn < INVALID_LOW) | (s2_dn > INVALID_HIGH)).any() or \
        ((ven_dn < INVALID_LOW) | (ven_dn > INVALID_HIGH)).any()
    if invalid:
        flags.append("invalid")
    s2, ven = s2_dn / DN_SCALE, ven_dn / DN_SCALE
    mean = float(ven.mean())
    out["ven_mean"], out["ven_std"] = mean, float(ven.std())
    out["ven_relative_std"] = out["ven_std"] / (abs(mean) if abs(mean) > 1e-12 else 1.0)
    if out["ven_relative_std"] < MIN_RELATIVE_STD:
        flags.append("flat")
    out["bright_fraction"] = float((ven > BRIGHT_THRESHOLD).mean())
    if out["bright_fraction"] > MAX_BRIGHT_FRACTION:
        flags.append("bright")
    ven10 = _block2(ven)
    a_s2 = _inner(_gauss(s2, COARSE_FWHM_M, S2_PIXEL_M), COARSE_FWHM_M, S2_PIXEL_M).ravel()
    a_ven = _inner(_gauss(ven10, COARSE_FWHM_M, S2_PIXEL_M), COARSE_FWHM_M, S2_PIXEL_M).ravel()
    corr = float(np.corrcoef(a_s2, a_ven)[0, 1]) if a_s2.std() > 0 and a_ven.std() > 0 else np.nan
    out["coarse_corr"] = corr
    if not corr >= MIN_COARSE_CORR:
        flags.append("coarse_mismatch")
    slope, intercept = np.polyfit(a_ven, a_s2, 1)
    out["fit_a"], out["fit_b"] = float(slope), float(intercept)
    try:
        (drow, dcol), dev = estimate_shift(s2, ven)
    except Exception:
        drow = dcol = dev = np.nan
    out["shift_row_px"], out["shift_col_px"], out["shift_quadrant_dev_px"] = \
        float(drow), float(dcol), float(dev)
    if not np.hypot(drow, dcol) <= MAX_SHIFT_PX:
        flags.append("shift_large")
    if not dev <= MAX_QUADRANT_DEV_PX:
        flags.append("shift_unstable")
    out["flags"] = ";".join(flags)
    out["usable"] = not flags
    return out


def screen_site(site, data_dir=DATA_DIR, force=False):
    """Filter descriptors of every patch of one site, cached per site."""
    import pandas as pd
    path = rt.ensure_out_dir("e5_screen") / f"{site}.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    idx = _index(site, data_dir)
    archives = {}
    t0 = time.time()
    rows = []
    for row in idx.itertuples(index=False):
        s2, ven, *_ = read_patch(row, data_dir, archives)
        rows.append(patch_quality(row, s2, ven))
    for z in archives.values():
        z.close()
    frame = pd.DataFrame(rows)
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    print(f"[{site}] {len(idx)} patches screened, {int(frame.usable.sum())} usable "
          f"({time.time() - t0:.0f} s)", flush=True)
    return frame


def build_manifest(sites=E5_SITES, data_dir=DATA_DIR, shard=""):
    """Screen every patch of every site (``shard`` "k/n": only sites k, k+n,
    ... and no sample), then draw the protocol sample over all sites."""
    import pandas as pd
    if shard:
        k, n = (int(v) for v in shard.split("/"))
        for site in list(sites)[k::n]:
            screen_site(site, data_dir)
        return None
    man = pd.concat([screen_site(site, data_dir) for site in sites], ignore_index=True)
    man["usable"] = man["usable"].astype(bool)
    man["selected"] = False
    rng = np.random.default_rng(SAMPLE_SEED)
    for site in sites:
        ok = man[(man.site == site) & man.usable]
        pairs = sorted(ok.pair_id.unique())
        queues = []
        for p in (pairs[i] for i in rng.permutation(len(pairs))):
            ids = sorted(ok.loc[ok.pair_id == p, "patch_id"])
            queues.append([ids[i] for i in rng.permutation(len(ids))])
        chosen = []
        while len(chosen) < PATCHES_PER_SITE and any(queues):
            for q in queues:
                if q and len(chosen) < PATCHES_PER_SITE:
                    chosen.append(q.pop(0))
        man.loc[man.patch_id.isin(chosen), "selected"] = True
    path = rt.ensure_out_dir() / "e5_manifest.csv"
    man.to_csv(path, index=False)
    summary = man.groupby("site").agg(n=("patch_id", "size"), usable=("usable", "sum"),
                                      selected=("selected", "sum"))
    print(summary.to_string())
    flags = man.loc[~man.usable, "flags"].str.split(";").explode().value_counts()
    print("exclusions (a patch can carry several flags):")
    print(flags.to_string())
    print(f"-> {path}")
    return man


def selected_patches(data_dir=DATA_DIR):
    import pandas as pd
    man = pd.read_csv(rt.OUT_DIR / "e5_manifest.csv")
    man = man[man.selected].sort_values(["site", "patch_id"]).reset_index(drop=True)
    parts = []
    for site in man.site.unique():
        idx = _index(site, data_dir)
        parts.append(idx.merge(man[man.site == site], on=["site", "patch_id", "pair_id"]))
    return list(pd.concat(parts, ignore_index=True).itertuples(index=False))


# =============================================================================
# Geometry: evaluation cells, reference, readout points
# =============================================================================
def _hp():
    import healpix_geo
    return healpix_geo.nested


def evaluation_cells(t10, shape, crs, level, margin_m=EDGE_MARGIN_M, step_m=2.0):
    """Cells of ``level`` whose centre lies >= margin_m inside the footprint."""
    import pyproj
    x0, y1 = t10.c, t10.f
    x1 = x0 + t10.a * shape[1]
    y0 = y1 + t10.e * shape[0]
    xs = np.arange(x0 + margin_m - 10.0, x1 - margin_m + 10.0, step_m)
    ys = np.arange(y0 + margin_m - 10.0, y1 - margin_m + 10.0, step_m)
    xx, yy = np.meshgrid(xs, ys)
    to_ll = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
    lon, lat = to_ll.transform(xx.ravel(), yy.ravel())
    cells = np.unique(np.asarray(_hp().lonlat_to_healpix(lon, lat, level, ellipsoid=ELLIPSOID),
                                 dtype=np.int64))
    clon, clat = _hp().healpix_to_lonlat(cells.astype(np.uint64), level, ellipsoid=ELLIPSOID)
    to_utm = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
    cx, cy = to_utm.transform(np.asarray(clon), np.asarray(clat))
    inside = ((cx >= x0 + margin_m) & (cx <= x1 - margin_m)
              & (cy >= y0 + margin_m) & (cy <= y1 - margin_m))
    return cells[inside]


def child_points_utm(cells, level, sublevel, crs):
    """UTM (x, y) of the ``sublevel`` descendants of ``cells`` (nested), shape
    (n_cells, 4**(sublevel - level))."""
    import pyproj
    k = 4 ** (sublevel - level)
    kids = (np.asarray(cells, dtype=np.int64)[:, None] * k + np.arange(k)[None, :]).ravel()
    lon, lat = _hp().healpix_to_lonlat(kids.astype(np.uint64), sublevel, ellipsoid=ELLIPSOID)
    to_utm = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
    x, y = to_utm.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x).reshape(len(cells), k), np.asarray(y).reshape(len(cells), k)


def reference_values(ven_refl, t05, x, y, shift_rc, fit_ab):
    """Cell averages of the harmonized VENmuS image, constant per 5 m pixel,
    read at the sub-cell points (x, y) moved by the registration shift:
    the VENmuS value matching Sentinel-2 position (x, y) sits at
    (x - 10 dcol, y + 10 drow)."""
    drow, dcol = shift_rc
    xv = x - S2_PIXEL_M * dcol
    yv = y + S2_PIXEL_M * drow
    col = np.floor((xv - t05.c) / t05.a).astype(np.int64)
    row = np.floor((yv - t05.f) / t05.e).astype(np.int64)
    n, m = ven_refl.shape
    if col.min() < 0 or row.min() < 0 or col.max() >= m or row.max() >= n:
        raise RuntimeError("reference point outside the VENmuS patch")
    a, b = fit_ab
    return (a * ven_refl[row, col] + b).mean(axis=1)


def s2_centres(t10, shape):
    """UTM axes (ascending) of the Sentinel-2 pixel centres."""
    xs = t10.c + t10.a * (np.arange(shape[1]) + 0.5)
    ys = t10.f + t10.e * (np.arange(shape[0]) + 0.5)
    return xs, ys


def readout(fields, xs, ys, x, y, method="cubic", chunk=8):
    """Cell means of raster fields read at sub-cell points (paper estimand)."""
    from scipy.interpolate import RegularGridInterpolator
    order = np.argsort(ys)
    q = np.column_stack([y.ravel(), x.ravel()])
    out = []
    for i in range(0, len(fields), chunk):
        stack = np.stack([f[order] for f in fields[i:i + chunk]], axis=-1)
        vals = RegularGridInterpolator((ys[order], xs), stack, method=method,
                                       bounds_error=True)(q)
        out.append(vals.reshape(x.shape[0], x.shape[1], -1).mean(axis=1))
    return np.concatenate(out, axis=1)


# =============================================================================
# Runner (one patch)
# =============================================================================
E5_COLUMNS = rt.RESULT_COLUMNS + ["level"]


E5_PARTS = {"main": "e5", "ext": "e5_ext"}


def patch_csv_path(patch_id, part="main"):
    return rt.ensure_out_dir(E5_PARTS[part]) / f"{patch_id}.csv"


def _psf_grid(part):
    """psf_aware (lam, n_iter) pairs of one part.  The extension holds the
    pairs absent from the E1 grid; x_ref (0 iterations) does not depend on
    lam and is recorded once, with lam = 0."""
    main = [(lam, it) for lam in rt.PSF_LAMBDAS for it in rt.PSF_ITERS]
    if part == "main":
        return main
    lams = rt.PSF_LAMBDAS + E5_EXT_PSF_LAMBDAS
    iters = tuple(sorted(set(rt.PSF_ITERS) | set(E5_EXT_PSF_ITERS)))
    ext = [(lam, it) for lam in lams for it in iters if (lam, it) not in main and it > 0]
    return [(0.0, 0)] + ext


def run_patch(row, force=False, data_dir=DATA_DIR, part="main"):
    """Score every method on one patch.  ``part="main"`` is the pre-registered
    grid; ``part="ext"`` only the configurations amendment 3 adds."""
    import pandas as pd
    import pyproj
    import torch
    import healpix_resample as hr

    out_path = patch_csv_path(row.patch_id, part)
    if out_path.exists() and not force:
        return pd.read_csv(out_path)
    s2_dn, ven_dn, t10, t05, crs = read_patch(row, data_dir)
    s2, ven = s2_dn / DN_SCALE, ven_dn / DN_SCALE
    meta = dict(scene=row.patch_id, scene_class=row.site, region_id=row.site,
                patch_id=row.patch_id)
    shift = (row.shift_row_px, row.shift_col_px)
    fit = (row.fit_a, row.fit_b)
    xs, ys = s2_centres(t10, s2.shape)
    xx, yy = np.meshgrid(xs, ys)
    to_ll = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
    lon, lat = (np.asarray(v, dtype=np.float64) for v in to_ll.transform(xx.ravel(), yy.ravel()))
    y_obs = s2.ravel()

    levels = {}
    for level in E5_LEVELS:
        cells = evaluation_cells(t10, s2.shape, crs, level)
        rx, ry = child_points_utm(cells, level, REF_SUBLEVEL, crs)
        ref = reference_values(ven, t05, rx, ry, shift, fit)
        qx, qy = child_points_utm(cells, level, level + READOUT_DEPTH, crs)
        levels[level] = dict(cells=cells, ref=ref, qx=qx, qy=qy)

    rows = []

    def score(level, cell_ids, values, family, method, **params):
        L = levels[level]
        est = pd.Series(np.asarray(values, dtype=np.float64),
                        index=np.asarray(cell_ids, dtype=np.int64))
        est = est[~est.index.duplicated()].reindex(L["cells"]).to_numpy()
        m = np.isfinite(est) & np.isfinite(L["ref"])
        rmse = float(np.sqrt(np.mean((est[m] - L["ref"][m]) ** 2))) if m.any() else np.nan
        rows.append(dict(rt._row("sen2venus", meta, family, method, rmse=rmse,
                                 n_cells=int(m.sum()), corr=rt._corr(est, L["ref"]), **params),
                         level=level))

    if part == "main":
        for level, L in levels.items():
            for method in ("nearest", "linear", "cubic"):
                v = readout([s2], xs, ys, L["qx"], L["qy"], method=method)[:, 0]
                score(level, L["cells"], v, f"geo_{method}", f"classical_{method}")

    widths = E5_WIDTHS if part == "main" else E5_WIDTHS + E5_EXT_WIDTHS
    for f in widths:
        fwhm = f * S2_NOMINAL_FWHM_M
        common = dict(width_factor=f, fwhm_m=fwhm)
        scale = hr.fwhm_to_scale(fwhm)
        # psf_aware keeps f >= 0.6 (amendment 3: its f choice is interior)
        for level in (E5_LEVELS if f in E5_WIDTHS else ()):
            npt = int(hr.recommend_npt(scale, level, target_mass=TARGET_KERNEL_MASS)["npt"])
            t0 = time.perf_counter()
            rs = hr.PSFResampler(lon, lat, level, sigma_m=scale, Npt=npt, threshold=THRESHOLD,
                                 ellipsoid=ELLIPSOID, device="cpu", verbose=False)
            build_s = time.perf_counter() - t0
            for lam, it in _psf_grid(part):
                t1 = time.perf_counter()
                res = rs.resample(y_obs, lam=lam, max_iter=it, tol=rt.PSF_TOL)
                score(level, rt.to_numpy(res.cell_ids), rt.to_numpy(res.cell_data).ravel(),
                      "psf_aware", "psf_aware", lam=lam, n_iter=it,
                      runtime_s=build_s + time.perf_counter() - t1, **common)
            del rs
        kernel = rt.gaussian_kernel(fwhm, S2_PIXEL_M)
        if part == "main":
            fields, runtimes = rt.deconvolution_fields(s2, kernel, rl_variants=("clip", "minmax"))
        elif f in E5_EXT_WIDTHS:       # whole deconvolution grid, RL from 1 iteration
            fields, runtimes = rt.deconvolution_fields(
                s2, kernel, rl_variants=("clip", "minmax"),
                rl_iters=tuple(sorted(set(rt.RL_ITERS) | set(E5_EXT_RL_ITERS))))
        else:                          # RL at 1 and 2 iterations only
            fields, runtimes = {}, {}
            for variant in ("clip", "minmax"):
                img01, lo, sc = rt.rl_normalize(s2, variant)
                t0 = time.perf_counter()
                snaps = rt.rl_snapshots(img01, kernel, E5_EXT_RL_ITERS)
                for k, est in snaps.items():
                    fields[("rl2", variant, np.nan, k)] = est * sc + lo
                    runtimes[("rl2", variant, np.nan, k)] = (time.perf_counter() - t0) / len(snaps)
        keys = list(fields)
        for level, L in levels.items():
            vals = readout([fields[k] for k in keys], xs, ys, L["qx"], L["qy"])
            for j, key in enumerate(keys):
                family, method = rt._family_method(key)
                score(level, L["cells"], vals[:, j], family, method, variant=key[1],
                      lam=key[2], n_iter=key[3], runtime_s=runtimes[key], **common)
    frame = pd.DataFrame(rows, columns=E5_COLUMNS)
    tmp = out_path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(out_path)
    return frame


# =============================================================================
# Analysis
# =============================================================================
E5_FAMILIES = ("psf_aware", "rl2", "tik_id", "tik_lap", "wiener_hunt",
               "geo_nearest", "geo_linear", "geo_cubic")


def collect(parts=("main", "ext")):
    """Raw E5 rows: the pre-registered run plus, when present, the amendment 3
    extension (its patches must match the main run's)."""
    import pandas as pd
    frames = {}
    for part in parts:
        files = sorted(rt.ensure_out_dir(E5_PARTS[part]).glob("*.csv"))
        if files:
            frames[part] = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    if "main" not in frames:
        raise FileNotFoundError("no E5 results")
    if "ext" in frames:
        missing = set(frames["main"].patch_id) ^ set(frames["ext"].patch_id)
        if missing:
            raise RuntimeError(f"E5 extension incomplete: {len(missing)} patch(es) differ")
    return pd.concat(list(frames.values()), ignore_index=True)


def site_frame(patches):
    """Per-site means over patches of the RMSE and of the bicubic-relative
    score, in the layout `revision_tools.select` expects (one class)."""
    df = patches.copy()
    df["variant"] = df["variant"].fillna("")
    df["config"] = rt._config_label(df)
    bic = df[df.family == "geo_cubic"].set_index(["patch_id", "level"])["rmse"]
    df["rel"] = df["rmse"].to_numpy() / bic.reindex(
        list(zip(df["patch_id"], df["level"]))).to_numpy()
    keys = ["level", "region_id", "family", "method", "variant", "width_factor", "lam",
            "n_iter", "omega", "config"]
    site = (df.groupby(keys, dropna=False)
            .agg(rmse=("rmse", "mean"), rel=("rel", "mean"), n_patches=("patch_id", "nunique"))
            .reset_index())
    site["scene_class"] = "all"
    return df, site


def _transfer_configs():
    """Mode A configuration of each family minimizing the mean relative score
    over the 40 E1 synthetic regions (same 10 m -> L20 geometry)."""
    df = rt.prepare(rt.collect_results("synthetic", "e1e2"))
    df = df[df["width_factor"].isna() | np.isclose(df["width_factor"], 1.0)]
    out = {}
    for fam in E5_FAMILIES:
        sub = df[df.family == fam]
        if len(sub):
            out[fam] = sub.groupby("config")["rel"].mean().idxmin()
    return out


def _paired(ours, theirs):
    rows = [r for r in rt.paired_stats(ours, theirs) if r["scene_class"] == "all"]
    return rows[0]


def e5_analyze(level_main=20):
    import pandas as pd
    patches, sites_all = site_frame(collect())
    out = rt.ensure_out_dir("analysis_e5")
    transfer = _transfer_configs()
    comps = [f for f in E5_FAMILIES if f != "psf_aware"]
    sel_rows, cmp_rows, curve_rows, pw_rows = [], [], [], []
    for level in E5_LEVELS:
        site = sites_all[sites_all.level == level]
        tuned = {}
        for fam in E5_FAMILIES:
            for mode in ("A", "B"):
                s = rt.select(site, fam, mode, "loo_class")
                if len(s):
                    tuned[(fam, mode, "loso")] = s
            o = rt.select(site, fam, "B", "oracle")
            if len(o):
                tuned[(fam, "B", "oracle")] = o
            if fam in transfer:
                sub = site[(site.family == fam) & (site.config == transfer[fam])]
                if len(sub):
                    t = sub[["region_id", "scene_class", "config", "rmse"]].copy()
                    t["family"], t["mode"], t["scheme"] = fam, "A", "transfer"
                    tuned[(fam, "A", "transfer")] = t
        for (fam, mode, scheme), s in tuned.items():
            sel_rows.append(s.assign(level=level))
        for mode, scheme in (("A", "loso"), ("B", "loso"), ("A", "transfer")):
            ours = tuned.get(("psf_aware", mode, scheme))
            if ours is None:
                continue
            best = None
            for comp in comps:
                theirs = tuned.get((comp, mode, scheme))
                if theirs is None:
                    continue
                cmp_rows.append(dict(level=level, ours=f"{mode}/{scheme}", competitor=comp,
                                     **_paired(ours, theirs)))
                if comp in rt.E4_COMPETITORS:
                    r = theirs.set_index("region_id")["rmse"]
                    best = r if best is None else np.minimum(best, r.reindex(best.index))
            if best is not None:
                theirs = pd.DataFrame({"region_id": best.index, "rmse": best.to_numpy()})
                cmp_rows.append(dict(level=level, ours=f"{mode}/{scheme}",
                                     competitor="strongest", **_paired(ours, theirs)))
            # descriptive: patch-level wins with each site's selected configs
            cfg_o = ours.set_index("region_id")["config"]
            pl = patches[patches.level == level]
            mine = pl[pl.family == "psf_aware"]
            mine = mine[mine.config == mine.region_id.map(cfg_o)].set_index("patch_id")["rmse"]
            for comp in comps:
                theirs = tuned.get((comp, mode, scheme))
                if theirs is None:
                    continue
                cfg_t = theirs.set_index("region_id")["config"]
                t = pl[pl.family == comp]
                t = t[t.config == t.region_id.map(cfg_t)].set_index("patch_id")["rmse"]
                idx = mine.index.intersection(t.index)
                pw_rows.append(dict(level=level, ours=f"{mode}/{scheme}", competitor=comp,
                                    n_patches=len(idx), patch_win_frac=float((mine[idx] < t[idx]).mean())))
        # width curves: every family tuned by LOSO at each f
        relmap = site.set_index(["region_id", "family", "config"])["rel"]
        for f in sorted(site["width_factor"].dropna().unique()):
            at_f = site[np.isclose(site["width_factor"], f)]
            tf = {}
            for fam in ("psf_aware",) + rt.E4_COMPETITORS:
                s = rt.select(at_f[at_f.family == fam], fam, "B", "loo_class")
                if len(s):
                    tf[fam] = s
                    rel = pd.Series([relmap[(rid, fam, cfg)] for rid, cfg in
                                     zip(s["region_id"], s["config"])])
                    curve_rows.append(dict(level=level, f=f, family=fam,
                                           median_rel=float(rel.median()),
                                           q25_rel=float(rel.quantile(0.25)),
                                           q75_rel=float(rel.quantile(0.75))))
            if "psf_aware" not in tf:      # f < 0.6 (amendment 3)
                continue
            best = pd.concat({k: v.set_index("region_id")["rmse"] for k, v in tf.items()
                              if k != "psf_aware"}, axis=1).min(axis=1)
            o = tf["psf_aware"].set_index("region_id")["rmse"].reindex(best.index)
            curve_rows.append(dict(level=level, f=f, family="psf_aware_vs_strongest",
                                   wins=int((o < best).sum()), n_sites=int(len(best)),
                                   mean_rel_margin=float(((best - o) / best).mean())))
    sel = pd.concat(sel_rows, ignore_index=True)
    comp = pd.DataFrame(cmp_rows)
    curves = pd.DataFrame(curve_rows)
    pwins = pd.DataFrame(pw_rows)
    sel.to_csv(out / "e5_selected_configs.csv", index=False)
    comp.to_csv(out / "e5_comparisons.csv", index=False)
    curves.to_csv(out / "e5_width_curves.csv", index=False)
    pwins.to_csv(out / "e5_patch_wins.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{patches.patch_id.nunique()} patches, {sites_all.region_id.nunique()} sites")
    print("transferred configurations:", transfer)
    print(comp[["level", "ours", "competitor", "wins", "n_regions", "sign_test_p",
                "mean_rel_margin", "ci_low", "ci_high"]].round(4).to_string(index=False))
    c = curves[curves.family == "psf_aware_vs_strongest"]
    print(c[["level", "f", "wins", "n_sites", "mean_rel_margin"]].round(4).to_string(index=False))
    e5_figure()
    print(f"-> {out}")
    return sel, comp, curves


def e5_figure():
    """Fig. E5: relative RMSE versus assumed width (median, IQR over sites),
    each method tuned by leave-one-site-out at each f; L20 and L19."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.edgecolor": rt._E4_INK2, "axes.labelcolor": rt._E4_INK,
                         "xtick.color": rt._E4_INK2, "ytick.color": rt._E4_INK2,
                         "pdf.fonttype": 42})
    cur = pd.read_csv(rt.OUT_DIR / "analysis_e5" / "e5_width_curves.csv")
    fig, axes = plt.subplots(1, len(E5_LEVELS), figsize=(7.16, 2.8), constrained_layout=True,
                             squeeze=False, sharey=True)
    cell_m = {20: 6.2, 19: 12.4}
    for c, level in enumerate(E5_LEVELS):
        ax = axes[0, c]
        g0 = cur[cur.level == level]
        for fam, (label, color, marker, ls) in rt._E4_SERIES.items():
            g = g0[g0.family == fam].sort_values("f")
            if g.empty:
                continue
            ax.fill_between(g["f"], g["q25_rel"], g["q75_rel"], color=color, alpha=0.15, lw=0)
            ax.plot(g["f"], g["median_rel"], color=color, lw=1.6, ls=ls, marker=marker, ms=4,
                    mec="white", mew=0.6, label=label)
        w = g0[g0.family == "psf_aware_vs_strongest"].sort_values("f")
        for _, r in w.iterrows():
            ax.text(r["f"], 1.0, f"{int(r['wins'])}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6, color=rt._E4_INK2)
        ax.axhline(1.0, color=rt._E4_INK2, lw=0.6)
        ax.set_xlabel("assumed width factor f (× 12.5 m)")
        ax.set_title(f"L{level} ({cell_m.get(level, 0):.1f} m cells); "
                     "top row: sites won by PSF-aware", loc="left", pad=12)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].set_ylabel("RMSE / bicubic RMSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
    figdir = rt.ensure_out_dir("figures")
    fig.savefig(figdir / "e5_width_curves.pdf")
    fig.savefig(figdir / "e5_width_curves.png", dpi=200)
    plt.close(fig)
    print(f"-> {figdir / 'e5_width_curves.pdf'}")
