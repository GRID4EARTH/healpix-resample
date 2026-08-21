"""real_groundtruth_common_tools.py

All function/constant definitions for `notebooks/real_groundtruth_downscale.ipynb`
(the real-data, HEALPix-cell-space analogue of the paper's controlled
synthetic-detector experiment), factored out of the notebook so the notebook
itself only contains step-by-step calls and inline inspection of
intermediate results.

Usage from the notebook (which runs with cwd = notebooks/, one level below
the repo root -- see DATA_DIR/FIG_DIR/TABLE_DIR below, which are relative
paths resolved against that cwd)::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("..").resolve()))
    import tests.real_groundtruth_common_tools as gt

Import the *module*, not `from ... import *`: every function below reads its
configuration (BLOCK, REF_LEVEL, RECON_FWHM_M, ...) from this module's own
globals at call time, so overriding e.g. ``gt.REF_LEVEL = 17`` before calling
``gt.reconstruct_psf_aware(...)`` actually changes behaviour -- the same
effect as editing this file, without leaving the notebook. `from ... import
*` would copy the *values* into the notebook's namespace instead, and
reassigning the copy would not affect what the functions below see.

Protocol (steps 1-7, see the notebook's title cell for the full rationale
and revision history):
  1. fetch a real, native 10 m Sentinel-2 B04 UTM patch                      -> extract_bench_data / load_scene_patch
  2. PSF-aware resample the real (undegraded) data to NATIVE_LEVEL           -> build_reference (step 2 of 2-3)
  3. plain (no-PSF) NESTED downgrade to REF_LEVEL -- the reference           -> build_reference (step 3 of 2-3)
  4. Gaussian-blur the real data to the coarse effective resolution         -> degrade_to_coarse (step 4 of 4-5)
  5. point-sample (decimate) every BLOCK-th pixel                            -> build_coarse_grid (step 5 of 4-5)
  6. PSF-aware resample the coarse samples directly onto REF_LEVEL           -> reconstruct_psf_aware
  7. compare against the reference, cell-for-cell, at REF_LEVEL              -> align_and_mask + compute_metrics
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import torch
import xarray as xr
import pyproj
import pystac_client
import healpix_geo
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from healpix_resample import (
        PSFResampler, BicubicResampler, fwhm_to_scale, recommend_npt, cell_size_m,
    )
except Exception as exc:
    PSFResampler = BicubicResampler = fwhm_to_scale = recommend_npt = cell_size_m = None
    warnings.warn(f"Could not import healpix_resample: {exc}")

try:
    from skimage.restoration import richardson_lucy
except Exception as exc:
    richardson_lucy = None
    warnings.warn(f"Could not import richardson_lucy: {exc}")

# =============================================================================
# Configuration -- see the notebook's configuration section for the full
# reasoning behind each default. Every function below reads these as module
# globals, so `gt.CONST = value` before a call changes what that call does.
# =============================================================================

# DATA_DIR is the SAME cache directory used by test-resample-paper.ipynb.
# Relative paths, resolved against the *caller's* cwd -- intended to be run
# from notebooks/, one level below this file.
DATA_DIR  = Path("data")
FIG_DIR   = Path("figures")
TABLE_DIR = Path("tables")

RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)

# -- Real data -----------------------------------------------------------------
MIRROR_BASE = ("https://grid4earth.s3.gra.io.cloud.ovh.net"
               "/public/eopf-mirror/sentinel-2-l2a/patches")
PATCH_SIZE   = 256          # native 10 m pixels per side, as fetched/cached
BAND         = "b04"        # Sentinel-2 red band, 10 m native
PIXEL_SIZE_M = 10.0
ELLIPSOID    = "WGS84"
DEVICE       = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# -- Steps 2-3: reference construction (real native data, no degradation) ------
NATIVE_LEVEL  = 20     # matches the level used for 10 m data everywhere else
                        # in the paper
NATIVE_FWHM_M = 12.5    # the paper's own empirically calibrated 10 m
                         # effective response, reused here to build the best
                         # available PSF-aware estimate of the true field
                         # from real, undegraded 10 m data
NATIVE_LAMBDA = 0.01
REF_LEVEL = 19          # target level for the reference AND for the
                         # reconstruction under test (step 6)
MIN_CHILDREN_FRAC = 0.5 # healpix_down() drops a level-18 cell if fewer than
                         # this fraction of its 16 level-20 children were
                         # actually retained by the level-20 operator

# -- Steps 4-5: degradation (real native data -> coarse UTM samples) -----------
BLOCK = 2                                    # 10 m -> 20 m
CENTRAL_SIZE = (PATCH_SIZE // BLOCK) * BLOCK  # largest multiple of BLOCK <= PATCH_SIZE
PIXEL_SIZE_COARSE_M = PIXEL_SIZE_M * BLOCK

# Gaussian blur, then point-sample (decimate) every BLOCK-th pixel --
# deliberately NOT a block average (see degrade_to_coarse's docstring).
DEGRADE_FWHM_M = 1.25 * PIXEL_SIZE_COARSE_M

# -- Step 6: reconstruction under test (20 m samples -> REF_LEVEL) -------------
RECON_FWHM_M = DEGRADE_FWHM_M   # assumed effective response;
                                              # deliberately != DEGRADE_FWHM_M
LAMBDA_COARSE = 0.01
MAX_ITER      = 100
THRESHOLD     = 0.1
RICHARDSON_LUCY_ITER = 100

# -- Evaluation: geometric edge margin (metres), not a pixel crop --------------
EDGE_MARGIN_M = 8 * RECON_FWHM_M

CLASSICAL_METHODS = ["nearest", "linear", "cubic"]
CLASSICAL_LABELS = {"nearest": "Nearest", "linear": "Bilinear", "cubic": "Bicubic"}

def describe_config():
    """Print the physical meaning of the current level/size configuration.
    Call this explicitly after changing any constant above -- it is not run
    automatically on import."""
    if cell_size_m is None:
        print("healpix_resample.cell_size_m is not available.")
        return
    print(f"NATIVE_LEVEL={NATIVE_LEVEL}: cell size ~= {cell_size_m(NATIVE_LEVEL):.2f} m "
          f"(reference built from real {PIXEL_SIZE_M:.0f} m data, step 2)")
    print(f"REF_LEVEL={REF_LEVEL}:    cell size ~= {cell_size_m(REF_LEVEL):.2f} m "
          f"(comparison level for every method, vs. {PIXEL_SIZE_COARSE_M:.0f} m "
          f"coarse/degraded pixel, step 7)")
    print(f"EDGE_MARGIN_M={EDGE_MARGIN_M:.0f} m excluded from every metric "
          f"(8x RECON_FWHM_M={RECON_FWHM_M:.1f} m)")


def _cache_suffix():
    """Every parameter that changes a *result* (not just plotting) is baked
    into intermediate .npz cache filenames, so changing the configuration
    always recomputes rather than silently reusing a stale result from a
    different configuration."""
    return (f"native{NATIVE_LEVEL}_ref{REF_LEVEL}_block{BLOCK}"
            f"_fwhm{int(round(RECON_FWHM_M))}")


# =============================================================================
# Benchmark scenes
# =============================================================================

benchmark_coordinates = {
    "urban": {
        "location": "Paris, France",
        "wgs84": {"lat": 48.8566, "lon": 2.3522},
        "recommended_date": "2026-05-01/2026-05-30",
        "product_id": "S2B_MSIL2A_20260527T105619_N0512_R094_T31UDQ_20260527T133135",
        "cloud": 20,
    },
    "water": {
        "location": "Lake Geneva, Switzerland",
        "wgs84": {"lat": 46.4983, "lon": 6.6327},
        "recommended_date": "2026-06-01/2026-06-24",
        "product_id": "S2A_MSIL2A_20260624T103041_N0512_R108_T32TLS_20260624T182910",
        "cloud": 20,
    },
    "forest": {
        "location": "Black Forest, Germany",
        "wgs84": {"lat": 48.265, "lon": 8.016},
        "recommended_date": "2026-07-01/2026-07-31",
        "product_id": "S2A_MSIL2A_20260704T102701_N0512_R108_T32UMU_20260704T170718",
        "cloud": 20,
    },
    "agriculture": {
        "location": "Beauce plains, France",
        "wgs84": {"lat": 48.258, "lon": 1.499},
        "recommended_date": "2026-06-01/2026-06-30",
        "product_id": "S2B_MSIL2A_20260623T104619_N0512_R051_T31UCP_20260623T132606",
        "cloud": 20,
    },
}


# =============================================================================
# Step 1: Sentinel-2 fetch utilities
# =============================================================================

def _crs_of(obj):
    """CRS of an EOPF product or a cached patch.

    `obj.crs_code` is not provided by xarray-eopf 0.3.0, and the metadata key
    is `horizontal_CRS_code` (capital CRS), not `horizontal_crs_code`. The CF
    grid-mapping coordinate is the stable place to look, so prefer it.
    """
    try:
        return pyproj.CRS.from_wkt(obj.spatial_ref.attrs["crs_wkt"])
    except Exception:
        try:
            meta = getattr(obj, "attrs", {}).get("other_metadata", {})
            return pyproj.CRS.from_user_input(meta["horizontal_CRS_code"])
        except Exception:
            meta = getattr(obj, "attrs", {}).get("other_metadata", {})
            return pyproj.CRS.from_user_input(meta["horizontal_crs_code"])


def _add_latlon(ds, transformer):
    xx, yy = np.meshgrid(ds.x.values, ds.y.values)
    lon, lat = transformer.transform(xx, yy)
    ds = ds.assign_coords(longitude=(["y", "x"], lon), latitude=(["y", "x"], lat))
    return ds


def extract_bench_data(scene, patch_size=None, force=False, coords=None, band=None):
    """Fetch (or load from cache) the native 10 m patch for one scene.

    `coords` defaults to `benchmark_coordinates[scene]`. Pass it explicitly to
    fetch a one-off location without registering it in `benchmark_coordinates`.
    """
    if patch_size is None:
        patch_size = PATCH_SIZE
    if band is None:
        band = BAND

    DATA_DIR.mkdir(exist_ok=True)
    xr.set_options(keep_attrs=True, display_expand_attrs=False)
    catalog = pystac_client.Client.open("https://stac.core.eopf.eodc.eu")
    path = DATA_DIR / f"{scene}_data.zarr"
    if path.exists() and not force:
        try:
            import zarr as _zarr
            _a = _zarr.open_group(str(path), mode="r").attrs.asdict()
            if _a.get("source_item_id"):
                print(f"[{scene}] cached  <- {_a['source_item_id']}"
                      f"  ({_a.get('source_datetime', '?')},"
                      f" cloud {_a.get('source_cloud_cover', '?')}%)")
            else:
                print(f"[{scene}] cached  <- provenance not recorded "
                      f"(cache predates source_item_id; delete it to refresh)")
        except Exception as _exc:
            print(f"[{scene}] cached  <- could not read provenance: {_exc}")
        return
    coords = benchmark_coordinates[scene] if coords is None else coords
    lat0, lon0 = coords["wgs84"]["lat"], coords["wgs84"]["lon"]

    pinned = coords.get("product_id")
    if pinned:
        try:
            import fsspec
            _ds = xr.open_zarr(
                fsspec.get_mapper(f"{MIRROR_BASE}/{pinned}.zarr"), consolidated=True
            )
            _ds.to_zarr(path, mode="w")
            print(f"[{scene}] mirror  <- {pinned}"
                  f"  (cloud {_ds.attrs.get('source_cloud_cover', '?')}%)")
            return
        except Exception as exc:
            print(f"[{scene}] mirror unavailable ({type(exc).__name__}), "
                  f"falling back to the STAC search: {str(exc)[:70]}")

    items = list(catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[lon0 - 0.05, lat0 - 0.05, lon0 + 0.05, lat0 + 0.05],
        datetime=coords["recommended_date"],
        query={"eo:cloud_cover": {"lt": coords["cloud"]}},
    ).get_all_items())
    if not items:
        raise RuntimeError(f"No items for {scene}")
    item = items[0]

    _props = item.properties
    print(f"[{scene}] {len(items)} product(s) matched; using {item.id}"
          f"  ({_props.get('datetime', '?')},"
          f" cloud {_props.get('eo:cloud_cover', '?')}%)")
    if len(items) > 1:
        print(f"[{scene}]   others: {', '.join(i.id for i in items[1:4])}"
              f"{' ...' if len(items) > 4 else ''}")

    ds = xr.open_dataset(
        item.assets["product"].href,
        engine="eopf-zarr",
        resolution=10,
        variables=[band],
        chunks={},
    )

    try:
        src_crs = pyproj.CRS.from_wkt(ds.spatial_ref.attrs["crs_wkt"])
    except Exception:
        try:
            src_crs = pyproj.CRS.from_user_input(
                ds.attrs["other_metadata"]["horizontal_CRS_code"]
            )
        except Exception:
            src_crs = pyproj.CRS.from_user_input(
                ds.attrs["other_metadata"]["horizontal_crs_code"]
            )
    transformer = pyproj.Transformer.from_crs(
        src_crs, pyproj.CRS.from_epsg(4326), always_xy=True
    )

    _fwd = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(4326), _crs_of(ds), always_xy=True
    )
    x_c, y_c = _fwd.transform(lon0, lat0)
    half = patch_size // 2 * 10
    ds_patch = ds.sel(
        x=slice(x_c - half, x_c + half),
        y=slice(y_c + half, y_c - half),
    )
    ds_patch = _add_latlon(ds_patch, transformer)
    ds_patch.attrs["source_item_id"] = item.id
    ds_patch.attrs["source_collection"] = "sentinel-2-l2a"
    ds_patch.attrs["source_stac_endpoint"] = "https://stac.core.eopf.eodc.eu"
    ds_patch.attrs["source_datetime"] = str(_props.get("datetime", ""))
    ds_patch.attrs["source_cloud_cover"] = str(_props.get("eo:cloud_cover", ""))
    ds_patch.to_zarr(path, mode="w")


# =============================================================================
# Common utilities: loading, UTM axes, decimation, the no-PSF HEALPix
# downgrade, and the geometric helpers needed to evaluate every method --
# HEALPix-native or raster-based -- at exactly the same set of points.
# =============================================================================

def fill_nan_with_mean(img):
    img = np.asarray(img, dtype=np.float32).copy()
    mask = np.isfinite(img)
    if not mask.any():
        raise ValueError("Image contains no finite pixels.")
    img[~mask] = np.nanmean(img)
    return img


def load_scene_patch(scene_name):
    path = DATA_DIR / f"{scene_name}_data.zarr"
    dt = xr.open_datatree(path, engine="zarr", consolidated=False, chunks={})
    da = dt[BAND]
    img = fill_nan_with_mean(da.values)
    lon = da["longitude"].values.astype(np.float64)
    lat = da["latitude"].values.astype(np.float64)
    return img, lon, lat, da


def central_crop(img, size=None):
    if size is None:
        size = CENTRAL_SIZE
    ny, nx = img.shape
    cy, cx = ny // 2, nx // 2
    h = size // 2
    return img[cy - h:cy + h, cx - h:cx + h]


def get_utm_axes(da, central_size=None):
    """1-D UTM x and y axes for the central crop of a DataArray."""
    if central_size is None:
        central_size = CENTRAL_SIZE
    x_full = da.x.values
    y_full = da.y.values
    nx, ny = x_full.size, y_full.size
    cx, cy = nx // 2, ny // 2
    h = central_size // 2
    x0 = x_full[cx - h:cx + h]
    y0 = y_full[cy - h:cy + h]
    return x0, y0


def decimate_2d(arr, block=None, offset=None):
    """Point-sample every `block`-th row/column of a 2-D array, at a fixed
    sub-pixel offset (defaults to the block centre, `block // 2`).

    Deliberately NOT a block average: averaging mixes in a boxcar (square,
    non-circular) component on top of whatever blur was already applied,
    which a circular-kernel resampler (PSFResampler) has no way to
    represent. Decimating an already-blurred array keeps the *shape* of the
    true degradation kernel equal to the shape of that blur alone.
    """
    if block is None:
        block = BLOCK
    if offset is None:
        offset = block // 2
    return arr[offset::block, offset::block]


def decimate_1d(arr, block=None, offset=None):
    """1-D analogue of decimate_2d, for UTM axis arrays: the geolocation of
    each decimated sample must be the true location of the pixel that was
    point-sampled, not a block-averaged position."""
    if block is None:
        block = BLOCK
    if offset is None:
        offset = block // 2
    return arr[offset::block]


def healpix_down(cell_ids_fine, cell_data_fine, level_fine, level_coarse,
                  min_children_frac=None):
    """Standard NESTED-scheme HEALPix downgrade: the unweighted mean of the
    4**(level_fine - level_coarse) child cells belonging to each coarser
    parent, with NO spatial-response weighting -- unlike PSFResampler, this
    is a purely geometric aggregation.

    Stand-in for `healpix_analyse`'s own `down()` (not installed in this
    environment); swap in the real call if you have that package available.
    The NESTED parent/child relationship this relies on -- child_id in
    [parent_id * 4**dl, (parent_id + 1) * 4**dl) -- is a fixed property of
    the indexing scheme itself, not something specific to this notebook.

    A parent cell is dropped if fewer than `min_children_frac` of its
    nominal 4**dl children are actually present in `cell_ids_fine` (e.g.
    near the edge of a finite patch, or where the fine-level operator's own
    threshold pruned some children) -- this avoids quietly averaging over
    just 1-2 children and calling it a level_coarse value.
    """
    if min_children_frac is None:
        min_children_frac = MIN_CHILDREN_FRAC
    dl = level_fine - level_coarse
    if dl <= 0:
        raise ValueError("level_fine must be strictly greater than level_coarse")
    factor = 4 ** dl

    ids = np.asarray(cell_ids_fine, dtype=np.int64)
    vals = np.asarray(cell_data_fine, dtype=np.float64)
    parents = ids // factor

    uniq_parents, inv = np.unique(parents, return_inverse=True)
    sums = np.bincount(inv, weights=vals, minlength=uniq_parents.size)
    counts = np.bincount(inv, minlength=uniq_parents.size)
    means = sums / np.maximum(counts, 1)

    keep = counts >= (min_children_frac * factor)
    return uniq_parents[keep].astype(np.int64), means[keep].astype(np.float32)


def cell_centers_lonlat(cell_ids, level, ellipsoid=None):
    """HEALPix NESTED cell centres (lon, lat, degrees) for arbitrary cell ids."""
    if ellipsoid is None:
        ellipsoid = ELLIPSOID
    ids = np.asarray(cell_ids, dtype=np.uint64)
    lon, lat = healpix_geo.nested.healpix_to_lonlat(ids, level, ellipsoid=ellipsoid)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def lonlat_to_utm(lon_deg, lat_deg, da):
    """Project lon/lat (degrees) onto the UTM CRS of a cached patch's
    DataArray -- used to evaluate raster-based baselines (which only know
    how to interpolate on a UTM grid) at HEALPix cell centres."""
    fwd = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(4326), _crs_of(da), always_xy=True
    )
    x, y = fwd.transform(np.asarray(lon_deg), np.asarray(lat_deg))
    return np.asarray(x), np.asarray(y)


def interior_mask(x, y, x_axis, y_axis, margin_m=None):
    """True for points at least `margin_m` inside the UTM bounding box
    spanned by `x_axis`/`y_axis` -- the geometric analogue of a pixel-based
    border crop, applicable to an unstructured set of HEALPix cell centres
    rather than a regular image array."""
    if margin_m is None:
        margin_m = EDGE_MARGIN_M
    xmin, xmax = min(x_axis.min(), x_axis.max()), max(x_axis.min(), x_axis.max())
    ymin, ymax = min(y_axis.min(), y_axis.max()), max(y_axis.min(), y_axis.max())
    return ((x >= xmin + margin_m) & (x <= xmax - margin_m) &
            (y >= ymin + margin_m) & (y <= ymax - margin_m))


# =============================================================================
# Steps 2-3: reference construction from real native data
# =============================================================================

def build_reference(scene_name, force=False):
    """Steps 1-3: real native 10 m patch -> PSF-aware HEALPix field at
    NATIVE_LEVEL (the best available PSF-aware estimate of the true field,
    built from real, undegraded data using the paper's own calibrated 10 m
    response) -> plain (no-PSF) NESTED downgrade to REF_LEVEL.
    """
    TABLE_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    out_npz = DATA_DIR / f"{scene_name}_real_downscale_reference_{_cache_suffix()}.npz"
    if out_npz.exists() and not force:
        return dict(np.load(out_npz, allow_pickle=True))
    if PSFResampler is None:
        raise ImportError("PSFResampler is not available. Install healpix-resample.")

    img_n, lon_n, lat_n, da = load_scene_patch(scene_name)
    img_n = central_crop(img_n)
    lon_n = central_crop(lon_n)
    lat_n = central_crop(lat_n)

    scale_m = fwhm_to_scale(NATIVE_FWHM_M)
    npt = recommend_npt(scale_m, NATIVE_LEVEL)["npt"]
    resampler = PSFResampler(
        lon_deg=lon_n.reshape(-1), lat_deg=lat_n.reshape(-1),
        level=NATIVE_LEVEL, sigma_m=scale_m, Npt=npt,
        threshold=THRESHOLD, device=DEVICE, verbose=True,
    )
    res = resampler.resample(img_n.reshape(-1), lam=NATIVE_LAMBDA, max_iter=MAX_ITER)

    cell_ids_ref, cell_data_ref = healpix_down(
        res.cell_ids, res.cell_data, NATIVE_LEVEL, REF_LEVEL,
    )

    np.savez_compressed(
        out_npz, scene=scene_name,
        cell_ids_native=np.asarray(res.cell_ids),
        cell_data_native=np.asarray(res.cell_data),
        cell_ids_ref=cell_ids_ref, cell_data_ref=cell_data_ref,
    )
    return dict(np.load(out_npz, allow_pickle=True))


# =============================================================================
# Steps 4-5: degradation (real native data -> coarse UTM samples)
# =============================================================================

def _fwhm_to_gauss_sigma_px(fwhm_m, pixel_size_m=None):
    """Standard Gaussian FWHM -> sigma (pixels), for scipy.ndimage.gaussian_filter.

    NOTE: this is the ordinary sigma = FWHM / (2*sqrt(2 ln 2)) convention --
    NOT the healpix_resample package's own `s = FWHM / sqrt(2 ln 2)` scale
    (see healpix_resample.psf_geometry's module docstring). The two are
    unrelated conventions used in two different places here and must not be
    swapped.
    """
    if pixel_size_m is None:
        pixel_size_m = PIXEL_SIZE_M
    sigma_m = fwhm_m / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return sigma_m / pixel_size_m


def degrade_to_coarse(img_native, fwhm_m=None, block=None):
    """Gaussian blur (the only shaping applied), then point-sample
    (decimate) every `block`-th pixel -- deliberately NOT a block average."""
    if fwhm_m is None:
        fwhm_m = DEGRADE_FWHM_M
    if block is None:
        block = BLOCK
    sigma_px = _fwhm_to_gauss_sigma_px(fwhm_m)
    blurred = gaussian_filter(img_native, sigma=sigma_px, mode="reflect")
    return decimate_2d(blurred, block=block)


def build_coarse_grid(scene_name):
    """Steps 4-5: real native truth -> Gaussian blur -> decimate every
    BLOCK-th pixel, giving the synthetic coarse observation and its exact
    geolocation (also decimated, not averaged)."""
    img_n, lon_n, lat_n, da = load_scene_patch(scene_name)
    img_n = central_crop(img_n)
    lon_n = central_crop(lon_n)
    lat_n = central_crop(lat_n)

    img_c = degrade_to_coarse(img_n)
    lon_c = decimate_2d(lon_n)
    lat_c = decimate_2d(lat_n)

    x_n, y_n = get_utm_axes(da)
    x_c = decimate_1d(x_n)
    y_c = decimate_1d(y_n)

    return dict(img_native=img_n, x_native=x_n, y_native=y_n,
                img_coarse=img_c, lon_coarse=lon_c, lat_coarse=lat_c,
                x_coarse=x_c, y_coarse=y_c, da=da)


# =============================================================================
# Step 6: PSF-aware reconstruction at REF_LEVEL
# =============================================================================

def reconstruct_psf_aware(scene_name, force=True):
    """out_cell_ids=ref["cell_ids_ref"] restricts the operator's own
    retained cells to (a subset of) the reference's cells directly at
    construction time -- reconstruction and reference already live at the
    same HEALPix level, so no separate readout/interpolation is needed."""
    DATA_DIR.mkdir(exist_ok=True)
    out_npz = DATA_DIR / f"{scene_name}_real_downscale_psf_aware_{_cache_suffix()}.npz"
    if out_npz.exists() and not force:
        return dict(np.load(out_npz, allow_pickle=True))

    ref = build_reference(scene_name, force=force)
    g = build_coarse_grid(scene_name)

    scale_m = fwhm_to_scale(RECON_FWHM_M)
    npt = recommend_npt(scale_m, REF_LEVEL)["npt"]

    t0 = time.time()
    resampler = PSFResampler(
        lon_deg=g["lon_coarse"].reshape(-1), lat_deg=g["lat_coarse"].reshape(-1),
        level=REF_LEVEL, sigma_m=scale_m, Npt=npt,
        out_cell_ids=ref["cell_ids_ref"],
        threshold=THRESHOLD, device=DEVICE, verbose=False,
    )
    build_time = time.time() - t0

    t1 = time.time()
    res = resampler.resample(g["img_coarse"].reshape(-1), lam=LAMBDA_COARSE, max_iter=MAX_ITER)
    solve_time = time.time() - t1

    np.savez_compressed(
        out_npz, scene=scene_name,
        cell_ids=np.asarray(res.cell_ids), cell_data=np.asarray(res.cell_data),
        build_time=build_time, solve_time=solve_time,
    )
    return dict(np.load(out_npz, allow_pickle=True))


# =============================================================================
# Baselines, evaluated at the same reference cell centres
# =============================================================================

def _axes_ascending(y_ax, x_ax, img):
    """RegularGridInterpolator requires ascending axes; UTM y is usually
    descending (north-up raster). Flip whichever axis needs it."""
    if y_ax[0] > y_ax[-1]:
        y_ax, img = y_ax[::-1], img[::-1, :]
    if x_ax[0] > x_ax[-1]:
        x_ax, img = x_ax[::-1], img[:, ::-1]
    return y_ax, x_ax, img


def reconstruct_classical(scene_name, method="linear", force=False):
    """Classical baseline (nearest/bilinear/bicubic), read out exactly at
    the reference's own HEALPix cell centres (projected to UTM) -- the
    same evaluation points as every other method."""
    DATA_DIR.mkdir(exist_ok=True)
    out_npz = DATA_DIR / f"{scene_name}_real_downscale_classical_{method}_{_cache_suffix()}.npz"
    if out_npz.exists() and not force:
        return dict(np.load(out_npz, allow_pickle=True))

    ref = build_reference(scene_name, force=force)
    g = build_coarse_grid(scene_name)

    y_ax, x_ax, img_ax = _axes_ascending(g["y_coarse"], g["x_coarse"], g["img_coarse"])
    interp = RegularGridInterpolator((y_ax, x_ax), img_ax, method=method,
                                      bounds_error=False, fill_value=None)

    lon_ref, lat_ref = cell_centers_lonlat(ref["cell_ids_ref"], REF_LEVEL)
    x_ref, y_ref = lonlat_to_utm(lon_ref, lat_ref, g["da"])
    vals = interp(np.stack([y_ref, x_ref], axis=-1))

    np.savez_compressed(
        out_npz, scene=scene_name, method=method,
        cell_ids=ref["cell_ids_ref"], cell_data=vals.astype(np.float32),
    )
    return dict(np.load(out_npz, allow_pickle=True))


def reconstruct_richardson_lucy(scene_name, force=False):
    """Two-stage baseline: deconvolve the coarse observation with the
    assumed Gaussian response (RECON_FWHM_M), then read out at the
    reference's own HEALPix cell centres, same as the classical baselines."""
    DATA_DIR.mkdir(exist_ok=True)
    out_npz = DATA_DIR / f"{scene_name}_real_downscale_richardson_lucy_{_cache_suffix()}.npz"
    if out_npz.exists() and not force:
        return dict(np.load(out_npz, allow_pickle=True))
    if richardson_lucy is None:
        raise ImportError("scikit-image richardson_lucy is not available.")

    ref = build_reference(scene_name, force=force)
    g = build_coarse_grid(scene_name)
    img_c = g["img_coarse"]

    lo, hi = np.percentile(img_c, [0.5, 99.5])
    hi = max(hi, lo + 1e-6)
    img01 = np.clip((img_c - lo) / (hi - lo), 0, 1)

    sigma_px = _fwhm_to_gauss_sigma_px(RECON_FWHM_M, pixel_size_m=PIXEL_SIZE_COARSE_M)
    radius = max(1, int(np.ceil(4 * sigma_px)))
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma_px ** 2)
    psf /= psf.sum()

    deconv01 = richardson_lucy(img01, psf, num_iter=RICHARDSON_LUCY_ITER, clip=False)
    deconv_c = deconv01 * (hi - lo) + lo

    y_ax, x_ax, img_ax = _axes_ascending(g["y_coarse"], g["x_coarse"], deconv_c)
    interp = RegularGridInterpolator((y_ax, x_ax), img_ax, method="cubic",
                                      bounds_error=False, fill_value=None)

    lon_ref, lat_ref = cell_centers_lonlat(ref["cell_ids_ref"], REF_LEVEL)
    x_ref, y_ref = lonlat_to_utm(lon_ref, lat_ref, g["da"])
    vals = interp(np.stack([y_ref, x_ref], axis=-1))

    np.savez_compressed(
        out_npz, scene=scene_name,
        cell_ids=ref["cell_ids_ref"], cell_data=vals.astype(np.float32),
    )
    return dict(np.load(out_npz, allow_pickle=True))


# =============================================================================
# Step 7: alignment and metrics, entirely in HEALPix cell space
# =============================================================================

def align_and_mask(cell_ids, cell_data, ref, g):
    """Restrict `(cell_ids, cell_data)` to cells also present in the
    reference and inside the geometric interior margin. Returns
    (estimate, reference, cell_ids) arrays of equal length, ready for
    compute_metrics()."""
    ref_ids = np.asarray(ref["cell_ids_ref"])
    ref_vals = np.asarray(ref["cell_data_ref"])
    pos = {int(c): i for i, c in enumerate(ref_ids)}

    cell_ids = np.asarray(cell_ids)
    cell_data = np.asarray(cell_data)
    in_ref = np.array([int(c) in pos for c in cell_ids])
    cell_ids = cell_ids[in_ref]
    cell_data = cell_data[in_ref]
    ref_sel = np.array([pos[int(c)] for c in cell_ids], dtype=np.int64)
    ref_vals_sel = ref_vals[ref_sel]

    lon_c, lat_c = cell_centers_lonlat(cell_ids, REF_LEVEL)
    x_c, y_c = lonlat_to_utm(lon_c, lat_c, g["da"])
    interior = interior_mask(x_c, y_c, g["x_native"], g["y_native"])

    return cell_data[interior], ref_vals_sel[interior], cell_ids[interior]


def compute_metrics(estimate, reference, scene, method):
    e = np.asarray(estimate, dtype=np.float64)
    t = np.asarray(reference, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(e)
    tv, ev = t[mask], e[mask]
    err = ev - tv

    if tv.size == 0:
        return dict(scene=scene, method=method, n_cells=0,
                    rmse=np.nan, mae=np.nan, bias=np.nan, corr=np.nan, psnr=np.nan)

    rmse = float(np.sqrt(mean_squared_error(tv, ev)))
    mae = float(mean_absolute_error(tv, ev))
    bias = float(np.mean(err))
    corr = float(np.corrcoef(tv, ev)[0, 1]) if tv.size > 1 else np.nan
    dr = float(np.nanmax(t) - np.nanmin(t))
    psnr = (float("inf") if rmse == 0
            else float(20 * np.log10(dr / rmse)) if dr > 0 else np.nan)

    return dict(scene=scene, method=method, n_cells=int(tv.size),
                rmse=rmse, mae=mae, bias=bias, corr=corr, psnr=psnr)


def run_all(force=False):
    TABLE_DIR.mkdir(exist_ok=True)
    rows = []
    for scene in benchmark_coordinates:
        ref = build_reference(scene, force=force)
        g = build_coarse_grid(scene)

        try:
            d = reconstruct_psf_aware(scene, force=force)
            e, t, _ = align_and_mask(d["cell_ids"], d["cell_data"], ref, g)
            rows.append(compute_metrics(e, t, scene, "psf_aware"))
        except Exception as exc:
            print(f"[WARN] PSF-aware failed for {scene}: {exc}")

        for method in CLASSICAL_METHODS:
            try:
                d = reconstruct_classical(scene, method=method, force=force)
                e, t, _ = align_and_mask(d["cell_ids"], d["cell_data"], ref, g)
                rows.append(compute_metrics(e, t, scene, f"classical_{method}"))
            except Exception as exc:
                print(f"[WARN] classical {method} failed for {scene}: {exc}")

        try:
            d = reconstruct_richardson_lucy(scene, force=force)
            e, t, _ = align_and_mask(d["cell_ids"], d["cell_data"], ref, g)
            rows.append(compute_metrics(e, t, scene, "richardson_lucy"))
        except Exception as exc:
            print(f"[WARN] Richardson-Lucy failed for {scene}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "real_groundtruth_downscale_metrics.csv", index=False)
    return df


def format_table(df):
    methods = ["psf_aware"] + [f"classical_{m}" for m in CLASSICAL_METHODS] + ["richardson_lucy"]
    labels = {
        "psf_aware": "PSF-aware",
        "classical_nearest": "Nearest-neighbor",
        "classical_linear": "Bilinear",
        "classical_cubic": "Bicubic",
        "richardson_lucy": "Richardson-Lucy",
    }
    records = []
    for scene in benchmark_coordinates:
        row = {"Scene": scene}
        for m in methods:
            sub = df[(df["scene"] == scene) & (df["method"] == m)]
            if not sub.empty:
                row[f"{labels[m]} RMSE"] = sub.iloc[0]["rmse"]
                row[f"{labels[m]} Corr"] = sub.iloc[0]["corr"]
                row[f"{labels[m]} n"] = sub.iloc[0]["n_cells"]
            else:
                row[f"{labels[m]} RMSE"] = np.nan
                row[f"{labels[m]} Corr"] = np.nan
                row[f"{labels[m]} n"] = 0
        records.append(row)
    table = pd.DataFrame(records)
    TABLE_DIR.mkdir(exist_ok=True)
    table.to_csv(TABLE_DIR / "real_groundtruth_downscale_table.csv", index=False)
    return table


# =============================================================================
# Plotting -- FOR VISUAL INSPECTION ONLY. Nothing below feeds back into any
# metric computed above.
# =============================================================================

def project_for_display(cell_ids, cell_data, lon_grid, lat_grid, level=None):
    """Read a HEALPix cell-space field out at arbitrary lon/lat query
    points, FOR DISPLAY ONLY -- never used in any metric. Same
    out_cell_ids + BicubicResampler.invert() composition used elsewhere in
    this package to sample a reconstructed field at points that were never
    part of the operator's own construction.
    """
    if level is None:
        level = REF_LEVEL
    readout = BicubicResampler(
        lon_deg=lon_grid.reshape(-1), lat_deg=lat_grid.reshape(-1),
        level=level, out_cell_ids=cell_ids, Npt=16,
        threshold=THRESHOLD, device=DEVICE, verbose=False,
    )
    dest_ids = readout.get_cell_ids()
    pos = {int(c): i for i, c in enumerate(np.asarray(cell_ids))}
    sel = np.array([pos[int(c)] for c in dest_ids], dtype=np.int64)
    aligned = np.asarray(cell_data)[sel]
    out = readout.invert(aligned)
    return np.asarray(out).reshape(lon_grid.shape)


def interior_display_grid(scene_name):
    """A regular native-resolution grid restricted to the same interior
    margin (EDGE_MARGIN_M) used by every metric -- for qualitative,
    side-by-side plotting only."""
    img_n, lon_n, lat_n, da = load_scene_patch(scene_name)
    img_n = central_crop(img_n)
    lon_n = central_crop(lon_n)
    lat_n = central_crop(lat_n)
    x_n, y_n = get_utm_axes(da)

    xx, yy = np.meshgrid(x_n, y_n)
    mask = interior_mask(xx, yy, x_n, y_n)
    if not mask.any():
        raise RuntimeError("EDGE_MARGIN_M leaves no interior pixels for this patch size.")
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    return dict(img_native=img_n[r0:r1 + 1, c0:c1 + 1],
                lon=lon_n[r0:r1 + 1, c0:c1 + 1], lat=lat_n[r0:r1 + 1, c0:c1 + 1])


def plot_panels(force=False):
    scenes = list(benchmark_coordinates.keys())
    fig, axes = plt.subplots(len(scenes), 7, figsize=(2.6 * 7, 2.6 * len(scenes)))
    if len(scenes) == 1:
        axes = axes[np.newaxis, :]

    for r, scene in enumerate(scenes):
        disp = interior_display_grid(scene)
        truth = disp["img_native"]
        vmin, vmax = np.nanpercentile(truth, [1, 99])
        g = build_coarse_grid(scene)

        d_psf = reconstruct_psf_aware(scene, force=force)
        img_psf = project_for_display(d_psf["cell_ids"], d_psf["cell_data"], disp["lon"], disp["lat"])

        panels = [truth, g["img_coarse"], img_psf]
        col_labels = [
            f"Truth (native, {truth.shape[0]}x{truth.shape[1]} px)",
            f"Degraded ({PIXEL_SIZE_COARSE_M:.0f} m, {g['img_coarse'].shape[0]}x{g['img_coarse'].shape[1]} px)",
            f"PSF-aware (level {REF_LEVEL})",
        ]
        for method in CLASSICAL_METHODS:
            d_cl = reconstruct_classical(scene, method=method, force=force)
            panels.append(project_for_display(d_cl["cell_ids"], d_cl["cell_data"], disp["lon"], disp["lat"]))
            col_labels.append(CLASSICAL_LABELS[method])
        d_rl = reconstruct_richardson_lucy(scene, force=force)
        panels.append(project_for_display(d_rl["cell_ids"], d_rl["cell_data"], disp["lon"], disp["lat"]))
        col_labels.append("Richardson-Lucy")

        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_labels[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(scene, fontsize=10)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "real_groundtruth_downscale_panels.pdf", dpi=200)
    return fig


def plot_scatter(force=False):
    """Predicted vs. reference, per method -- the direct visualization of
    what compute_metrics() measures. Every point is one HEALPix cell."""
    methods = ["psf_aware"] + [f"classical_{m}" for m in CLASSICAL_METHODS] + ["richardson_lucy"]
    labels = {
        "psf_aware": "PSF-aware", "classical_nearest": "Nearest",
        "classical_linear": "Bilinear", "classical_cubic": "Bicubic",
        "richardson_lucy": "Richardson-Lucy",
    }
    scenes = list(benchmark_coordinates.keys())
    fig, axes = plt.subplots(len(scenes), len(methods), figsize=(2.6 * len(methods), 2.6 * len(scenes)))
    if len(scenes) == 1:
        axes = axes[np.newaxis, :]

    for r, scene in enumerate(scenes):
        ref = build_reference(scene, force=force)
        g = build_coarse_grid(scene)
        getters = {
            "psf_aware": lambda: reconstruct_psf_aware(scene, force=force),
            "classical_nearest": lambda: reconstruct_classical(scene, "nearest", force=force),
            "classical_linear": lambda: reconstruct_classical(scene, "linear", force=force),
            "classical_cubic": lambda: reconstruct_classical(scene, "cubic", force=force),
            "richardson_lucy": lambda: reconstruct_richardson_lucy(scene, force=force),
        }
        for c, m in enumerate(methods):
            d = getters[m]()
            e, t, _ = align_and_mask(d["cell_ids"], d["cell_data"], ref, g)
            ax = axes[r, c]
            ax.scatter(t, e, s=2, alpha=0.15, color="tab:blue", rasterized=True)
            lo = min(np.nanmin(t), np.nanmin(e)) if t.size else 0.0
            hi = max(np.nanmax(t), np.nanmax(e)) if t.size else 1.0
            ax.plot([lo, hi], [lo, hi], color="black", lw=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            if r == 0:
                ax.set_title(labels[m], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{scene}\nestimate", fontsize=9)
            if r == len(scenes) - 1:
                ax.set_xlabel("reference (step 3)", fontsize=9)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "real_groundtruth_downscale_scatter.pdf", dpi=200)
    return fig


def plot_rmse_summary(df):
    labels = {
        "psf_aware": "PSF-aware",
        "classical_nearest": "Nearest",
        "classical_linear": "Bilinear",
        "classical_cubic": "Bicubic",
        "richardson_lucy": "Richardson-Lucy",
    }
    methods = list(labels.keys())
    scenes = list(benchmark_coordinates.keys())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.15
    x = np.arange(len(scenes))
    for i, m in enumerate(methods):
        vals = [df[(df["scene"] == s) & (df["method"] == m)]["rmse"].mean() for s in scenes]
        ax.bar(x + (i - len(methods) / 2) * width, vals, width, label=labels[m])
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylabel(f"RMSE vs. reference (HEALPix level {REF_LEVEL})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "real_groundtruth_downscale_rmse_summary.pdf", dpi=200)
    return fig
