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

Two control experiments address known asymmetries of this protocol; both are
driven by `run_variant()` and compared with `rank_table()`:

  * `run_variant(reference_mode="geometric")` rebuilds the reference by direct
    unweighted aggregation of the native 10 m pixels onto REF_LEVEL cells, with
    no spatial-response model at any stage. Steps 2-3 otherwise use the same
    PSF-aware operator family as the method under test in step 6, so a
    systematic signature of that operator would appear in both the reference
    and the PSF-aware estimate but not in the baselines. Absolute RMSE is not
    comparable between the two reference modes (the geometric reference targets
    the observed rather than the latent field) -- compare the *ranking*.

  * `run_variant(baseline_estimand="point_sample")` reproduces the earlier
    behaviour, where raster baselines were read out as a single sample at each
    cell centre while the reference is a child-cell average. The default,
    "cell_average", puts both sides on the same estimand.
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

# Publication default: inputs must come from the versioned DOI archive.
# Set to False only when intentionally constructing that frozen input bundle.
OFFLINE = True

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

# Shape of the *imposed* degradation.  The publication configuration remains
# isotropic.  The anisotropic values define a deliberately severe shape-
# mismatch control: the current public PSFResampler interface still accepts a
# single isotropic Gaussian width, so the reconstruction below remains
# isotropic even when the simulated observation is blurred by 24 m x 45 m.
DEGRADE_PSF_MODE = "isotropic"
ANISOTROPIC_DEGRADE_FWHM_X_M = 24.0
ANISOTROPIC_DEGRADE_FWHM_Y_M = 45.0

# -- Step 6: reconstruction under test (20 m samples -> REF_LEVEL) -------------
# Assumed effective response for the reconstruction. Set equal to
# DEGRADE_FWHM_M: this is the *matched* configuration, the primary case used
# throughout the paper. Set it to a different value to run a PSF-mismatch
# variant of this experiment (the cache suffix already encodes it, so results
# will not be silently reused).
RECON_FWHM_M = DEGRADE_FWHM_M
LAMBDA_COARSE = 0.01
# Matches the paper's stated solver configuration ("at most 14 iterations"),
# which the paper also relies on as an implicit iterative regularization.
# Earlier runs of this notebook used 100, i.e. a *different* regularization
# regime from every other experiment in the paper; keep this at 14 unless you
# deliberately want to characterize that difference.
MAX_ITER      = 14
THRESHOLD     = 0.1
RICHARDSON_LUCY_ITER = 100

# -- Evaluation: geometric edge margin (metres), not a pixel crop --------------
EDGE_MARGIN_FACTOR = 8.0
EDGE_MARGIN_M = EDGE_MARGIN_FACTOR * RECON_FWHM_M

# How the raster-based baselines are read out at each REF_LEVEL cell:
#   "cell_average" -- mean of the 4**(NATIVE_LEVEL-REF_LEVEL) child-cell
#                     centres, i.e. the SAME estimand as the reference, which
#                     is itself a child-cell average (healpix_down). This is
#                     the estimand the paper uses for every method in the
#                     synthetic experiment.
#   "point_sample" -- a single sample at the parent cell centre. Cheaper, but
#                     a different estimand from the reference (the paper
#                     measures a 2.3-2.5x RMSE difference between the two in
#                     the synthetic setting).
# Run both and compare: for smooth interpolators the difference is small here,
# because a field interpolated from a 20 m raster varies slowly across a
# 12.44 m cell, but "cell_average" is the symmetric choice.
BASELINE_ESTIMAND = "cell_average"

# Reference construction mode (step 2-3):
#   "psf_aware" -- PSFResampler at NATIVE_LEVEL, then healpix_down. Uses the
#                  paper's own calibrated 10 m response to estimate the latent
#                  field, but shares its estimator family with the PSF-aware
#                  method under test.
#   "geometric" -- direct, unweighted average of the native 10 m pixels whose
#                  centres fall in each REF_LEVEL cell. NO spatial-response
#                  model at any stage, so it shares nothing with any method
#                  under test. Control experiment for the shared-estimator
#                  concern; note it targets the *observed* (still 12.5 m
#                  PSF-blurred) field rather than the latent one, so absolute
#                  RMSE values are not comparable to the "psf_aware" mode --
#                  only the *ranking* of methods is.
REFERENCE_MODE = "psf_aware"

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
    fwhm_x_m, fwhm_y_m = degradation_fwhm_xy()
    print(f"DEGRADE_PSF_MODE={DEGRADE_PSF_MODE!r}: "
          f"FWHM_x={fwhm_x_m:.1f} m, FWHM_y={fwhm_y_m:.1f} m; "
          f"isotropic reconstruction FWHM={RECON_FWHM_M:.1f} m")
    print(f"EDGE_MARGIN_M={EDGE_MARGIN_M:.0f} m excluded from every metric")
    print(f"REFERENCE_MODE={REFERENCE_MODE!r}, BASELINE_ESTIMAND={BASELINE_ESTIMAND!r}, "
          f"MAX_ITER={MAX_ITER}")


def _cache_suffix():
    """Every parameter that changes a *result* (not just plotting) is baked
    into intermediate .npz cache filenames, so changing the configuration
    always recomputes rather than silently reusing a stale result from a
    different configuration."""
    degradation_tag = ""
    if DEGRADE_PSF_MODE == "anisotropic":
        degradation_tag = (
            f"_deganiso{ANISOTROPIC_DEGRADE_FWHM_X_M:g}x"
            f"{ANISOTROPIC_DEGRADE_FWHM_Y_M:g}"
        )
    return (f"native{NATIVE_LEVEL}_ref{REF_LEVEL}_block{BLOCK}"
            f"_fwhm{int(round(RECON_FWHM_M))}_it{MAX_ITER}"
            f"_ref{REFERENCE_MODE}_est{BASELINE_ESTIMAND}{degradation_tag}")


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
    path = DATA_DIR / f"{scene}_data.zarr"
    # OFFLINE takes precedence over force: never refresh a frozen input.
    if path.exists() and (OFFLINE or not force):
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
    if OFFLINE:
        raise FileNotFoundError(
            f"Missing frozen Sentinel-2 input: {path}. Download the DOI data "
            "archive and place its data/ directory under notebooks/."
        )
    catalog = pystac_client.Client.open("https://stac.core.eopf.eodc.eu")
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

def build_reference_geometric(img_n, lon_n, lat_n):
    """Control reference (REFERENCE_MODE='geometric'): assign every native
    10 m pixel to the REF_LEVEL cell containing its centre and take the plain
    unweighted mean per cell.

    No spatial-response model, no inversion, no PSFResampler -- so this
    reference shares nothing with any method under test, which removes the
    shared-estimator concern affecting the 'psf_aware' mode.

    Caveat, and the reason this is a control rather than the default: it
    estimates the *observed* field (still carrying the sensor's own ~12.5 m
    response) averaged over each cell, not the latent field. Every method is
    then scored against a blurrier target, so absolute RMSE values are not
    comparable with the 'psf_aware' mode. What IS comparable, and what this
    control is for, is the *ranking* of the methods.

    Cells receiving fewer than MIN_CHILDREN_FRAC of the pixels they would get
    under uniform coverage are dropped, mirroring healpix_down()'s pruning.
    """
    ids = healpix_geo.nested.lonlat_to_healpix(
        np.asarray(lon_n, dtype=np.float64).reshape(-1),
        np.asarray(lat_n, dtype=np.float64).reshape(-1),
        REF_LEVEL, ellipsoid=ELLIPSOID,
    )
    ids = np.asarray(ids, dtype=np.int64)
    vals = np.asarray(img_n, dtype=np.float64).reshape(-1)

    uniq, inv = np.unique(ids, return_inverse=True)
    sums = np.bincount(inv, weights=vals, minlength=uniq.size)
    counts = np.bincount(inv, minlength=uniq.size)
    means = sums / np.maximum(counts, 1)

    # Expected pixels per cell under uniform coverage: (cell area)/(pixel area).
    expected = max(1.0, (cell_size_m(REF_LEVEL) / PIXEL_SIZE_M) ** 2)
    keep = counts >= (MIN_CHILDREN_FRAC * expected)
    return uniq[keep].astype(np.int64), means[keep].astype(np.float32)


def build_reference(scene_name, force=False):
    """Steps 1-3: real native 10 m patch -> reference field at REF_LEVEL.

    REFERENCE_MODE='psf_aware' (default): PSF-aware HEALPix field at
    NATIVE_LEVEL (the best available estimate of the latent field, built from
    real, undegraded data using the paper's own calibrated 10 m response),
    then a plain (no-PSF) NESTED downgrade to REF_LEVEL.

    REFERENCE_MODE='geometric': direct pixel aggregation onto REF_LEVEL, with
    no spatial-response model anywhere -- see build_reference_geometric().
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

    if REFERENCE_MODE == "geometric":
        cell_ids_ref, cell_data_ref = build_reference_geometric(img_n, lon_n, lat_n)
        np.savez_compressed(
            out_npz, scene=scene_name,
            cell_ids_native=np.asarray([], dtype=np.int64),
            cell_data_native=np.asarray([], dtype=np.float32),
            cell_ids_ref=cell_ids_ref, cell_data_ref=cell_data_ref,
        )
        return dict(np.load(out_npz, allow_pickle=True))
    if REFERENCE_MODE != "psf_aware":
        raise ValueError(f"Unknown REFERENCE_MODE {REFERENCE_MODE!r}")

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


def degradation_fwhm_xy():
    """Return imposed degradation widths along projected x/east and y/north.

    The anisotropic branch changes only the synthetic 10 m -> 20 m
    degradation.  Reconstruction intentionally keeps the current isotropic
    public operator, making this a PSF-shape-mismatch test rather than a claim
    that arbitrary kernels are already exposed by the package API.
    """
    if DEGRADE_PSF_MODE == "isotropic":
        return float(DEGRADE_FWHM_M), float(DEGRADE_FWHM_M)
    if DEGRADE_PSF_MODE == "anisotropic":
        return (
            float(ANISOTROPIC_DEGRADE_FWHM_X_M),
            float(ANISOTROPIC_DEGRADE_FWHM_Y_M),
        )
    raise ValueError(f"Unknown DEGRADE_PSF_MODE={DEGRADE_PSF_MODE!r}")


def degrade_to_coarse(img_native, fwhm_m=None, block=None,
                      fwhm_x_m=None, fwhm_y_m=None):
    """Gaussian blur, then point-sample every `block`-th pixel.

    With the default isotropic mode this reproduces the publication protocol.
    With ``DEGRADE_PSF_MODE='anisotropic'`` (or explicit x/y widths), SciPy
    receives ``sigma=(sigma_y, sigma_x)`` in array-axis order.  No block
    average is applied, so this Gaussian is the only imposed response shape.
    """
    if fwhm_x_m is None or fwhm_y_m is None:
        if fwhm_m is not None:
            fwhm_x_m = fwhm_y_m = fwhm_m
        else:
            fwhm_x_m, fwhm_y_m = degradation_fwhm_xy()
    if block is None:
        block = BLOCK
    sigma_x_px = _fwhm_to_gauss_sigma_px(fwhm_x_m)
    sigma_y_px = _fwhm_to_gauss_sigma_px(fwhm_y_m)
    blurred = gaussian_filter(
        img_native, sigma=(sigma_y_px, sigma_x_px), mode="reflect"
    )
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

def _readout_points(cell_ids):
    """Query points at which a raster baseline is sampled to produce a value
    for each REF_LEVEL cell, plus the grouping needed to reduce them.

    Returns (lon, lat, n_per_cell). With BASELINE_ESTIMAND='point_sample'
    there is one query point per cell (the cell centre). With
    'cell_average' there is one per NATIVE_LEVEL child cell, and the caller
    averages each consecutive block of n_per_cell samples -- giving the same
    child-cell-average estimand as the reference itself, instead of a single
    point sample at a different estimand.
    """
    ids = np.asarray(cell_ids, dtype=np.int64)
    if BASELINE_ESTIMAND == "point_sample":
        lon, lat = cell_centers_lonlat(ids, REF_LEVEL)
        return lon, lat, 1
    if BASELINE_ESTIMAND != "cell_average":
        raise ValueError(f"Unknown BASELINE_ESTIMAND {BASELINE_ESTIMAND!r}")

    dl = NATIVE_LEVEL - REF_LEVEL
    if dl <= 0:
        raise ValueError("NATIVE_LEVEL must exceed REF_LEVEL for cell averaging")
    factor = 4 ** dl
    # NESTED indexing: the children of cell c at depth dl are exactly
    # [c*factor, (c+1)*factor) -- the same relation healpix_down() inverts.
    children = (ids[:, None] * factor + np.arange(factor)[None, :]).reshape(-1)
    lon, lat = cell_centers_lonlat(children, NATIVE_LEVEL)
    return lon, lat, factor


def _reduce_readout(vals, n_per_cell):
    """Average consecutive blocks of n_per_cell samples back to one value
    per REF_LEVEL cell (a no-op when n_per_cell == 1)."""
    vals = np.asarray(vals, dtype=np.float64)
    if n_per_cell == 1:
        return vals
    return vals.reshape(-1, n_per_cell).mean(axis=1)


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

    lon_q, lat_q, n_per_cell = _readout_points(ref["cell_ids_ref"])
    x_q, y_q = lonlat_to_utm(lon_q, lat_q, g["da"])
    vals = _reduce_readout(interp(np.stack([y_q, x_q], axis=-1)), n_per_cell)

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

    lon_q, lat_q, n_per_cell = _readout_points(ref["cell_ids_ref"])
    x_q, y_q = lonlat_to_utm(lon_q, lat_q, g["da"])
    vals = _reduce_readout(interp(np.stack([y_q, x_q], axis=-1)), n_per_cell)

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


def run_all(force=False, csv_name=None):
    """Run every method on every scene and return the metrics DataFrame.

    `csv_name` overrides the output filename. run_variant() passes a
    variant-specific name so that a control run never overwrites the main
    result file -- the default name is the one the paper's table is built
    from.
    """
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
    df.to_csv(TABLE_DIR / (csv_name or "real_groundtruth_downscale_metrics.csv"),
              index=False)
    return df


def run_variant(reference_mode=None, baseline_estimand=None, max_iter=None,
                 degrade_psf_mode=None, degrade_fwhm_x_m=None,
                 degrade_fwhm_y_m=None, force=False, label=None):
    """Run the whole protocol under a temporarily overridden configuration and
    return its metrics DataFrame, with the configuration recorded in extra
    columns. Restores the previous configuration afterwards, including on
    error, so it is safe to call several times in a row from a notebook.

    Used for the control experiments discussed in the paper's limitations:
    `reference_mode='geometric'` removes the shared-estimator component
    between the reference and the PSF-aware method under test, and
    `baseline_estimand='point_sample'` reproduces the earlier, asymmetric
    baseline readout. ``degrade_psf_mode='anisotropic'`` imposes a 24 m x
    45 m Gaussian degradation while retaining the scalar isotropic
    reconstruction width.  This last variant probes response-shape mismatch;
    it does not imply that the public API already accepts an arbitrary kernel.
    Cache filenames encode every result-changing setting, so variants never
    collide with each other or with the main run.
    """
    global REFERENCE_MODE, BASELINE_ESTIMAND, MAX_ITER
    global DEGRADE_PSF_MODE, ANISOTROPIC_DEGRADE_FWHM_X_M
    global ANISOTROPIC_DEGRADE_FWHM_Y_M, EDGE_MARGIN_M
    saved = (
        REFERENCE_MODE, BASELINE_ESTIMAND, MAX_ITER, DEGRADE_PSF_MODE,
        ANISOTROPIC_DEGRADE_FWHM_X_M, ANISOTROPIC_DEGRADE_FWHM_Y_M,
        EDGE_MARGIN_M,
    )
    try:
        if reference_mode is not None:
            REFERENCE_MODE = reference_mode
        if baseline_estimand is not None:
            BASELINE_ESTIMAND = baseline_estimand
        if max_iter is not None:
            MAX_ITER = max_iter
        if degrade_psf_mode is not None:
            DEGRADE_PSF_MODE = degrade_psf_mode
        if degrade_fwhm_x_m is not None:
            ANISOTROPIC_DEGRADE_FWHM_X_M = float(degrade_fwhm_x_m)
        if degrade_fwhm_y_m is not None:
            ANISOTROPIC_DEGRADE_FWHM_Y_M = float(degrade_fwhm_y_m)
        degradation_x_m, degradation_y_m = degradation_fwhm_xy()
        EDGE_MARGIN_M = EDGE_MARGIN_FACTOR * max(
            RECON_FWHM_M, degradation_x_m, degradation_y_m
        )
        # Never write over the main metrics CSV: that file is the one the
        # paper's table is generated from, and a control run must not be able
        # to silently replace it.
        df = run_all(
            force=force,
            csv_name=f"real_groundtruth_downscale_metrics_{_cache_suffix()}.csv",
        )
        df["reference_mode"] = REFERENCE_MODE
        df["baseline_estimand"] = BASELINE_ESTIMAND
        df["max_iter"] = MAX_ITER
        df["degrade_psf_mode"] = DEGRADE_PSF_MODE
        df["degrade_fwhm_x_m"] = degradation_x_m
        df["degrade_fwhm_y_m"] = degradation_y_m
        df["recon_fwhm_m"] = RECON_FWHM_M
        df["edge_margin_m"] = EDGE_MARGIN_M
        df["variant"] = label or f"{REFERENCE_MODE}/{BASELINE_ESTIMAND}/it{MAX_ITER}"
        return df
    finally:
        (
            REFERENCE_MODE, BASELINE_ESTIMAND, MAX_ITER, DEGRADE_PSF_MODE,
            ANISOTROPIC_DEGRADE_FWHM_X_M,
            ANISOTROPIC_DEGRADE_FWHM_Y_M, EDGE_MARGIN_M,
        ) = saved


def run_anisotropic_psf_mismatch(force=False, fwhm_x_m=24.0, fwhm_y_m=45.0):
    """Run the deliberately non-matched 24 m x 45 m degradation control.

    The observation is blurred anisotropically in projected UTM axes, whereas
    PSF-aware reconstruction and Richardson--Lucy retain the current scalar
    isotropic response.  Use this to quantify robustness to a response shape
    not implemented by the released interface, not as an arbitrary-PSF test.
    """
    return run_variant(
        degrade_psf_mode="anisotropic",
        degrade_fwhm_x_m=fwhm_x_m,
        degrade_fwhm_y_m=fwhm_y_m,
        force=force,
        label=f"anisotropic degradation {fwhm_x_m:g}m x {fwhm_y_m:g}m",
    )


def rank_table(df):
    """Rank of every method by RMSE within each scene (1 = best).

    This is the output that matters for the `reference_mode='geometric'`
    control: that reference targets the observed rather than the latent
    field, so its absolute RMSE values are not comparable with the default
    configuration -- but the ranking is. If PSF-aware stays first in every
    scene under both references, the shared-estimator concern is addressed.
    """
    out = df.copy()
    out["rank"] = out.groupby("scene")["rmse"].rank(method="min").astype(int)
    return out.pivot_table(index="method", columns="scene", values="rank")


def format_table(df, csv_name="real_groundtruth_downscale_table.csv"):
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
    table.to_csv(TABLE_DIR / csv_name, index=False)
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
