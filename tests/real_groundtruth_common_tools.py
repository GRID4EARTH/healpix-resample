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
import hashlib
import shutil
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


# Additional scenes registered at run time by load_region_sites(): the 40
# geographically distributed regions of the multi-region synthetic validation,
# reused here so the real-data protocol can be run on the same site list.
# Kept separate from benchmark_coordinates so that the four paper scenes stay
# the default everywhere and nothing silently changes.
region_coordinates = {}

# The 40 region patches are real Sentinel-2 acquisitions, like the four
# benchmark scenes -- not the Esri textures used by the synthetic
# multi-region validation. They live in their own subdirectory of the frozen
# input bundle so that `data/` keeps one store per paper scene at top level:
#
#   data/multi_patch_sentinel2/region__<class>__<region_id>_data.zarr
#
# They are part of the archived input set and are covered by
# notebooks/build_data_manifest.py, so a normal run reads them offline exactly
# like every other primary input.
REGION_DATA_SUBDIR = "multi_patch_sentinel2"

# Acquisition window and cloud ceiling used when searching Sentinel-2 products
# for a region that has no pinned product_id. Only consulted by the one-off
# acquisition step that builds the bundle, never by an offline run.
REGION_DATE_WINDOW = "2023-01-01/2026-09-30"
REGION_CLOUD_MAX = 10

# Cloud ceilings tried in order, per region. The strict value is attempted at
# every lattice position first; only a region with no product at all below it
# escalates. Persistently cloudy sites (Borneo, equatorial forest) otherwise
# drop out entirely, which biases the sample towards dry regions -- a worse
# problem than admitting a cloudier scene, because the patch-level quality
# screen still has to pass either way.
#
# Regions acquired at a relaxed ceiling are recorded in the products CSV
# (`cloud_max_used`) so the paper can state how many needed it.
REGION_CLOUD_RELAXED = 25
REGION_CLOUD_LADDER = [REGION_CLOUD_MAX, REGION_CLOUD_RELAXED]


# =============================================================================
# Sentinel-2 source for the 40 multi-region sites
#
# The four paper scenes come from the EOPF STAC, a European demonstration
# catalogue. Queried at the 40 multi-region anchors it returns nothing outside
# Europe and part of North America -- Amazon, Borneo, Congo, Punjab, Pampas,
# Tokyo, Singapore and most of the southern hemisphere all yield "No items"
# even with a four-year window and a 40% cloud ceiling. That is a catalogue
# coverage limit, not a weather one, and no filter relaxation can fix it.
#
# Element 84's earth-search holds the complete Sentinel-2 L2A archive as
# Cloud-Optimized GeoTIFFs, so the multi-region experiment reads from there
# instead and can use exactly the same 40 sites as the synthetic validation.
# =============================================================================

REGION_STAC_SOURCE = "earth-search"        # or "eopf" for the original path
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
EARTH_SEARCH_COLLECTION = "sentinel-2-l2a"
EARTH_SEARCH_ASSET = "red"                 # B04, 10 m
# Public bucket hosting the COGs; only used to rewrite s3:// hrefs to HTTPS.
AWS_PUBLIC_REGION = "us-west-2"
S2_REFLECTANCE_SCALE = 1.0 / 10000.0
S2_NODATA_DN = 0
# Processing baseline 04.00 (2022-01-25) added a +1000 DN offset to L2A. Some
# STAC items record whether it has already been removed; when they do not, the
# baseline decides. Ignoring this silently biases reflectance by 0.1, which is
# comparable to the signal itself over water.
S2_BOA_OFFSET_DN = -1000.0
S2_OFFSET_BASELINE = "04.00"


def _boa_offset_dn(properties):
    """DN offset to apply to a Sentinel-2 L2A COG, given its STAC properties."""
    applied = properties.get("earthsearch:boa_offset_applied")
    if applied is True:
        return 0.0
    baseline = str(properties.get("s2:processing_baseline", "") or "")
    if applied is False:
        return S2_BOA_OFFSET_DN
    # Property absent: fall back to the baseline string, which sorts correctly
    # as zero-padded "MM.mm".
    if baseline and baseline >= S2_OFFSET_BASELINE:
        return S2_BOA_OFFSET_DN
    return 0.0


def _search_earth_search(lon0, lat0, coords, limit=200):
    """Cloud-sorted Sentinel-2 L2A items covering a point.

    Sorting by cloud cover then datetime then id makes the choice deterministic
    and dependent only on the data, so re-running picks the same product and
    the selection cannot drift as new acquisitions are published.
    """
    catalog = pystac_client.Client.open(EARTH_SEARCH_URL)
    search = catalog.search(
        collections=[EARTH_SEARCH_COLLECTION],
        intersects={"type": "Point", "coordinates": [float(lon0), float(lat0)]},
        datetime=coords["recommended_date"],
        query={"eo:cloud_cover": {"lt": coords["cloud"]}},
        limit=limit,
    )
    items = list(search.items())
    items.sort(key=lambda it: (it.properties.get("eo:cloud_cover", 100.0),
                               str(it.properties.get("datetime", "")), it.id))
    return items


def _public_https_href(href):
    """Rewrite an ``s3://bucket/key`` href to its public HTTPS equivalent.

    The Sentinel-2 COGs live in a public bucket, but an ``s3://`` href sends
    GDAL through /vsis3/, which signs the request with whatever AWS
    credentials it finds in the environment. Stale or unrelated keys then
    produce ``The AWS Access Key Id you provided does not exist in our
    records`` even though no credentials are needed at all. Plain HTTPS avoids
    signing entirely.
    """
    if not str(href).startswith("s3://"):
        return href
    bucket, _, key = str(href)[len("s3://"):].partition("/")
    return f"https://{bucket}.s3.{AWS_PUBLIC_REGION}.amazonaws.com/{key}"


def _rasterio_public_env():
    """GDAL settings for anonymous reads of a public bucket.

    ``AWS_NO_SIGN_REQUEST`` is the one that matters: it stops GDAL using
    ambient credentials. The other two just avoid pointless directory listings
    and HEAD requests over HTTP, which make a 256x256 window read noticeably
    slower.
    """
    import rasterio
    return rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
    )


def warn_about_aws_credentials():
    """Flag ambient AWS credentials, which break reads of a public bucket."""
    import os
    present = [k for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                            "AWS_SESSION_TOKEN", "AWS_PROFILE")
               if os.environ.get(k)]
    if present:
        print(f"[earth-search] NOTE: {', '.join(present)} set in the "
              f"environment. The Sentinel-2 COGs are public and are read "
              f"unsigned, so these are ignored here; if you still see "
              f"'AWS Access Key Id ... does not exist', unset them.")
    return present


def _read_cog_patch(href, lon0, lat0, patch_size, properties):
    """Read a patch_size x patch_size window centred on (lon0, lat0) from a COG.

    Returns (reflectance, x, y, crs). No-data is returned as NaN rather than 0
    so a patch that falls partly outside the granule is caught by the quality
    screen instead of entering the experiment as very dark ground.
    """
    import rasterio
    from rasterio.windows import Window

    href = _public_https_href(href)
    with _rasterio_public_env():
        with rasterio.open(href) as src:
            to_src = pyproj.Transformer.from_crs(
                pyproj.CRS.from_epsg(4326), src.crs, always_xy=True
            )
            cx, cy = to_src.transform(float(lon0), float(lat0))
            row, col = src.index(cx, cy)
            half = patch_size // 2
            window = Window(col - half, row - half, patch_size, patch_size)
            dn = src.read(1, window=window, boundless=True,
                          fill_value=S2_NODATA_DN).astype(np.float64)
            transform = src.window_transform(window)
            crs = src.crs

    dn[dn == S2_NODATA_DN] = np.nan
    refl = (dn + _boa_offset_dn(properties)) * S2_REFLECTANCE_SCALE

    cols = np.arange(patch_size)
    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (cols + 0.5) * transform.e   # transform.e is negative
    return refl.astype(np.float32), x, y, crs


def _fetch_patch_earth_search(scene, coords, patch_size, band):
    """Build the frozen-bundle Zarr for one scene from an earth-search COG."""
    lat0, lon0 = coords["wgs84"]["lat"], coords["wgs84"]["lon"]
    items = _search_earth_search(lon0, lat0, coords)
    if not items:
        raise RuntimeError(
            f"No earth-search items for {scene} in {coords['recommended_date']} "
            f"below {coords['cloud']}% cloud"
        )
    item = items[0]
    props = item.properties
    if EARTH_SEARCH_ASSET not in item.assets:
        raise KeyError(f"asset {EARTH_SEARCH_ASSET!r} missing from {item.id}")

    print(f"[{scene}] {len(items)} product(s); using {item.id} "
          f"({props.get('datetime', '?')}, cloud "
          f"{props.get('eo:cloud_cover', '?')}%, baseline "
          f"{props.get('s2:processing_baseline', '?')})")

    refl, x, y, crs = _read_cog_patch(
        item.assets[EARTH_SEARCH_ASSET].href, lon0, lat0, patch_size, props
    )

    to_wgs = pyproj.Transformer.from_crs(
        crs, pyproj.CRS.from_epsg(4326), always_xy=True
    )
    xx, yy = np.meshgrid(x, y)
    lon, lat = to_wgs.transform(xx, yy)

    ds = xr.Dataset(
        {band: (("y", "x"), refl)},
        coords={"x": x, "y": y,
                "longitude": (("y", "x"), lon),
                "latitude": (("y", "x"), lat),
                "spatial_ref": ((), 0)},
    )
    ds.spatial_ref.attrs["crs_wkt"] = crs.to_wkt()
    ds.attrs.update({
        "source_item_id": item.id,
        "source_collection": EARTH_SEARCH_COLLECTION,
        "source_stac_endpoint": EARTH_SEARCH_URL,
        "source_datetime": str(props.get("datetime", "")),
        "source_cloud_cover": str(props.get("eo:cloud_cover", "")),
        "source_processing_baseline": str(props.get("s2:processing_baseline", "")),
        "boa_offset_dn_applied": str(_boa_offset_dn(props)),
    })
    if scene in region_coordinates:
        region_coordinates[scene]["product_id"] = item.id
    return ds


def check_eopf_engine(raise_on_missing=False):
    """Is xarray's ``eopf-zarr`` backend registered?

    Reading Sentinel-2 from the EOPF STAC needs ``engine="eopf-zarr"``, which
    comes from the `xarray-eopf` package. Two distinct failures look identical
    from the notebook:

    * the package is simply absent (wrong environment), or
    * it is installed but its backend never registered -- the classic cause is
      a PyPI `pyproj` wheel shadowing conda's `libproj`, which breaks PROJ's
      database lookup and silently unregisters the entry point. `pixi.toml`
      carries this warning next to the dependency for exactly that reason.

    Both surface as ``ValueError: unrecognized engine 'eopf-zarr'`` only once
    a download is attempted, i.e. after a STAC query per scene. Call this
    first instead.
    """
    try:
        from xarray.backends import list_engines
        engines = sorted(list_engines())
    except Exception as exc:                      # pragma: no cover
        engines = []
        print(f"[eopf-zarr] could not list xarray engines: {exc}")

    ok = "eopf-zarr" in engines
    if ok:
        print("[eopf-zarr] backend registered.")
        return True

    try:
        import xarray_eopf  # noqa: F401
        installed = True
    except Exception:
        installed = False

    msg = [
        "The xarray 'eopf-zarr' backend is NOT registered, so Sentinel-2 "
        "products cannot be opened from the EOPF STAC.",
        f"  engines available: {engines}",
    ]
    if installed:
        msg += [
            "  xarray_eopf IS importable, so this is a registration failure,",
            "  not a missing package. The usual cause is a PyPI pyproj wheel",
            "  shadowing conda's libproj (see the comment in pixi.toml):",
            "    pixi run -e notebooks python -c \"import pyproj; print(pyproj.__file__)\"",
            "  It must resolve inside the pixi environment, not site-packages.",
        ]
    else:
        msg += [
            "  xarray_eopf is not importable: you are in the wrong environment.",
            "  Run this notebook with:  pixi run -e notebooks jupyter lab",
        ]
    text = "\n".join(msg)
    if raise_on_missing:
        raise RuntimeError(text)
    print(text)
    return False


def scene_zarr_path(scene):
    """Location of a scene's frozen Sentinel-2 patch.

    The four paper scenes sit directly in `data/`; the 40 multi-region scenes
    sit in `data/<REGION_DATA_SUBDIR>/`, keeping the top level readable.
    """
    if scene in region_coordinates or scene.startswith("region__"):
        return DATA_DIR / REGION_DATA_SUBDIR / f"{scene}_data.zarr"
    return DATA_DIR / f"{scene}_data.zarr"


_PATCH_FINGERPRINT_CACHE = {}


def patch_fingerprint(scene, refresh=False):
    """Short hash identifying the patch a derived cache was computed FROM.

    `_cache_suffix()` already bakes every *configuration* parameter into the
    intermediate filenames, so changing a constant recomputes instead of
    silently reusing a foreign result. The patch itself was the one input not
    covered: caches were keyed by scene name alone, so re-acquiring a region
    -- at a fallback lattice position, or from a different catalogue -- left
    the previous reference, degraded grid and per-method estimates in place,
    still keyed by the same name and still perfectly loadable.

    That is not hypothetical. It cost two regions: western_australia was
    re-fetched at lattice position r1c0 while its cached reference stayed at
    r1c1, putting the reference cells 4,000 m east of the raster (measured:
    4,001.6 m, against 4,000.0 m predicted from the two anchors), so the
    interior mask retained nothing. Nothing raised an error, because a
    perfectly valid reference for the wrong ground is still a perfectly valid
    array.

    Hashing the patch identity into the filename makes the failure
    impossible rather than merely detectable: a re-acquired patch produces a
    different cache name, so it always recomputes, and the stale files stay
    on disk for forensics instead of being overwritten.
    """
    if not refresh and scene in _PATCH_FINGERPRINT_CACHE:
        return _PATCH_FINGERPRINT_CACHE[scene]
    path = scene_zarr_path(scene)
    if not path.exists():
        return "nopatch"
    try:
        dt = xr.open_datatree(path, engine="zarr", consolidated=False,
                              chunks={})
        attrs = dict(dt.attrs) or dict(dt[BAND].attrs)
        da = dt[BAND]
        # Product identity AND ground position: a re-acquisition can keep the
        # product and move the window, or keep the window and change the
        # product. Both must invalidate.
        key = "|".join([
            str(attrs.get("source_item_id", "")),
            str(attrs.get("source_stac_endpoint", "")),
            f"{float(da.x.values[0]):.1f},{float(da.y.values[0]):.1f}",
            f"{da.sizes.get('x', 0)}x{da.sizes.get('y', 0)}",
        ])
    except Exception:
        key = f"unreadable:{path.stat().st_mtime_ns if path.exists() else 0}"
    fp = hashlib.sha256(key.encode()).hexdigest()[:8]
    _PATCH_FINGERPRINT_CACHE[scene] = fp
    return fp


def _cache_npz_path(scene, stem):
    """Location of a derived intermediate cache (.npz), not a primary input.

    Region intermediates go next to their patch rather than into the top-level
    `data/`, which would otherwise accumulate 40 scenes x 5 methods of derived
    files alongside the four archived paper inputs.

    The name carries both the configuration suffix and the patch fingerprint,
    so a cache is only ever reused for the same configuration AND the same
    underlying acquisition. See `patch_fingerprint`.
    """
    if scene in region_coordinates or scene.startswith("region__"):
        base = DATA_DIR / REGION_DATA_SUBDIR
    else:
        base = DATA_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{scene}_{stem}_{_cache_suffix()}_p{patch_fingerprint(scene)}.npz"


def _coords_for(scene):
    """Look a scene up in the paper's four benchmark scenes first, then in the
    regions registered by load_region_sites()."""
    if scene in benchmark_coordinates:
        return benchmark_coordinates[scene]
    if scene in region_coordinates:
        return region_coordinates[scene]
    raise KeyError(
        f"Unknown scene {scene!r}. Call load_region_sites() first if this is "
        f"one of the multi-region site ids."
    )


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
    path = scene_zarr_path(scene)
    path.parent.mkdir(parents=True, exist_ok=True)
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
            "archive by running every cell of "
            "notebooks/load_data_in_zenodo.ipynb first."
        )
    coords = _coords_for(scene) if coords is None else coords

    # Multi-region scenes read Cloud-Optimized GeoTIFFs from earth-search,
    # which has global coverage; the four paper scenes keep the original EOPF
    # path so their archived inputs stay bit-identical.
    is_region = scene in region_coordinates or scene.startswith("region__")
    if is_region and REGION_STAC_SOURCE == "earth-search":
        ds_patch = _fetch_patch_earth_search(scene, coords, patch_size, band)
        ds_patch.to_zarr(path, mode="w")
        return

    catalog = pystac_client.Client.open("https://stac.core.eopf.eodc.eu")
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

    # Pin the resolved product so a re-run is reproducible: the STAC query is
    # time-dependent and "first match" drifts as new acquisitions appear.
    if scene in region_coordinates:
        region_coordinates[scene]["product_id"] = item.id

    try:
        ds = xr.open_dataset(
            item.assets["product"].href,
            engine="eopf-zarr",
            resolution=10,
            variables=[band],
            chunks={},
        )
    except ValueError as exc:
        if "eopf-zarr" in str(exc):
            check_eopf_engine(raise_on_missing=True)
        raise

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
    path = scene_zarr_path(scene_name)
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
    out_npz = _cache_npz_path(scene_name, "real_downscale_reference")
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
    out_npz = _cache_npz_path(scene_name, "real_downscale_psf_aware")
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
    out_npz = _cache_npz_path(scene_name, f"real_downscale_classical_{method}")
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
    out_npz = _cache_npz_path(scene_name, "real_downscale_richardson_lucy")
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


def run_all(force=False, csv_name=None, scenes=None):
    """Run every method on every scene and return the metrics DataFrame.

    `csv_name` overrides the output filename. run_variant() passes a
    variant-specific name so that a control run never overwrites the main
    result file -- the default name is the one the paper's table is built
    from.

    `scenes` overrides which scenes are processed; it defaults to the four
    benchmark scenes. The multi-region driver passes the 40 region ids
    registered by load_region_sites().
    """
    TABLE_DIR.mkdir(exist_ok=True)
    rows = []
    for scene in (scenes if scenes is not None else benchmark_coordinates):
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
                 degrade_fwhm_y_m=None, recon_fwhm_m=None, edge_margin_m=None,
                 force=False, label=None, scenes=None, csv_name=None):
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
    global ANISOTROPIC_DEGRADE_FWHM_Y_M, EDGE_MARGIN_M, RECON_FWHM_M
    saved = (
        REFERENCE_MODE, BASELINE_ESTIMAND, MAX_ITER, DEGRADE_PSF_MODE,
        ANISOTROPIC_DEGRADE_FWHM_X_M, ANISOTROPIC_DEGRADE_FWHM_Y_M,
        EDGE_MARGIN_M, RECON_FWHM_M,
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
        if recon_fwhm_m is not None:
            # Width mismatch: the reconstruction assumes a different response
            # width from the one actually imposed by the degradation. The
            # cache suffix already encodes RECON_FWHM_M, so nothing is reused.
            RECON_FWHM_M = float(recon_fwhm_m)
        degradation_x_m, degradation_y_m = degradation_fwhm_xy()
        if edge_margin_m is not None:
            # Pinned by the caller. Needed by the width-mismatch arms: the
            # margin normally scales with RECON_FWHM_M, so letting it float
            # would change the set of evaluated cells at the same time as the
            # assumed width, confounding the two. Every arm must score the
            # same footprint for the comparison to isolate the width error.
            EDGE_MARGIN_M = float(edge_margin_m)
        else:
            EDGE_MARGIN_M = EDGE_MARGIN_FACTOR * max(
                RECON_FWHM_M, degradation_x_m, degradation_y_m
            )
        # Never write over the main metrics CSV: that file is the one the
        # paper's table is generated from, and a control run must not be able
        # to silently replace it.
        df = run_all(
            force=force,
            scenes=scenes,
            csv_name=csv_name
            or f"real_groundtruth_downscale_metrics_{_cache_suffix()}.csv",
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
            ANISOTROPIC_DEGRADE_FWHM_Y_M, EDGE_MARGIN_M, RECON_FWHM_M,
        ) = saved


def run_width_mismatch(scale=1.5, force=False, scenes=None, pin_edge_margin=True):
    """Width-mismatch variant of the real-data downscaling protocol.

    The degradation stays at DEGRADE_FWHM_M; only the *assumed* reconstruction
    width changes, to `scale * DEGRADE_FWHM_M`. This is the real-data analogue
    of the paper's +-50% synthetic mismatch arms: `scale=1.5` overestimates
    the response, `scale=0.5` underestimates it.

    Note the asymmetry that makes this test meaningful: only \\psf-aware and
    Richardson--Lucy use the assumed width at all, so the geometric baselines
    are unaffected by the mismatch. A width error therefore costs the two
    response-modelling methods and nothing else -- the opposite of a
    handicap for the baselines.

    `pin_edge_margin` (default True) holds the evaluation footprint at the
    matched-configuration value. Without it, EDGE_MARGIN_M scales with
    RECON_FWHM_M, so the +50% arm would silently evaluate a smaller interior
    (300 m margin instead of 200 m) and every method's RMSE would move --
    including the geometric baselines, which do not use the assumed width at
    all. That is a confound, not a result: the baselines shifting under a
    change that cannot affect them is the diagnostic that the footprint, not
    the width, moved. Leave this True unless you specifically want to measure
    the footprint effect.
    """
    matched_margin = EDGE_MARGIN_FACTOR * max(DEGRADE_FWHM_M,
                                               *degradation_fwhm_xy())
    return run_variant(
        recon_fwhm_m=scale * DEGRADE_FWHM_M,
        edge_margin_m=matched_margin if pin_edge_margin else None,
        force=force,
        scenes=scenes,
        label=f"recon width x{scale:g} ({scale * DEGRADE_FWHM_M:g} m assumed)",
    )


# =============================================================================
# Multi-region real-data downscaling
#
# The four-scene protocol above is a case study: tens of thousands of cells,
# but four patches and one sensor. This section reruns exactly the same seven
# steps on the 40 geographically distributed regions already used by the
# synthetic multi-region validation, so the real-data claim gets the same
# statistical treatment as the synthetic one.
#
# Statistical unit: the region. One patch per region, ten regions per scene
# class, so the per-class bootstrap below resamples 10 independent values --
# it is a plain bootstrap over regions, not a cluster bootstrap (there is no
# within-region clustering left to account for with a single patch each).
# =============================================================================

# Order in which the 3x3 lattice positions of a region are tried when the
# preferred one has no usable Sentinel-2 acquisition. Centre first, then the
# edge midpoints, then the corners: a fixed, data-independent ordering, so
# which patch ends up representing a region never depends on any method's
# score.
LATTICE_FALLBACK_ORDER = [(1, 1), (0, 1), (1, 0), (1, 2), (2, 1),
                          (0, 0), (0, 2), (2, 0), (2, 2)]

# Below this many regions in a class, a per-class bootstrap interval is not
# reported: with n=2 or 3 the percentile interval describes which two or three
# sites happened to be usable, not the class. Sentinel-2 availability makes an
# unbalanced sample the normal outcome, so the pooled statistic is primary and
# per-class numbers are descriptive.
MIN_REGIONS_FOR_CI = 6

REGION_SITES_CSV = "multi_patch_sites.csv"
REGION_BOOTSTRAP_REPLICATES = 5000


def load_region_sites(csv_name=None, date_window=None, cloud_max=None,
                       patch_row=1, patch_col=1, allow_fallback=True):
    """Register the multi-region sites so the real-data protocol can run on
    them, and return the list of registered scene ids.

    Reads `tables/multi_patch_sites.csv` (written by
    `multi_patch_latitude_validation.ipynb`) and keeps **one patch per
    region**, so regions stay the independent statistical unit.

    The preferred patch is the centre of the 3x3 lattice
    (`patch_row=patch_col=1`, the region anchor). With `allow_fallback`, the
    other eight positions are registered as ordered alternates, tried only if
    the preferred one yields no usable acquisition. The order is fixed in
    advance (LATTICE_FALLBACK_ORDER) and the criterion is data availability,
    evaluated before any reconstruction, so the selection cannot favour any
    method.

    Returns scene ids of the form ``region__<scene_class>__<region_id>``.
    """
    path = TABLE_DIR / (csv_name or REGION_SITES_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run multi_patch_latitude_validation.ipynb "
            "first: it writes the site manifest this experiment reuses."
        )
    sites = pd.read_csv(path)

    if allow_fallback:
        order = [(patch_row, patch_col)] + [p for p in LATTICE_FALLBACK_ORDER
                                             if p != (patch_row, patch_col)]
    else:
        order = [(patch_row, patch_col)]

    registered = []
    for region_id, grp in sites.groupby("region_id", sort=False):
        by_pos = {(int(r.patch_row), int(r.patch_col)): r
                  for _, r in grp.iterrows()}
        candidates = [by_pos[p] for p in order if p in by_pos]
        if not candidates:
            continue
        head = candidates[0]
        scene = f"region__{head.scene_class}__{region_id}"
        region_coordinates[scene] = {
            "location": head.location,
            "wgs84": {"lat": float(head.patch_lat), "lon": float(head.patch_lon)},
            "recommended_date": date_window or REGION_DATE_WINDOW,
            "cloud": cloud_max if cloud_max is not None else REGION_CLOUD_MAX,
            "scene_class": head.scene_class,
            "region_id": region_id,
            "patch_id": head.patch_id,
            "candidates": [
                {"patch_id": c.patch_id,
                 "patch_row": int(c.patch_row), "patch_col": int(c.patch_col),
                 "wgs84": {"lat": float(c.patch_lat), "lon": float(c.patch_lon)}}
                for c in candidates
            ],
        }
        registered.append(scene)

    per_class = (pd.Series([region_coordinates[s]["scene_class"]
                            for s in registered]).value_counts().to_dict())
    print(f"Registered {len(registered)} regions ({len(per_class)} classes, "
          f"{per_class}); {len(order)} candidate patch position(s) each.")
    return registered


def check_region_sites(scenes=None):
    """Report which region patches are present in the frozen bundle.

    Read-only: never downloads. This is what a normal, offline run calls.
    """
    scenes = scenes or sorted(region_coordinates)
    rows = []
    for scene in scenes:
        path = scene_zarr_path(scene)
        rows.append({"scene": scene, "path": str(path),
                     "present": path.exists()})
    df = pd.DataFrame(rows)
    n = int(df.present.sum())
    print(f"{n}/{len(df)} region patches present under "
          f"{DATA_DIR / REGION_DATA_SUBDIR}.")
    if n < len(df):
        print("Missing patches: install the frozen bundle "
              "(notebooks/load_data_in_zenodo.ipynb). If they are not in the "
              "archive yet, build them once with acquire_region_sites().")
    return df


def region_provenance(scenes=None, verbose=True):
    """Report where each region patch actually came from, store by store.

    The bundle was built in two passes: an initial one against the EOPF
    demonstration catalogue, and a second against earth-search after the
    former turned out to cover only Europe and North America. A store
    written by the first pass is still a perfectly valid patch, so nothing
    downstream complains -- but it carries a different endpoint, a different
    product, and (before the BOA-offset fix) a different radiometric
    convention from its 36 neighbours. Mixed provenance inside one
    experiment is exactly the kind of thing a referee is entitled to ask
    about, and the answer should not be "we think they are all the same".

    Returns a DataFrame with one row per region and, in `needs_refetch`, the
    stores that did NOT come from the current REGION_STAC_SOURCE. Pass that
    list straight to `acquire_region_sites(scenes=..., force=True)`.
    """
    scenes = scenes or sorted(region_coordinates)
    want = EARTH_SEARCH_URL if REGION_STAC_SOURCE == "earth-search" else None
    rows = []
    for scene in scenes:
        path = scene_zarr_path(scene)
        row = {"scene": scene, "present": path.exists(), "endpoint": "",
               "product_id": "", "baseline": "", "boa_offset_dn": "",
               "homogeneous": False}
        if path.exists():
            try:
                dt = xr.open_datatree(path, engine="zarr",
                                      consolidated=False, chunks={})
                a = dict(dt.attrs)
                if not a:                      # attrs can sit on the band node
                    a = dict(dt[BAND].attrs)
                row.update(
                    endpoint=str(a.get("source_stac_endpoint", "")),
                    product_id=str(a.get("source_item_id", "")),
                    baseline=str(a.get("source_processing_baseline", "")),
                    boa_offset_dn=str(a.get("boa_offset_dn_applied", "")),
                )
            except Exception as exc:           # unreadable store: refetch it
                row["endpoint"] = f"<unreadable: {type(exc).__name__}>"
        row["homogeneous"] = bool(
            row["present"] and want and row["endpoint"] == want
            and row["product_id"]
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    needs = df.loc[~df.homogeneous, "scene"].tolist()
    if verbose:
        n_ok = int(df.homogeneous.sum())
        print(f"Provenance: {n_ok}/{len(df)} stores come from "
              f"{REGION_STAC_SOURCE} ({want}).")
        for _, r in df[~df.homogeneous].iterrows():
            why = ("absent" if not r.present
                   else r.endpoint or "no source_stac_endpoint recorded")
            print(f"  {r.scene:42s} {why}")
        if needs:
            print("\nTo make provenance homogeneous, re-acquire exactly these:")
            print("  gt.acquire_region_sites(scenes=needs_refetch, force=True)")
            print("Then rebuild the manifest and re-run the protocol: the "
                  "product ids change, so the paper's numbers must be "
                  "re-derived rather than patched.")
    df.attrs["needs_refetch"] = needs
    return df


def acquire_region_sites(scenes=None, stop_on_error=False, force=False,
                          screen=True, cloud_ladder=None):
    """ONE-OFF: download the 40 region patches into the frozen-bundle layout.

    This is the only function here that touches the network, and it exists to
    *construct* an input bundle, not to run an experiment. The 40 Sentinel-2
    region patches are a new primary input: after running this once, rebuild
    and publish the manifest, then set OFFLINE back to True so every later run
    reads the archived copies.

        gt.OFFLINE = False
        gt.acquire_region_sites()
        gt.OFFLINE = True
        # then, from the repository root:
        #   python notebooks/build_data_manifest.py --doi <new DOI>
        #   python notebooks/build_data_manifest.py --check --doi <new DOI>

    Patches are written to `data/<REGION_DATA_SUBDIR>/`, and each records the
    Sentinel-2 product id it came from, so the manifest can pin provenance.
    A region with no cloud-free acquisition in REGION_DATE_WINDOW simply fails
    here and is reported; that criterion depends only on data availability and
    is evaluated long before any method is scored.
    """
    if OFFLINE:
        raise RuntimeError(
            "acquire_region_sites() needs OFFLINE = False. It builds a new "
            "input bundle rather than reading the frozen one; set it back to "
            "True immediately afterwards."
        )
    # Fail immediately, with a usable diagnosis, rather than after a STAC query
    # per region: a missing reader kills all 40 scenes with the same generic
    # error otherwise.
    if REGION_STAC_SOURCE == "earth-search":
        try:
            import rasterio  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "REGION_STAC_SOURCE='earth-search' reads Cloud-Optimized "
                f"GeoTIFFs and needs rasterio, which is not importable ({exc}).\n"
                "  pixi run -e notebooks jupyter lab\n"
                "rasterio is declared in the 'notebooks' feature of pixi.toml."
            ) from exc
        print(f"[earth-search] {EARTH_SEARCH_COLLECTION} at {EARTH_SEARCH_URL}, "
              f"asset {EARTH_SEARCH_ASSET!r}")
        warn_about_aws_credentials()
    else:
        check_eopf_engine(raise_on_missing=True)

    scenes = scenes or sorted(region_coordinates)
    rows = []
    for i, scene in enumerate(scenes, 1):
        meta = region_coordinates.get(scene, {})
        candidates = meta.get("candidates") or [
            {"patch_id": meta.get("patch_id", scene),
             "patch_row": -1, "patch_col": -1, "wgs84": meta.get("wgs84", {})}
        ]
        last_exc, errors, ok_row = None, [], None
        # Cloud ceilings outermost: a clean patch at another lattice position
        # is preferred over a cloudier one at the centre. The screen decides
        # either way, but this keeps "as clean as available" the first choice.
        ladder = [c for c in (cloud_ladder or REGION_CLOUD_LADDER)
                  if c is not None]
        attempts = [(c, cand) for c in ladder for cand in candidates]
        for attempt, (cloud_max, cand) in enumerate(attempts, 1):
            coords = dict(meta, wgs84=cand["wgs84"], cloud=cloud_max)
            coords.pop("product_id", None)   # resolve afresh for this position
            try:
                extract_bench_data(scene, coords=coords, force=force)
                meta["patch_id"] = cand["patch_id"]
                meta["wgs84"] = cand["wgs84"]

                # A patch can download perfectly and still be unusable: mostly
                # no-data at a granule edge, or entirely under a cloud the
                # scene-level percentage did not reveal. Advancing only on a
                # download *exception* would leave those in, so the quality
                # screen drives the fallback too. All criteria read the input
                # alone, so this cannot select for any method.
                quality = patch_quality(scene) if screen else {"usable": True}
                if not quality.get("usable", False):
                    errors.append(f"quality: {quality.get('flags', '?')}")
                    shutil.rmtree(scene_zarr_path(scene), ignore_errors=True)
                    continue

                ok_row = {
                    "scene": scene, "status": "ok",
                    "scene_class": meta.get("scene_class", ""),
                    "region_id": meta.get("region_id", ""),
                    "patch_id": cand["patch_id"],
                    "patch_row": cand["patch_row"], "patch_col": cand["patch_col"],
                    "attempt": attempt,
                    "cloud_max_used": cloud_max,
                    "product_id": meta.get("product_id", ""),
                    "quality_flags": quality.get("flags", ""),
                    "error": "",
                }
                rows.append(ok_row)
                notes = []
                if cand["patch_id"] != candidates[0]["patch_id"]:
                    notes.append(f"position {cand['patch_id']}")
                if cloud_max != ladder[0]:
                    notes.append(f"cloud<{cloud_max}%")
                extra = f"  ({'; '.join(notes)})" if notes else ""
                print(f"[{i:3d}/{len(scenes)}] {scene}: ok{extra}")
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                errors.append(f"{type(exc).__name__}: {exc}")
        if ok_row is None and last_exc is None and errors:
            # Every position downloaded but none passed the screen.
            last_exc = RuntimeError(errors[-1])
        if last_exc is not None:
            # Keep every distinct reason: "no items" (catalogue has no product
            # there at all) is a completely different problem from "no finite
            # pixels" (anchor outside the granule) or a missing band node, and
            # only the first is unfixable by trying another position.
            kinds = sorted(set(e.split(":")[0] for e in errors))
            rows.append({
                "scene": scene, "status": "failed",
                "scene_class": meta.get("scene_class", ""),
                "region_id": meta.get("region_id", ""),
                "patch_id": "", "patch_row": -1, "patch_col": -1,
                "attempt": len(attempts), "cloud_max_used": np.nan,
                "product_id": "",
                "error": f"{len(candidates)} position(s) tried; {'; '.join(kinds)}"
                          f" | last: {errors[-1]}",
            })
            print(f"[{i:3d}/{len(scenes)}] [FAIL] {scene} after "
                  f"{len(attempts)} attempt(s) "
                  f"({len(candidates)} position(s) x {len(ladder)} cloud "
                  f"ceiling(s)): {errors[-1]}")
            if stop_on_error:
                raise last_exc
    df = pd.DataFrame(rows)
    n_ok = int((df.status == "ok").sum())
    print(f"\n{n_ok}/{len(df)} region patches acquired into "
          f"{DATA_DIR / REGION_DATA_SUBDIR}.")
    if n_ok and "cloud_max_used" in df.columns:
        used = df.loc[df.status == "ok", "cloud_max_used"].value_counts().sort_index()
        print("  cloud ceiling actually used: "
              + ", ".join(f"{int(c)}%: {n}" for c, n in used.items()))
        relaxed = df[(df.status == "ok")
                     & (df.cloud_max_used > (cloud_ladder or REGION_CLOUD_LADDER)[0])]
        if len(relaxed):
            print(f"  {len(relaxed)} region(s) needed a relaxed ceiling: "
                  + ", ".join(relaxed.region_id.astype(str)))

    # Persist the resolved product ids. The STAC query is time-dependent -- the
    # "first matching item" can change as new acquisitions are published -- so
    # an archived patch must record which product it came from, otherwise the
    # bundle is not reproducible.
    if n_ok:
        TABLE_DIR.mkdir(exist_ok=True)
        out = TABLE_DIR / "real_groundtruth_multiregion_products.csv"
        df.to_csv(out, index=False)
        print(f"Resolved product ids written to {out}.")
    print("Next: rebuild the manifest and re-publish the archive, then set "
          "OFFLINE = True.")
    return df


def _region_metadata(scene):
    c = region_coordinates.get(scene, {})
    return c.get("scene_class", "unknown"), c.get("region_id", scene)


def run_multiregion(force=False, scenes=None, label="multiregion",
                     csv_prefix="real_groundtruth_multiregion"):
    """Run the full seven-step protocol on every registered region.

    Scenes whose patch is missing or degenerate are skipped and recorded,
    exactly as in the synthetic multi-region validation: the skip criterion
    depends only on the input or on the reference, never on a method score.
    """
    scenes = scenes or sorted(region_coordinates)
    if not scenes:
        raise RuntimeError("No regions registered. Call load_region_sites() first.")
    TABLE_DIR.mkdir(exist_ok=True)

    rows, failures = [], []
    for i, scene in enumerate(scenes, 1):
        scene_class, region_id = _region_metadata(scene)
        try:
            df = run_all(
                force=force, scenes=[scene],
                csv_name=f"{csv_prefix}_scratch_{_cache_suffix()}.csv",
            )
            if df.empty:
                raise RuntimeError("no metric rows produced")
            df = df.copy()
            df["scene_class"] = scene_class
            df["region_id"] = region_id
            rows.append(df)
            print(f"[{i:3d}/{len(scenes)}] {scene}: {len(df)} rows")
        except Exception as exc:
            failures.append({"scene": scene, "scene_class": scene_class,
                              "region_id": region_id,
                              "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{i:3d}/{len(scenes)}] [FAIL] {scene}: "
                  f"{type(exc).__name__}: {exc}")

    metrics = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    fail_df = pd.DataFrame(failures)
    suffix = _cache_suffix()
    metrics.to_csv(TABLE_DIR / f"{csv_prefix}_metrics_{suffix}.csv", index=False)
    fail_df.to_csv(TABLE_DIR / f"{csv_prefix}_failures_{suffix}.csv", index=False)
    print(f"\n{metrics.region_id.nunique() if len(metrics) else 0} regions analysed, "
          f"{len(fail_df)} failed.")
    return metrics, fail_df


def _bootstrap_ci(values, n_rep=None, seed=20260818, alpha=0.05):
    """Percentile bootstrap over independent regions."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n_rep = n_rep or REGION_BOOTSTRAP_REPLICATES
    draws = rng.integers(0, v.size, size=(n_rep, v.size))
    means = v[draws].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 100 * alpha / 2)), \
           float(np.percentile(means, 100 * (1 - alpha / 2)))


# --- patch quality screening -------------------------------------------------
#
# The STAC cloud filter is a *scene-level* percentage: a granule reported at
# 8% cloud can still have our 2.56 km patch entirely under a cloud, and a
# patch near a granule edge can be mostly no-data. Both would enter the
# experiment as legitimate-looking inputs. Every criterion below reads only
# the input patch -- never a reconstruction or a score -- and is applied
# before any method runs, so screening cannot favour any method.
MIN_FINITE_FRACTION = 0.99   # essentially complete patch
MIN_RELATIVE_STD = 1e-3      # reject flat patches (the degenerate-reference case)
MAX_BRIGHT_FRACTION = 0.30   # B04 reflectance > BRIGHT_THRESHOLD => likely cloud
BRIGHT_THRESHOLD = 0.30


def patch_quality(scene):
    """Quality descriptors for one acquired patch, read from the input only."""
    row = {"scene": scene, "readable": False, "shape": "", "n_pixels": 0,
           "finite_fraction": np.nan, "mean": np.nan, "std": np.nan,
           "relative_std": np.nan, "p01": np.nan, "p99": np.nan,
           "bright_fraction": np.nan, "flags": "", "usable": False}
    meta = region_coordinates.get(scene, {})
    row["scene_class"] = meta.get("scene_class", "")
    row["region_id"] = meta.get("region_id", scene)
    row["patch_id"] = meta.get("patch_id", "")
    try:
        dt = xr.open_datatree(scene_zarr_path(scene), engine="zarr",
                               consolidated=False, chunks={})
        img = np.asarray(dt[BAND].values, dtype=np.float64)
    except Exception as exc:
        row["flags"] = f"unreadable: {type(exc).__name__}"
        return row

    row["readable"] = True
    row["shape"] = "x".join(str(n) for n in img.shape)
    row["n_pixels"] = int(img.size)
    finite = np.isfinite(img)
    row["finite_fraction"] = float(finite.mean())

    flags = []
    if min(img.shape[-2:]) < CENTRAL_SIZE:
        flags.append(f"too small (<{CENTRAL_SIZE})")
    if row["finite_fraction"] < MIN_FINITE_FRACTION:
        flags.append(f"finite {100 * row['finite_fraction']:.1f}%")

    if finite.any():
        v = img[finite]
        row["mean"] = float(v.mean())
        row["std"] = float(v.std())
        row["p01"], row["p99"] = (float(x) for x in np.percentile(v, [1, 99]))
        denom = abs(row["mean"]) if abs(row["mean"]) > 1e-12 else 1.0
        row["relative_std"] = float(row["std"] / denom)
        row["bright_fraction"] = float((v > BRIGHT_THRESHOLD).mean())
        if row["relative_std"] < MIN_RELATIVE_STD:
            flags.append("flat (degenerate reference)")
        if row["bright_fraction"] > MAX_BRIGHT_FRACTION:
            flags.append(f"bright {100 * row['bright_fraction']:.0f}% "
                          f"(cloud/snow?)")
    else:
        flags.append("no finite pixels")

    row["flags"] = "; ".join(flags)
    row["usable"] = not flags
    return row


def screen_region_patches(scenes=None, csv_name=None, verbose=True):
    """Quality-screen every acquired region patch before running anything.

    Returns (usable_scenes, report). The report is written to
    `tables/real_groundtruth_multiregion_quality.csv` so the exclusions are
    auditable: a reader can check that each one was rejected on an input
    property, not on how a method scored.
    """
    scenes = scenes or sorted(region_coordinates)
    rows = [patch_quality(s) for s in scenes if scene_zarr_path(s).exists()]
    report = pd.DataFrame(rows)
    if report.empty:
        print("No acquired patches to screen.")
        return [], report

    TABLE_DIR.mkdir(exist_ok=True)
    report.to_csv(TABLE_DIR / (csv_name or
                                "real_groundtruth_multiregion_quality.csv"),
                  index=False)
    usable = report.loc[report.usable, "scene"].tolist()

    if verbose:
        print(f"{len(usable)}/{len(report)} patches pass the input-quality "
              f"screen.")
        bad = report[~report.usable]
        if len(bad):
            print("\nRejected (input properties only, before any method ran):")
            for _, r in bad.iterrows():
                print(f"  {r.scene:44s} {r.flags}")
        print("\nPer class among the usable patches:")
        keep = report[report.usable]
        if len(keep):
            for cls, n in keep.scene_class.value_counts().sort_index().items():
                print(f"  {cls:14s} {n:3d}")
    return usable, report


def plot_paired_summary(metrics, pooled=None, fname="real_groundtruth_multiregion_paired.pdf"):
    """Two-panel view of the pooled paired result.

    Left: every region as one point, PSF-aware RMSE against each competitor's,
    with the 1:1 line. Points above the line are regions where PSF-aware wins,
    so the claim is readable directly rather than resting on a summary number.

    Right: mean paired difference per competitor with its bootstrap interval;
    the zero line is the null. Annotated with wins/n and the exact sign test.
    """
    if pooled is None:
        pooled = paired_summary(metrics)
    wide = metrics.pivot_table(index=["scene_class", "region_id"],
                                columns="method", values="rmse")
    competitors = [c for c in wide.columns if c != "psf_aware"]
    labels = {"classical_nearest": "Nearest", "classical_linear": "Bilinear",
              "classical_cubic": "Bicubic", "richardson_lucy": "Richardson-Lucy"}
    classes = sorted(metrics.scene_class.unique())
    markers = dict(zip(classes, ["o", "s", "^", "D", "v", "P"]))
    colours = dict(zip(competitors, plt.rcParams["axes.prop_cycle"].by_key()["color"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    for comp in competitors:
        d = wide[["psf_aware", comp]].dropna()
        for cls in classes:
            sub = d[d.index.get_level_values("scene_class") == cls]
            if sub.empty:
                continue
            ax1.scatter(sub["psf_aware"], sub[comp], s=34, alpha=0.85,
                        marker=markers.get(cls, "o"), color=colours[comp],
                        edgecolor="k", linewidth=0.3,
                        label=f"{labels.get(comp, comp)} / {cls}")
    # Multiplicative padding: the axes are logarithmic, so an additive margin
    # can put the lower limit at or below zero, which collapses the whole
    # panel into one corner.
    finite = wide.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    lim = [finite.min() / 1.6, finite.max() * 1.6]
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.plot(lim, lim, "k-", lw=0.9, zorder=0)
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("PSF-aware RMSE"); ax1.set_ylabel("competitor RMSE")
    ax1.set_title("One point per region\n(above the line = PSF-aware wins)",
                  fontsize=10)
    handles = [plt.Line2D([], [], marker=markers.get(c, "o"), ls="", color="grey",
                          markeredgecolor="k", label=c) for c in classes]
    handles += [plt.Line2D([], [], marker="o", ls="", color=colours[c],
                           label=labels.get(c, c)) for c in competitors]
    ax1.legend(handles=handles, fontsize=7, loc="upper left", ncol=2)

    y = np.arange(len(pooled))
    ax2.errorbar(pooled.mean_delta_rmse, y,
                 xerr=[pooled.mean_delta_rmse - pooled.delta_ci95_low,
                       pooled.delta_ci95_high - pooled.mean_delta_rmse],
                 fmt="o", capsize=3, color="tab:blue")
    ax2.axvline(0, color="k", lw=0.9)
    ax2.set_yticks(y)
    ax2.set_yticklabels([labels.get(c, c) for c in pooled.competitor])
    ax2.invert_yaxis()
    ax2.set_xlabel("mean paired RMSE difference\n(competitor $-$ PSF-aware)")
    ax2.set_title("Pooled over regions, 95% bootstrap", fontsize=10)
    for yi, (_, r) in zip(y, pooled.iterrows()):
        ax2.annotate(f"{int(r.psf_aware_wins)}/{int(r.n_regions)}, "
                     f"$p$={r.sign_test_p:.1e}",
                     (r.delta_ci95_high, yi), textcoords="offset points",
                     xytext=(6, 0), va="center", fontsize=8)
    ax2.margins(x=0.35)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / fname, dpi=200)
    return fig


def align_synthetic_to_real(real_metrics,
                             synthetic_csv="multi_patch_latitude_metrics.csv",
                             csv_prefix="real_groundtruth_multiregion"):
    """Restrict the synthetic multi-region results to the regions that also
    produced a usable real Sentinel-2 patch.

    Sentinel-2 availability decides which regions survive, and it decides it
    for reasons unrelated to the method: catalogue coverage, cloud, granule
    edges. Comparing a 40-region synthetic result against a 32-region real one
    would leave open whether any difference comes from the method or from the
    two experiments sitting on different sites. Filtering the synthetic set to
    the same regions removes that question.

    Returns (aligned_synthetic, real_regions). The synthetic frame keeps its
    own per-patch structure -- nine patches per region -- because that is how
    it was run; only the region list is intersected.
    """
    path = TABLE_DIR / synthetic_csv
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run multi_patch_latitude_validation.ipynb "
            "first, or pass synthetic_csv=."
        )
    syn = pd.read_csv(path)
    real_regions = sorted(real_metrics.region_id.unique())
    aligned = syn[syn.region_id.isin(real_regions)].copy()

    missing = sorted(set(real_regions) - set(syn.region_id.unique()))
    dropped = sorted(set(syn.region_id.unique()) - set(real_regions))
    print(f"Synthetic aligned to the real sample: "
          f"{aligned.region_id.nunique()}/{syn.region_id.nunique()} regions kept.")
    if dropped:
        print(f"  {len(dropped)} synthetic-only region(s) dropped "
              f"(no usable Sentinel-2 patch): {', '.join(dropped[:8])}"
              f"{' ...' if len(dropped) > 8 else ''}")
    if missing:
        print(f"  WARNING: {len(missing)} real region(s) absent from the "
              f"synthetic run: {', '.join(missing)}")
    print("  per class:",
          aligned.groupby('scene_class').region_id.nunique().to_dict())

    TABLE_DIR.mkdir(exist_ok=True)
    aligned.to_csv(TABLE_DIR / f"{csv_prefix}_synthetic_aligned.csv", index=False)
    return aligned, real_regions


def compare_real_and_synthetic(real_metrics, aligned_synthetic,
                                csv_prefix="real_groundtruth_multiregion"):
    """Side-by-side recovery of the same regions, synthetic versus real.

    The two experiments measure different things -- the synthetic one recovers
    a known latent texture, the real one an independently built HEALPix
    reference -- so absolute RMSE is not comparable. What is comparable, and
    what this tabulates, is the *ranking* and the paired margin over a common
    set of sites.
    """
    rows = []
    syn_method = {"psf_aware_matched": "psf_aware", "nearest": "classical_nearest",
                  "bilinear": "classical_linear", "bicubic": "classical_cubic"}
    syn = aligned_synthetic.copy()
    syn["method_common"] = syn.method.map(syn_method)
    syn = syn.dropna(subset=["method_common"])
    syn_region = (syn.groupby(["scene_class", "region_id", "method_common"])
                     .rmse_vs_latent.mean().reset_index())

    for source, frame, col in (("synthetic", syn_region, "rmse_vs_latent"),
                                ("real", real_metrics.rename(
                                    columns={"method": "method_common"}), "rmse")):
        wide = frame.pivot_table(index=["scene_class", "region_id"],
                                  columns="method_common", values=col)
        if "psf_aware" not in wide.columns:
            continue
        for comp in [c for c in wide.columns if c != "psf_aware"]:
            d = (wide[comp] - wide["psf_aware"]).dropna()
            if d.empty:
                continue
            rows.append({"source": source, "competitor": comp,
                          "n_regions": int(d.size),
                          "psf_aware_wins": int((d > 0).sum()),
                          "win_fraction": float((d > 0).mean()),
                          "mean_delta_rmse": float(d.mean())})
    out = pd.DataFrame(rows)
    TABLE_DIR.mkdir(exist_ok=True)
    out.to_csv(TABLE_DIR / f"{csv_prefix}_real_vs_synthetic.csv", index=False)
    return out


def paired_summary(metrics, csv_prefix="real_groundtruth_multiregion"):
    """Pooled, distribution-light comparison across all available regions.

    This is the statistic to quote when Sentinel-2 availability leaves an
    unbalanced or small sample: every region counts once, each competitor is
    paired against PSF-aware *within the same region* so between-site scene
    variability cancels, and no per-class claim is made.

    Reports, per competitor: regions compared, PSF-aware wins, an exact
    two-sided sign test, and the bootstrap interval of the mean paired RMSE
    difference. `n_regions` here is the number of regions with a finite value
    for *both* methods, which can be smaller than the count in
    summarize_multiregion() if a region produced NaN metrics.
    """
    wide = metrics.pivot_table(index=["scene_class", "region_id"],
                                columns="method", values="rmse")
    if "psf_aware" not in wide.columns:
        raise ValueError("No psf_aware rows to compare against.")

    from math import comb
    rows = []
    for method in [c for c in wide.columns if c != "psf_aware"]:
        delta = (wide[method] - wide["psf_aware"]).dropna()
        n = int(delta.size)
        if n == 0:
            continue
        wins = int((delta > 0).sum())
        tail = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n
        mean, lo, hi = _bootstrap_ci(delta.values)
        rows.append({"competitor": method, "n_regions": n,
                      "psf_aware_wins": wins, "win_fraction": wins / n,
                      "sign_test_p": min(1.0, 2 * tail),
                      "mean_delta_rmse": mean,
                      "delta_ci95_low": lo, "delta_ci95_high": hi})
    out = pd.DataFrame(rows)
    TABLE_DIR.mkdir(exist_ok=True)
    out.to_csv(TABLE_DIR / f"{csv_prefix}_pooled_{_cache_suffix()}.csv",
               index=False)
    return out


# A region must evaluate at least this fraction of the median cell count to
# enter the analysis. The input-quality screen looks at the patch; nothing
# looked at how many cells the *protocol* actually managed to score. A region
# whose reference and reconstruction barely overlap can pass the input screen
# and still contribute a handful of cells, then carry the same weight as a
# full one in the region-level mean -- or contribute zero cells and silently
# shrink every paired comparison.
#
# `n_cells` is identical across all five methods within a region (they are
# scored on the same cells), so this criterion cannot favour any method.
MIN_EVALUATED_CELL_FRACTION = 0.5


def screen_region_metrics(metrics, min_fraction=None, verbose=True):
    """Drop regions that evaluated too few HEALPix cells to be meaningful.

    Returns (kept, rejected). Rejection depends only on how much of the patch
    survived the protocol's own geometry -- never on a method's score.
    """
    min_fraction = (MIN_EVALUATED_CELL_FRACTION if min_fraction is None
                    else min_fraction)
    per_region = metrics.groupby("region_id").n_cells.max()
    median = float(per_region[per_region > 0].median()) if (per_region > 0).any() else 0.0
    floor = min_fraction * median

    bad = set(per_region[per_region < floor].index)
    bad |= set(metrics.loc[~np.isfinite(pd.to_numeric(metrics.rmse,
                                                       errors="coerce")),
                            "region_id"].unique())
    kept = metrics[~metrics.region_id.isin(bad)].copy()
    rejected = metrics[metrics.region_id.isin(bad)].copy()

    if verbose:
        print(f"Evaluated-cell screen: median {median:,.0f} cells per region, "
              f"floor {floor:,.0f} ({100 * min_fraction:.0f}%).")
        if bad:
            print(f"  {len(bad)} region(s) dropped:")
            for rid in sorted(bad):
                n = int(per_region.get(rid, 0))
                cls = metrics.loc[metrics.region_id == rid, "scene_class"].iloc[0]
                why = "no evaluable cells" if n == 0 else f"{n:,} cells ({100*n/median:.1f}% of median)"
                print(f"    {cls:12s} {rid:20s} {why}")
        print(f"  {kept.region_id.nunique()} region(s) retained.")
        if bad:
            print("  Run diagnose_alignment(sorted(bad)) to see WHICH stage "
                  "of the geometry collapsed before accepting the loss.")
    return kept, rejected


def diagnose_alignment(scenes, verbose=True):
    """Account for every cell lost between the reference support and the
    scored sample, one stage at a time.

    `screen_region_metrics` tells us a region evaluated too few cells; it
    does not tell us why, and "the two supports did not overlap" is a
    description rather than a diagnosis. Because `n_cells` is identical
    across all five methods, whatever goes wrong happens in the single code
    path they share -- `align_and_mask` followed by the finite mask in
    `compute_metrics`. There are only four candidate stages, and this
    function measures all four:

      n_ref       cells in the reference support (build_reference)
      n_in_ref    ... also present in the estimate support (always == n_ref
                  for the classical baselines, which are read out ON the
                  reference cells; a drop here means build_reference and the
                  operator disagree about the patch)
      n_interior  ... surviving the EDGE_MARGIN_M interior mask, i.e. whose
                  centres reproject inside the patch's own UTM box
      n_finite    ... where the reference value is finite (weakly
                  constrained cells come back NaN by design)

    It also reports the offset between the centroid of the reprojected cell
    centres and the centre of the UTM box. A patch-scale offset there means
    the cell centres and the raster axes are not describing the same ground
    -- a georeferencing problem -- whereas a near-zero offset with a
    collapsed n_finite means the reference itself failed to constrain, which
    is a radiometry or coverage problem. The two have opposite fixes.

    Read-only and cache-backed: no download, no re-solve.
    """
    if isinstance(scenes, str):
        scenes = [scenes]
    rows = []
    for scene in scenes:
        row = {"scene": scene, "n_ref": 0, "n_in_ref": 0, "n_interior": 0,
               "n_finite": 0, "offset_x_m": np.nan, "offset_y_m": np.nan,
               "box_w_m": np.nan, "box_h_m": np.nan, "verdict": ""}
        try:
            ref = build_reference(scene)
            g = build_coarse_grid(scene)
            ref_ids = np.asarray(ref["cell_ids_ref"])
            ref_vals = np.asarray(ref["cell_data_ref"])
            row["n_ref"] = int(ref_ids.size)
            row["n_in_ref"] = int(ref_ids.size)   # baselines read out on ref

            lon_c, lat_c = cell_centers_lonlat(ref_ids, REF_LEVEL)
            x_c, y_c = lonlat_to_utm(lon_c, lat_c, g["da"])
            x_ax, y_ax = g["x_native"], g["y_native"]
            interior = interior_mask(x_c, y_c, x_ax, y_ax)
            row["n_interior"] = int(interior.sum())
            row["n_finite"] = int(np.isfinite(ref_vals[interior]).sum())

            bx = 0.5 * (float(x_ax.min()) + float(x_ax.max()))
            by = 0.5 * (float(y_ax.min()) + float(y_ax.max()))
            row["offset_x_m"] = float(np.median(x_c) - bx)
            row["offset_y_m"] = float(np.median(y_c) - by)
            row["box_w_m"] = float(x_ax.max() - x_ax.min())
            row["box_h_m"] = float(y_ax.max() - y_ax.min())

            off = max(abs(row["offset_x_m"]), abs(row["offset_y_m"]))
            half = 0.5 * min(row["box_w_m"], row["box_h_m"])
            if row["n_ref"] == 0:
                row["verdict"] = "reference support empty"
            elif off > 0.25 * half:
                row["verdict"] = (f"georeferencing: cell centres offset "
                                  f"{off:,.0f} m from a {half:,.0f} m half-box")
            elif row["n_interior"] < 0.5 * row["n_ref"]:
                row["verdict"] = "interior margin removed most of the support"
            elif row["n_finite"] < 0.5 * max(row["n_interior"], 1):
                row["verdict"] = "reference values non-finite (weak constraint)"
            else:
                row["verdict"] = "healthy"
        except Exception as exc:
            row["verdict"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    df = pd.DataFrame(rows)
    if verbose:
        for _, r in df.iterrows():
            print(f"{r.scene}")
            print(f"    ref {r.n_ref:>7,}  -> in_ref {r.n_in_ref:>7,}"
                  f"  -> interior {r.n_interior:>7,}  -> finite {r.n_finite:>7,}")
            print(f"    centre offset ({r.offset_x_m:+,.0f}, {r.offset_y_m:+,.0f}) m "
                  f"in a {r.box_w_m:,.0f} x {r.box_h_m:,.0f} m box")
            print(f"    -> {r.verdict}")
    return df


def region_availability(metrics):
    """Regions surviving per scene class, and whether a per-class interval is
    defensible. Also flags regions whose metrics came back non-finite, which
    otherwise silently shrink the paired comparisons."""
    n = (metrics.groupby("scene_class").region_id.nunique()
         .rename("n_regions").reset_index())
    n["per_class_ci_reportable"] = n.n_regions >= MIN_REGIONS_FOR_CI
    print(f"{int(n.n_regions.sum())} regions analysed. Per-class breakdown:")
    for _, r in n.iterrows():
        flag = "" if r.per_class_ci_reportable else \
            f"  <- below MIN_REGIONS_FOR_CI={MIN_REGIONS_FOR_CI}, descriptive only"
        print(f"  {r.scene_class:14s} {r.n_regions:3d}{flag}")

    bad = metrics[~np.isfinite(metrics.rmse)]
    if len(bad):
        print(f"\n{bad.region_id.nunique()} region(s) produced non-finite RMSE "
              f"and will drop out of the paired comparisons:")
        for rid, grp in bad.groupby("region_id"):
            print(f"  {rid}: {sorted(grp.method.unique())}")

    if not n.per_class_ci_reportable.all():
        print("\nQuote the pooled paired_summary() result as the primary "
              "statistic; report per-class means as descriptive, without "
              "intervals.")
    return n


def summarize_multiregion(metrics, csv_prefix="real_groundtruth_multiregion"):
    """Region-level summary and paired comparisons against PSF-aware.

    Mirrors the synthetic multi-region analysis: per scene class, the mean
    RMSE of each method over regions with a bootstrap interval, plus the
    paired per-region RMSE difference (competitor minus PSF-aware), its
    bootstrap interval, and the win fraction.
    """
    if metrics.empty:
        raise ValueError("Empty metrics frame.")
    TABLE_DIR.mkdir(exist_ok=True)
    suffix = _cache_suffix()

    summary = []
    for (cls, method), grp in metrics.groupby(["scene_class", "method"]):
        n_reg = grp.region_id.nunique()
        reportable = n_reg >= MIN_REGIONS_FOR_CI
        mean, lo, hi = _bootstrap_ci(grp.rmse.values)
        summary.append({"scene_class": cls, "method": method,
                         "n_regions": n_reg,
                         "mean_rmse": mean,
                         # Suppressed rather than printed: below n=6 the
                         # interval describes which sites were usable, not
                         # the class, and a reader would take it at face value.
                         "rmse_ci95_low": lo if reportable else np.nan,
                         "rmse_ci95_high": hi if reportable else np.nan,
                         "ci_reportable": reportable,
                         "between_region_sd": float(np.std(grp.rmse.values, ddof=1))
                         if grp.rmse.size > 1 else np.nan})
    summary = pd.DataFrame(summary).sort_values(["scene_class", "method"])

    wide = metrics.pivot_table(index=["scene_class", "region_id"],
                                columns="method", values="rmse")
    comparisons = []
    if "psf_aware" in wide.columns:
        for method in [c for c in wide.columns if c != "psf_aware"]:
            for cls, grp in wide.groupby(level="scene_class"):
                delta = (grp[method] - grp["psf_aware"]).dropna()
                if delta.empty:
                    continue
                mean, lo, hi = _bootstrap_ci(delta.values)
                comparisons.append({
                    "scene_class": cls, "competitor": method,
                    "n_regions": int(delta.size),
                    "mean_delta_rmse_competitor_minus_matched": mean,
                    "delta_ci95_low": lo, "delta_ci95_high": hi,
                    "matched_win_fraction": float((delta > 0).mean()),
                })
    comparisons = pd.DataFrame(comparisons)

    summary.to_csv(TABLE_DIR / f"{csv_prefix}_summary_{suffix}.csv", index=False)
    comparisons.to_csv(TABLE_DIR / f"{csv_prefix}_comparisons_{suffix}.csv",
                        index=False)
    return summary, comparisons


def format_multiregion_table(summary, comparisons,
                              csv_name="real_groundtruth_multiregion_table.csv"):
    """Paper-shaped table: per scene class, PSF-aware mean RMSE and, for each
    competitor, its mean RMSE, the paired delta with interval, and the win
    fraction."""
    labels = {"psf_aware": "PSF-aware", "classical_nearest": "Nearest-neighbor",
              "classical_linear": "Bilinear", "classical_cubic": "Bicubic",
              "richardson_lucy": "Richardson-Lucy"}
    records = []
    for cls in sorted(summary.scene_class.unique()):
        row = {"Scene class": cls}
        s = summary[summary.scene_class == cls].set_index("method")
        if "psf_aware" in s.index:
            row["Regions"] = int(s.loc["psf_aware", "n_regions"])
            row["PSF-aware RMSE"] = s.loc["psf_aware", "mean_rmse"]
        for m, lab in labels.items():
            if m == "psf_aware" or m not in s.index:
                continue
            row[f"{lab} RMSE"] = s.loc[m, "mean_rmse"]
            c = comparisons[(comparisons.scene_class == cls)
                             & (comparisons.competitor == m)]
            if not c.empty:
                row[f"{lab} delta"] = c.mean_delta_rmse_competitor_minus_matched.iloc[0]
                row[f"{lab} delta lo"] = c.delta_ci95_low.iloc[0]
                row[f"{lab} delta hi"] = c.delta_ci95_high.iloc[0]
                row[f"{lab} win"] = c.matched_win_fraction.iloc[0]
        records.append(row)
    table = pd.DataFrame(records)
    TABLE_DIR.mkdir(exist_ok=True)
    table.to_csv(TABLE_DIR / csv_name, index=False)
    return table


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
