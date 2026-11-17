"""E6 of the JSTARS revision: does the E1 result hold outside its conditions?

Arms (protocol section E6 of ``review/08_protocole_preenregistre.md``):

* ``a_b02``, ``a_b03``, ``a_b08``  -- other 10 m bands, same products and windows;
* ``b_2_18``, ``b_4_19``          -- pixel / target-cell ratio 0.8 and 3.2;
* ``c_random``                    -- 60 regions drawn at random on land;
* ``d_l18``, ``d_l19``            -- Landsat 8/9 OLI red band, ratio 1.2 and 2.4;
* ``e_512``, ``e_1024``           -- patch size (8 regions).

Every arm runs the E1 machinery (`revision_tools.run_real_region`) on the
paper's real-data module with a few of its constants changed (`gt_config`).
The data steps (``fetch_*``) need the network and run on a login node; the
arms run offline.
"""
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import revision_tools as rt

# -- constants fixed by the protocol ------------------------------------------
E6_SEED = 20261117
N_RANDOM = 60
N_DRAWS = 2000
RANDOM_LAT = (-56.0, 84.0)
N_ITEMS_TRY = 5
MIN_FINITE = 0.99
STAC_CLOUD = 25
E6E_PER_CLASS = 2
FROZEN_SUBDIR = "multi_patch_sentinel2"
FROZEN_BAND = "b04"
FROZEN_SIZE = 256
E6_SUBDIR = "revision_e6"
ES_ASSETS = {"b02": "blue", "b03": "green", "b04": "red", "b08": "nir"}

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2"
LANDSAT_COLLECTION = "landsat-c2-l2"
LANDSAT_PLATFORMS = ("landsat-8", "landsat-9")
LANDSAT_SCALE = 2.75e-5
LANDSAT_OFFSET = -0.2
LANDSAT_NODATA = 0

E6_COMPETITORS = ("rl2", "tik_id", "tik_lap", "wiener_hunt", "geo_cubic")


@dataclass(frozen=True)
class Arm:
    name: str
    dataset: str | None = None      # None: the frozen E1 bundle
    band: str = "b04"
    patch: int = 256
    pixel_m: float = 10.0
    block: int = 2
    native_level: int = 20
    ref_level: int = 19
    native_fwhm_m: float = 12.5
    bright_screen: bool = True
    widths: tuple | None = None     # None: the E1 grid (rt.WIDTH_FACTORS["real"])
    scenes: str = "e1"              # "e1", "e6e" or "random"

    @property
    def study(self):
        return f"e6{self.name}"


ARMS = {a.name: a for a in (
    Arm("a_b02", "s2_b02_256", band="b02", bright_screen=False),
    Arm("a_b03", "s2_b03_256", band="b03", bright_screen=False),
    Arm("a_b08", "s2_b08_256", band="b08", bright_screen=False),
    Arm("b_2_18", None, ref_level=18),
    Arm("b_4_19", "s2_b04_512", patch=512, block=4),
    Arm("c_random", "s2_random_b04_256", scenes="random"),
    Arm("d_l18", "landsat_red_256", band="red", pixel_m=30.0, native_level=18,
        ref_level=17, native_fwhm_m=37.5),
    Arm("d_l19", "landsat_red_256", band="red", pixel_m=30.0, native_level=19,
        ref_level=18, native_fwhm_m=37.5),
    Arm("e_512", "s2_b04_512", patch=512, widths=(1.0,), scenes="e6e"),
    Arm("e_1024", "s2_b04_1024", patch=1024, widths=(1.0,), scenes="e6e"),
)}
# The E1 set-up itself, as an arm (reference for the analysis and the tests).
E1_ARM = Arm("e1")
DATASETS = {
    "s2_b02_256": "s2_same", "s2_b03_256": "s2_same", "s2_b08_256": "s2_same",
    "s2_b04_512": "s2_same", "s2_b04_1024": "s2_same",
    "s2_random_b04_256": "s2_random", "landsat_red_256": "landsat",
}


def dataset_arm(dataset):
    """The first arm reading ``dataset``: its band, size and screen."""
    return next(a for a in ARMS.values() if a.dataset == dataset)


# =============================================================================
# Configuration of the paper's real-data module for one arm
# =============================================================================
GT_NAMES = ("REGION_DATA_SUBDIR", "BAND", "PATCH_SIZE", "CENTRAL_SIZE", "PIXEL_SIZE_M",
            "BLOCK", "PIXEL_SIZE_COARSE_M", "NATIVE_LEVEL", "NATIVE_FWHM_M", "REF_LEVEL",
            "DEGRADE_FWHM_M", "RECON_FWHM_M", "EDGE_MARGIN_M", "MAX_BRIGHT_FRACTION")


def arm_constants(gt, arm):
    """Values of GT_NAMES for ``arm``, derived with the module's own rules
    (FWHM = 1.25 x coarse pixel, margin = EDGE_MARGIN_FACTOR x FWHM)."""
    coarse = arm.pixel_m * arm.block
    fwhm = 1.25 * coarse
    return {
        "REGION_DATA_SUBDIR": (f"{E6_SUBDIR}/{arm.dataset}" if arm.dataset
                               else FROZEN_SUBDIR),
        "BAND": arm.band,
        "PATCH_SIZE": arm.patch,
        "CENTRAL_SIZE": (arm.patch // arm.block) * arm.block,
        "PIXEL_SIZE_M": arm.pixel_m,
        "BLOCK": arm.block,
        "PIXEL_SIZE_COARSE_M": coarse,
        "NATIVE_LEVEL": arm.native_level,
        "NATIVE_FWHM_M": arm.native_fwhm_m,
        "REF_LEVEL": arm.ref_level,
        "DEGRADE_FWHM_M": fwhm,
        "RECON_FWHM_M": fwhm,
        "EDGE_MARGIN_M": gt.EDGE_MARGIN_FACTOR * fwhm,
        "MAX_BRIGHT_FRACTION": gt.MAX_BRIGHT_FRACTION if arm.bright_screen else 1.0,
    }


@contextlib.contextmanager
def gt_config(gt, arm):
    """Set the module constants of ``arm`` and restore them on exit.  The
    patch-fingerprint cache is keyed by scene only, so it is cleared both
    ways: the same scene name points at another patch inside an arm."""
    saved = {n: getattr(gt, n) for n in GT_NAMES}
    try:
        for n, v in arm_constants(gt, arm).items():
            setattr(gt, n, v)
        gt._PATCH_FINGERPRINT_CACHE.clear()
        yield gt
    finally:
        for n, v in saved.items():
            setattr(gt, n, v)
        gt._PATCH_FINGERPRINT_CACHE.clear()


# =============================================================================
# Scene lists
# =============================================================================
def acquisition_csv(dataset):
    return rt.ensure_out_dir() / f"e6_acquisition_{dataset}.csv"


def manifest_csv():
    return rt.ensure_out_dir() / "e6c_manifest.csv"


def e6e_scenes(gt, scenes):
    """Per class, the first E6E_PER_CLASS region ids in alphabetical order."""
    by_class = {}
    for s in scenes:
        meta = gt.region_coordinates[s]
        by_class.setdefault(meta["scene_class"], []).append((str(meta["region_id"]), s))
    out = []
    for cls in rt.CLASSES:
        out += [s for _, s in sorted(by_class.get(cls, []))[:E6E_PER_CLASS]]
    return out


def register_random(gt, usable_only=True):
    """Register the E6c regions from the manifest; return their scene ids."""
    import pandas as pd
    man = pd.read_csv(manifest_csv())
    if usable_only:
        man = man[man["usable"].fillna(False).astype(bool)]
    scenes = []
    for r in man.itertuples(index=False):
        if not isinstance(r.scene, str) or not r.scene:
            continue
        gt.region_coordinates[r.scene] = random_coords(r.draw, r.lat, r.lon)
        scenes.append(r.scene)
    return scenes


def random_coords(draw, lat, lon):
    rid = f"rnd{int(draw):04d}"
    return {"location": f"random draw {int(draw)}", "wgs84": {"lat": float(lat), "lon": float(lon)},
            "recommended_date": None, "cloud": STAC_CLOUD, "scene_class": "random",
            "region_id": rid, "patch_id": rid}


def arm_scenes(gt, arm):
    """Regions of ``arm``; call with the module's default constants."""
    import pandas as pd
    if arm.scenes == "random":
        return register_random(gt)
    scenes = rt.real_usable_scenes(gt)
    if arm.scenes == "e6e":
        scenes = e6e_scenes(gt, scenes)
    if arm.dataset:
        acq = pd.read_csv(acquisition_csv(arm.dataset))
        ok = set(acq.loc[acq["ok"].fillna(False).astype(bool), "scene"])
        scenes = [s for s in scenes if s in ok]
    return scenes


# =============================================================================
# Data: same product and window as the frozen B04 patch (E6a, E6b, E6e)
# =============================================================================
def frozen_path(gt, scene):
    return Path(gt.DATA_DIR) / FROZEN_SUBDIR / f"{scene}_data.zarr"


def frozen_patch(gt, scene):
    import xarray as xr
    dt = xr.open_datatree(frozen_path(gt, scene), engine="zarr", consolidated=False, chunks={})
    da = dt[FROZEN_BAND]
    attrs = dict(dt.attrs) or dict(da.attrs)
    return dict(item_id=str(attrs.get("source_item_id", "")),
                endpoint=str(attrs.get("source_stac_endpoint", "")),
                x=np.asarray(da.x.values, dtype=np.float64),
                y=np.asarray(da.y.values, dtype=np.float64),
                values=np.asarray(da.values), crs=gt._crs_of(da))


def window_offsets(x_first, y_first, transform, pixel_m=None, size=FROZEN_SIZE,
                   new_size=FROZEN_SIZE):
    """Pixel offsets (col, row) in a raster of affine ``transform`` of the
    ``new_size`` window centred like the ``size`` window whose first pixel
    centre is (x_first, y_first)."""
    col = (x_first - 0.5 * transform.a - transform.c) / transform.a
    row = (y_first - 0.5 * transform.e - transform.f) / transform.e
    if abs(col - round(col)) > 1e-6 or abs(row - round(row)) > 1e-6:
        raise RuntimeError(f"window not on the raster grid (col {col}, row {row})")
    shift = (new_size - size) // 2
    return int(round(col)) - shift, int(round(row)) - shift


def _read_s2_window(gt, href, col_off, row_off, size, props):
    """Reflectance window of a Sentinel-2 L2A COG, computed exactly as
    `gt._read_cog_patch` does (DN 0 -> NaN, BOA offset, 1e-4 scale)."""
    import rasterio
    from rasterio.windows import Window
    href = gt._public_https_href(href)
    with gt._rasterio_public_env():
        with rasterio.open(href) as src:
            window = Window(col_off, row_off, size, size)
            dn = src.read(1, window=window, boundless=True,
                          fill_value=gt.S2_NODATA_DN).astype(np.float64)
            transform = src.window_transform(window)
            crs = src.crs
    dn[dn == gt.S2_NODATA_DN] = np.nan
    refl = (dn + gt._boa_offset_dn(props)) * gt.S2_REFLECTANCE_SCALE
    cols = np.arange(size)
    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (cols + 0.5) * transform.e
    return refl.astype(np.float32), x, y, crs


def _raster_transform(gt, href):
    import rasterio
    with gt._rasterio_public_env():
        with rasterio.open(gt._public_https_href(href)) as src:
            return src.transform


def write_patch(gt, path, band, refl, x, y, crs, attrs):
    """Write a patch with the layout of the frozen bundle (gt's fetcher)."""
    import pyproj
    import xarray as xr
    to_wgs = pyproj.Transformer.from_crs(crs, pyproj.CRS.from_epsg(4326), always_xy=True)
    xx, yy = np.meshgrid(x, y)
    lon, lat = to_wgs.transform(xx, yy)
    ds = xr.Dataset({band: (("y", "x"), refl)},
                    coords={"x": x, "y": y, "longitude": (("y", "x"), lon),
                            "latitude": (("y", "x"), lat), "spatial_ref": ((), 0)})
    ds.spatial_ref.attrs["crs_wkt"] = crs.to_wkt()
    ds.attrs.update({k: str(v) for k, v in attrs.items()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode="w", **gt._zarr_v2_kwargs())


def _earth_search_item(gt, item_id):
    import pystac_client
    cat = pystac_client.Client.open(gt.EARTH_SEARCH_URL)
    items = list(cat.search(collections=[gt.EARTH_SEARCH_COLLECTION], ids=[item_id]).items())
    if len(items) != 1:
        raise RuntimeError(f"{len(items)} earth-search item(s) for {item_id}")
    return items[0]


def _quality(gt, dataset, scene):
    with gt_config(gt, dataset_arm(dataset)):
        q = gt.patch_quality(scene)
    return dict(usable=bool(q["usable"]), flags=q["flags"],
                finite_fraction=q["finite_fraction"], relative_std=q["relative_std"],
                bright_fraction=q["bright_fraction"])


def fetch_s2_same(gt, dataset, scenes):
    """Read ``dataset`` for each scene from the frozen patch's own product,
    after checking that B04 re-read by this code equals the frozen patch."""
    import pandas as pd
    arm = dataset_arm(dataset)
    rows = []
    for i, scene in enumerate(scenes, 1):
        rec = dict(dataset=dataset, scene=scene,
                   region_id=gt.region_coordinates[scene]["region_id"],
                   scene_class=gt.region_coordinates[scene]["scene_class"],
                   ok=False, reason="", item_id="", b04_check_max_abs=np.nan)
        try:
            fz = frozen_patch(gt, scene)
            rec["item_id"] = fz["item_id"]
            item = _earth_search_item(gt, fz["item_id"])
            props = item.properties
            transform = _raster_transform(gt, item.assets["red"].href)
            col, row = window_offsets(fz["x"][0], fz["y"][0], transform)
            b04, x4, y4, _ = _read_s2_window(gt, item.assets["red"].href, col, row,
                                             FROZEN_SIZE, props)
            diff = np.abs(b04.astype(np.float64) - fz["values"].astype(np.float64))
            same_nan = np.array_equal(np.isnan(b04), np.isnan(fz["values"]))
            rec["b04_check_max_abs"] = float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0
            if not (same_nan and rec["b04_check_max_abs"] == 0.0
                    and np.allclose(x4, fz["x"]) and np.allclose(y4, fz["y"])):
                rec["reason"] = "B04 re-read differs from the frozen patch"
                rows.append(rec)
                continue
            asset = ES_ASSETS[arm.band]
            c2, r2 = window_offsets(fz["x"][0], fz["y"][0], transform, new_size=arm.patch)
            refl, x, y, crs = _read_s2_window(gt, item.assets[asset].href, c2, r2,
                                              arm.patch, props)
            with gt_config(gt, arm):
                path = gt.scene_zarr_path(scene)
            write_patch(gt, path, arm.band, refl, x, y, crs, {
                "source_item_id": item.id, "source_collection": gt.EARTH_SEARCH_COLLECTION,
                "source_stac_endpoint": gt.EARTH_SEARCH_URL, "source_asset": asset,
                "source_datetime": props.get("datetime", ""),
                "source_cloud_cover": props.get("eo:cloud_cover", ""),
                "source_processing_baseline": props.get("s2:processing_baseline", ""),
                "boa_offset_dn_applied": gt._boa_offset_dn(props),
                "window": f"{c2},{r2},{arm.patch}"})
            q = _quality(gt, dataset, scene)
            rec.update(q, ok=q["usable"], reason=q["flags"])
        except Exception as exc:
            rec["reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(rec)
        print(f"[{dataset} {i:2d}/{len(scenes)}] {scene}: "
              f"{'ok' if rec['ok'] else 'EXCLUDED ' + rec['reason']}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(acquisition_csv(dataset), index=False)
    print(f"{int(frame['ok'].sum())}/{len(frame)} usable -> {acquisition_csv(dataset)}")
    return frame


# =============================================================================
# Data: regions drawn at random on land (E6c)
# =============================================================================
def draw_points(seed=E6_SEED, n=N_DRAWS, lat_range=RANDOM_LAT):
    """Points uniform in area on the sphere between the two latitudes."""
    rng = np.random.default_rng(seed)
    s0, s1 = np.sin(np.radians(lat_range))
    u = rng.uniform(s0, s1, n)
    lon = rng.uniform(-180.0, 180.0, n)
    return np.degrees(np.arcsin(u)), lon


def land_mask(lat, lon):
    """Inside the Natural Earth 1:50m land polygons (downloaded by cartopy)."""
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from shapely.prepared import prep
    path = shpreader.natural_earth(resolution="50m", category="physical", name="land")
    land = prep(unary_union(list(shpreader.Reader(path).geometries())))
    return np.array([land.contains(Point(float(lo), float(la))) for la, lo in zip(lat, lon)])


def _search_sorted(search_items):
    items = list(search_items)
    items.sort(key=lambda it: (it.properties.get("eo:cloud_cover", 100.0),
                               str(it.properties.get("datetime", "")), it.id))
    return items


def fetch_random(gt):
    """Draw, land test, acquisition and screen, in draw order, until
    N_RANDOM usable patches; every draw is logged in the manifest."""
    import pandas as pd
    import pystac_client
    arm = ARMS["c_random"]
    lat, lon = draw_points()
    land = land_mask(lat, lon)
    cat = pystac_client.Client.open(gt.EARTH_SEARCH_URL)
    rows, n_ok = [], 0
    for d in range(N_DRAWS):
        rec = dict(draw=d, lat=lat[d], lon=lon[d], land=bool(land[d]), status="",
                   item_id="", cloud=np.nan, scene="", usable=False, flags="")
        if n_ok >= N_RANDOM:
            rec["status"] = "not processed (target reached)"
            rows.append(rec)
            continue
        if not land[d]:
            rec["status"] = "sea"
            rows.append(rec)
            continue
        scene = f"region__random__rnd{d:04d}"
        gt.region_coordinates[scene] = random_coords(d, lat[d], lon[d])
        try:
            search = cat.search(collections=[gt.EARTH_SEARCH_COLLECTION],
                                intersects={"type": "Point", "coordinates": [float(lon[d]), float(lat[d])]},
                                datetime=gt.REGION_DATE_WINDOW,
                                query={"eo:cloud_cover": {"lt": STAC_CLOUD}}, limit=200)
            items = _search_sorted(search.items())
            if not items:
                rec["status"] = "no product"
                rows.append(rec)
                continue
            chosen = None
            for item in items[:N_ITEMS_TRY]:
                refl, x, y, crs = gt._read_cog_patch(item.assets["red"].href, lon[d], lat[d],
                                                     arm.patch, item.properties)
                if np.isfinite(refl).mean() >= MIN_FINITE:
                    chosen = (item, refl, x, y, crs)
                    break
            if chosen is None:
                rec["status"] = f"window incomplete in the first {N_ITEMS_TRY} products"
                rows.append(rec)
                continue
            item, refl, x, y, crs = chosen
            props = item.properties
            with gt_config(gt, arm):
                path = gt.scene_zarr_path(scene)
            write_patch(gt, path, "b04", refl, x, y, crs, {
                "source_item_id": item.id, "source_collection": gt.EARTH_SEARCH_COLLECTION,
                "source_stac_endpoint": gt.EARTH_SEARCH_URL, "source_asset": "red",
                "source_datetime": props.get("datetime", ""),
                "source_cloud_cover": props.get("eo:cloud_cover", ""),
                "source_processing_baseline": props.get("s2:processing_baseline", ""),
                "boa_offset_dn_applied": gt._boa_offset_dn(props)})
            q = _quality(gt, arm.dataset, scene)
            rec.update(status="acquired", item_id=item.id,
                       cloud=props.get("eo:cloud_cover", np.nan), scene=scene,
                       usable=q["usable"], flags=q["flags"])
            n_ok += int(q["usable"])
        except Exception as exc:
            rec["status"] = f"error: {type(exc).__name__}: {exc}"
        rows.append(rec)
        print(f"[draw {d:4d}] {rec['status']} {rec['flags']} ({n_ok}/{N_RANDOM})", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(manifest_csv(), index=False)
    print(f"{n_ok} usable random regions -> {manifest_csv()}")
    return frame


# =============================================================================
# Data: Landsat 8/9 OLI (E6d)
# =============================================================================
def _pc_token():
    import json
    import urllib.request
    with urllib.request.urlopen(PC_TOKEN_URL, timeout=60) as r:
        return json.load(r)["token"]


def landsat_reflectance(dn):
    """Collection 2 level-2 surface reflectance; DN 0 is no data."""
    dn = np.asarray(dn, dtype=np.float64).copy()
    dn[dn == LANDSAT_NODATA] = np.nan
    return dn * LANDSAT_SCALE + LANDSAT_OFFSET


def _read_landsat_window(href, lon0, lat0, size):
    import pyproj
    import rasterio
    from rasterio.windows import Window
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4",
                      GDAL_HTTP_RETRY_DELAY="2"):
        with rasterio.open(href) as src:
            to_src = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), src.crs,
                                                 always_xy=True)
            cx, cy = to_src.transform(float(lon0), float(lat0))
            row, col = src.index(cx, cy)
            half = size // 2
            window = Window(col - half, row - half, size, size)
            dn = src.read(1, window=window, boundless=True, fill_value=LANDSAT_NODATA)
            transform = src.window_transform(window)
            crs = src.crs
    cols = np.arange(size)
    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (cols + 0.5) * transform.e
    return landsat_reflectance(dn).astype(np.float32), x, y, crs


def patch_centre_lonlat(fz):
    import pyproj
    n = len(fz["x"])
    cx = 0.5 * (fz["x"][n // 2 - 1] + fz["x"][n // 2])
    cy = 0.5 * (fz["y"][n // 2 - 1] + fz["y"][n // 2])
    to_wgs = pyproj.Transformer.from_crs(fz["crs"], pyproj.CRS.from_epsg(4326), always_xy=True)
    lon, lat = to_wgs.transform(cx, cy)
    return float(lon), float(lat)


def fetch_landsat(gt, scenes):
    import pandas as pd
    import pystac_client
    dataset = "landsat_red_256"
    arm = dataset_arm(dataset)
    cat = pystac_client.Client.open(PC_STAC)
    token, t_token = _pc_token(), time.time()
    rows = []
    for i, scene in enumerate(scenes, 1):
        rec = dict(dataset=dataset, scene=scene,
                   region_id=gt.region_coordinates[scene]["region_id"],
                   scene_class=gt.region_coordinates[scene]["scene_class"],
                   ok=False, reason="", item_id="", platform="", cloud=np.nan)
        try:
            if time.time() - t_token > 1800:
                token, t_token = _pc_token(), time.time()
            lon0, lat0 = patch_centre_lonlat(frozen_patch(gt, scene))
            rec.update(lon=lon0, lat=lat0)
            search = cat.search(collections=[LANDSAT_COLLECTION],
                                intersects={"type": "Point", "coordinates": [lon0, lat0]},
                                datetime=gt.REGION_DATE_WINDOW,
                                query={"eo:cloud_cover": {"lt": STAC_CLOUD}}, limit=200)
            items = [it for it in _search_sorted(search.items())
                     if it.properties.get("platform") in LANDSAT_PLATFORMS]
            chosen = None
            for item in items[:N_ITEMS_TRY]:
                href = item.assets["red"].href
                href = f"{href}{'&' if '?' in href else '?'}{token}"
                refl, x, y, crs = _read_landsat_window(href, lon0, lat0, arm.patch)
                if np.isfinite(refl).mean() >= MIN_FINITE:
                    chosen = (item, refl, x, y, crs)
                    break
            if chosen is None:
                rec["reason"] = (f"no complete window in the first {N_ITEMS_TRY} of "
                                 f"{len(items)} product(s)")
            else:
                item, refl, x, y, crs = chosen
                props = item.properties
                with gt_config(gt, arm):
                    path = gt.scene_zarr_path(scene)
                write_patch(gt, path, arm.band, refl, x, y, crs, {
                    "source_item_id": item.id, "source_collection": LANDSAT_COLLECTION,
                    "source_stac_endpoint": PC_STAC, "source_asset": "red",
                    "source_platform": props.get("platform", ""),
                    "source_datetime": props.get("datetime", ""),
                    "source_cloud_cover": props.get("eo:cloud_cover", ""),
                    "reflectance": f"DN*{LANDSAT_SCALE}{LANDSAT_OFFSET:+}"})
                q = _quality(gt, dataset, scene)
                rec.update(q, ok=q["usable"], reason=q["flags"], item_id=item.id,
                           platform=props.get("platform", ""),
                           cloud=props.get("eo:cloud_cover", np.nan))
        except Exception as exc:
            rec["reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(rec)
        print(f"[{dataset} {i:2d}/{len(scenes)}] {scene}: "
              f"{'ok' if rec['ok'] else 'EXCLUDED ' + str(rec['reason'])}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(acquisition_csv(dataset), index=False)
    print(f"{int(frame['ok'].sum())}/{len(frame)} usable -> {acquisition_csv(dataset)}")
    return frame


def fetch(gt, dataset):
    kind = DATASETS[dataset]
    if kind == "s2_random":
        return fetch_random(gt)
    scenes = rt.real_usable_scenes(gt)
    if dataset == "s2_b04_1024":
        scenes = e6e_scenes(gt, scenes)
    if kind == "landsat":
        return fetch_landsat(gt, scenes)
    return fetch_s2_same(gt, dataset, scenes)


# =============================================================================
# Runner
# =============================================================================
def central_domain(gt, scene):
    """E6e common domain: the cells E1 evaluates on the 256^2 patch (the
    interior, at the E1 margin, of the central 256^2 window).  Call inside
    the arm's gt_config."""
    _, _, _, da = gt.load_scene_patch(scene)
    n = int(da.sizes["x"])
    h0 = (n - FROZEN_SIZE) // 2
    x256 = np.asarray(da.x.values[h0:h0 + FROZEN_SIZE])
    y256 = np.asarray(da.y.values[h0:h0 + FROZEN_SIZE])
    margin = gt.EDGE_MARGIN_FACTOR * 1.25 * 20.0     # E1: 8 x 25 m

    def keep(ids):
        lon, lat = gt.cell_centers_lonlat(np.asarray(ids), gt.REF_LEVEL)
        x, y = gt.lonlat_to_utm(lon, lat, da)
        return gt.interior_mask(x, y, x256, y256, margin_m=margin)
    return keep


def run_arm(gt, arm_name, *, shard="", force=False, limit=0):
    import traceback
    arm = ARMS[arm_name]
    scenes = arm_scenes(gt, arm)
    if limit:
        scenes = scenes[:limit]
    if shard:
        k, n = (int(v) for v in shard.split("/"))
        scenes = scenes[k::n]
        print(f"shard {k}/{n}: {len(scenes)} region(s)")
    failures = []
    with gt_config(gt, arm):
        for i, scene in enumerate(scenes, 1):
            t0 = time.perf_counter()
            try:
                domains = ({"central": central_domain(gt, scene)}
                           if arm.scenes == "e6e" else None)
                frame = rt.run_real_region(gt, scene, do_e2=False, force=force,
                                           widths=arm.widths, study=arm.study,
                                           domains=domains)
                print(f"[{i:3d}/{len(scenes)}] {arm.name} {scene}: {len(frame)} rows "
                      f"({time.perf_counter() - t0:.1f} s)", flush=True)
            except Exception as exc:
                failures.append(scene)
                print(f"[{i:3d}/{len(scenes)}] [FAIL] {arm.name} {scene}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
    print(f"\n{len(scenes) - len(failures)}/{len(scenes)} regions done; failures: {failures}")
    return 1 if failures else 0


# =============================================================================
# Analysis
# =============================================================================
def transfer_configs():
    """Per (family, mode): the configuration minimizing the mean relative
    score over the 40 E1 real regions (mode A: f = 1; mode B: any f)."""
    df = rt.prepare(rt.collect_results("real", "e1e2"))
    out = {}
    for fam in rt.E1_FAMILIES:
        for mode in ("A", "B"):
            sub = rt._width_filter(df[df.family == fam], mode)
            if len(sub):
                out[(fam, mode)] = sub.groupby("config")["rel"].mean().idxmin()
    return out


def tuned_frames(df, transfer, schemes=("transfer", "5fold")):
    import pandas as pd
    out = {}
    for fam in rt.E1_FAMILIES:
        if fam not in set(df["family"]):
            continue
        for mode in ("A", "B"):
            if "transfer" in schemes and (fam, mode) in transfer:
                sub = df[(df.family == fam) & (df.config == transfer[(fam, mode)])]
                if len(sub):
                    t = sub[["region_id", "scene_class", "config", "rmse"]].copy()
                    t["family"], t["mode"], t["scheme"] = fam, mode, "transfer"
                    out[(fam, mode, "transfer")] = t
            if "5fold" in schemes:
                s = rt.select(df, fam, mode, "5fold")
                if len(s):
                    out[(fam, mode, "5fold")] = s
        o = rt.select(df, fam, "B", "oracle")
        if len(o):
            out[(fam, "B", "oracle")] = o
    return out


def paired(ours, theirs):
    """`rt.paired_stats` plus a bootstrap CI of the mean relative margin."""
    rows = rt.paired_stats(ours, theirs)
    joined = ours.set_index("region_id")[["rmse", "scene_class"]].join(
        theirs.set_index("region_id")[["rmse"]], rsuffix="_comp", how="inner")
    for r in rows:
        grp = joined if r["scene_class"] == "all" else joined[joined.scene_class == r["scene_class"]]
        rel = ((grp["rmse_comp"] - grp["rmse"]) / grp["rmse_comp"]).to_numpy()
        _, lo, hi = rt.bootstrap_mean_ci(rel)
        r.update(rel_ci_low=lo, rel_ci_high=hi)
    return rows


def compare(tuned, schemes=("transfer", "5fold")):
    import pandas as pd
    rows = []
    for scheme in schemes:
        for mode in ("A", "B"):
            ours = tuned.get(("psf_aware", mode, scheme))
            if ours is None:
                continue
            best = None
            for comp in E6_COMPETITORS:
                theirs = tuned.get((comp, mode, scheme))
                if theirs is None:
                    continue
                rows += [dict(scheme=scheme, mode=mode, competitor=comp, **r)
                         for r in paired(ours, theirs)]
                if comp in rt.E4_COMPETITORS:
                    r_ = theirs.set_index("region_id")[["rmse", "scene_class"]]
                    best = r_ if best is None else best.assign(
                        rmse=np.minimum(best["rmse"], r_["rmse"].reindex(best.index)))
            if best is not None:
                theirs = best.reset_index()
                rows += [dict(scheme=scheme, mode=mode, competitor="strongest", **r)
                         for r in paired(ours, theirs)]
    return pd.DataFrame(rows)


def arm_frame(arm):
    df = rt.prepare(rt.collect_results("real", "e1e2" if arm.name == "e1" else arm.study))
    return df[df["family"].isin(rt.E1_FAMILIES)].copy()


def e6_analyze():
    import pandas as pd
    out = rt.ensure_out_dir("analysis_e6")
    transfer = transfer_configs()
    pd.Series({f"{k[0]}|{k[1]}": v for k, v in transfer.items()}).to_csv(
        out / "e6_transfer_configs.csv", header=["config"])
    comps, curves, sizes = [], [], []
    for arm in (E1_ARM,) + tuple(a for a in ARMS.values() if a.scenes != "e6e"):
        try:
            df = arm_frame(arm)
        except FileNotFoundError:
            print(f"[{arm.name}] no results")
            continue
        tuned = tuned_frames(df, transfer)
        c = compare(tuned)
        c.insert(0, "arm", arm.name)
        comps.append(c)
        rel = df.set_index(["region_id", "family", "config"])["rel"]
        for (fam, mode, scheme), t in tuned.items():
            r = pd.Series([rel.get((rid, fam, cfg), np.nan)
                           for rid, cfg in zip(t["region_id"], t["config"])], dtype=float)
            curves.append(dict(arm=arm.name, family=fam, mode=mode, scheme=scheme,
                               n_regions=int(r.notna().sum()), median_rel=float(r.median()),
                               q25_rel=float(r.quantile(0.25)), q75_rel=float(r.quantile(0.75))))
    comp = pd.concat(comps, ignore_index=True)
    comp.to_csv(out / "e6_comparisons.csv", index=False)
    summary = pd.DataFrame(curves)
    summary.to_csv(out / "e6_relative_rmse.csv", index=False)
    repro = reproduction(comp)
    repro.to_csv(out / "e6_reproduction.csv", index=False)
    size_tab = e6e_analyze(transfer)
    if size_tab is not None:
        size_tab.to_csv(out / "e6e_patch_size.csv", index=False)
    pd.set_option("display.width", 220)
    head = comp[(comp.scene_class == "all")]
    print(head[["arm", "scheme", "mode", "competitor", "wins", "n_regions",
                "mean_rel_margin", "rel_ci_low", "rel_ci_high"]].round(4).to_string(index=False))
    print(repro.to_string(index=False))
    e6_figure(comp)
    print(f"-> {out}")
    return comp


def reproduction(comp):
    """Protocol criterion: same sign as E1 and a 95 % CI excluding 0
    (internal 5-fold, mode B); and sign kept under transfer, mode A."""
    import pandas as pd
    allc = comp[comp.scene_class == "all"].set_index(["arm", "scheme", "mode", "competitor"])
    rows = []
    for (arm, scheme, mode, compt), r in allc.iterrows():
        if arm == "e1" or (scheme, mode) not in (("5fold", "B"), ("transfer", "A")):
            continue
        key = ("e1", scheme, mode, compt)
        if key not in allc.index:
            continue
        sign_e1 = np.sign(allc.loc[key, "mean_rel_margin"])
        same = np.sign(r["mean_rel_margin"]) == sign_e1
        excl = (r["rel_ci_low"] > 0) or (r["rel_ci_high"] < 0)
        rows.append(dict(arm=arm, scheme=scheme, mode=mode, competitor=compt,
                         e1_margin=allc.loc[key, "mean_rel_margin"],
                         margin=r["mean_rel_margin"], ci_low=r["rel_ci_low"],
                         ci_high=r["rel_ci_high"], wins=r["wins"], n=r["n_regions"],
                         same_sign=bool(same),
                         reproduced=bool(same and excl) if scheme == "5fold" else np.nan))
    return pd.DataFrame(rows)


def e6e_analyze(transfer):
    """Patch size: each family at its transferred mode A configuration, on
    the common domain (relative change against 256^2) and on the full
    interior; PSF-aware cost against the number of samples."""
    import pandas as pd
    studies = {256: ("e1e2", "e1e2"), 512: ("e6e_512", "e6e_512_central"),
               1024: ("e6e_1024", "e6e_1024_central")}
    frames = {}
    for size, (full, central) in studies.items():
        try:
            frames[(size, "full")] = rt.prepare(rt.collect_results("real", full))
            frames[(size, "central")] = rt.prepare(rt.collect_results("real", central))
        except FileNotFoundError:
            continue
    if len(frames) < 4:
        return None
    regions = sorted(set(frames[(512, "full")]["region_id"]))
    rows = []
    for fam in rt.E1_FAMILIES:
        cfg = transfer.get((fam, "A"))
        if cfg is None:
            continue
        base = None
        for size in sorted({s for s, _ in frames}):
            for dom in ("central", "full"):
                df = frames[(size, dom)]
                sub = df[(df.family == fam) & (df.config == cfg) & df.region_id.isin(regions)]
                r = sub.set_index("region_id")["rmse"].reindex(regions)
                rel = sub.set_index("region_id")["rel"].reindex(regions)
                row = dict(family=fam, size=size, domain=dom, n_regions=int(r.notna().sum()),
                           median_rmse=float(r.median()), median_rel=float(rel.median()))
                if dom == "central":
                    if size == 256:
                        base = r
                    d = (r - base) / base
                    row.update(max_abs_change_vs_256=float(np.nanmax(np.abs(d))),
                               median_change_vs_256=float(np.nanmedian(d)))
                if fam == "psf_aware" and dom == "full":
                    t = sub.set_index("region_id")["runtime_s"].reindex(regions)
                    row.update(median_runtime_s=float(t.median()),
                               n_samples=int((size // 2) ** 2))
                rows.append(row)
    tab = pd.DataFrame(rows)
    pa = tab[(tab.family == "psf_aware") & (tab.domain == "full")].dropna(subset=["median_runtime_s"])
    if len(pa) >= 2:
        slope = np.polyfit(np.log(pa["n_samples"]), np.log(pa["median_runtime_s"]), 1)[0]
        tab["psf_runtime_loglog_slope"] = float(slope)
    return tab


ARM_LABELS = {
    "e1": "E1 (B04, 1.6)", "a_b02": "B02", "a_b03": "B03", "a_b08": "B08",
    "b_2_18": "ratio 0.8", "b_4_19": "ratio 3.2", "c_random": "random regions",
    "d_l18": "Landsat, 1.2", "d_l19": "Landsat, 2.4",
}


E6_XLIM = (-30.0, 35.0)     # margin axis (%); values beyond are drawn at the edge


def e6_figure(comp):
    """Mean relative margin of PSF-aware (95 % CI) per arm, against the
    bicubic, the tuned RL, the tuned Tikhonov and the strongest tuned
    deconvolution: transfer (mode A) and internal cross-validation (mode B)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = (("geo_cubic", "bicubic"), ("rl2", "RL"), ("tik_id", "Tikhonov"),
              ("strongest", "strongest deconvolution"))
    styles = ((("transfer", "A"), "transfer from E1, f = 1", "#2a78d6", "o"),
              (("5fold", "B"), "5-fold CV in the arm, f tuned", "#eb6834", "s"))
    arms = [a for a in ARM_LABELS if a in set(comp["arm"])]
    allc = comp[comp.scene_class == "all"]
    plt.rcParams.update({"axes.edgecolor": rt._E4_INK2, "axes.labelcolor": rt._E4_INK,
                         "xtick.color": rt._E4_INK2, "ytick.color": rt._E4_INK2,
                         "text.color": rt._E4_INK})
    fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 0.32 * len(arms) + 1.1),
                             sharey=True, constrained_layout=True)
    ypos = {a: i for i, a in enumerate(arms[::-1])}
    for ax, (compt, title) in zip(axes, panels):
        ax.axvline(0.0, color=rt._E4_INK2, lw=0.8, zorder=1)
        for k, ((scheme, mode), label, color, marker) in enumerate(styles):
            sub = allc[(allc.competitor == compt) & (allc.scheme == scheme) & (allc["mode"] == mode)]
            for r in sub.itertuples(index=False):
                if r.arm not in ypos:
                    continue
                y = ypos[r.arm] + (0.17 if k == 0 else -0.17)
                lo, hi = (float(np.clip(100 * v, *E6_XLIM)) for v in (r.rel_ci_low, r.rel_ci_high))
                ax.plot([lo, hi], [y, y], color=color, lw=1.2, zorder=2)
                m = 100 * r.mean_rel_margin
                if E6_XLIM[0] <= m <= E6_XLIM[1]:
                    ax.plot(m, y, marker=marker, ms=4, color=color, mec="white", mew=0.6,
                            zorder=3, label=label if r.arm == arms[0] else None)
                else:       # off scale: arrow at the edge, value written
                    edge = E6_XLIM[0] if m < E6_XLIM[0] else E6_XLIM[1]
                    ax.plot(edge, y, marker="<" if m < 0 else ">", ms=5, color=color,
                            zorder=3, clip_on=False)
                    ax.annotate(f"{m:.0f} %", (edge, y), xytext=(6 if m < 0 else -6, 0),
                                textcoords="offset points", fontsize=6, va="center",
                                ha="left" if m < 0 else "right", color=rt._E4_INK2)
        ax.set_title(f"vs {title}", fontsize=8)
        ax.set_xlim(*E6_XLIM)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", color="#e4e3df", lw=0.6)
        ax.set_xlabel("margin of PSF-aware (%)", fontsize=7)
    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels([ARM_LABELS[a] for a in ypos], fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=7, frameon=False)
    fdir = rt.ensure_out_dir("figures")
    for ext in ("pdf", "png"):
        fig.savefig(fdir / f"e6_generalization.{ext}", dpi=200)
    plt.close(fig)
