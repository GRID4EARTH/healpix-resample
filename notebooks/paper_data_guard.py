"""Shared offline-data guard for the paper notebooks.

The publication notebooks are deliberately network-free.  The only supported
online entry point is ``load_data_in_zenodo.ipynb``.
"""

from __future__ import annotations

from pathlib import Path


ZENODO_RECORD_ID = "22210945"  # fresh record; previous records withdrawn (Esri remediation)
ZENODO_DOI = "10.5281/zenodo.22210945"
ZENODO_RECORD_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"

SCENES = ("urban", "water", "forest", "agriculture")


# --- Esri World Imagery: pinned Wayback release ------------------------------
#
# The Esri-derived latent textures are NOT redistributed (Esri's terms grant
# redistribution for static map images, not for machine-readable derived
# datasets, and the underlying pixels belong to third-party providers).
# Instead, every user regenerates them locally. Determinism across users and
# years is provided by Esri World Imagery WAYBACK: dated, versioned snapshots
# of the basemap whose tiles do not change after publication, unlike the live
# World_Imagery service. The release below is therefore part of the
# experiment definition, exactly like a Sentinel-2 product identifier, and
# the SHA-256 values in data_manifest.csv pin the expected regeneration.
#
# Do not bump this number casually: a different release can contain updated
# imagery for some patches, which changes the synthetic experiments' inputs
# and requires re-deriving every synthetic result in the paper.
#
# Release catalogue: https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json
WAYBACK_RELEASE = 26334
WAYBACK_RELEASE_DATE = "2026-08-05"
ESRI_WAYBACK_TILE_URL = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
    "World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/"
    f"{WAYBACK_RELEASE}/{{z}}/{{y}}/{{x}}"
)


# --- Zarr format-2 invariant --------------------------------------------------
#
# The frozen bundle is uniformly Zarr FORMAT 2, readable by zarr-python 2.x
# and 3.x alike. That invariant once broke silently: region patches acquired
# on a machine with zarr-python 3 were written as format 3, which zarr 2
# cannot read, and the failure surfaced weeks later on a reproducer's machine
# looking exactly like data corruption. Environment pins cannot prevent a
# recurrence, because the acquisition machine is not always the pixi
# environment -- so the invariant is enforced HERE, at every writer. Any
# notebook or helper that creates a store must go through one of these two
# functions instead of calling zarr/xarray directly.

def sph_ang2pix(nside, lon, lat, lonlat=True, nest=True):
    """Drop-in replacement for ``healpy.ang2pix`` backed by healpix-geo.

    The notebooks historically used healpy for the synthetic HEALPix
    indexing; the paper's stack is healpix-geo, so healpy was a second,
    redundant dependency. Equivalence is exact, not approximate: with
    ``ellipsoid="sphere"`` (healpy's convention), healpix-geo returns
    bit-identical cell ids (verified on 20k random points at levels 20 and
    22) and centres agreeing to ~1e-14 degrees.

    The SPHERICAL convention is preserved deliberately. The real-data
    pipeline uses the WGS84 authalic convention (ELLIPSOID in
    tests/real_groundtruth_common_tools.py); the synthetic experiments were
    defined on the sphere and stay there -- switching them would change
    every synthetic cell assignment. Each pipeline is internally consistent.
    """
    import numpy as np
    from healpix_geo import nested

    if not (lonlat and nest):
        raise ValueError("only the lonlat=True, nest=True convention is supported")
    level = int(nside).bit_length() - 1
    if 2 ** level != int(nside):
        raise ValueError(f"nside={nside} is not a power of two")
    ids = nested.lonlat_to_healpix(
        np.asarray(lon, dtype=np.float64).ravel(),
        np.asarray(lat, dtype=np.float64).ravel(),
        level, ellipsoid="sphere",
    )
    return np.asarray(ids, dtype=np.int64)


def sph_pix2ang(nside, ipix, lonlat=True, nest=True):
    """Drop-in replacement for ``healpy.pix2ang`` (see sph_ang2pix)."""
    import numpy as np
    from healpix_geo import nested

    if not (lonlat and nest):
        raise ValueError("only the lonlat=True, nest=True convention is supported")
    level = int(nside).bit_length() - 1
    if 2 ** level != int(nside):
        raise ValueError(f"nside={nside} is not a power of two")
    lon, lat = nested.healpix_to_lonlat(
        np.asarray(ipix, dtype=np.uint64).ravel(), level, ellipsoid="sphere",
    )
    return np.asarray(lon), np.asarray(lat)


def zarr_v2_kwargs() -> dict:
    """Kwargs forcing ``Dataset.to_zarr``/``DataTree.to_zarr`` to format 2.

    Under zarr-python >= 3 this passes ``zarr_format=2`` explicitly; under
    zarr-python 2 (which can only write format 2, and does not know the
    kwarg) it passes nothing.
    """
    import zarr

    if int(str(zarr.__version__).split(".")[0]) >= 3:
        return {"zarr_format": 2}
    return {}


def open_zarr_group_w_v2(path):
    """``zarr.open_group(path, mode="w")`` pinned to format 2.

    For the raw zarr-python API (used by the Esri patch-cache writers),
    which has its own default format independent of xarray's.
    """
    import zarr

    if int(str(zarr.__version__).split(".")[0]) >= 3:
        return zarr.open_group(str(path), mode="w", zarr_format=2)
    return zarr.open_group(str(path), mode="w")


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the checkout independently of the Jupyter working directory."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "healpix_resample").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise RuntimeError(
        f"Could not locate the healpix-resample repository above {current}."
    )


def _valid_zarr(path: Path) -> bool:
    return path.is_dir() and any(
        (path / marker).exists() for marker in (".zgroup", "zarr.json", ".zarray")
    )


def _sentinel_scene_paths() -> list[str]:
    return [f"notebooks/data/{scene}_data.zarr" for scene in SCENES]


def _esri_latent_paths() -> list[str]:
    return [
        f"notebooks/data/esri_latent/{scene}__z17_n256_os4.zarr"
        for scene in SCENES
    ]


# Requirement categories:
#   paths/files/globs  -- archived inputs from the Zenodo bundle. Missing =>
#                         hard failure pointing at load_data_in_zenodo.ipynb.
#   regen_paths/regen_globs -- Esri-derived stores that are NOT redistributed
#                         (Esri terms) and that THIS notebook regenerates
#                         itself from the pinned Wayback release. Missing =>
#                         a notice, and the notebook proceeds to regenerate.
#   prereq_globs       -- Esri-derived stores this notebook READS but cannot
#                         produce. Missing => hard failure naming the
#                         notebook that regenerates them.
NOTEBOOK_REQUIREMENTS = {
    "test-resample-paper.ipynb": {
        "paths": _sentinel_scene_paths(),
        "regen_paths": _esri_latent_paths(),
    },
    "noise_sensitivity.ipynb": {
        "paths": _sentinel_scene_paths(),
        "regen_paths": _esri_latent_paths(),
    },
    "real_groundtruth_downscale.ipynb": {
        "paths": [f"notebooks/data/{scene}_data.zarr" for scene in SCENES],
    },
    "multi_patch_latitude_validation.ipynb": {
        "regen_globs": [("notebooks/data/multi_patch_latitude/esri_patch_cache/*.zarr", 360)],
    },
    # No hard requirement: the benchmark prefers the 40 archived Sentinel-2
    # region patches (Zenodo bundle, redistributable), then the Esri cache,
    # then the git-versioned site manifest with deterministic synthetic
    # textures -- construction timings depend on geometry only.
    "throughput_scaling_benchmark.ipynb": {},
    "conservative_flux_ERA5.ipynb": {
        "files": ["notebooks/outputFLUX.grib"],
    },
    # This diagnostic is synthetic and has no external input, but it still
    # declares OFFLINE=True in its setup cell.
    "effective_kernel_geometry.ipynb": {},
}


def require_paper_data(notebook_name: str) -> Path:
    """Fail early with one actionable installation message when data are absent."""
    root = find_repo_root()
    requirements = NOTEBOOK_REQUIREMENTS.get(notebook_name)
    if requirements is None:
        raise KeyError(f"No frozen-data requirements registered for {notebook_name!r}.")

    missing: list[str] = []
    for relative in requirements.get("paths", []):
        path = root / relative
        if not _valid_zarr(path):
            missing.append(relative)

    for relative in requirements.get("files", []):
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative)

    for pattern, expected_count in requirements.get("globs", []):
        stores = [path for path in root.glob(pattern) if _valid_zarr(path)]
        if len(stores) < expected_count:
            missing.append(f"{pattern} ({len(stores)}/{expected_count} valid stores)")

    # Esri-derived stores this notebook reads but cannot produce: hard
    # failure, but pointing at the regenerating notebook, not at Zenodo.
    for pattern, expected_count, hint in requirements.get("prereq_globs", []):
        stores = [path for path in root.glob(pattern) if _valid_zarr(path)]
        if len(stores) < expected_count:
            raise FileNotFoundError(
                f"{notebook_name} needs {expected_count} Esri-derived stores "
                f"({pattern}; {len(stores)} present). These are NOT in the "
                f"Zenodo archive -- {hint}."
            )

    # Esri-derived stores THIS notebook regenerates itself: absence is the
    # normal fresh-install state, not an error. Announce and proceed.
    regen_missing: list[str] = []
    for relative in requirements.get("regen_paths", []):
        if not _valid_zarr(root / relative):
            regen_missing.append(relative)
    for pattern, expected_count in requirements.get("regen_globs", []):
        stores = [path for path in root.glob(pattern) if _valid_zarr(path)]
        if len(stores) < expected_count:
            regen_missing.append(
                f"{pattern} ({len(stores)}/{expected_count} valid stores)")
    if regen_missing:
        preview = "\n".join(f"  - {item}" for item in regen_missing[:6])
        if len(regen_missing) > 6:
            preview += f"\n  - ... and {len(regen_missing) - 6} more"
        print(
            f"[offline data] {notebook_name}: {len(regen_missing)} "
            "Esri-derived input(s) absent -- expected on a fresh install, "
            "since they are not redistributed (Esri terms). This notebook's "
            "acquisition cells will regenerate them from the pinned World "
            "Imagery Wayback release "
            f"{WAYBACK_RELEASE} ({WAYBACK_RELEASE_DATE}): with the default "
            "OFFLINE = False, simply run the notebook top to bottom (network "
            "required once; the fetch is cache-first and touches only missing "
            "stores). Set OFFLINE = True afterwards to enforce strictly "
            "network-free reruns.\n" + preview
        )

    if missing:
        preview = "\n".join(f"  - {item}" for item in missing[:12])
        if len(missing) > 12:
            preview += f"\n  - ... and {len(missing) - 12} more"
        raise FileNotFoundError(
            "Frozen publication data are missing or incomplete.\n\n"
            "Run every cell of notebooks/load_data_in_zenodo.ipynb first.\n"
            f"Zenodo record: {ZENODO_RECORD_URL}\n"
            f"Dataset DOI: {ZENODO_DOI}\n\n"
            f"Missing requirements for {notebook_name}:\n{preview}"
        )

    print(f"[offline data] {notebook_name}: required frozen inputs are available")
    return root
