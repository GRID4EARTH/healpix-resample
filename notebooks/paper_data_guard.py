"""Shared offline-data guard for the paper notebooks.

The publication notebooks are deliberately network-free.  The only supported
online entry point is ``load_data_in_zenodo.ipynb``.
"""

from __future__ import annotations

from pathlib import Path


ZENODO_RECORD_ID = "22107490"
ZENODO_DOI = "10.5281/zenodo.22107490"
ZENODO_RECORD_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"

SCENES = ("urban", "water", "forest", "agriculture")


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


def _core_scene_paths() -> list[str]:
    sentinel = [f"notebooks/data/{scene}_data.zarr" for scene in SCENES]
    esri = [
        f"notebooks/data/esri_latent/{scene}__z17_n256_os4.zarr"
        for scene in SCENES
    ]
    return sentinel + esri


NOTEBOOK_REQUIREMENTS = {
    "test-resample-paper.ipynb": {
        "paths": _core_scene_paths(),
    },
    "noise_sensitivity.ipynb": {
        "paths": _core_scene_paths(),
    },
    "real_groundtruth_downscale.ipynb": {
        "paths": [f"notebooks/data/{scene}_data.zarr" for scene in SCENES],
    },
    "multi_patch_latitude_validation.ipynb": {
        "globs": [("notebooks/data/multi_patch_latitude/esri_patch_cache/*.zarr", 360)],
    },
    "throughput_scaling_benchmark.ipynb": {
        "globs": [("notebooks/data/multi_patch_latitude/esri_patch_cache/*.zarr", 360)],
    },
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
