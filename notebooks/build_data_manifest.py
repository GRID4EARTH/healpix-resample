"""Build and verify the immutable input-data manifest for the paper notebooks.

Run from either the repository root or ``notebooks/``::

    python notebooks/build_data_manifest.py
    python notebooks/build_data_manifest.py --check
    python notebooks/build_data_manifest.py --doi 10.5281/zenodo.22083697

Directory hashes include every contained relative filename and byte, in sorted
order.  Consequently a changed Zarr chunk, attribute, or metadata file changes
the recorded SHA-256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"

SENTINEL_PRODUCTS = {
    "urban": "S2B_MSIL2A_20260527T105619_N0512_R094_T31UDQ_20260527T133135",
    "water": "S2A_MSIL2A_20260624T103041_N0512_R108_T32TLS_20260624T182910",
    "forest": "S2A_MSIL2A_20260704T102701_N0512_R108_T32UMU_20260704T170718",
    "agriculture": "S2B_MSIL2A_20260623T104619_N0512_R051_T31UCP_20260623T132606",
}

FIELDS = [
    "asset_id", "category", "source", "source_identifier", "relative_path",
    "kind", "required_by", "expected_shape", "status", "bytes", "sha256",
    "archive_doi", "notes",
]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_directory(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = item.stat().st_size
        total += size
        digest.update(size.to_bytes(8, "big"))
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return total, digest.hexdigest()


def asset(relative_path: str, **values):
    row = {field: "" for field in FIELDS}
    row.update(values)
    row["relative_path"] = relative_path.replace("\\", "/")
    return row


def expected_assets(doi: str):
    rows = []
    for scene, product in SENTINEL_PRODUCTS.items():
        rows.append(asset(
            f"notebooks/data/{scene}_data.zarr",
            asset_id=f"sentinel2_{scene}", category="primary_input",
            source="Copernicus Sentinel-2 L2A", source_identifier=product,
            kind="zarr", required_by=(
                "test-resample-paper.ipynb;noise_sensitivity.ipynb;"
                "real_groundtruth_downscale.ipynb"
            ), expected_shape="256x256 native 10 m patch",
            archive_doi=doi,
            notes="Must retain B04, projected/geographic coordinates, CRS and source provenance attributes.",
        ))
        rows.append(asset(
            f"notebooks/data/esri_latent/{scene}__z17_n256_os4.zarr",
            asset_id=f"esri_latent_{scene}", category="primary_input",
            source="Esri World Imagery", source_identifier=f"zoom=17;scene={scene}",
            kind="zarr", required_by="test-resample-paper.ipynb;noise_sensitivity.ipynb",
            expected_shape="latent_hr:1024x1024 float32;x_hr:1024;y_hr:1024",
            archive_doi=doi,
            notes="PSF-independent luminance texture sampled on the frozen Sentinel grid.",
        ))

    sites_path = NOTEBOOKS / "tables" / "multi_patch_sites.csv"
    with sites_path.open(newline="", encoding="utf-8") as stream:
        for site in csv.DictReader(stream):
            patch_id = site["patch_id"]
            rows.append(asset(
                "notebooks/data/multi_patch_latitude/esri_patch_cache/"
                f"{patch_id}__z17_n256_os4_gsd10.zarr",
                asset_id=f"esri_multipatch_{patch_id}", category="primary_input",
                source="Esri World Imagery",
                source_identifier=(
                    f"zoom=17;lat={site['patch_lat']};lon={site['patch_lon']}"
                ), kind="zarr",
                required_by=(
                    "multi_patch_latitude_validation.ipynb;"
                    "throughput_scaling_benchmark.ipynb"
                ), expected_shape="latent_hr:1024x1024 float32",
                archive_doi=doi,
                notes="One of 360 selected patches; retain even when the reference degradation is degenerate.",
            ))

    # Real Sentinel-2 patches at the 40 region anchors, used by the
    # multi-region real-data downscaling experiment. One patch per region,
    # taken at the centre of each 3x3 lattice (patch_row = patch_col = 1), so
    # the site selection matches the synthetic multi-region manifest exactly.
    with sites_path.open(newline="", encoding="utf-8") as stream:
        for site in csv.DictReader(stream):
            if (site.get("patch_row"), site.get("patch_col")) != ("1", "1"):
                continue
            scene = f"region__{site['scene_class']}__{site['region_id']}"
            rows.append(asset(
                f"notebooks/data/multi_patch_sentinel2/{scene}_data.zarr",
                asset_id=f"sentinel2_{scene}", category="primary_input",
                source="Copernicus Sentinel-2 L2A (earth-search COG)",
                source_identifier=(
                    f"lat={site['patch_lat']};lon={site['patch_lon']};"
                    "product recorded in the store's source_item_id attribute"
                ), kind="zarr",
                required_by="real_groundtruth_multiregion.ipynb",
                expected_shape="256x256 native 10 m patch",
                archive_doi=doi,
                notes=(
                    "One of 40 region anchors. Acquired once via "
                    "real_groundtruth_common_tools.acquire_region_sites(); "
                    "a region with no cloud-free acquisition in the search "
                    "window is absent by design and is skipped downstream."
                ),
            ))

    rows.append(asset(
        "notebooks/outputFLUX.grib", asset_id="era5_output_flux",
        category="primary_input", source="ERA5 reanalysis-era5-complete",
        source_identifier="2024-06-01T06:00;stream=enda;members=0-9;N256",
        kind="grib", required_by="conservative_flux_ERA5.ipynb",
        expected_shape="10 ensemble members; steps 3h and 6h; sshf/slhf/ssr",
        archive_doi=doi,
        notes="The cfgrib .idx file is derived and is intentionally not archived.",
    ))
    return rows


def inspect(row):
    path = REPO / row["relative_path"]
    if not path.exists():
        row.update(status="missing", bytes="", sha256="")
    elif path.is_dir():
        size, digest = hash_directory(path)
        row.update(status="available", bytes=str(size), sha256=digest)
    else:
        row.update(
            status="available", bytes=str(path.stat().st_size), sha256=hash_file(path)
        )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if an input is missing")
    parser.add_argument("--doi", default="", help="DOI assigned to the frozen archive")
    parser.add_argument("--output", type=Path, default=REPO / "data_manifest.csv")
    args = parser.parse_args()

    rows = [inspect(row) for row in expected_assets(args.doi)]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    available = sum(row["status"] == "available" for row in rows)
    missing = [row for row in rows if row["status"] != "available"]
    print(f"Wrote {args.output}: {available}/{len(rows)} assets available")
    if missing:
        print(f"Missing: {len(missing)}")
        for row in missing[:20]:
            print(f"  {row['relative_path']}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    if args.check and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
