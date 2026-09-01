"""Build and verify the immutable input-data manifest for the paper notebooks.

Run from either the repository root or ``notebooks/``::

    python notebooks/build_data_manifest.py
    python notebooks/build_data_manifest.py --check
    python notebooks/build_data_manifest.py --doi 10.5281/zenodo.22107490

Directory hashes include every contained relative filename and byte, in sorted
order.  Consequently a changed Zarr chunk, attribute, or metadata file changes
the recorded SHA-256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data_guard import WAYBACK_RELEASE, WAYBACK_RELEASE_DATE  # noqa: E402


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
            asset_id=f"esri_latent_{scene}", category="regenerated_input",
            source=(f"Esri World Imagery Wayback release {WAYBACK_RELEASE} "
                    f"({WAYBACK_RELEASE_DATE})"),
            source_identifier=f"zoom=17;scene={scene};wayback={WAYBACK_RELEASE}",
            kind="zarr", required_by="test-resample-paper.ipynb;noise_sensitivity.ipynb",
            expected_shape="latent_hr:1024x1024 float32;x_hr:1024;y_hr:1024",
            archive_doi="",
            notes=("NOT redistributed (Esri terms). Regenerated locally by the "
                   "notebook's acquisition cells from the pinned Wayback "
                   "release; the SHA-256 here pins the expected regeneration."),
        ))

    sites_path = NOTEBOOKS / "tables" / "multi_patch_sites.csv"
    with sites_path.open(newline="", encoding="utf-8") as stream:
        for site in csv.DictReader(stream):
            patch_id = site["patch_id"]
            rows.append(asset(
                "notebooks/data/multi_patch_latitude/esri_patch_cache/"
                f"{patch_id}__z17_n256_os4_gsd10.zarr",
                asset_id=f"esri_multipatch_{patch_id}",
                category="regenerated_input",
                source=(f"Esri World Imagery Wayback release {WAYBACK_RELEASE} "
                        f"({WAYBACK_RELEASE_DATE})"),
                source_identifier=(
                    f"zoom=17;lat={site['patch_lat']};lon={site['patch_lon']};"
                    f"wayback={WAYBACK_RELEASE}"
                ), kind="zarr",
                required_by=(
                    "multi_patch_latitude_validation.ipynb;"
                    "throughput_scaling_benchmark.ipynb"
                ), expected_shape="latent_hr:1024x1024 float32",
                archive_doi="",
                notes=("NOT redistributed (Esri terms). One of 360 patches "
                       "regenerated locally from the pinned Wayback release; "
                       "the SHA-256 here pins the expected regeneration."),
            ))

    # Real Sentinel-2 patches at the 40 region anchors, used by the
    # multi-region real-data downscaling experiment: one patch per region.
    #
    # The lattice position is NOT assumed to be the region anchor (1,1). Where
    # the anchor yielded no usable acquisition, the pre-declared fallback order
    # moved the window to a neighbouring position, and four regions were in
    # fact acquired off-anchor. Deriving the recorded coordinates from (1,1)
    # would therefore publish a location the archived array does not cover --
    # the one field a reader would use to verify the patch. The catalogue is
    # likewise not assumed: four stores predate the switch to earth-search and
    # come from the EOPF Zarr Sample Service.
    #
    # Both facts are read from the acquisition record written by
    # acquire_region_sites(), and the per-store provenance attributes remain
    # the authority inside each Zarr.
    products = {}
    products_path = NOTEBOOKS / "tables" / "real_groundtruth_multiregion_products.csv"
    if products_path.exists():
        with products_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                products[row["scene"]] = row

    sites_by_pos = {}
    with sites_path.open(newline="", encoding="utf-8") as stream:
        for site in csv.DictReader(stream):
            sites_by_pos[(site["region_id"], site["patch_row"],
                          site["patch_col"])] = site

    seen = set()
    with sites_path.open(newline="", encoding="utf-8") as stream:
        for site in csv.DictReader(stream):
            region_id = site["region_id"]
            if region_id in seen:
                continue
            seen.add(region_id)
            scene = f"region__{site['scene_class']}__{region_id}"
            rec = products.get(scene, {})
            acquired = sites_by_pos.get(
                (region_id, rec.get("patch_row"), rec.get("patch_col")))
            lat = (acquired or site)["patch_lat"]
            lon = (acquired or site)["patch_lon"]
            product_id = rec.get("product_id") or ""
            catalogue = ("EOPF Zarr Sample Service"
                         if not product_id else "earth-search COG")
            position = (f"r{rec['patch_row']}c{rec['patch_col']}"
                        if rec.get("patch_row") not in (None, "", "-1")
                        else "recorded in the store attributes")
            rows.append(asset(
                f"notebooks/data/multi_patch_sentinel2/{scene}_data.zarr",
                asset_id=f"sentinel2_{scene}", category="primary_input",
                source=f"Copernicus Sentinel-2 L2A ({catalogue})",
                source_identifier=(
                    f"lat={lat};lon={lon};lattice={position};"
                    + (f"product={product_id}" if product_id
                       else "product recorded in the store's "
                            "source_item_id attribute")
                ), kind="zarr",
                required_by="real_groundtruth_multiregion.ipynb",
                expected_shape="256x256 native 10 m patch",
                archive_doi=doi,
                notes=(
                    "One of 40 regions, one patch each. Acquired via "
                    "real_groundtruth_common_tools.acquire_region_sites(); "
                    "where the region anchor yielded no usable acquisition, "
                    "a pre-declared fallback position within the same region "
                    "was used, and the coordinates above are those of the "
                    "patch actually archived."
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

    archived = [r for r in rows if r["category"] != "regenerated_input"]
    regen = [r for r in rows if r["category"] == "regenerated_input"]
    a_ok = sum(r["status"] == "available" for r in archived)
    r_ok = sum(r["status"] == "available" for r in regen)
    print(f"Wrote {args.output}: {a_ok}/{len(archived)} archived assets, "
          f"{r_ok}/{len(regen)} regenerable Esri assets available")
    missing = [r for r in archived if r["status"] != "available"]
    if missing:
        print(f"Missing ARCHIVED assets: {len(missing)}")
        for row in missing[:20]:
            print(f"  {row['relative_path']}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    if r_ok < len(regen):
        print(f"NOTE: {len(regen) - r_ok} Esri-derived asset(s) not present. "
              "They are NOT in the Zenodo archive (Esri terms); regenerate "
              "them by running the acquisition cells of "
              "test-resample-paper.ipynb, noise_sensitivity.ipynb and "
              "multi_patch_latitude_validation.ipynb against the pinned "
              "Wayback release, then re-run this check.")
    # --check fails on missing archived assets only: regenerables are
    # expected to be absent on a fresh install, by design.
    if args.check and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
