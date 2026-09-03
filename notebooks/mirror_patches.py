"""Mirror the paper's four Sentinel-2 scenes as patches into the GRID4EARTH bucket.

Run this only to (re)create the mirrored inputs; the notebook reads the result
and does not need this script. Requires the `grid4earth` AWS profile with write
access (see project-guidelines/docs/aws-setup.md -- note the working endpoint is
s3.gra.io.cloud.ovh.net).

Why the notebook reads a mirror rather than searching the EOPF STAC directly:
that catalogue is a curated view whose holdings change. Three of these four
scenes stopped resolving there entirely, and items are marked deprecated over
time, so a search-based notebook cannot be relied on to return the same data
twice. The mirrored patches are pinned by product id and publicly readable
without credentials.

Each patch holds only what the notebook uses -- the b04 band at 10 m cropped to
256x256, plus coordinates and the CRS -- about 1 MB per scene. It is NOT a full
product mirror; the `note` attribute on each store says so.
"""
import numpy as np
import pyproj
import pystac_client
import s3fs
import xarray as xr

DEST = "grid4earth/public/eopf-mirror/sentinel-2-l2a/patches"
ENDPOINT = "https://s3.gra.io.cloud.ovh.net"
STAC = "https://stac.core.eopf.eodc.eu"
HALF = 128 * 10          # the notebook's half-width, metres
BAND = "b04"

SCENES = {
    "urban":       (48.8566, 2.3522, "2026-05-01/2026-05-30", "Paris, France"),
    "water":       (46.4983, 6.6327, "2026-06-01/2026-06-30", "Lake Geneva, Switzerland"),
    "forest":      (48.265, 8.016, "2026-07-01/2026-07-31", "Black Forest, Germany"),
    "agriculture": (48.258, 1.499, "2026-06-01/2026-06-30", "Beauce plains, France"),
}

cat = pystac_client.Client.open(STAC)
fs = s3fs.S3FileSystem(profile="grid4earth",
                       client_kwargs={"endpoint_url": ENDPOINT},
                       s3_additional_kwargs={"ACL": "public-read"})

pinned = {}
for scene, (lat, lon, window, place) in SCENES.items():
    items = list(cat.search(collections=["sentinel-2-l2a"],
                            bbox=[lon - .03, lat - .03, lon + .03, lat + .03],
                            datetime=window,
                            query={"eo:cloud_cover": {"lt": 20}}).items())
    if not items:
        print(f"{scene}: NO CANDIDATES"); continue
    item = min(items, key=lambda i: i.properties["eo:cloud_cover"])   # clearest

    ds = xr.open_dataset(item.assets["product"].href, engine="eopf-zarr",
                         resolution=10, variables=[BAND], chunks={})
    crs = pyproj.CRS.from_wkt(ds.spatial_ref.attrs["crs_wkt"])
    fwd = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs, always_xy=True)
    back = pyproj.Transformer.from_crs(crs, pyproj.CRS.from_epsg(4326), always_xy=True)
    xc, yc = fwd.transform(lon, lat)
    patch = ds.sel(x=slice(xc - HALF, xc + HALF), y=slice(yc + HALF, yc - HALF))
    if patch[BAND].shape != (256, 256):
        print(f"{scene}: unexpected shape {patch[BAND].shape}, skipping"); continue

    xx, yy = np.meshgrid(patch.x.values, patch.y.values)
    lo, la = back.transform(xx, yy)
    out = xr.Dataset(
        {BAND: (("y", "x"), patch[BAND].values.astype("float64"))},
        coords={"x": patch.x.values, "y": patch.y.values,
                "longitude": (("y", "x"), lo), "latitude": (("y", "x"), la)},
    )
    out["spatial_ref"] = xr.DataArray(0)
    out["spatial_ref"].attrs["crs_wkt"] = crs.to_wkt()
    out = out.set_coords("spatial_ref")   # coordinate, so it survives dt[BAND]
    out.attrs.update(
        note=("256x256 patch extracted for the healpix-resample paper -- "
              "NOT a full product mirror. Contains only the b04 band at 10 m "
              "plus coordinates, which is all the paper's notebook uses."),
        scene_name=scene, scene_location=place, band=BAND, patch_size=256,
        source_item_id=item.id,
        source_datetime=str(item.properties.get("datetime", "")),
        source_cloud_cover=str(item.properties.get("eo:cloud_cover", "")),
        source_collection="sentinel-2-l2a", source_stac_endpoint=STAC,
        centre_lon=lon, centre_lat=lat,
        # Which processor produced the product this patch came from. Recorded
        # so the paper can state the provenance rather than just the scene.
        processing_software=str(item.properties.get("processing:software", "")),
        processing_version=str(item.properties.get("processing:version", "")),
        processing_facility=str(item.properties.get("processing:facility", "")),
        processing_level=str(item.properties.get("processing:level", "")),
        processing_lineage=str(item.properties.get("processing:lineage", "")),
        platform=str(item.properties.get("platform", "")),
        instruments=str(item.properties.get("instruments", "")),
        gsd_m=str(item.properties.get("gsd", "")),
        extracted_by="healpix-resample paper preparation",
    )
    dest = f"{DEST}/{item.id}.zarr"
    # Format 2 pinned: the mirror is read back by environments running
    # zarr-python 2.x (see paper_data_guard.zarr_v2_kwargs).
    from paper_data_guard import zarr_v2_kwargs
    out.to_zarr(fs.get_mapper(dest), mode="w", consolidated=True,
                **zarr_v2_kwargs())
    pinned[scene] = item.id
    print(f"{scene:12s} {item.properties['eo:cloud_cover']:5.2f}%  -> {dest}")

print("\npin these in benchmark_coordinates:")
for s, pid in pinned.items():
    print(f'    "{s}": {{..., "product_id": "{pid}"}},')
