# Frozen input-data bundle

The recommended installer is `notebooks/load_data_in_zenodo.ipynb`. Run all of
its cells once; it downloads and verifies the immutable archives from
https://doi.org/10.5281/zenodo.22083697 and restores the layout below:

```text
data/
├── urban_data.zarr
├── water_data.zarr
├── forest_data.zarr
├── agriculture_data.zarr
├── esri_latent/
│   ├── urban__z17_n256_os4.zarr
│   ├── water__z17_n256_os4.zarr
│   ├── forest__z17_n256_os4.zarr
│   └── agriculture__z17_n256_os4.zarr
├── multi_patch_latitude/
│   └── esri_patch_cache/
│       └── 360 stores named
│           <patch_id>__z17_n256_os4_gsd10.zarr
└── multi_patch_sentinel2/
    └── 40 stores named
        region__<scene_class>__<region_id>_data.zarr
```

`multi_patch_sentinel2/` holds real Sentinel-2 patches at the 40 region
anchors, one per region, taken at the centre of each 3x3 lattice. They are
used only by `real_groundtruth_multiregion.ipynb`, and are **Sentinel-2**
acquisitions -- not the Esri textures in `multi_patch_latitude/`, with which
they share only the site list. Derived `.npz` intermediates for these scenes
are written alongside them and are not archived.

To create this directory the first time, run the acquisition cell of
`real_groundtruth_multiregion.ipynb` once (the only network access outside
`load_data_in_zenodo.ipynb`), then rebuild the manifest and publish a new
archive version. A region with no cloud-free acquisition in the search window
is legitimately absent and is skipped by the experiment.

The ERA5 input `outputFLUX.grib` belongs one level above this directory, next
to `conservative_flux_ERA5.ipynb`.

The publication notebooks default to `OFFLINE=True`. Missing primary inputs
therefore raise `FileNotFoundError` instead of silently querying a mutable STAC
catalogue, object-store mirror, or Esri tile service.

After copying or creating the bundle, rebuild and verify the manifest from the
repository root:

```bash
python notebooks/build_data_manifest.py --doi 10.5281/zenodo.22083697
python notebooks/build_data_manifest.py --check --doi 10.5281/zenodo.22083697
```

`data_manifest.csv` records the expected path, original source identifier,
byte size, and a deterministic SHA-256 tree hash for every Zarr store. Do not
archive transient `.tmp` stores, metric/result NPZ files, figures, or cfgrib
`.idx` files as primary inputs.
