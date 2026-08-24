# Frozen input-data bundle

Copy the contents of the versioned DOI archive into this directory while
preserving the layout below:

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
└── multi_patch_latitude/
    └── esri_patch_cache/
        └── 360 stores named
            <patch_id>__z17_n256_os4_gsd10.zarr
```

The ERA5 input `outputFLUX.grib` belongs one level above this directory, next
to `conservative_flux_ERA5.ipynb`.

The publication notebooks default to `OFFLINE=True`. Missing primary inputs
therefore raise `FileNotFoundError` instead of silently querying a mutable STAC
catalogue, object-store mirror, or Esri tile service.

After copying or creating the bundle, rebuild and verify the manifest from the
repository root:

```bash
python notebooks/build_data_manifest.py --doi 10.5281/zenodo.RECORD
python notebooks/build_data_manifest.py --check --doi 10.5281/zenodo.RECORD
```

`data_manifest.csv` records the expected path, original source identifier,
byte size, and a deterministic SHA-256 tree hash for every Zarr store. Do not
archive transient `.tmp` stores, metric/result NPZ files, figures, or cfgrib
`.idx` files as primary inputs.
