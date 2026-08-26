# Frozen data deposit for the paper notebooks

## Recommended repository

Deposit the immutable input data as a **Dataset** on Zenodo (or an equivalent
institutional research-data repository).  Zotero is useful for keeping the
resulting DOI and citation in the bibliography, but it is not the preservation
repository for this bundle.

Before publishing the record, verify that redistribution of every Sentinel-2
and Esri-derived asset is compatible with its source terms and record the
applicable attribution/licence in the Zenodo metadata and in this archive.

## Archive layout

The archive must reproduce the repository paths so that it can be extracted at
the root of a clean checkout without editing a notebook:

```text
healpix-resample-paper-data-v1/
|-- data_manifest.csv
|-- DATA_README.md
`-- notebooks/
    |-- outputFLUX.grib
    `-- data/
        |-- README.md
        |-- urban_data.zarr/
        |-- water_data.zarr/
        |-- forest_data.zarr/
        |-- agriculture_data.zarr/
        |-- esri_latent/
        |   |-- urban__z17_n256_os4.zarr/
        |   |-- water__z17_n256_os4.zarr/
        |   |-- forest__z17_n256_os4.zarr/
        |   `-- agriculture__z17_n256_os4.zarr/
        |-- multi_patch_latitude/esri_patch_cache/
        |   `-- 360 stores <patch_id>__z17_n256_os4_gsd10.zarr/
        `-- multi_patch_sentinel2/
            `-- 40 stores region__<class>__<region_id>_data.zarr/
```

Do not include `*.idx`, `*.tmp`, generated figures, tables, notebook outputs,
or result caches. They are derived products, not immutable primary inputs.

Because Zarr stores contain many small files, package the tree in ZIP archives
instead of uploading every chunk separately. A practical split is:

1. `healpix-resample-paper-core-data-v2.zip`: the four Sentinel-2 stores, four
   scene-level Esri stores, ERA5 GRIB, README and manifest.
2. `healpix-resample-paper-esri-multipatch-v2.zip`: the 360 patch stores.
   Rebuilt by the packing script; the hand-packed v1 file is superseded, not
   carried over, so that every published archive is reproducible from the
   documented procedure.
3. `healpix-resample-paper-sentinel2-regions-v1.zip`: the 40 real Sentinel-2
   region patches under `notebooks/data/multi_patch_sentinel2/`.

Keeping the region patches in their own archive leaves the new inputs
separately citable and lets a reader fetch them without the ~1 GB Esri bundle.
Thirty-six patches come from earth-search and four from the EOPF Zarr Sample
Service; the originating catalogue and product identifier are recorded both in
each store's attributes and in `data_manifest.csv`.

Both ZIP files must preserve the paths beginning with `notebooks/`.

## Publication workflow

1. Copy or generate all expected stores at the paths above.
2. Create a Zenodo draft of type **Dataset** and reserve its DOI.
3. Insert the reserved DOI and rebuild the manifest:

   ```bash
   python notebooks/build_data_manifest.py --doi 10.5281/zenodo.22107490
   python notebooks/build_data_manifest.py --check --doi 10.5281/zenodo.22107490
   ```

   The second command must exit successfully and report `409/409 assets
   available`.
4. Build the ZIP archive(s) only after that check, and include the final
   `data_manifest.csv` and `git_commit.txt` in the core archive.
   `notebooks/load_data_in_zenodo.ipynb` verifies that the set of published
   files matches `EXPECTED_ARCHIVES` exactly, so update that dictionary with
   the name, byte size and MD5 of every archive in the new version before
   the record is used offline.
5. On a different machine, clone the exact Git commit, extract the archives at
   the repository root, rerun the manifest check, and execute the notebooks
   with their default `OFFLINE=True`.
6. Publish the Zenodo record, then cite its dataset DOI in the paper and add the
   DOI record to Zotero.

Keep the code/notebooks in the Git repository and identify the exact commit in
the dataset metadata. If an input file changes later, publish a new dataset
version and regenerate every checksum rather than replacing the frozen bundle.
