# Reproducing the paper's results from scratch

This recipe takes a clean machine to every table and figure of
*healpix-resample: PSF-Aware Resampling of Earth-Observation Imagery onto
HEALPix* (main paper and supplement), using only the public repository and
the frozen Zenodo data bundle. Two one-time downloads are needed -- the
Zenodo archives, and the Esri-derived textures regenerated from a pinned
World Imagery Wayback release -- after which everything runs offline.

## 0. Requirements

- Linux x86-64 (the pixi environment is pinned to `linux-64`).
- An NVIDIA GPU with a working CUDA driver. The paper's timings use an
  L4, but any CUDA GPU with ≥ 4 GB free memory reproduces the *results*
  (peak GPU memory of the main configuration is ~1.4 GiB); without a GPU
  the code falls back to CPU and runs, slowly.
- ~6 GB of free disk: 1.2 GB of compressed archives, ~2.5 GB extracted,
  plus derived caches.
- [pixi](https://pixi.sh) (installs its own Python and all dependencies;
  nothing else is needed system-wide).

## 1. Get the code at the paper's commit

```bash
git clone https://github.com/GRID4EARTH/healpix-resample.git
cd healpix-resample
git checkout jstars-lastreview-before-submission   # or the commit below
pixi install -e notebooks
```

The authoritative code version is the one recorded *inside the data
bundle*: after step 2, `git_commit.txt` at the repository root gives the
exact commit hash the archives were packed against. If in doubt, check out
that hash rather than a tag.

The `notebooks` environment contains everything the paper notebooks need
(PyTorch, rasterio, xarray-eopf, scikit-image 0.26.0 for Richardson–Lucy,
etc.). Start Jupyter with:

```bash
pixi run -e notebooks jupyter lab
```

## 2. Install the frozen data (one-time, ~1.2 GB download)

Run `notebooks/load_data_in_zenodo.ipynb` top to bottom. It downloads the
two archives of the Zenodo record referenced in that notebook (the record
DOI and per-archive MD5 values are pinned there), refuses to proceed on any
mismatch, and extracts them at the repository paths the notebooks expect.
Then verify the installation:

```bash
pixi run -e notebooks python notebooks/build_data_manifest.py --check \
    --doi 10.5281/zenodo.22210945
```

This must report all **45 archived assets available** (each checked by
SHA-256); the 364 Esri-derived assets will be listed as regenerable and
absent — that is expected, they are produced in the next step. If an
*archived* asset is missing, stop and fix that first.

## 2b. Regenerate the Esri-derived textures (one-time, ~1–2 GB download)

The Esri World Imagery derivatives are **not redistributed** (Esri's terms
cover static map images, not machine-readable derived datasets). They are
regenerated locally, deterministically, from the pinned **Wayback release
26334 (2026-08-05)** — a dated, immutable snapshot of World Imagery, so
every user fetches identical tiles regardless of when they run this step.
The pin lives in `notebooks/paper_data_guard.py` and is part of the
experiment definition, like a Sentinel-2 product identifier.

`test-resample-paper.ipynb` (scene textures) and
`multi_patch_latitude_validation.ipynb` (the 360 patches) handle this
themselves: their setup cells default to `OFFLINE = False`, and their
acquisition cells are cache-first — they fetch only what is absent, so the
first run downloads the textures and later runs touch nothing. Just run
them top to bottom (network required for this first pass). Set
`OFFLINE = True` in the setup cells afterwards if you want to *enforce*
strictly network-free reruns. Then re-run the manifest
check: it must now also report all 364 regenerated assets available with
matching SHA-256 — the manifest pins the expected regeneration, so a
mismatch means the fetch differed and must be investigated, not accepted.

**Zarr format note.** Every store in the bundle is **Zarr format 2**,
readable by any zarr-python ≥ 2.11 (including 3.x). This is deliberate and
enforced: the acquisition code refuses to write any other format, and
`notebooks/convert_zarr_stores_to_v2.py --report` audits the installed
bundle. If that report ever shows a format-3 store, run the converter — it
rewrites losslessly and verifies bit-for-bit — rather than changing any
environment pin.

## 3. Run the experiment notebooks

Run each notebook top to bottom with a fresh kernel, in this order (the
order only matters where noted). Once step 2b has populated the Esri
caches, no notebook fetches anything: the acquisition cells are
cache-first and find every store present, so all runs are effectively
network-free (set `OFFLINE = True` in the setup cells of notebooks 1, 2
and 5 to enforce it).

| # | Notebook | Reproduces | Notes |
|---|----------|------------|-------|
| 1 | `test-resample-paper.ipynb` | Four-scene synthetic results, round-trip check, estimand and hold-out controls, spectral diagnostics (supplement tables and figures; PSF spectral-consistency figure of the main paper) | |
| 2 | `multi_patch_latitude_validation.ipynb` | 40-region synthetic validation (main paper multi-region table and spectral-uncertainty figure) | Longest synthetic run: 360 patches |
| 3 | `real_groundtruth_downscale.ipynb` | Four-scene controlled reduced-resolution results and the response-width sweep (supplement; width-sensitivity numbers of the main paper) | |
| 4 | `real_groundtruth_multiregion.ipynb` | The paper's main real-data result: 40-region reduced-resolution validation, estimator-independent geometric control, zero-correction (By) ablation, geometry-integrity check | Needs the region patches from step 2's archive; independent of notebooks 1–3 except the site list, which is versioned in `notebooks/tables/multi_patch_sites.csv` |
| 5 | `noise_sensitivity.ipynb` | Noise × damping sweeps, noise × PSF-mismatch interaction, geolocation-jitter sensitivity (supplement) | |
| 6 | `throughput_scaling_benchmark.ipynb` | Batch-scaling benchmark (supplement) | Timing values are hardware-dependent; expect the *ratios*, not the absolute times, to reproduce. Runs from the Zenodo bundle alone: its patch pool is the 40 archived Sentinel-2 region patches (fallbacks: Esri cache, then synthetic textures on the versioned site-manifest geometries) |
| 7 | `conservative_flux_ERA5.ipynb` | ERA5 flux-conservation results (supplement) | |

On a single L4 the full set is on the order of a few hours; notebooks 2 and 4
dominate. Intermediate results are cached in `.npz` files whose names
encode every result-changing parameter *and* a fingerprint of the input
patch, so a partial rerun resumes where it stopped and can never silently
reuse a result computed from different data.

Each experiment notebook ends by calling
`notebooks/publish_paper_assets.py`, which copies every figure cited by
`tex/main.tex` and `tex/supplement.tex`, and every declared table CSV, into
`tex/`. After all notebooks have run, that call must report no
`missing-source` and no `would-update` entries — that is the machine check
that the paper's assets match your run.

## 4. Compile the paper and compare

```bash
cd tex
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
pdflatex supplement.tex && bibtex supplement && pdflatex supplement.tex \
    && pdflatex supplement.tex
```

(The supplement needs no compile order: its cross-references to the main
paper use the frozen numbers in `main-refs.tex`.)

## 5. What you should see

Key results to check your run against (all from the CSVs under
`notebooks/tables/`, which back the paper's tables):

- **40-region synthetic** (`multi_patch_two_stage_40_summary.csv`):
  recovery fraction 15.4–19.6 % by scene class, positive region-cluster
  bootstrap interval in every class, 100 % win fraction.
- **40-region real reduced-resolution**
  (`real_groundtruth_multiregion_*_pooled_*.csv`): PSF-aware lowest RMSE in
  all 40 regions against each of four competitors, exact sign test
  p = 1.8×10⁻¹², margins 11.0–17.9 % over Richardson–Lucy by class.
- **Zero-correction ablation**
  (`real_groundtruth_multiregion_xref_ablation.csv`): the aggregate By
  alone is the weakest method tested; the full solution beats it in all 40
  regions with 21.6–33.5 % mean relative gains.
- **Geometry integrity**: the multiregion notebook's alignment cell must
  print `All 40 regions: reference and raster agree to within … m (< 1
  native pixel)` — if it raises instead, a cache predates your data
  install; rerun the protocol cell with `force=True`.

Statistical quantities (bootstrap intervals, sign tests) use fixed seeds
and reproduce exactly. Solver outputs are floating-point GPU computations:
last-digit differences across CUDA/driver versions are possible and
harmless; every reported effect is orders of magnitude above that level.

## Troubleshooting

- **Errors opening the `.zarr` stores** — symptoms include
  `KeyError: '.zgroup'`, `GroupNotFoundError`, or messages about an
  unsupported/unknown Zarr format: a store is not in the expected format 2.
  Audit with `python notebooks/convert_zarr_stores_to_v2.py --report` and,
  if any format-3 store is listed, run the converter (see the Zarr format
  note above). Do not re-download the store and do not change the zarr
  pin: the data are healthy, only the container is wrong.
- **CUDA not found**: the code runs on CPU automatically (`DEVICE` in
  `tests/real_groundtruth_common_tools.py`); only the timings change.
- **A notebook downloads imagery after step 2b**: it should not — the
  acquisition cells are cache-first and fetch only *missing* Esri stores.
  A fetch after the caches are complete means a store was deleted or its
  content diverged; re-run the manifest check to find which, rather than
  letting it silently refetch. The Sentinel-2/ERA5 acquisition cells are
  separately guarded by `RUN_ACQUISITION = False` flags; leave them as
  they are — the frozen bundle is the experiment's input, not a cache of it.
- **`eopf-zarr` backend unregistered** (only matters if you deliberately
  re-acquire from the EOPF STAC, which reproduction never does): you mixed
  a PyPI `pyproj` wheel into the conda environment; keep every geo
  dependency on conda-forge as pinned in `pixi.toml`.
