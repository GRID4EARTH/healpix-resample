# Reproducing the paper's figures and tables

This tutorial walks through reproducing **every figure and table** of the
*healpix-resample* paper (main text and supplement) from a clean machine,
using the public repository and the frozen Zenodo data bundle. It is a
condensed version of [`REPRODUCING.md`](https://github.com/GRID4EARTH/healpix-resample/blob/main/REPRODUCING.md)
at the repository root, which remains the authoritative recipe.

The pages of this tutorial are documentation only: none of the commands run
during the documentation build, because the full reproduction needs a CUDA
GPU and a one-time ~1.2 GB download.

## Requirements

- Linux x86-64, an NVIDIA GPU with a working CUDA driver (any GPU with
  ≥ 4 GB free memory reproduces the *results*; only timings change), ~6 GB
  of free disk, and [pixi](https://pixi.sh).

## 1. Code at the paper's commit

```{code-block} bash
git clone https://github.com/GRID4EARTH/healpix-resample.git
cd healpix-resample
git checkout jstars-lastreview-before-submission
pixi install -e notebooks
pixi run -e notebooks jupyter lab
```

The authoritative code version is recorded *inside the data bundle*: after
step 2, `git_commit.txt` at the repository root gives the exact commit the
archives were packed against.

## 2. Install the frozen data

Run `notebooks/load_data_in_zenodo.ipynb` top to bottom. It downloads the
two archives of [Zenodo record 22210945](https://doi.org/10.5281/zenodo.22210945),
verifies their published size and MD5, and extracts them at the repository
paths the notebooks expect. Then verify:

```{code-block} bash
pixi run -e notebooks python notebooks/build_data_manifest.py --check \
    --doi 10.5281/zenodo.22210945
```

This must report all **45 archived assets available**. The 364 Esri-derived
assets are listed as regenerable and absent — expected on a fresh install.

## 3. Regenerate the Esri-derived textures

The Esri World Imagery derivatives are **not redistributed** (Esri's terms
cover static map images, not machine-readable derived datasets). They
regenerate locally and deterministically from the pinned World Imagery
**Wayback release 26334 (2026-08-05)**, an immutable dated snapshot, so
every user fetches identical tiles whenever this step runs.

`test-resample-paper.ipynb` (scene textures) and
`multi_patch_latitude_validation.ipynb` (the 360 patches) handle this
themselves: they default to `OFFLINE = False` and their acquisition cells
are cache-first, so the first top-to-bottom run downloads the textures once
and later runs touch nothing. After both have run, re-run the manifest
check: it must now also report all **364 regenerated assets** available with
matching SHA-256. (Set `OFFLINE = True` in the setup cells to *enforce*
network-free reruns afterwards.)

## 4. Run the experiment notebooks

Each notebook runs top to bottom with a fresh kernel; after step 3, the acquisition caches are complete and no notebook fetches anything:

| # | Notebook | Reproduces |
|---|----------|------------|
| 1 | `test-resample-paper.ipynb` | Four-scene synthetic results, round-trip, estimand and hold-out controls, spectral diagnostics |
| 2 | `multi_patch_latitude_validation.ipynb` | 40-region synthetic validation |
| 3 | `real_groundtruth_downscale.ipynb` | Four-scene reduced-resolution results, response-width sweep |
| 4 | `real_groundtruth_multiregion.ipynb` | Main real-data result: 40-region reduced-resolution validation and ablations |
| 5 | `noise_sensitivity.ipynb` | Noise × damping, noise × PSF-mismatch, geolocation-jitter sweeps |
| 6 | `throughput_scaling_benchmark.ipynb` | Batch-scaling benchmark (see the [dedicated tutorial](throughput_benchmark.md)) |
| 7 | `conservative_flux_ERA5.ipynb` | ERA5 flux-conservation results |

Each experiment notebook ends by calling
`notebooks/publish_paper_assets.py`, which copies every figure and table CSV
cited by the paper into `tex/`. After all notebooks have run, that call must
report no `missing-source` and no `would-update` entries — the machine check
that the paper's assets match your run.

## 5. Compile and compare

```{code-block} bash
cd tex
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
pdflatex supplement.tex && bibtex supplement && pdflatex supplement.tex && pdflatex supplement.tex
```

Key checkpoints (from the CSVs under `notebooks/tables/`): 40-region
synthetic recovery of 15.4–19.6 % by scene class with 100 % win fraction;
lowest RMSE in all 40 regions of the real reduced-resolution validation
(sign test p = 1.8×10⁻¹²). Statistical quantities use fixed seeds and
reproduce exactly; GPU floating-point outputs may differ in the last digit
across CUDA versions. See `REPRODUCING.md` for the full list and
troubleshooting.
