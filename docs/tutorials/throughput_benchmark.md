# Running only the throughput benchmark

`notebooks/throughput_scaling_benchmark.ipynb` measures operator-construction
and solve throughput as a function of batch size, under three regimes: one
operator **shared** across a batch of repeated acquisitions of the same
footprint, the same footprint with the operator **rebuilt** every item, and a
batch of **distinct** geographic patches. It is the paper's batch-scaling
benchmark (supplement), and it can be run on its own — none of the other
experiment notebooks are prerequisites.

The commands below are documentation only; they do not run during the
documentation build (the benchmark needs a CUDA GPU to be meaningful).

## Setup

```{code-block} bash
git clone https://github.com/GRID4EARTH/healpix-resample.git
cd healpix-resample
pixi install -e notebooks
pixi run -e notebooks jupyter lab
```

## Choosing the patch pool

The benchmark selects its patch pool automatically, in order of preference:

1. **Archived Sentinel-2 regions** (the paper's configuration): the 40 real
   region patches of [Zenodo record 22210945](https://doi.org/10.5281/zenodo.22210945).
   Install them by running `notebooks/load_data_in_zenodo.ipynb` first.
2. **Esri-derived cache**: the 360 regenerated multi-region patches, if
   present (see the [full reproduction tutorial](reproduce_paper.md)).
3. **Synthetic fallback**: deterministic synthetic textures on the versioned
   site-manifest geometries — *no download needed at all*.

Operator-construction cost depends on patch geometry only, so the scaling
*shape* is meaningful in every tier; to reproduce the paper's numbers, use
tier 1. The chosen pool is recorded as `patch_pool_source` in
`notebooks/tables/throughput_scaling_environment.csv`.

## Running

Open `throughput_scaling_benchmark.ipynb` and run it top to bottom. Two
cells deserve attention before the sweep:

- the **cost-estimator cell** prints a runtime estimate extrapolated over
  `BATCH_SIZES` and `N_BATCH_REPEATS`, so you can decide whether to shorten
  `BATCH_SIZES` first (the full 1–2048 sweep takes a few hours on a
  datacenter GPU);
- `N_BATCH_REPEATS = 1` reproduces the paper's single descriptive sweep;
  raise it for repeat-to-repeat uncertainty at the cost of runtime.

## Outputs and what to expect

The run writes, under `notebooks/`:

- `tables/throughput_scaling_by_batch.csv` — one row per regime × batch size,
- `tables/throughput_scaling_by_item.csv` — per-item build and solve timings,
- `tables/throughput_scaling_environment.csv` — hardware and configuration snapshot,
- `figures/throughput_scaling_curve.pdf` and `figures/throughput_scaling_memory.pdf`.

Timing values are hardware-dependent: expect the **ratios**, not the
absolute times, to reproduce. On the paper's reference system (NVIDIA L4,
Sentinel-2 pool), sharing one operator across a batch of 2048 repeated
acquisitions reaches 0.075 s/patch against ~0.86 s/patch for both rebuild
regimes — an ~11.5× amortized speedup — while batches of distinct
geographic patches gain nothing, since each needs its own operator.

Two reading rules from the notebook's reporting guidance: the
`different_pointing` regime samples the pool *with replacement* once the
batch exceeds the pool size (check `unique_patches_used` before citing such
a point), and this is a single-GPU sequential benchmark — do not
extrapolate to multi-GPU or parallelized deployments.
