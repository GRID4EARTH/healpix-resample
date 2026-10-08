# Tutorials

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Real Sentinel-2 resampler tour
:link: zenodo_resamplers
:link-type: doc
Download one small frozen Zenodo archive and compare every resampler on the same real Sentinel-2 scene.
:::

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc
Learn to resample scattered point data onto a HEALPix grid in four steps.
:::

:::{grid-item-card} Synthetic resampler reference
:link: 4resamplers
:link-type: doc
Compact synthetic examples for every continuous, conservative, categorical, and bitmask resampler.
:::

:::{grid-item-card} Parent-cell subsetting
:link: 5parent_cell_subsetting
:link-type: doc
Process one coarse HEALPix cell at a time and verify the reassembled result matches a global run.
:::

:::{grid-item-card} Reproduce the paper
:link: reproduce_paper
:link-type: doc
Rebuild every figure and table of the healpix-resample paper from the frozen Zenodo bundle.
:::

:::{grid-item-card} Throughput benchmark
:link: throughput_benchmark
:link-type: doc
Run only the batch-scaling throughput benchmark — with the archived Sentinel-2 pool or a zero-download synthetic fallback.
:::
::::

```{toctree}
:hidden:
:maxdepth: 1

zenodo_resamplers
quickstart
4resamplers
5parent_cell_subsetting
reproduce_paper
throughput_benchmark
```
