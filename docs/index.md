# healpix-resample

**Reproject unstructured lon/lat data onto a HEALPix grid — fast, sparse, GPU-ready.**

`healpix-resample`  takes a cloud of geographic measurements (each with a longitude, a latitude, and a value) and maps them onto a uniform spherical grid called HEALPix. It is designed for large datasets and runs efficiently on both CPU and GPU via PyTorch sparse operators.

```
lon/lat points (N)  →  sparse operator M  →  HEALPix cells (K)
```

HEALPix is a standard equal-area pixelization of the sphere used in astrophysics, climatology and oceanography. The resolution is controlled by a **level parameter**: nside = 2**level, so level=10 gives ~12 million cells.

## Where to go next

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Start with a real Sentinel-2 scene from Zenodo and compare every resampler.
:::

:::{grid-item-card} User Guide
:link: user-guide/index
:link-type: doc
In-depth explanation of each resampler and key parameters.
:::

:::{grid-item-card} API Reference
:link: api
:link-type: doc
Complete documentation for every public class and function.
:::

:::{grid-item-card} Installation
:link: installation
:link-type: doc
How to install the package in your environment.
:::

::::

```{toctree}
---
maxdepth: 2
caption: User guide
hidden: true
---
installation
user-guide/index
tutorials/index
```

```{toctree}
---
maxdepth: 2
caption: Reference
hidden: true
---
api
terminology
```
