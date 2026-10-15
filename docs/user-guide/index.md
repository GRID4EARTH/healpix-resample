# User Guide

Everything you need to understand how `healpix-resample` works and how to configure it.

## Output format

All resamplers return a `ResampleResults` dataclass:

| Field                | Shape              | Description                                      |
|----------------------|--------------------|--------------------------------------------------|
| `cell_data`          | `(K,)` or `(B, K)` | Resampled values on HEALPix cells                |
| `cell_ids`           | `(K,)`             | HEALPix cell indices (nested scheme)             |
| `cg_residual_norms`  | `(iters,)`         | CG convergence history *(PSFResampler only)*     |
| `cg_niters`          | scalar             | Number of CG iterations *(PSFResampler only)*    |

## Key parameters (all resamplers)

| Parameter     | Default   | Description                                            |
|---------------|-----------|--------------------------------------------------------|
| `level`       | —         | HEALPix resolution. `nside = 2**level`                 |
| `threshold`   | `0.1`     | Minimum weight sum to keep a cell                      |
| `sigma_m`     | auto      | Gaussian scale in metres (defaults to pixel size)      |
| `out_cell_ids`| `None`    | Restrict output to a specific subset of cells          |
| `ellipsoid`   | `"WGS84"` | Geodetic ellipsoid                                     |
| `device` | auto | `"cpu"` or `"cuda"`. Auto-detected if not set. |
| `dtype` | `float64` | PyTorch dtype. Use `float32` for speed, `float64` for precision. |

### When no requested cell has support

With `out_cell_ids`, it can happen that none of the requested cells passes `threshold`, typically
for a chunk at the edge of the input whose nearby samples all lie just outside it. This is not an
error: the KNN-based resamplers (`BilinearResampler`, `BicubicResampler`, `PSFResampler`) are built
with `K = 0` and `resample()` returns an empty result, `cell_ids` of size 0 and `cell_data` of shape
`(B, 0)`, so a chunked caller simply keeps its fill value for that chunk. `NearestResampler` instead
fills every requested cell with its nearest sample (its `out_cell_ids` fallback).

Without `out_cell_ids`, every sample supports its own cell, so an empty result means the
configuration is wrong: the constructor raises `RuntimeError("No HEALPix cell passed the
threshold ...")`; lower `threshold` or widen the neighbourhood rings.

## GPU usage

Pass `device="cuda"` to any resampler to run on GPU. The operators `M` and `MT` stay in GPU memory between calls to `resample()`, so the cost is paid only once at construction time.

```python
from healpix_resample import BilinearResampler
import torch

nr = BilinearResampler(
    lon_deg=lon, lat_deg=lat,
    level=13,
    device="cuda",
    dtype=torch.float32,
)

result = nr.resample(val)  # runs on GPU
```

## What next

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} BilinearResampler
:link: regrid_to_healpix_bilinear
:link-type: doc
4-point weighted interpolation. 
:::

:::{grid-item-card} NearestResampler
:link: regrid_to_healpix_nearest
:link-type: doc
Fastest option. One cell per sample, no interpolation.
:::

:::{grid-item-card} BicubicResampler
:link: regrid_to_healpix_bicubic
:link-type: doc
16-point radial cubic-convolution interpolation. Smoother than bilinear, no CG solve.
:::

:::{grid-item-card} CloughTocherResampler
:link: regrid_to_healpix_clough_tocher
:link-type: doc
Delaunay triangulation + Clough-Tocher C1 cubic. Genuine bivariate interpolant, no extrapolation outside the sample convex hull.
:::

:::{grid-item-card} PSFResampler
:link: regrid_to_healpix_psf
:link-type: doc
Gaussian kernel + conjugate gradient. Best quality.
:::

:::{grid-item-card} OverlapConservativeResampler
:link: regrid_to_healpix_overlap_conservative
:link-type: doc
First-order overlap-area conservative remapping from a lat/lon grid (the ESMF/xESMF formulation), and how it differs from the other two conservative modes.
:::

:::{grid-item-card} Parent-cell subsetting
:link: regrid_to_healpix_parent_cell_subsetting
:link-type: doc
Process one coarse HEALPix cell at a time for large-scale/global datasets.
:::

:::{grid-item-card} Mask-like / categorical data
:link: regrid_to_healpix_mask
:link-type: doc
BitmaskResampler (independent flags) and CategoricalResampler (mutually-exclusive classes, argmax).
:::
::::

```{toctree}
:hidden:
:maxdepth: 1

regrid_to_healpix_bilinear
regrid_to_healpix_nearest
regrid_to_healpix_bicubic
regrid_to_healpix_clough_tocher
regrid_to_healpix_psf
regrid_to_healpix_overlap_conservative
regrid_to_healpix_parent_cell_subsetting
regrid_to_healpix_mask
```
