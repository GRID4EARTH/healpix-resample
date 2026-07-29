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

:::{grid-item-card} PSFResampler
:link: regrid_to_healpix_psf
:link-type: doc
Gaussian kernel + conjugate gradient. Best quality.
:::
::::

```{toctree}
:hidden:
:maxdepth: 1

regrid_to_healpix_bilinear
regrid_to_healpix_nearest
regrid_to_healpix_bicubic
regrid_to_healpix_psf
```
