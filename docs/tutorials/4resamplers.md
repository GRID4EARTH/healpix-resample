---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# The resamplers

This notebook runs all resamplers on the same dataset so you can compare their behaviour, accuracy, and speed.

## Setup

```{code-cell} python
import numpy as np
from healpix_resample import (
    NearestResampler,
    BilinearResampler,
    BicubicResampler,
    CloughTocherResampler,
    PSFResampler,
    CellPointResampler,
    ConservativeResampler,
    BitmaskResampler,
    CategoricalResampler,
)

# Shared dataset: a small structured grid near the origin
ndata = 128
lon_grid, lat_grid = np.meshgrid(
    0.3 * np.arange(ndata) / ndata,
    0.3 * np.arange(ndata) / ndata,
)
lon = lon_grid.ravel()
lat = lat_grid.ravel()
val = lon  # simple field: value = longitude

level = 15  # nside = 32768 — high resolution
```
### `NearestResampler` 

Each point is assigned to its single nearest HEALPix cell. Fast and simple, but can produce blocky results.

```{code-cell} python
nr_nearest = NearestResampler(lon_deg=lon, lat_deg=lat, level=level)
res_nearest = nr_nearest.resample(val)

rval_nearest = nr_nearest.invert(res_nearest.cell_data)
mse_nearest = np.mean((rval_nearest - val) ** 2)
print(f"Nearest  — output cells: {res_nearest.cell_data.shape[0]}, MSE: {mse_nearest:.2e}")
```

### `BilinearResampler` 

Uses the **4 nearest cells** with distance-based weights. Smoother than nearest, good for locally grid-like data.

```{code-cell} python
nr_bili = BilinearResampler(lon_deg=lon, lat_deg=lat, level=level)
res_bili = nr_bili.resample(val, lam=0.0)

rval_bili = nr_bili.invert(res_bili.cell_data)
mse_bili = np.mean((rval_bili - val) ** 2)
print(f"Bilinear — output cells: {res_bili.cell_data.shape[0]}, MSE: {mse_bili:.2e}")
```

### `BicubicResampler`

Uses the **16 nearest cells** with a radial generalization of Keys' cubic convolution kernel. Smoother/sharper than bilinear on fields with curvature, still a fixed non-iterative interpolation (no CG solve).

```{code-cell} python
nr_bicubic = BicubicResampler(lon_deg=lon, lat_deg=lat, level=level)
res_bicubic = nr_bicubic.resample(val, lam=0.0)

rval_bicubic = nr_bicubic.invert(res_bicubic.cell_data)
mse_bicubic = np.mean((rval_bicubic - val) ** 2)
print(f"Bicubic  — output cells: {res_bicubic.cell_data.shape[0]}, MSE: {mse_bicubic:.2e}")
```

### `CloughTocherResampler`

A **Delaunay triangulation + Clough-Tocher C1 cubic** interpolant — a genuine bivariate interpolant (exact
at input points, C1 continuous across triangle edges), rather than a radial kernel sum like
`BicubicResampler`. On fields with real curvature this shows fewer small-scale artifacts than
`BicubicResampler`, because its discrete KNN neighbour-set membership can flip between adjacent output
cells; Delaunay/CT has no such failure mode. Only resamples cells whose center falls **inside the convex
hull** of the input samples (no extrapolation), and is intended for regional/local input extents (it
projects samples to a local tangent plane — see `docs/user-guide/regrid_to_healpix_clough_tocher.md`).
`invert()` is not implemented for this class (see its docstring).

```{code-cell} python
nr_ct = CloughTocherResampler(lon_deg=lon, lat_deg=lat, level=level, verbose=False)
res_ct = nr_ct.resample(val)

print(f"Clough-Tocher — output cells: {res_ct.cell_data.shape[0]} (only cells inside the sample convex hull)")
```

### Conservative mode (`BilinearResampler` / `BicubicResampler`)

Both resamplers above interpolate: each cell's value is renormalized against whichever samples happen to
link to it, which does *not* guarantee the total is preserved. Passing `area=` and
`resample(conservative=True)` switches to redistributing each sample's own value across its nearest cells
instead, so the global total is conserved exactly (see `docs/user-guide/regrid_to_healpix_bilinear.md` for
the full derivation — this is [issue #44](https://github.com/GRID4EARTH/healpix-resample/issues/44),
"conservative bi-linear is missing").

```{code-cell} python
rng = np.random.default_rng(0)
area = rng.uniform(0.5, 2.0, size=lon.shape)
target = (val * area).sum()

nr_bili_area = BilinearResampler(lon_deg=lon, lat_deg=lat, level=level, area=area)
res_bili_interp = nr_bili_area.resample(val)                          # conservative=False (default)
res_bili_cons = nr_bili_area.resample(val, conservative=True)

print(f"Input               sum(val*area)            = {target:.6f}")
print(f"Bilinear interpolating (conservative=False)   = {res_bili_interp.cell_data.sum():.6f}  (not expected to match)")
print(f"Bilinear conservative  (conservative=True)     = {res_bili_cons.cell_data.sum():.6f}  (should match exactly)")
```

### `PSFResampler`

Applies a **Gaussian PSF kernel** around each sample and solves a damped least-squares problem with Conjugate Gradient. Best reconstruction quality — especially when data is dense or the field has fine structure.

```{code-cell} python
nr_psf = PSFResampler(lon_deg=lon, lat_deg=lat, level=level, threshold=0.5, verbose=False)
res_psf = nr_psf.resample(val, lam=0.0)

rval_psf = nr_psf.invert(res_psf.cell_data)
mse_psf = np.mean((rval_psf - val) ** 2)
print(f"PSF      — output cells: {res_psf.cell_data.shape[0]}, MSE: {mse_psf:.2e}")
print(f"           CG iterations: {res_psf.cg_niters}")
```
### `CellPointResampler`

Special mode: encodes each point as a HEALPix cell ID at level 29. No interpolation — used for exact point indexing.

```{code-cell} python
nr_zuniq = CellPointResampler(lon_deg=lon, lat_deg=lat)
res_zuniq = nr_zuniq.resample(val)

rval_zuniq = nr_zuniq.invert(res_zuniq.cell_data)
max_err = np.max(np.abs(rval_zuniq - val))
print(f"Zuniq    — output cells: {res_zuniq.cell_data.shape[0]}, max error: {max_err:.2e}")
```

### `ConservativeResampler`

Bins each point into its containing HEALPix cell and accumulates an **area-weighted sum**, so the total
integrated quantity (`sum(val * area)`) is preserved exactly between the sample-space and HEALPix-cell
representations — unlike the other resamplers, which interpolate *values* rather than conserve a *flux*.

Use this when `val` is an intensive/density quantity (flux per m², temperature, ...) measured over
pixels of known — possibly non-uniform — footprint `area`. If your samples already carry an extensive,
pre-integrated quantity (a total, e.g. counts), leave `area` at its default of `1.0` and plain summation
is exactly conservative regardless of how footprint sizes vary.

```{code-cell} python
# Give samples a non-uniform footprint area to demonstrate the area weighting.
rng = np.random.default_rng(0)
area = rng.uniform(0.5, 2.0, size=lon.shape)

nr_cons = ConservativeResampler(lon_deg=lon, lat_deg=lat, level=level, area=area)
res_cons = nr_cons.resample(val)

rval_cons = nr_cons.invert(res_cons.cell_data)

flux_in = np.sum(val * area)
flux_out = np.sum(res_cons.cell_data)
flux_back = np.sum(rval_cons * area)
print(f"Conservative — output cells: {res_cons.cell_data.shape[0]}")
print(f"  sum(val*area)         = {flux_in:.6f}")
print(f"  sum(hval)              = {flux_out:.6f}  (should equal the line above)")
print(f"  sum(invert(hval)*area) = {flux_back:.6f}  (should equal the line above)")
```

### `CategoricalResampler`

For mutually-exclusive class labels (issue [#43](https://github.com/GRID4EARTH/healpix-resample/issues/43)): resamples a one-hot indicator per class through `BilinearResampler` (by default) and picks the argmax per cell. `return_scores=True` also returns a softmax-normalized per-class confidence.

```{code-cell} python
# Three "land-cover" classes split by longitude tercile.
land_cover = np.digitize(lon, np.quantile(lon, [1 / 3, 2 / 3])).astype(np.int64)

nr_cat = CategoricalResampler(lon_deg=lon, lat_deg=lat, level=level)
res_cat = nr_cat.resample(land_cover, return_scores=True)

winner_score = res_cat.scores.max(axis=0)  # (K,) softmax score of the winning class per cell
print(f"Categorical — output cells: {res_cat.cell_data.shape[0]}, classes found: {res_cat.classes}")
print(f"Winning-class softmax score range: [{winner_score.min():.3f}, {winner_score.max():.3f}]")
```

### `BitmaskResampler`

For independent, co-occurring boolean flags packed into an integer (e.g. an 8-bit quality/cloud mask): each bit is resampled and thresholded independently, then the bits are recombined.

```{code-cell} python
bit0 = (lon > np.median(lon)).astype(np.int64)   # e.g. "cloud" flag
bit1 = (lat > np.median(lat)).astype(np.int64)   # e.g. "cloud-shadow" flag, independent of bit0
quality_mask = bit0 | (bit1 << 1)

nr_bitmask = BitmaskResampler(lon_deg=lon, lat_deg=lat, level=level, n_bits=2)
res_bitmask = nr_bitmask.resample(quality_mask)

print(f"Bitmask — output cells: {res_bitmask.cell_data.shape[0]}")
print(f"Distinct output values: {sorted(set(res_bitmask.cell_data.tolist()))}  (subset of [0, 1, 2, 3])")
```