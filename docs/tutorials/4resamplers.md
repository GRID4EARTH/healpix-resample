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
    PSFResampler,
    CellPointResampler,
    ConservativeResampler,
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