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

# Quickstart

All resamplers share the same two-step API: build once, resample many times.

## Setup
```{code-cell} python
import numpy as np
from healpix_resample import BilinearResampler
```

## 1. Your data: N points with lon/lat coordinates and values

```{code-cell} python

# 1. Your data: N points with lon/lat coordinates and values
lon = np.random.uniform(-180, 180, 10000)
lat = np.random.uniform(-90,  90,  10000)
val = np.sin(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))

```

## 2. Build the resampler (done once, reusable)
```{code-cell} python
nr = BilinearResampler(lon_deg=lon, lat_deg=lat, level=8)
```

## 3. Resample
```{code-cell} python
result = nr.resample(val)

print("HEALPix values shape:", result.cell_data.shape)   # (K,)
print("HEALPix cell IDs shape:", result.cell_ids.shape)  # (K,)
```

`result.cell_data` contains the values projected onto the HEALPix grid.
`result.cell_ids` contains the corresponding HEALPix cell indices (nested scheme).

## 4. Project back to original points (optional)

`invert()` reprojects the HEALPix field back to the original sample locations.
This is useful to check reconstruction quality.

```{code-cell} python
val_reconstructed = nr.invert(result.cell_data)

# Mean squared reconstruction error
mse = np.mean((val_reconstructed - val) ** 2)
print(f"Reconstruction MSE: {mse:.2e}")
```

## Batch mode

Pass a `(B, N)` array to process multiple fields at once (e.g. B time steps) **without rebuilding the operator**:

```{code-cell} python
# 10 time steps, same spatial points
val_batch = np.stack([val * (1 + 0.1 * i) for i in range(10)])  # (10, N)

result_batch = nr.resample(val_batch)
print("Batch output shape:", result_batch.cell_data.shape)  # (10, K)
```

## Next steps

- Run the {doc}`zenodo_resamplers` notebook to compare every method on the
  same frozen Sentinel-2 scene available from Zenodo.
- Use the {doc}`4resamplers` notebook as a compact synthetic API reference.
- Read the {doc}`../user-guide/index` for a detailed explanation of parameters.
