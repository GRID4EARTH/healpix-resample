# `healpix_resample.nearest` (nearest-neighbour HEALPix regridding)

`healpix_resample.nearest` provides the **simplest and fastest** mapping from unstructured geographic samples
(**longitude/latitude**) to a **subset of HEALPix pixels** at a target resolution (`nside = 2**level`).

Each HEALPix cell takes the value of **its nearest sample**. No sparse matrix is built: the class stores one
source index per cell, so `resample()` is a direct gather. The implementation is designed for **large N**,
optional **batched data** `(B, N)`, and **CUDA acceleration**.

> Note on geodesy: the package is intended to manage the **HEALPix authalic definition** and the Earth ellipsoid
> with **WGS84** through its geometry helper.

---

## What the class does

Given:
- sample coordinates `(lon, lat)` of shape `(N,)`
- sample values `val` of shape `(N,)` or `(B, N)`
- a HEALPix `level` (thus `nside = 2**level`)

The class:

1. Runs the shared KNN search: each sample finds its `Npt` nearest kept HEALPix cells (default `Npt = 9`, so
   that every cell is reached even when the output grid is finer than the input).
2. For each kept cell, keeps the sample with the largest Gaussian weight, i.e. the nearest one: `hi_k` `(K,)`.
3. With `out_cell_ids`, fills the requested cells the KNN rings did not reach with their nearest sample (see
   `max_distance_m` below).

Nearest-neighbour is equivalent to a **piecewise-constant** remapping on the HEALPix grid.

---

## Constructor

```python
from healpix_resample import NearestResampler

op = NearestResampler(
    lon_deg=lon,
    lat_deg=lat,
    level=level,
    Npt=9,
    out_cell_ids=None,
    max_distance_m=None,
    device="cuda",
    dtype=torch.float32,
    nest=True,
    verbose=False,
)
```

### Key parameters

- **`lon_deg, lat_deg`**: sample coordinates in degrees, shape `(N,)`. `float32` and `float64` are both
  accepted; the geometry is computed in `float64`.
- **`level`**: HEALPix level (`nside = 2**level`).
- **`Npt`**: number of HEALPix cells each sample reaches in the KNN search (default 9).
- **`out_cell_ids`**: restrict the output to a caller-specified subset of `level` cells (e.g. one chunk).
  Requested cells the KNN rings did not reach are filled with their nearest sample.
- **`max_distance_m`** (default `None`): only used with `out_cell_ids`. A requested cell is filled by that
  fallback only if its nearest sample lies within `max_distance_m` metres of the cell centre; otherwise it is
  left out of the result, like the cells the interpolating resamplers cannot compute. With `None`, every
  requested cell is filled, however far its nearest sample is.
- **`nest`**: HEALPix indexing scheme (nested if `True`).
- **`device`, `dtype`**: PyTorch device and dtype.
- **`verbose`**: diagnostic prints, including how many requested cells the fallback filled or left out.

### Chunked processing and `max_distance_m`

When a dataset is processed chunk by chunk with the chunk's cells as `out_cell_ids`, a chunk that is only
partly covered by the input has requested cells far from any sample. Without `max_distance_m`, they are all
filled from the samples at the edge of the input, which extends those values over the whole chunk. Set
`max_distance_m` to a few input pixel sizes to leave such cells out, so that they keep the caller's fill
value:

```python
op = NearestResampler(lon_deg=lon, lat_deg=lat, level=20, out_cell_ids=chunk_cells, max_distance_m=60.0)
res = op.resample(mask)   # res.cell_ids holds only the cells within 60 m of a sample
```

---

## Stored attributes

After initialization:

- **`N`**: number of samples.
- **`K`**: number of output HEALPix cells.
- **`cell_ids`**: `(K,)` HEALPix cell ids, sorted.
- **`hi_k`**: `(K,)` index of the nearest sample for each cell.

---

## Methods

### `resample(val)`

Samples → HEALPix cells: `hval[:, k] = val[:, hi_k[k]]`.

- **Input**: `val` `(N,)` or `(B, N)`
- **Output**: a `ResampleResults` with `cell_data` `(K,)` or `(B, K)` and `cell_ids` `(K,)`

### `invert(hval)`

HEALPix cells → samples: each sample gets the mean of the cells it is the nearest sample of.

- **Input**: `hval` `(K,)` or `(B, K)`
- **Output**: values at samples `(N,)` or `(B, N)`

---

## Typical workflow

```python
import torch
from healpix_resample import NearestResampler

op = NearestResampler(lon_deg=lon, lat_deg=lat, level=level, device="cuda", dtype=torch.float32)

res = op.resample(val)          # samples -> HEALPix cells
hval, cell_ids = res.cell_data, res.cell_ids

val_hat = op.invert(hval)       # HEALPix cells -> samples
```

---

## Notes and practical tips

- Nearest-neighbour is **fast** and **memory-light**, but it can be **blocky** and introduce discretization artifacts.
- It is the right choice for masks, flags and classification maps, whose values must not be interpolated.
- Like the other operators, it returns a **subset** of HEALPix pixels (`cell_ids`) rather than a full-sky map.
