# `healpix_resample.psf` (PSF / multi-point HEALPix regridding)

`healpix_resample.psf` provides **GPU-friendly sparse regridding** from unstructured geographic samples
(**longitude/latitude**) to a **subset of HEALPix pixels** at a target resolution (`nside = 2**level`).

In contrast to a pure nearest-neighbor operator, this class builds a **local, multi-point Gaussian kernel**
around each sample (a “PSF”-like footprint) and can **solve an inverse problem** to estimate a HEALPix field
that best explains the observed samples.

The implementation is designed for **large N** and **batched values** `(B, N)` on **CUDA** using **PyTorch sparse**
operators.

---

## What the class does

Given:
- sample coordinates `(lon, lat)` of shape `(N,)`
- sample values `val` of shape `(N,)` or `(B, N)`
- a HEALPix `level` (thus `nside = 2**level`)
- a neighbourhood size `Npt` (number of nearby HEALPix cells per sample)

The class:

1. **Selects nearby HEALPix cells** for each sample using local neighbourhood search (avoids building an `N × Npix`
   distance matrix).
2. Computes **Gaussian weights** as a function of distance (meters) with scale `sigma_m`.
3. Builds two sparse operators:
   - **`M`** of shape `(N, K)` : maps a HEALPix field on *K kept pixels* to sample points (forward model is via `MT` below)
   - **`MT`** of shape `(K, N)` : maps sample values back to the HEALPix subset (adjoint-like accumulation)

4. Provides:
   - **`resample(val)`**: estimate the HEALPix field (`hval`) from samples by solving a **damped least-squares**
     problem with **Conjugate Gradient (CG)**
   - **`invert(hval)`**: project a HEALPix field back to sample locations

> Note on geodesy: distances are computed in meters and the class supports the **Earth ellipsoid WGS84** and the
> **HEALPix authalic definition** through its geometry helper.

---

## Mathematical view (high level)

Let:
- `y` be the sample values `(B, N)`
- `h` be the unknown HEALPix field `(B, K)` on the kept pixels
- `M` be `(N, K)` and `MT` be `(K, N)`

A reference field is computed by weighted back-projection:
- `x_ref = y @ M`  (shape `(B, K)`)

Then the solver estimates an update `delta` by minimizing a damped normal equation:
- minimize `|| (x_ref + delta) @ MT - y ||^2 + lam * ||delta||^2`

This is solved with CG using matrix-vector products only:
- `A(v) = (v @ MT) @ M + lam * v`

Finally:
- `h = x_ref + delta`

`M` and `MT` are *not* Euclidean transposes of each other (each is normalized against a
different axis of the raw weight matrix -- per-HEALPix-cell for `M`, per-source-sample for
`MT`). They are, however, an exact adjoint pair with respect to a pair of weighted inner
products induced by those same normalizers (`Dx` on the HEALPix side, `Dy` on the source-sample
side). CG's own dot products are evaluated in the `Dx`-weighted metric (`self.cell_weight`)
rather than the Euclidean one, which is what makes CG a theoretically justified solver here
rather than an empirical choice -- see the accompanying paper for the full derivation.

---

## Conservative rebinning: weighting by pixel area

By default, `M`'s per-cell normalization treats every source sample as contributing equally,
regardless of how much physical area it actually represents. On a grid where pixel area varies
appreciably (e.g. a reduced Gaussian grid, where longitude spacing shrinks towards the poles),
this introduces a small but systematic bias: the local average is really an average over
*samples*, not over *area*.

Passing `area` bakes the per-sample pixel area directly into `M`'s construction (raw weights
`g_ik` are multiplied by the sample's area `a_i` before normalizing), making the reconstruction
an area-weighted local average instead -- a **conservative rebinning**. `MT` is unaffected: the
area factor cancels out of its own per-sample normalization algebraically, which matches the
fact that the HEALPix side needs no such weight (HEALPix cells are equal-area / iso-surface by
construction).

```python
op = PSFResampler(lon_deg=lon, lat_deg=lat, level=level, area=area)     # explicit weights
op = PSFResampler(lon_deg=lon, lat_deg=lat, level=level)                # area="auto" (default)
```

- **`area=None` (default) or `area="auto"`**: the per-sample area is estimated automatically
  from the grid geometry, assuming samples share latitude "rings" (regular lat/lon grids,
  reduced Gaussian grids such as ECMWF's N-grids). If no such structure is detected (e.g. a
  grid regular in a different projection such as UTM, or scattered points), silently falls back
  to a uniform weight of `1.0` per sample -- the same as before this option existed.
- **`area=<array>`**: use an explicit per-sample area/weight instead (own units; only ratios
  matter).

Local area-weighting alone does not guarantee *exact* global conservation -- it removes the
local bias, but small residual imbalances can remain (e.g. near the edge of the retained-cell
set). For an exact guarantee, combine it with `resample(..., conservative=True)`:

```python
res = op.resample(val, conservative=True)
```

which applies a minimum-distortion correction (a Lagrange-multiplier solve reusing the same
weighted CG machinery) so that `mean(hval)` exactly matches the true area-weighted mean of
`val` -- equivalent to conserving the area-integrated total, since HEALPix cells are
equal-area. See the accompanying paper (Section "Optional Flux-Conservation Constraint") for
the full derivation, and `docs/tutorials` for a worked example on real ERA5 data.

---

## Bounded fit: `clip`

Solving for a field sharper than the samples' PSF is a deconvolution: across a sharp,
high-contrast edge the solution over- and undershoots, which can give negative values for a
non-negative field (reflectance, concentration, precipitation) and values above the input
maximum. `lam` damps it but does not remove it.

`clip=(lo, hi)` adds hard bounds to the fit:

```python
res = op.resample(val, lam=0.1, clip=(0.0, None))          # non-negative field
res = op.resample(val, lam=0.1, clip=(None, 1.0))          # upper bound only
res = op.resample(val, lam=0.1, clip=("input", "input"))   # within the input range, per batch row
```

Each side is `None` (not constrained), a number, or `"input"` (the min, or max, of the row's
valid samples). The default `clip=None` is the unconstrained fit.

The fit then solves the same weighted least-squares problem under `lo <= h <= hi`:

    minimize || h @ MT - y ||^2_Dy + lam * || h - x_ref ||^2_Dx   subject to   lo <= h <= hi

with an active-set loop around the same weighted CG:

1. a cell at a bound whose gradient points out of the box is **held** at that bound and removed
   from the free unknowns;
2. CG solves for the free cells, with the held cells **still in the forward model** `h @ MT`;
3. cells that step out of the box are clamped and held at the next pass;
4. a held cell whose gradient points back into the box is released (Karush-Kuhn-Tucker
   conditions); the loop stops when nothing changes, or after `max_outer` passes (default 10).

Because a held cell still enters the forward model, the samples it can no longer explain leave a
residual that its free neighbours take up. A cell held at 0 pulls its neighbours down slightly,
instead of being zeroed after the fact; the result is the minimum of the constrained problem,
which clipping the unconstrained result is not.

**Lower and upper bounds are not equivalent.** A lower bound such as 0 is usually physical. An
upper bound is a stronger choice: a bright, sharp feature smoothed by the input PSF is
legitimately reconstructed above the input values. Capping it at the input maximum spreads the
energy the capped cells cannot carry over their neighbours, which widens the feature. Bounding
one side also reduces the ringing on the other side, since the oscillation that follows an
overshoot is no longer driven.

With `conservative=True`, the conservation correction is computed on the free cells only;
free cells it would take out of the box are held at their bound and the correction is computed
again. The output then satisfies both the bounds and the conservation constraint whenever the
target mean is reachable inside the box.

`ResampleResults.n_clipped_low` and `n_clipped_high` give the number of cells held at each bound
(per batch row); they are `None` without `clip`.

---

## Constructor

```python
PSFResampler(
    lon_deg, lat_deg,
    level,
    out_cell_ids=None,
    Npt=9,
    sigma_m=None,
    threshold=0.1,
    area=None,
    fill_missing_out_cells=False,
    nest=True,
    radius=6371000.0,
    ellipsoid="WGS84",
    dtype=torch.float32,
    device="cpu",
    # forwarded to KNeighborsResampler through **kwargs (see "ring_* parameters" below):
    # ring_weight, ring_search_init, ring_search_max, num_threads
    verbose=False,
)
```

### Key parameters

- **`lon_deg, lat_deg`**: sample coordinates in degrees, shape `(N,)`, `float32` or `float64`
  (the geometry is computed in `float64`).
- **`level`**: HEALPix level (`nside = 2**level`).
- **`out_cell_ids`**: restrict the output to a caller-specified subset of `level`-resolution
  cells (e.g. from `subset_for_parent_cell` when processing one coarse region at a time). See
  `fill_missing_out_cells` below for what happens when a requested cell has too little real
  kernel support. If *none* of the requested cells has support (e.g. a chunk at the edge of the
  input whose nearby samples all lie just outside it), the constructor does not raise: it builds
  an operator with `K = 0`, and `resample()` returns `cell_ids` of size 0 and `cell_data` of
  shape `(B, 0)`, with or without `fill_missing_out_cells`. Without `out_cell_ids`, an empty
  result still raises `RuntimeError("No HEALPix cell passed the threshold ...")`.
- **`Npt`**: number of neighbouring HEALPix cells used per sample.
- **`sigma_m`**: Gaussian length scale in meters.
  - If `None`, a default scale based on the HEALPix pixel area is used:
    `sigma = sqrt(4*pi / (12*4**level)) * R`.
- **`threshold`**: global pruning threshold on accumulated raw (area-independent) weights;
  pixels with too little kernel support are discarded. This reduces `K` and keeps the operator
  compact.
- **`area`**: per-sample pixel area/weight of the source grid, for conservative rebinning (see
  above). `None`/`"auto"` (default) estimates it automatically when possible, else falls back
  to uniform.
- **`fill_missing_out_cells`** (default `False`): only relevant with `out_cell_ids`. Some
  requested output cells can end up with too little real kernel support -- an empty or
  near-empty `M` column, typically because `out_cell_ids` force-included a cell the KNN search
  wouldn't have retained on its own (e.g. a river-mouth cell in an ocean model, the original
  motivating case for this fallback). Filling such a cell requires an expensive, unvectorized
  per-cell fallback search.
  - `False` (default): skip that fallback entirely. Weakly-supported requested cells come back
    as `nan` in `resample()`'s output rather than an approximate value -- correct and fast, and
    the recommended default especially when combined with `subset_for_parent_cell` (which can
    force-include many such cells at once).
  - `True`: restore the original approximate-fallback-fill behaviour (a non-NaN but approximate
    value for these cells). Opt into this only if a value is specifically needed instead of a
    gap, and the construction-time cost is acceptable.
- **`nest`**: HEALPix indexing scheme (nested if `True`).
- **`device`, `dtype`**: PyTorch device and dtype for all matrices and computations.
- **`ring_*` parameters**: control the local neighbourhood expansion strategy in the geometry helper.
  `ring_weight` is the neighbourhood used to decide which cells are kept; each sample then looks
  for its `Npt` nearest kept cells in rings growing from `ring_search_init` to `ring_search_max`
  (when `None`, `ring_search_max` is `ring_search_init + 2`, and a value below `ring_search_init`
  is raised to it). A sample that does not find `Npt` cells within `ring_search_max` rings keeps
  the ones it found; this happens when the samples are much sparser than the cells. If no sample
  reaches any kept cell, the constructor raises a `RuntimeError` instead of returning an empty
  operator.
- **`verbose`**: prints CG progress and area-estimation diagnostics.

---

## Stored attributes (after initialization)

- **`N`**: number of samples.
- **`K`**: number of kept HEALPix pixels.
- **`cell_ids`**: `(K,)` HEALPix pixel ids retained after thresholding.
- **`hi`**: `(N, Npt)` indices into `cell_ids` for each sample (the chosen neighbours).
- **`d_m`**: `(N, Npt)` distances in meters for each neighbour link.
- **`area`**: `(N,)` per-sample pixel area/weight used to build `M` (see above).
- **`cell_weight`**: `(K,)` the `Dx` weight (raw, area-weighted per-cell accumulated weight)
  CG uses for its inner product.
- **`M`**: sparse CSR `(N, K)` operator.
- **`MT`**: sparse CSR `(K, N)` operator.

---

## Methods

### `resample(val, lam=0.0, max_iter=100, tol=1e-8, x0=None, return_info=False, conservative=False, clip=None, max_outer=10)`

Estimate a HEALPix field from samples.

- **Input**:
  - `val`: `(N,)` or `(B, N)`
  - `lam`: damping / Tikhonov regularization
  - `x0`: optional initial guess for `delta` (shape `(B, K)`)
  - `conservative`: apply the exact minimum-distortion flux-conservation correction (see above)
  - `clip`: `None` or `(lo, hi)`, hard bounds imposed inside the fit (see "Bounded fit" above)
  - `max_outer`: maximum number of passes of the bounded fit
- **Output**: a `ResampleResults` with `cell_data` (`hval`, shape `(K,)` or `(B, K)`), `cell_ids`,
  CG diagnostics (`cg_residual_norms`, `cg_niters`) and, with `clip`, the number of cells held at
  each bound (`n_clipped_low`, `n_clipped_high`)

### `invert(hval)`

Project HEALPix field(s) back to sample locations.

- **Input**: `hval` `(K,)` or `(B, K)`
- **Output**: reconstructed samples `(N,)` or `(B, N)`

### `get_cell_ids()`

Return the kept HEALPix pixel ids as a NumPy array `(K,)`.

---

## Typical workflow

```python
import torch
from healpix_resample import PSFResampler

op = PSFResampler(
    lon_deg=lon, lat_deg=lat,
    level=level, Npt=9,
    device="cuda", dtype=torch.float32,
    area=area,             # optional; omit for "auto" (see above)
)

# Estimate HEALPix field on the kept pixels, exactly conservative
res = op.resample(val, lam=1e-3, max_iter=200, tol=1e-7, conservative=True)
hval = res.cell_data

# Reconstruct values at the original sample points
val_hat = op.invert(hval)

# Access the HEALPix pixel ids corresponding to hval
cell_ids = op.get_cell_ids()
```

---

## Notes and practical tips

- **Choose `Npt`** according to the desired smoothness / footprint:
  - small `Npt` → more local, less smooth
  - larger `Npt` → smoother but more compute
- **Tune `sigma_m`**:
  - smaller `sigma_m` → sharper PSF, more local influence
  - larger `sigma_m` → smoother field but can blur features
- **Use `lam`** to stabilize inversion when sampling is sparse/irregular:
  - `lam = 0` is pure least squares
  - `lam > 0` damps high-frequency or poorly constrained modes
- **Use `clip`**, not a larger `lam`, to keep the output physical at sharp edges (see
  "Bounded fit" above):
  - `lam` reduces the over- and undershoots but also smooths the whole field; `clip` only acts
    on the cells that leave the box
  - for a non-negative field, `clip=(0.0, None)` is the natural setting; add an upper bound only
    when values above it are known to be impossible
  - check `res.n_clipped_low` / `res.n_clipped_high`: a large count means the fit is strongly
    constrained, and may call for a larger `sigma_m` or `lam`
  - each pass of the bounded fit runs one CG solve, so `clip` costs up to `max_outer` times
    the unconstrained fit
- The operator only returns a **subset** of HEALPix pixels (`cell_ids`), not the full sky map.
  This is intentional for memory/performance on regional problems.

