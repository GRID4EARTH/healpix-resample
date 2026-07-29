# `healpix_resample.bicubic` (radial cubic-convolution HEALPix regridding)

`healpix_resample.bicubic.BicubicResampler` maps unstructured `(lon_deg, lat_deg, val)` samples onto a
HEALPix grid using a **radial generalization of Keys' cubic convolution kernel** — the same interpolation
kernel behind `cv2.INTER_CUBIC` / `PIL.Image.BICUBIC` on regular pixel grids, adapted to scattered data.

It sits between `NearestResampler` / `BilinearResampler` (fixed, non-iterative, few neighbours) and
`PSFResampler` (iterative CG deconvolution): a fixed, non-iterative local interpolation using more
neighbours than bilinear, for users who want smoother results than bilinear without paying for a full
CG solve.

> Note on geodesy: the package manages the **HEALPix authalic definition** and the Earth ellipsoid with
> **WGS84** through its geometry helper (`healpix_geo`).

---

## What "bicubic" means here

This package operates on **unstructured** samples found via a distance-based Gaussian-neighbourhood
search (`healpix_weighted_nearest` in `knn.py`), not a structured pixel grid with clean `(fx, fy)`
fractional offsets — so a textbook 2D bicubic convolution kernel doesn't directly apply. What *does*
generalize cleanly to scattered data is the **radial profile** of Keys' cubic convolution kernel: replace
Euclidean pixel distance with the geodesic distance `self.d_m` and length scale `self.sigma_m` already used
by every other resampler in this package, and evaluate the standard piecewise-cubic weight (`a = -0.5`, the
common default matching `PIL`/`cv2`):

```
u = d / sigma
w(u) = (a+2)|u|^3 - (a+3)|u|^2 + 1        for |u| <= 1
     = a|u|^3 - 5a|u|^2 + 8a|u| - 4a       for 1 < |u| < 2
     = 0                                   for |u| >= 2
```

`Npt = 16` is the default neighbourhood size — the natural analogue of the classic 4x4 bicubic stencil on
a structured grid, since Keys' kernel has support `|u| < 2` (roughly twice the reach of bilinear's
`|u| < 1`-ish support).

Unlike the nonnegative Gaussian/inverse-distance weights used elsewhere in this package, Keys' kernel is
**signed** — it goes negative for `1 < |u| < 2`, which is what gives cubic convolution its sharpening
property relative to bilinear. Two consequences:

- The per-cell/per-sample weight sums used to normalize the sparse operators can land close to zero from
  cancellation between the positive and negative lobes. `BicubicResampler.comp_matrix()` floors these sums
  (relative to the unsigned accumulated weight) before dividing, rather than dropping affected cells.
- `invert()` can genuinely overshoot/ring outside the local sample-value range — this is expected
  cubic-convolution behaviour, not a bug.

---

## Class hierarchy and construction

`BicubicResampler` is a `KNeighborsResampler` subclass, exactly like `BilinearResampler`: it overrides
`__init__` (to fix `Npt=16`, auto-correct `ring_search_max`, and accept `area=`), `comp_matrix()` (to build
the sparse operators from the cubic-convolution weight instead of the Gaussian/IDW weight, plus a
conservative-mode operator, see below), and `resample()` (to add `conservative=True`, see below). `invert()`
is inherited unchanged from `KNeighborsResampler`.

```python
from healpix_resample import BicubicResampler

op = BicubicResampler(
    lon_deg=lon,       # (N,) sample longitudes, degrees
    lat_deg=lat,       # (N,) sample latitudes, degrees
    level=level,       # HEALPix level, nside = 2**level
    device="cuda",
    dtype=torch.float32,
    nest=True,
)
```

### Key parameters

- **`lon_deg, lat_deg`**: unstructured sample coordinates in degrees, shape `(N,)`.
- **`level`**: HEALPix level (`nside = 2**level`).
- **`Npt`**: number of HEALPix neighbours per sample (default 16).
- **`nest`**: HEALPix indexing scheme.
- **`device`, `dtype`**: PyTorch placement and numerical type.
- **`threshold`**: minimum accumulated (unsigned, Gaussian-weighted) support for a HEALPix cell to be kept
  — same meaning as for every other resampler in this package.
- **`area`**: per-sample pixel area/weight, shape `(N,)`. Only used by `resample(conservative=True)` (see
  below); ignored otherwise. Defaults to `1.0` for every sample.

---

## Stored attributes

After construction, `BicubicResampler` exposes the same attributes as every `KNeighborsResampler`
subclass:

- **`N`**: number of samples.
- **`K`**: number of kept HEALPix cells (`cell_ids`).
- **`cell_ids`**: `(K,)` HEALPix cell ids retained.
- **`hi`**: `(N, Npt)` cell indices per sample (into `cell_ids`).
- **`d_m`**: `(N, Npt)` geodesic distances (metres) to those cells.
- **`M`**: sparse CSR `(N, K)` operator (per-cell-normalized), `hval = y @ M`.
- **`MT`**: sparse CSR `(K, N)` operator, `val_hat = hval @ MT`.
- **`M_cons`**: sparse CSR `(N, K)` operator (per-sample-normalized — a partition of unity per sample, up to
  the signed-kernel floor caveat below), used by `resample(conservative=True)`.
- **`area`**: `(N,)` per-sample area/weight (see above).

---

## Methods

### `resample(val, conservative=False)`

Sample space → HEALPix cell space.

- **`conservative=False`** (default): `hval = val @ self.M` (per-cell-normalized weights) — smooth, but not
  exactly mass-conserving.
- **`conservative=True`**: `hval = (val * self.area) @ self.M_cons` — each sample's own (area-weighted)
  value is redistributed across its 16 nearest cells using weights normalized so each *sample's* own
  weights sum to 1, guaranteeing `sum_k hval[k] == sum_i (valid i) val[i] * area[i]` exactly — see
  "Conservative mode" below.

Accepts `val` of shape `(N,)` or `(B, N)`, `np.ndarray` or `torch.Tensor`; returns the same type/batch shape
it was given, wrapped in a `ResampleResults(cell_data, cell_ids)`.

### `invert(hval)`

HEALPix cell space → sample space: `val_hat = hval @ self.MT` (inherited, no CG; unaffected by
`conservative`/`area`). Same `(K,)`/`(B, K)` and NumPy/Torch symmetry as `resample`.

---

## Conservative mode (`area=`, `resample(conservative=True)`)

Added for [issue #44](https://github.com/GRID4EARTH/healpix-resample/issues/44) ("conservative bi-linear is
missing"), and applied to `BicubicResampler` for consistency with `BilinearResampler`. Same construction as
the bilinear case (see `docs/user-guide/regrid_to_healpix_bilinear.md` for the full derivation): `M_cons`
reuses the exact weights already computed for `MT` (per-sample-normalized) under `M`'s `(sample, cell)`
index layout, so each sample's own contribution sums to 1 across the cells it links to and no value is
invented or lost globally.

**Caveat specific to this class's signed kernel**: unlike `BilinearResampler`'s non-negative
inverse-distance weights, Keys' cubic kernel is signed, so `M_cons`'s per-sample rows only sum to *exactly*
1 for samples whose `norm_row` wasn't floored by `_floor_signed` in `comp_matrix()` — true for the vast
majority of well-conditioned samples. For the rare, pathologically-cancelled sample that does hit the
floor, that sample's own contribution to the conservation identity is only approximate, not bit-exact —
the same accepted trade-off `_floor_signed` already makes for ordinary (non-conservative) interpolation.

NaN handling under `conservative=True` is the same as `BilinearResampler`'s: a NaN sample's value and area
are both excluded, so the conservation identity holds over exactly the valid samples; a batch row where
every sample is NaN comes back entirely `nan`.

---

## Notes and practical tips

- Bicubic gives no benefit over bilinear on a perfectly linear field (both interpolate a plane exactly);
  its advantage shows up on fields with curvature.
- Because the kernel is signed, `invert()` can produce values slightly outside the local sample range
  (overshoot/ringing) — expected, not a bug.
- If your data is very sparse relative to the HEALPix `level` (few samples within `2*sigma` of a cell),
  prefer `BilinearResampler` or `NearestResampler`, which need fewer nearby samples to produce a
  well-conditioned result.
