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

`BicubicResampler` is a `KNeighborsResampler` subclass, exactly like `BilinearResampler`: it only overrides
`__init__` (to fix `Npt=16` and auto-correct `ring_search_max`, following the same pattern as
`NearestResampler`) and `comp_matrix()` (to build the sparse operators from the cubic-convolution weight
instead of the Gaussian/IDW weight). `resample()` and `invert()` are inherited unchanged from
`KNeighborsResampler`.

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

---

## Stored attributes

After construction, `BicubicResampler` exposes the same attributes as every `KNeighborsResampler`
subclass:

- **`N`**: number of samples.
- **`K`**: number of kept HEALPix cells (`cell_ids`).
- **`cell_ids`**: `(K,)` HEALPix cell ids retained.
- **`hi`**: `(N, Npt)` cell indices per sample (into `cell_ids`).
- **`d_m`**: `(N, Npt)` geodesic distances (metres) to those cells.
- **`M`**: sparse CSR `(N, K)` operator, `hval = y @ M`.
- **`MT`**: sparse CSR `(K, N)` operator, `val_hat = hval @ MT`.

---

## Methods

### `resample(val)`

Sample space → HEALPix cell space: `hval = val @ self.M` (inherited from `KNeighborsResampler`, no CG).
Accepts `val` of shape `(N,)` or `(B, N)`, `np.ndarray` or `torch.Tensor`; returns the same type/batch shape
it was given, wrapped in a `ResampleResults(cell_data, cell_ids)`.

### `invert(hval)`

HEALPix cell space → sample space: `val_hat = hval @ self.MT` (inherited, no CG). Same `(K,)`/`(B, K)` and
NumPy/Torch symmetry as `resample`.

---

## Notes and practical tips

- Bicubic gives no benefit over bilinear on a perfectly linear field (both interpolate a plane exactly);
  its advantage shows up on fields with curvature.
- Because the kernel is signed, `invert()` can produce values slightly outside the local sample range
  (overshoot/ringing) — expected, not a bug.
- If your data is very sparse relative to the HEALPix `level` (few samples within `2*sigma` of a cell),
  prefer `BilinearResampler` or `NearestResampler`, which need fewer nearby samples to produce a
  well-conditioned result.
