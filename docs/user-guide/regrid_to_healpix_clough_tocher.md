# `healpix_resample.clough_tocher` (Delaunay / Clough-Tocher C1 cubic regridding)

`healpix_resample.clough_tocher.CloughTocherResampler` maps unstructured `(lon_deg, lat_deg, val)` samples
onto a HEALPix grid using a **Delaunay triangulation of the input samples, with a Clough-Tocher C1 cubic
Bezier macro-element fitted per triangle** — the same construction behind
`scipy.interpolate.griddata(method='cubic')`. Unlike `BicubicResampler`, this is a genuine bivariate
interpolant (exact at the input sample points), not a radial kernel sum.

> Note on geodesy: the package manages the **HEALPix authalic definition** and the Earth ellipsoid with
> **WGS84** through its geometry helper (`healpix_geo`).

---

## Why this exists

Comparing `BicubicResampler` against `scipy.interpolate.griddata(method='cubic')` on the same smooth,
curved synthetic field shows `griddata` produces visibly fewer small-scale artifacts. The likely cause:
`BicubicResampler`'s `Npt=16` nearest-neighbour search is **discrete** — which cells belong to a given
sample's neighbourhood can flip between one output cell and its immediate neighbour, causing small jumps
even though the weight formula itself is continuous for a *fixed* neighbour set. A Delaunay/Clough-Tocher
construction doesn't have this failure mode: adjacent triangles share two vertices, and the macro-element
is built, by construction, to be C1 continuous (value *and* gradient) across shared edges.

## What "Clough-Tocher" means here

1. **Local gnomonic (central) projection.** Delaunay triangulation and Clough-Tocher patches are inherently
   planar constructions, so samples are projected from `(lon_deg, lat_deg)` to a local tangent plane about
   the centroid of the sample set. Gnomonic projection is used specifically because it maps great-circle
   geodesics to straight lines, so triangle edges in the projected plane correspond to true geodesics on
   the sphere. **This distorts for large angular extents** — this resampler is intended for
   **regional/local** input extents (a single patch, not a global dataset in one construction call);
   `__init__` raises if any sample falls outside the projection's well-behaved hemisphere around the
   centroid. See `docs/user-guide/regrid_to_healpix_parent_cell_subsetting.md` for processing a large/global
   dataset one local patch at a time.
2. **Delaunay triangulation** (`scipy.spatial.Delaunay`, CPU/NumPy — the one unavoidable non-torch step;
   no mature GPU-native Delaunay library exists) of the projected samples.
3. **Gradient estimation**: a local least-squares plane fit at each triangulation vertex, using that
   vertex's direct Delaunay 1-ring neighbours. Precomputed as two sparse `(N, N)` matrices `Gx`, `Gy`
   (geometry only). This is **not** claimed to be bit-identical to
   `scipy.interpolate.CloughTocher2DInterpolator`, which uses a different (Nielson-style) gradient
   estimator — it's *a* correct, standard Clough-Tocher construction, not a scipy clone.
4. **Output validity = inside the convex hull.** Like `scipy.interpolate.griddata`, this resampler does
   not extrapolate: a HEALPix cell is only kept in `cell_ids` if its projected center falls inside the
   Delaunay triangulation.
5. **Per-triangle Clough-Tocher macro-element**: each triangle is split into 3 micro-triangles by
   connecting its vertices to the centroid; each micro-triangle carries a cubic Bezier patch built from
   corner values/gradients and a construction that guarantees C1 continuity across the internal split
   edges and across each triangle's shared edges with its neighbours.

Everything except the Delaunay triangulation itself is precompiled into one sparse `(N, K)` matrix `M`, so
`resample(val)` is a single batched sparse matmul (`hval = y @ M`), matching every other resampler in this
package.

---

## Class hierarchy and construction

`CloughTocherResampler` does **not** subclass `KNeighborsResampler` — that base class's KNN/Gaussian-
threshold geometry model (`Npt`, `sigma_m`, `threshold`) doesn't apply to a triangulation-based
construction.

```python
from healpix_resample import CloughTocherResampler

op = CloughTocherResampler(
    lon_deg=lon,       # (N,) sample longitudes, degrees -- regional/local extent
    lat_deg=lat,       # (N,) sample latitudes, degrees
    level=level,       # HEALPix level, nside = 2**level
    device="cuda",
    dtype=torch.float64,
    nest=True,
)
```

### Key parameters

- **`lon_deg, lat_deg`**: unstructured sample coordinates in degrees, shape `(N,)`. Must span a
  regional/local extent (see above).
- **`level`**: HEALPix level (`nside = 2**level`).
- **`nest`, `ellipsoid`**: same meaning as every other resampler.
- **`device`, `dtype`**: PyTorch placement and numerical type for `Gx`, `Gy`, `M`.
- **`out_cell_ids`**: optional caller-supplied subset of output cell ids, intersected with the
  convex-hull criterion.
- **`candidate_ring_expand`**: how many fine-`level` HEALPix rings around the sample footprint to
  consider as *candidate* output cells before filtering to "inside the convex hull" — doesn't affect
  correctness (over-generous candidates are simply dropped by the hull test), only whether the hull is
  fully covered.

---

## Stored attributes

- **`N`, `K`, `cell_ids`**: as in every other resampler.
- **`points2d`**: `(N, 2)` NumPy array — samples projected to the local gnomonic tangent plane.
- **`tri`**: the underlying `scipy.spatial.Delaunay` triangulation.
- **`Gx`, `Gy`**: sparse `(N, N)` torch tensors — `grad_x = Gx @ f`, `grad_y = Gy @ f` for any
  sample-space field `f`.
- **`M`**: sparse CSR `(N, K)` torch tensor — `hval = y @ M`.

---

## Methods

### `resample(val)`

Sample space → HEALPix cell space: `hval = val @ self.M`. Accepts `val` of shape `(N,)` or `(B, N)`,
`np.ndarray` or `torch.Tensor`; returns the same type/batch shape, wrapped in
`ResampleResults(cell_data, cell_ids)`.

### `invert(hval)`

**Not implemented.** Unlike the KNN-based resamplers (which get a natural `MT` "for free" from a
symmetric neighbour search), Delaunay/Clough-Tocher only defines a mapping from scattered samples to
arbitrary query points, not the reverse. Calling `invert()` raises `NotImplementedError` with a pointer
to `planning/05_clough_tocher_resampler.md` for the reasoning.

---

## Notes and practical tips

- No benefit over `BicubicResampler`/`BilinearResampler` on a perfectly affine field — all interpolate a
  plane exactly; the advantage shows up on fields with genuine curvature.
- Cells outside the sample convex hull are silently excluded from `cell_ids`, not given a garbage
  extrapolated value.
- For a large/global dataset, use `subset_for_parent_cell` (see
  `docs/user-guide/regrid_to_healpix_parent_cell_subsetting.md`) to process one regional patch at a time —
  this resampler is not designed to be constructed once over a global dataset.
