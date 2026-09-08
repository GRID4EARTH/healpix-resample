# `OverlapConservativeResampler` — first-order overlap-area conservative remapping

`OverlapConservativeResampler` implements the classical **first-order
conservative remapping** used in Earth-system modelling: each source cell's
value is redistributed over *every* HEALPix cell it overlaps, with weights
proportional to the source/target intersection areas.

This is the same mathematical formulation as ESMF/xESMF `conservative`
(and `conservative_normed`), applied to a HEALPix destination grid.

## Which conservative method do I want?

`healpix-resample` offers three different notions of conservation. They
are not interchangeable — they answer different questions.

| Class / mode | Geometry | Local redistribution | Conserves | PSF-aware |
|---|---|---|---|---|
| `ConservativeResampler` | sample → containing cell | no (hard binning) | exact discrete total | no |
| `OverlapConservativeResampler` | source/target cell intersections | yes, by overlap area | exact integral | no |
| `PSFResampler(..., conservative=True)` | observation operator (inverse problem) | implicit, through reconstruction | global constraint | yes |

Use `ConservativeResampler` for fast aggregation of scattered samples when
each sample is small compared to a HEALPix cell. Use
`OverlapConservativeResampler` when source cells are comparable to or
larger than the target cells and their content must be *split*
geometrically — the classical flux-remapping situation. Use the
constrained `PSFResampler` when you want to reconstruct a latent field
through an instrument response while imposing a global total.

The difference is easiest to see on a constant field: a constant remapped
by overlap areas stays constant everywhere, whereas hard binning gives
each target cell the sum of whatever samples happened to fall inside it.

## Usage

The source grid is described by its **cell boundaries** (not centres):

```python
import numpy as np
from healpix_resample import OverlapConservativeResampler

lon_bounds = np.arange(0.0, 360.0 + 1e-9, 1.0)   # 360 columns
lat_bounds = np.arange(-90.0, 90.0 + 1e-9, 1.0)  # 180 rows

remap = OverlapConservativeResampler(lon_bounds, lat_bounds, level=6)
result = remap.resample(field)          # field.shape == (180, 360)

result.cell_data   # values on HEALPix cells
result.cell_ids    # nested cell ids
```

The geometric weights are the expensive part and are computed once. Reuse
the same object for every variable and time step sharing that grid — the
sparse matrix is exposed as `remap.weights` if you need it directly.

Descending `lat_bounds` (the common convention for gridded climate data)
is detected and handled: pass the field in the same orientation as the
bounds you supplied.

## Normalization: `"destination"` vs `"covered"`

For target cells that are only partly covered by the source grid (a
regional grid, or a masked domain), two conventions exist:

```python
OverlapConservativeResampler(..., normalization="destination")  # default
OverlapConservativeResampler(..., normalization="covered")
```

- `"destination"` divides by the full HEALPix cell area, so uncovered area
  contributes zero. A constant field `C` reads as `C × (covered fraction)`
  on boundary cells. This is the convention that conserves the *integral*
  over the whole target grid — the equivalent of xESMF `conservative`.
- `"covered"` divides by the actually covered area, so a constant field
  reads as `C` everywhere it is defined. This is the equivalent of xESMF
  `conservative_normed`, and is usually what you want for an intensive
  physical field on a regional domain.

Both are identical wherever coverage is complete.

## Intensive vs extensive fields

```python
remap.resample(flux,   quantity="intensive")   # default: W/m², K, ...
remap.resample(energy, quantity="extensive")   # counts, per-cell totals
```

For an intensive field (a density) the result is the overlap-weighted
average over the target cell. For an extensive field (a quantity already
integrated over its source cell) the source total is split across target
cells in proportion to the overlap fraction *of the source cell*, so that
`result.cell_data.sum() == field.sum()` exactly. Getting this distinction
wrong silently changes the physics, so the parameter is explicit and has
no "guess" mode.

## Accuracy and scope

Overlap areas are computed **exactly on the sphere**, with no polygon
densification: the computation exploits the fact that in the HEALPix
equal-area projection, cells are exact squares and the images of source
lat/lon rectangles are straight-edged convex quadrilaterals — once each
rectangle is split at the two transition latitudes (|sin φ| = 2/3) and at
the polar-quadrant meridians. The intersections are then exact convex
polygon clippings, and the only error is floating-point rounding
(observed: global conservation to ~10⁻¹⁴, constant-field preservation to
~10⁻¹¹). Antimeridian crossings and polar cells need no special handling.

Current scope:

- rectilinear lat/lon source grids (regular or irregular), given by 1D
  boundary arrays;
- **spherical** Earth model (areas in steradians; multiply by `R²`);
- nested or ring HEALPix indexing, any level.

Curvilinear source grids and an authalic-ellipsoid variant are not
implemented yet. Note that this resampler is deliberately independent of
the `KNeighborsResampler` base class (it is a grid-to-grid geometric
operator, not a sample-based one), so it does not take `sigma_m`,
`threshold`, `device` or the other kernel parameters.
