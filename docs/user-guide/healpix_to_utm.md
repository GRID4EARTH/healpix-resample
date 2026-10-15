# HEALPix → UTM

`healpix_resample.utm` resamples a field defined on HEALPix cells onto a regular raster in a projected
CRS (typically a UTM zone).

Every resampler of this package maps lon/lat *samples* to HEALPix cells with `resample()`, and maps
HEALPix cells back to those samples with `invert()`. HEALPix → UTM is therefore the `invert()` of a
resampler whose samples are the **pixel centers of the target UTM grid**:

1. build the target grid — `UTMGrid`: the pixel-center `x`, `y` and the CRS of the tile;
2. reproject its pixel centers to lon/lat (`pyproj`);
3. construct a resampler on those points — **this determines which HEALPix cells the raster needs**;
4. read only those cells from the source dataset;
5. `resampler.invert(cell_data)`, reshaped to `(ny, nx)`.

`HealpixToUTM` does steps 2, 3 and 5. It is built from the geometry alone (grid and HEALPix level) and never
sees the source dataset, so it can be used on a store far larger than memory: its cost depends on the size of
the raster, not on the size of the archive. `pyproj` is an optional dependency:

```bash
pip install healpix-resample[utm]   # or pyproj from conda-forge in a conda/pixi environment
```

A complete, runnable example is in the {doc}`HEALPix → UTM tutorial <../tutorials/healpix_to_utm>`.

---

## Usage

```python
from healpix_resample import HealpixToUTM, UTMGrid

# target raster: pixel centers and CRS of the tile (y decreasing = north-up)
grid = UTMGrid(crs=32630, x=x, y=y)
# or from the outer pixel edges (xmin, ymin, xmax, ymax):
grid = UTMGrid.from_bounds(32630, (400000, 5358000, 402000, 5360000), resolution=10.0)

op = HealpixToUTM(grid, level, method="bilinear")   # geometry only, no data
op.cell_ids                         # (K,) int64, sorted: the cells the raster needs

data = ds.sel(cell_ids=op.cell_ids)["var"].values   # (B, K): read only those cells
img = op.resample(data)             # (B, ny, nx); (ny, nx) for data of shape (K,)

grid.x, grid.y, grid.crs            # coordinates of the output
op.valid                            # (ny, nx) bool: pixels the operator can fill
```

The operator can be reused for any other data defined on the same cells (other variables, other dates).

If the data are defined on other cells than exactly `op.cell_ids` — a superset, a different order, or a store
where some required cells are missing — pass their ids: `op.resample(data, cell_ids=ids)`. Required cells that
are absent are treated as NaN.

For data already in memory, `healpix_to_utm` does everything in one call, and can derive the grid from the
cells (bounding box of the cell centers, UTM zone of their centroid unless `crs=` is given, `resolution`
defaulting to the HEALPix cell size):

```python
from healpix_resample import healpix_to_utm

img, grid = healpix_to_utm(cell_data, cell_ids, level, resolution=10.0, method="bilinear")
```

Cell ids are in the nested scheme by default (`nest=False` for ring). NumPy arrays and torch tensors are both
accepted, and the output has the type of the input.

---

## Methods

| `method`            | Operator used                     | Behaviour                                                                 |
|---------------------|-----------------------------------|---------------------------------------------------------------------------|
| `"nearest"`         | `GroupByResampler.invert`         | Each pixel takes the value of the HEALPix cell containing its center.     |
| `"bilinear"` *(default)* | `BilinearResampler.invert`   | Weighted average of the 4 nearest cells.                                  |
| `"bicubic"`         | `BicubicResampler.invert`         | Weighted average of the 16 nearest cells (signed cubic-convolution kernel). |
| `"clough_tocher"`   | `CloughTocherResampler.invert`    | Delaunay triangulation of the cell centers + C1 cubic interpolant. Exact for an affine field; the most accurate on smooth fields. |
| a resampler class   | `cls(...).invert`                 | Any class accepting `lon_deg, lat_deg, level` and implementing `invert()`. |

Extra keyword arguments (`Npt`, `threshold`, `device`, `dtype`, …) are passed to the resampler.

`"nearest"` does not use `NearestResampler.invert`: that method sends each cell to its single nearest
sample, so on a raster finer than the cells most pixels would receive nothing.

---

## Notes

- **No extrapolation.** A pixel that depends on a cell without data — a NaN value, or a cell absent from the
  `cell_ids` passed to `resample` — is `NaN`. With `"clough_tocher"`, pixels outside the convex hull of the
  cell *centers* are `NaN` as well.
- **Each method needs its own cells.** `op.cell_ids` depends on `method` (the neighbourhood differs), so read
  the data with the `cell_ids` of the operator that will be applied.
- **`pad`** (Clough-Tocher only, automatic by default): the operator is built on a grid extended by two
  cell widths and cropped afterwards. `CloughTocherResampler` keeps a cell only when it lies inside the
  hull of its samples — here the pixel centers — so without this margin the cells surrounding the raster
  would be discarded and its border could not be interpolated.
- The weighted-average methods (`"bilinear"`, `"bicubic"`) are not affine-exact.
- Any projected CRS understood by `pyproj` works for the target grid; only the default-CRS helper is
  UTM-specific (standard 6° zones, without the Norway/Svalbard exceptions).
- Like `CloughTocherResampler` itself, `"clough_tocher"` is meant for regional extents (it works in a
  local gnomonic plane) — which is also the domain of validity of a single UTM zone.
