# HEALPix → UTM

`healpix_resample.utm` resamples a field defined on HEALPix cells onto a regular raster in a projected
CRS (typically a UTM zone).

Every resampler of this package maps lon/lat *samples* to HEALPix cells with `resample()`, and maps
HEALPix cells back to those samples with `invert()`. HEALPix → UTM is therefore the `invert()` of a
resampler whose samples are the **pixel centers of the target UTM grid**:

1. build the target grid — `UTMGrid`;
2. reproject its pixel centers to lon/lat (`pyproj`);
3. construct a resampler on those points, restricted to the cells that carry data
   (`out_cell_ids=cell_ids`);
4. `resampler.invert(cell_data)`, reshaped to `(ny, nx)`.

`HealpixToUTM` does steps 2–4. `pyproj` is an optional dependency:

```bash
pip install healpix-resample[utm]   # or pyproj from conda-forge in a conda/pixi environment
```

---

## Usage

```python
from healpix_resample import HealpixToUTM, UTMGrid, healpix_to_utm

# target raster: 10 m pixels in UTM 30N, bounds = outer pixel edges (xmin, ymin, xmax, ymax)
grid = UTMGrid.from_bounds(32630, (500000, 5340000, 512000, 5350000), resolution=10.0)

op = HealpixToUTM(cell_ids, level, grid, method="clough_tocher")
img = op.resample(cell_data)        # (ny, nx), or (B, ny, nx) for cell_data of shape (B, K)

grid.x, grid.y, grid.crs            # pixel-center eastings (nx,), northings (ny,, north-up), pyproj.CRS
op.valid                            # (ny, nx) bool: pixels that received a value
```

The operator depends only on the geometry (`cell_ids`, `level`, `grid`): build it once and apply it to
as many fields as needed. For a one-off call, with a grid derived from the cells themselves:

```python
img, grid = healpix_to_utm(cell_data, cell_ids, level, resolution=10.0, method="bilinear")
```

Here the grid covers the bounding box of the cell centers, in the UTM zone of their centroid
(`utm_crs_from_lonlat`), or in `crs=` if given. `resolution` defaults to the HEALPix cell size.

`cell_ids` may be in any order (nested scheme by default, `nest=False` for ring) but must be unique;
`cell_data` follows the same order. NumPy arrays and torch tensors are both accepted, and the output has
the type of the input.

---

## Methods

| `method`            | Operator used                     | Behaviour                                                                 |
|---------------------|-----------------------------------|---------------------------------------------------------------------------|
| `"nearest"`         | `GroupByResampler.invert`         | Each pixel takes the value of the HEALPix cell containing its center.     |
| `"bilinear"` *(default)* | `BilinearResampler.invert`   | Weighted average of the 4 nearest cells.                                  |
| `"bicubic"`         | `BicubicResampler.invert`         | Weighted average of the 16 nearest cells (signed cubic-convolution kernel). |
| `"clough_tocher"`   | `CloughTocherResampler.invert`    | Delaunay triangulation of the cell centers + C1 cubic interpolant. Exact for an affine field; the most accurate on smooth fields. |
| a resampler class   | `cls(...).invert`                 | Any class accepting `lon_deg, lat_deg, level, out_cell_ids` and implementing `invert()`. |

Extra keyword arguments (`Npt`, `threshold`, `device`, `dtype`, …) are passed to the resampler.

`"nearest"` does not use `NearestResampler.invert`: that method sends each cell to its single nearest
sample, so on a raster finer than the cells most pixels would receive nothing.

---

## Notes

- **No extrapolation.** A pixel gets a value only if its center lies in one of the cells of `cell_ids`;
  the other pixels are `NaN`. With `"clough_tocher"`, pixels outside the convex hull of the cell
  *centers* are `NaN` as well, which removes roughly half a cell along the edge of the data footprint.
- **NaN cells** propagate to the pixels they contribute to (a few cells around, depending on the method).
- **`pad`** (Clough-Tocher only, automatic by default): the operator is built on a grid extended by two
  cell widths and cropped afterwards. `CloughTocherResampler` keeps a cell only when it lies inside the
  hull of its samples — here the pixel centers — so without this margin the cells surrounding the raster
  would be discarded and its border could not be interpolated.
- The weighted-average methods are not affine-exact, and are less accurate along the edge of the data
  footprint where the neighbourhood is one-sided.
- Any projected CRS understood by `pyproj` works for the target grid; only the default-CRS helper is
  UTM-specific (standard 6° zones, without the Norway/Svalbard exceptions).
- Like `CloughTocherResampler` itself, `"clough_tocher"` is meant for regional extents (it works in a
  local gnomonic plane) — which is also the domain of validity of a single UTM zone.
