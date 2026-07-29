# `healpix_resample.bilinear` (`BilinearResampler`)

`BilinearResampler` regrids unstructured longitude/latitude samples onto a HEALPix grid using the
**4 nearest retained cells** per sample, weighted by inverse geodesic distance. It sits at the cheap,
non-iterative end of the package's resampler family: smoother than `NearestResampler` (which just
assigns each sample to a single cell), cheaper than `BicubicResampler` (16 neighbours) or
`PSFResampler` (a full CG-solved inverse problem).

```python
from healpix_resample import BilinearResampler

op = BilinearResampler(lon_deg=lon, lat_deg=lat, level=level)
res = op.resample(val)
hval, cell_ids = res.cell_data, res.cell_ids
```

`BilinearResampler` is a thin subclass of `KNeighborsResampler` with `Npt=4` fixed; it accepts every
constructor argument `KNeighborsResampler`/`PSFResampler` do (`out_cell_ids`, `threshold`, `sigma_m`,
`nest`, `ellipsoid`, `device`, `dtype`, `verbose`, ...) except the ones that are PSF-specific
(`area`, `fill_missing_out_cells`, CG parameters).

---

## Weighting

For each sample, the 4 nearest cells (among those retained by `threshold`) are combined with inverse
distance weights:

```
w = 1 / (eps + d_m / sigma_m)
```

where `d_m` is the geodesic distance (meters) to each candidate cell center and `sigma_m` is the
same length scale used elsewhere in the package (`sqrt(4*pi/(12*4**level)) * R` by default). This is
a fixed, non-iterative interpolation -- no CG solve, unlike `PSFResampler`.

---

## NaN handling and "orphaned" cells

`resample()`'s NaN-sample filtering follows the same pattern as the other KNN-mode resamplers: a
NaN-valued sample only affects the (up to 4) cells it actually links to, not the whole output.

A separate, more subtle failure mode -- and the direct cause of cells silently reading back as
exactly `0.0` (not `nan`) on otherwise perfectly finite input -- was fixed at the `KNeighborsResampler`
base class level and applies to `BilinearResampler` automatically:

- Which cells get **retained** (`cell_ids`) is decided by a wide-neighbourhood accumulated-weight
  threshold test.
- Which cells each **sample actually links to** is decided separately, by a narrower per-sample
  nearest-`Npt` search among the retained cells.
- These two criteria are not guaranteed consistent: a cell can pass the wide threshold test (many
  samples each contributing a small amount) while never being one of any single sample's own
  4-nearest retained cells -- leaving it an entirely empty column in `M`. Before the fix, such a
  column read back as exactly `0.0` for *any* input, including all-NaN input, which is what made
  this look like a NaN-propagation bug rather than a cell-retention inconsistency.

This is now fixed by pruning such "orphaned" cells out of `cell_ids` at construction time (when
`out_cell_ids` is not explicitly supplied) -- see `KNeighborsResampler.__init__` in `knn.py` and
`tests/test_bilinear.py` (`test_no_orphaned_cells_in_M`,
`test_all_nan_input_on_finite_data_is_not_silently_zero`) for the full write-up and regression
coverage. In practice this means: a handful of cells near the edge of well-covered regions, or at
target resolutions much finer than your input sampling density, can legitimately be absent from
`cell_ids` -- treat "not in `cell_ids`" as no-data (e.g. `nan`-fill when reassembling into a full
grid), rather than expecting every geometrically-plausible cell to appear.

If you're seeing more missing/no-data cells than expected at a given `level`, that's often a sign
the *target* resolution is finer than the input sampling can genuinely support there -- try a
coarser `level`, or switch to `PSFResampler`, which handles sparse/irregular coverage more robustly
via its iterative solve.

---

## Conservative mode (`area=`, `resample(conservative=True)`)

Added for [issue #44](https://github.com/GRID4EARTH/healpix-resample/issues/44) ("conservative bi-linear is
missing"). By default, `resample()` interpolates: each cell's value is a weighted blend of its 4 nearest
samples, normalized *per cell* (`self.M`) — smooth, but not exactly mass-conserving, since a cell's
normalization only depends on whichever samples happen to link to it, independent of any other cell.

```python
op = BilinearResampler(lon_deg=lon, lat_deg=lat, level=level, area=area)  # area optional, defaults to 1.0
res = op.resample(val, conservative=True)
```

With `conservative=True`, each sample's own value (scaled by its `area`) is instead redistributed across
its 4 nearest cells using `self.M_cons` — the *same* inverse-distance weights as the default path, but
normalized so each *sample's* own weights sum to exactly 1 (a partition of unity) instead of being
normalized per output cell. No value is invented or lost:

```
sum_k hval[k] == sum_i (valid i) val[i] * area[i]
```

exactly, regardless of how many samples any given cell receives contributions from. This is a
bilinear-weighted analogue of `ConservativeResampler`'s exact area conservation (see
`healpix_resample/conservative.py`), without that class's single-nearest-cell binning blockiness.

**NaN handling** under `conservative=True`: a NaN sample's value *and* its area are both excluded from
every cell's total, so the identity above holds over exactly the valid samples — no cell is forced to
`nan` just because one sample it shares a link with is missing (unlike `conservative=False`'s
interpolation path, where a NaN sample's linked cells lose their sample but keep an unadjusted
normalization — see the inherited `KNeighborsResampler.resample()` docstring for that caveat). A batch row
where every sample is NaN comes back entirely `nan`.

`area` defaults to `1.0` for every sample (equal-area pixels / already-extensive quantities, same
convention as `ConservativeResampler`) and is otherwise ignored when `conservative=False`.

---

## Practical tips

- Bilinear assumes the 4 nearest retained cells are a reasonable local neighbourhood for
  interpolation. On sparse or highly irregular sampling, prefer `PSFResampler`.
- Near the poles or across the dateline, the geodesic-distance weighting handles wrap-around
  correctly (no special-casing needed, unlike planar bilinear on a lon/lat array).
- See `docs/tutorials/4resamplers.md` for a runnable side-by-side comparison against the other
  resamplers on the same dataset.
