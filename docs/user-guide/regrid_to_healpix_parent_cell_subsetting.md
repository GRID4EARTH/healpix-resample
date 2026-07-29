# Processing one parent cell at a time (`subset_for_parent_cell`)

`KNeighborsResampler.__init__` computes a `healpix_geo` indexing pass and a KNN neighbourhood search
over **every** input sample up front, and `comp_matrix()` materializes sparse `(N, K)`/`(K, N)`
operators sized to that same `N`. For a global dataset, that doesn't fit in memory (and isn't
necessary) if you only want results for one region at a time.

`healpix_resample.subset_for_parent_cell` restricts both sides of the problem to one coarse HEALPix
"parent cell": the *output* cells (an exact, cheap lookup) and the *input* samples actually relevant
to them (the expensive part, and the actual point of this helper).

---

## Why `out_cell_ids` alone doesn't solve this

Every KNN-mode resampler already accepts an `out_cell_ids=` kwarg to restrict its *output*. But that
restriction is applied only *after* the expensive part -- indexing and the neighbourhood search still
run over the full, unfiltered input array. `subset_for_parent_cell` adds the missing piece: filtering
the *input* down to a local subset before a resampler is ever constructed, so `N` in the constructor
call is the local sample count, not the global one.

## Usage

```python
from healpix_resample import PSFResampler, subset_for_parent_cell

level_parent = 6   # coarse: nside = 64
level = 20          # fine target resolution

lon_sub, lat_sub, val_sub, out_ids = subset_for_parent_cell(
    lon, lat, val,
    parent_cell_id=parent_id,
    level_parent=level_parent,
    level=level,
)

op = PSFResampler(lon_deg=lon_sub, lat_deg=lat_sub, level=level, out_cell_ids=out_ids)
result = op.resample(val_sub)
```

Loop this over every parent cell at `level_parent` to process a full global dataset in bounded-memory
chunks. Reassembling results from multiple parent cells into one global map is not something this
function does for you -- it only makes *one* parent cell's computation tractable.

## What it returns

- `lon_sub, lat_sub, val_sub`: the input subset relevant to `parent_cell_id` -- samples whose own
  `level_parent` cell is `parent_cell_id` itself, plus a `margin_rings`-neighbour buffer around it
  (see below). `val_sub` is filtered along its *last* axis, so both `(N,)` and `(B, N)` input work.
- `out_cell_ids`: the `level`-resolution cells contained in `parent_cell_id`
  (`healpix_geo.*.zoom_to`), ready to pass straight into a KNN-mode resampler's `out_cell_ids=`.

## The margin is a correctness guarantee, not a performance knob

A sample can sit just outside `parent_cell_id`'s boundary, in a sibling `level_parent` cell, while
still being close enough in the fine-`level` kernel sense (`sigma_m`/`threshold`) to legitimately
contribute to a fine cell near the parent's edge. Filtering input strictly to "same parent cell id"
would silently starve edge cells of legitimate neighbours -- a subtle bug that only shows up as
slightly-degraded results near parent-cell boundaries, not as an error.

`margin_rings` (default `1`) includes not just `parent_cell_id` but its ring-`margin_rings`
neighbours at `level_parent` when filtering input. For `level_parent` many levels coarser than
`level` -- the intended use case, where a parent cell is already many multiples of a typical
`sigma_m` wide -- `margin_rings=1` is very likely generous. But this depends on *your* `sigma_m`/
`threshold` choice, not on this function's defaults: if you shrink `sigma_m` or loosen `threshold`
enough that the kernel's reach approaches a `level_parent` cell's own width, increase `margin_rings`
accordingly. As a rule of thumb, compare the parent cell's angular width
(`sqrt(4*pi/(12*4**level_parent))` radians) against the kernel's actual reach for your resampler, and
size the margin so the kernel can never reach past it.

## Resamplers that use `group_by=True`

`ConservativeResampler`, `GroupByResampler`, and `CellPointResampler` derive their cells purely from
the samples actually present in the (already margin-filtered) input -- they have no KNN neighbourhood
search and no `out_cell_ids` intersection logic to hook into:

- `ConservativeResampler` raises `NotImplementedError` if you pass `out_cell_ids` at all.
- `GroupByResampler`/`CellPointResampler` silently store but never use it (no error, no effect).

For these classes, use only `lon_sub`/`lat_sub`/`val_sub` from `subset_for_parent_cell` and do **not**
pass its `out_cell_ids` to their constructors -- the set of cells they produce is simply whatever the
filtered input actually hits.

```python
from healpix_resample import ConservativeResampler, subset_for_parent_cell

lon_sub, lat_sub, val_sub, _out_ids_unused = subset_for_parent_cell(
    lon, lat, val, parent_cell_id=parent_id, level_parent=6, level=20,
)
op = ConservativeResampler(lon_deg=lon_sub, lat_deg=lat_sub, level=20, area=area_sub)
result = op.resample(val_sub)
```
