---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Processing one parent cell at a time

This notebook is a runnable "test cell" for `subset_for_parent_cell`: it builds a synthetic dataset
spanning several coarse HEALPix parent cells, resamples it two ways — once globally, once one parent
cell at a time — and checks that the reassembled result matches the global one.

## Setup

```{code-cell} python
import numpy as np
import healpix_geo
from healpix_resample import NearestResampler, subset_for_parent_cell

# Shared dataset: wide enough to span several parent cells at level_parent.
level_parent = 4   # coarse: nside = 16
level = 7           # fine target resolution: nside = 128
ndata = 90
span = 8.0          # degrees

lon_grid, lat_grid = np.meshgrid(
    span * np.arange(ndata) / ndata,
    span * np.arange(ndata) / ndata,
)
lon = lon_grid.ravel()
lat = lat_grid.ravel()
val = np.sin(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))

parent_ids = np.unique(
    np.asarray(healpix_geo.nested.lonlat_to_healpix(lon, lat, level_parent, ellipsoid="WGS84")).astype(np.int64)
)
print(f"{len(parent_ids)} distinct parent cells in this dataset")
```

## Global resample (baseline)

```{code-cell} python
global_op = NearestResampler(lon_deg=lon, lat_deg=lat, level=level, verbose=False)
global_res = global_op.resample(val)
global_by_id = dict(zip(global_res.cell_ids.tolist(), global_res.cell_data.tolist()))

print(f"global run: {len(global_by_id)} cells from {len(lon)} samples")
```

## Parent-cell-by-parent-cell reassembly

```{code-cell} python
max_abs_diff = 0.0
n_cells_compared = 0
n_samples_seen = 0

for pid in parent_ids:
    lon_sub, lat_sub, val_sub, out_ids = subset_for_parent_cell(
        lon, lat, val,
        parent_cell_id=int(pid), level_parent=level_parent, level=level,
        margin_rings=1,
    )
    n_samples_seen += len(lon_sub)

    local_op = NearestResampler(
        lon_deg=lon_sub, lat_deg=lat_sub, level=level, out_cell_ids=out_ids, verbose=False
    )
    local_res = local_op.resample(val_sub)

    for cid, v_local in zip(local_res.cell_ids.tolist(), local_res.cell_data.tolist()):
        if cid in global_by_id:
            n_cells_compared += 1
            max_abs_diff = max(max_abs_diff, abs(v_local - global_by_id[cid]))

print(f"compared {n_cells_compared} cells across {len(parent_ids)} parent cells")
print(f"max |reassembled - global| = {max_abs_diff:.3e}  (expect ~0 for NearestResampler)")
print(f"total samples processed across all parent-cell subsets: {n_samples_seen} "
      f"(vs. {len(lon)} in the global run -- overlap comes from the margin buffer)")

assert max_abs_diff < 1e-9, "reassembled result should match the global run exactly for NearestResampler"
```

## Why the margin matters: shrinking it to 0 degrades boundary cells

```{code-cell} python
max_abs_diff_no_margin = 0.0

for pid in parent_ids:
    lon_sub, lat_sub, val_sub, out_ids = subset_for_parent_cell(
        lon, lat, val,
        parent_cell_id=int(pid), level_parent=level_parent, level=level,
        margin_rings=0,   # <-- no buffer: samples just across a parent boundary are dropped
    )
    local_op = NearestResampler(
        lon_deg=lon_sub, lat_deg=lat_sub, level=level, out_cell_ids=out_ids, verbose=False
    )
    local_res = local_op.resample(val_sub)

    for cid, v_local in zip(local_res.cell_ids.tolist(), local_res.cell_data.tolist()):
        if cid in global_by_id:
            max_abs_diff_no_margin = max(max_abs_diff_no_margin, abs(v_local - global_by_id[cid]))

print(f"max |reassembled - global| with margin_rings=0: {max_abs_diff_no_margin:.3e}")
print("(should be measurably worse than the margin_rings=1 result above)")
```
