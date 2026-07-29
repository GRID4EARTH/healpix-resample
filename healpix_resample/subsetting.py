"""
subsetting.py

Standalone helper for processing one coarse HEALPix "parent cell" at a time,
so a global input dataset never has to be loaded/searched all at once by
`KNeighborsResampler.__init__` (which computes a KNN neighbourhood search and
materializes sparse `(N, K)`/`(K, N)` operators sized to however many samples
are handed to it).

This module intentionally holds no resampler-specific logic: it only
computes (a) which fine-`level` HEALPix cells fall inside a given
`(parent_cell_id, level_parent)`, ready to pass as `out_cell_ids=` to any
resampler, and (b) an index into the *samples* (not into any particular
value array) selecting which ones are actually relevant to that parent cell,
so a resampler built from `lon_deg[sample_idx]`/`lat_deg[sample_idx]` only
ever sees a local subset, not the full global dataset. See
`subset_for_parent_cell`'s docstring for the full margin-correctness
discussion.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import healpix_geo


def subset_for_parent_cell(
    lon_deg,
    lat_deg,
    parent_cell_id: int,
    level_parent: int,
    level: int,
    *,
    nest: bool = True,
    ellipsoid: str = "WGS84",
    margin_rings: int = 1,
    num_threads: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Restrict processing to one parent cell: which samples, which output cells.

    This is the standalone helper for the "process one coarse cell at a
    time" workflow: it does *not* itself construct or call any resampler,
    and it does **not** take or return any value array (`val`) -- it only
    prepares the two pieces every resampler's constructor needs to do less
    work:

    1. **Output side** (cheap, exact, no new logic): the set of `level`
       -resolution cells contained in `parent_cell_id`, via
       `healpix_geo.*.zoom_to`. Pass this straight through as any
       resampler's ``out_cell_ids=`` kwarg -- see the "group_by resamplers"
       caveat below for the one place this doesn't apply.
    2. **Input side** (the actual point of this helper): an integer index
       into the *sample axis* -- not a filtered value array -- selecting
       which samples could plausibly matter for cells inside
       `parent_cell_id`.

    Why this returns an index rather than filtered arrays
    -------------------------------------------------------
    A single `(lon_deg, lat_deg)` grid is very commonly shared by many
    different value arrays (several variables, a time series of the same
    station network, ...). Taking a `val` in and handing back a filtered
    `val_sub` would mean recomputing this exact same geometric membership
    test again for every one of those arrays, even though the answer -- which
    samples are relevant to `parent_cell_id` -- only depends on
    `lon_deg`/`lat_deg` and never changes. Instead, `subset_for_parent_cell`
    returns `sample_idx`, an integer array into the sample axis: compute it
    once per `(lon_deg, lat_deg)` grid and reuse it to index `lon_deg`,
    `lat_deg`, and as many value arrays as you have that share those same
    coordinates:

    ```python
    sample_idx, out_ids = subset_for_parent_cell(
        lon, lat, parent_cell_id=pid, level_parent=6, level=20,
    )
    lon_sub, lat_sub = lon[sample_idx], lat[sample_idx]
    val_sub = val[..., sample_idx]           # any variable on the same grid
    other_val_sub = other_val[..., sample_idx]
    ```

    `sample_idx` indexes the *last* axis, so it applies uniformly whether a
    value array is `(N,)` or batched `(B, N)`, and works the same way for
    plain NumPy arrays or PyTorch tensors (fancy-indexing a tensor with a
    NumPy integer array works out of the box).

    Why input filtering needs a margin, not just "same parent cell id"
    --------------------------------------------------------------------
    A sample can sit just outside `parent_cell_id`'s boundary, in a
    *sibling* `level_parent` cell, while still being close enough in the
    fine-`level` kernel sense (`sigma_m`/`threshold` -- see
    `KNeighborsResampler`) to legitimately contribute to a fine cell near the
    parent's edge. Filtering input strictly to "same parent cell id" would
    silently starve edge cells of legitimate neighbours -- a subtle
    correctness bug that only shows up as slightly-degraded results near
    parent-cell boundaries, not as an error.

    `margin_rings` controls this, and is measured in **fine-`level`** HEALPix
    rings, not `level_parent` rings: samples are kept if their own `level`
    cell id is within `margin_rings` rings of *any* fine cell inside
    `parent_cell_id` (i.e. of `out_cell_ids`, expanded via
    `healpix_geo.*.kth_neighbourhood` at `level`). This matters because a
    fine cell is typically vastly smaller than a parent cell -- buffering by
    whole *parent-level* rings instead (one ring already means "pull in every
    neighbouring parent cell in full") would keep a hugely disproportionate
    number of irrelevant samples relative to the resampler's actual kernel
    reach at `level`. **This is a correctness guarantee, not just a
    performance knob**: the margin must be wide enough that the resampler's
    actual kernel reach (`sigma_m` and the effective radius implied by
    `threshold`) can never extend past it, but sized at the *fine* scale the
    kernel actually operates at. `margin_rings=1` (the default) is very
    likely generous relative to a fine cell's own width, but this scales
    with *your* `sigma_m`/`threshold` choice, not with this function's
    defaults -- if you shrink `sigma_m` or loosen `threshold` enough that the
    kernel reaches past a handful of fine cells, increase `margin_rings`
    accordingly. When in doubt, compare a fine cell's angular width (roughly
    `sqrt(4*pi/(12*4**level))` radians) against the kernel's actual reach for
    your resampler and size the margin so the kernel cannot reach past it.

    Resamplers using `group_by=True` (`ConservativeResampler`,
    `GroupByResampler`, `CellPointResampler`)
    ------------------------------------------------------------------------
    These resamplers derive their cells purely from `torch.unique` over the
    (already margin-filtered) input subset -- they have no KNN neighbourhood
    search and no `out_cell_ids` intersection logic to hook into:
    `ConservativeResampler` raises `NotImplementedError` if you pass
    `out_cell_ids` to it at all, and `GroupByResampler`/`CellPointResampler`
    silently store but never use it (no error, no effect). For these
    classes, use only `sample_idx` (piece 2) to slice `lon_deg`/`lat_deg`/
    `val` and do **not** pass the returned `out_cell_ids` (piece 1) to their
    constructors -- the set of cells they produce is whatever the filtered
    input actually hits, which will generally be a subset of (or, at the
    margin, extend slightly beyond) `out_cell_ids` and cannot be
    force-expanded to include empty parent-cell-interior cells the way
    KNN-mode resamplers can.

    Parameters
    ----------
    lon_deg, lat_deg : array-like, shape (N,)
        Full input sample coordinates in degrees.
    parent_cell_id : int
        HEALPix cell id at `level_parent` to restrict processing to.
    level_parent : int
        The coarse level `parent_cell_id` is expressed at (``nside =
        2**level_parent``). Should be well below `level`.
    level : int
        The target (fine) HEALPix level the eventual resampler will run at.
    nest : bool
        HEALPix indexing scheme, must match what the resampler will be
        constructed with.
    margin_rings : int
        Number of **fine-`level`** HEALPix rings around `out_cell_ids` to
        include when filtering input samples -- see "Why input filtering
        needs a margin" above (note: this is rings at `level`, not
        `level_parent`). ``0`` disables the margin entirely (kept only to
        make the "margin matters" failure mode easy to demonstrate/test; not
        recommended for real use).

    Returns
    -------
    sample_idx : numpy.ndarray of int
        Integer index into the sample axis (`lon_deg`/`lat_deg`, and any
        value array sharing that same axis) selecting the samples relevant
        to `parent_cell_id` (its own cells plus the `margin_rings`-neighbour
        buffer at `level`). Apply it yourself: ``lon_deg[sample_idx]``,
        ``val[..., sample_idx]``, etc.
    out_cell_ids : numpy.ndarray
        The `level`-resolution cells contained in `parent_cell_id`, ready to
        pass straight into a KNN-mode resampler's ``out_cell_ids=`` kwarg
        (not for `group_by=True` resamplers -- see above).
    """
    hp = healpix_geo.nested if nest else healpix_geo.ring

    lon_np = np.asarray(lon_deg, dtype=np.float64).reshape(-1)
    lat_np = np.asarray(lat_deg, dtype=np.float64).reshape(-1)

    parent_arr = np.asarray([parent_cell_id], dtype=np.uint64)

    # --- piece 1: output-side cell set (exact, no margin needed) -----------
    out_cell_ids = np.asarray(
        hp.zoom_to(parent_arr, level_parent, level, num_threads=num_threads)
    ).reshape(-1).astype(np.int64)

    # --- piece 2: input-side filtering, with a FINE-level margin ------------
    # The margin must scale with the resampler's actual kernel reach at
    # `level` (a handful of fine HEALPix rings), not with the *parent* cell's
    # own, much larger size -- expanding by whole neighbouring PARENT cells
    # would pull in a hugely disproportionate number of irrelevant samples.
    # So buffer `out_cell_ids` itself (already at `level`) by `margin_rings`
    # fine rings, rather than buffering `parent_cell_id` by parent rings.
    if margin_rings > 0:
        buffered_cells = np.asarray(
            hp.kth_neighbourhood(out_cell_ids.astype(np.uint64), level, margin_rings, num_threads=num_threads)
        ).reshape(-1).astype(np.int64)
        buffered_cells = buffered_cells[buffered_cells >= 0]  # kth_neighbourhood pads with -1
        keep_ids = np.unique(np.concatenate([out_cell_ids, buffered_cells]))
    else:
        keep_ids = out_cell_ids

    sample_fine_ids = np.asarray(
        hp.lonlat_to_healpix(lon_np, lat_np, level, num_threads=num_threads, ellipsoid=ellipsoid)
    ).astype(np.int64)

    keep_mask = np.isin(sample_fine_ids, keep_ids)
    sample_idx = np.nonzero(keep_mask)[0]

    return sample_idx, out_cell_ids
