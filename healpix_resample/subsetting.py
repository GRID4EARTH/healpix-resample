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
resampler, and (b) which input samples are actually relevant to that parent
cell, so the resampler constructor only ever sees a local subset of the full
`(lon_deg, lat_deg, val)` arrays. See `subset_for_parent_cell`'s docstring
for the full margin-correctness discussion.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import healpix_geo


def subset_for_parent_cell(
    lon_deg,
    lat_deg,
    val,
    parent_cell_id: int,
    level_parent: int,
    level: int,
    *,
    nest: bool = True,
    ellipsoid: str = "WGS84",
    margin_rings: int = 1,
    num_threads: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Restrict `(lon_deg, lat_deg, val)` and an output cell set to one parent cell.

    This is the standalone helper for the "process one coarse cell at a
    time" workflow: it does *not* itself construct or call any resampler --
    it only prepares the two pieces every resampler's constructor needs to
    do less work:

    1. **Output side** (cheap, exact, no new logic): the set of `level`
       -resolution cells contained in `parent_cell_id`, via
       `healpix_geo.*.zoom_to`. Pass this straight through as any
       resampler's ``out_cell_ids=`` kwarg -- see the "group_by resamplers"
       caveat below for the one place this doesn't apply.
    2. **Input side** (the actual point of this helper): a boolean-filtered
       subset of `lon_deg`/`lat_deg`/`val` containing only the samples that
       could plausibly matter for cells inside `parent_cell_id` -- so `N` in
       a resampler constructed from this subset is the *local* sample count,
       not the size of the full global dataset.

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

    `margin_rings` controls this: input samples are kept if their own
    `level_parent` cell id is `parent_cell_id` **or** within `margin_rings`
    HEALPix rings of it (`healpix_geo.*.kth_neighbourhood`), mirroring the
    same ring-growth pattern `healpix_weighted_nearest` already uses
    internally. **This is a correctness guarantee, not just a performance
    knob**: the margin must be wide enough that the resampler's actual
    kernel reach (`sigma_m` and the effective radius implied by
    `threshold`) can never extend past it. For `level_parent` many levels
    coarser than `level` (the intended use case -- a parent cell is already
    many multiples of a typical `sigma_m` wide), `margin_rings=1` (the
    default) is very likely generous, but this scales with *your*
    `sigma_m`/`threshold` choice, not with this function's defaults -- if
    you shrink `sigma_m` or loosen `threshold` enough that the kernel reaches
    close to a `level_parent` cell's own width, increase `margin_rings`
    accordingly. When in doubt, compare the parent cell's angular width
    (roughly `sqrt(4*pi/(12*4**level_parent))` radians) against the
    kernel's actual reach for your resampler and size the margin so the
    kernel cannot reach past it.

    Resamplers using `group_by=True` (`ConservativeResampler`,
    `GroupByResampler`, `CellPointResampler`)
    ------------------------------------------------------------------------
    These resamplers derive their cells purely from `torch.unique` over the
    (already margin-filtered) input subset -- they have no KNN neighbourhood
    search and no `out_cell_ids` intersection logic to hook into:
    `ConservativeResampler` raises `NotImplementedError` if you pass
    `out_cell_ids` to it at all, and `GroupByResampler`/`CellPointResampler`
    silently store but never use it (no error, no effect). For these
    classes, use only the returned `lon_sub`/`lat_sub`/`val_sub` (piece 2)
    and do **not** pass the returned `out_cell_ids` (piece 1) to their
    constructors -- the set of cells they produce is whatever the filtered
    input actually hits, which will generally be a subset of (or, at the
    margin, extend slightly beyond) `out_cell_ids` and cannot be
    force-expanded to include empty parent-cell-interior cells the way
    KNN-mode resamplers can.

    Parameters
    ----------
    lon_deg, lat_deg : array-like, shape (N,)
        Full input sample coordinates in degrees.
    val : array-like
        Full input values, shape ``(N,)`` or batched ``(B, N)``, filtered
        along the last axis in lockstep with `lon_deg`/`lat_deg`.
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
        Number of `level_parent` HEALPix rings around `parent_cell_id` to
        include when filtering input samples -- see "Why input filtering
        needs a margin" above. ``0`` disables the margin entirely (kept only
        to make the "margin matters" failure mode easy to demonstrate/test;
        not recommended for real use).

    Returns
    -------
    lon_sub, lat_sub, val_sub : numpy.ndarray
        The input subset relevant to `parent_cell_id` (its own cell plus the
        `margin_rings`-neighbour buffer at `level_parent`).
    out_cell_ids : numpy.ndarray
        The `level`-resolution cells contained in `parent_cell_id`, ready to
        pass straight into a KNN-mode resampler's ``out_cell_ids=`` kwarg
        (not for `group_by=True` resamplers -- see above).
    """
    hp = healpix_geo.nested if nest else healpix_geo.ring

    lon_np = np.asarray(lon_deg, dtype=np.float64).reshape(-1)
    lat_np = np.asarray(lat_deg, dtype=np.float64).reshape(-1)
    val_np = np.asarray(val)

    parent_arr = np.asarray([parent_cell_id], dtype=np.uint64)

    # --- piece 1: output-side cell set (exact, no margin needed) -----------
    out_cell_ids = np.asarray(
        hp.zoom_to(parent_arr, level_parent, level, num_threads=num_threads)
    ).reshape(-1).astype(np.int64)

    # --- piece 2: input-side filtering, with a margin -----------------------
    sample_parent_ids = np.asarray(
        hp.lonlat_to_healpix(lon_np, lat_np, level_parent, num_threads=num_threads, ellipsoid=ellipsoid)
    ).astype(np.int64)

    keep_ids = parent_arr.astype(np.int64)
    if margin_rings > 0:
        neighbours = np.asarray(
            hp.kth_neighbourhood(parent_arr, level_parent, margin_rings, num_threads=num_threads)
        ).reshape(-1).astype(np.int64)
        neighbours = neighbours[neighbours >= 0]  # kth_neighbourhood pads with -1
        keep_ids = np.unique(np.concatenate([keep_ids, neighbours]))

    keep_mask = np.isin(sample_parent_ids, keep_ids)

    lon_sub = lon_np[keep_mask]
    lat_sub = lat_np[keep_mask]
    val_sub = val_np[..., keep_mask]

    return lon_sub, lat_sub, val_sub, out_cell_ids
