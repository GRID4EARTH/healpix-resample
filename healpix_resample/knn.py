"""
knn.py

GPU-friendly sparse HEALPix regridding from unstructured lon/lat samples
to a subset of HEALPix pixels at a target resolution (nside = 2**level).

Core ideas:
- Use HEALPix local neighbourhoods (healpix_geo.kth_neighbourhood) to avoid N×npix distance matrices.
- Build sparse operators M (samples -> grid) and MT (grid -> samples) with Gaussian weights.
- Solve a damped least-squares problem with Conjugate Gradient (CG) on normal equations.

This module is designed for large N and batched values (B,N) on CUDA.
"""

import math
from typing import Generic, Tuple, Optional, Union

import healpix_geo
import numpy as np
import torch

from healpix_resample.base import ResampleResults, T_Array


def _lonlat_to_xyz(lon_rad: torch.Tensor, lat_rad: torch.Tensor) -> torch.Tensor:
    clat = torch.cos(lat_rad)
    return torch.stack([clat * torch.cos(lon_rad),
                        clat * torch.sin(lon_rad),
                        torch.sin(lat_rad)], dim=-1)  # (...,3)

@torch.no_grad()
def _safe_normalize(values: torch.Tensor, sums: torch.Tensor) -> torch.Tensor:
    """``values / sums`` with 0 (instead of NaN) wherever ``sums`` is 0.

    A zero sum means every weight behind it underflowed to 0 (float32 weights
    of samples far from every kept cell, #77), so the links it normalizes
    carry no information: a weight of 0 is the only value consistent with
    the rest of the operator.
    """
    nonzero = sums != 0
    return torch.where(nonzero, values / torch.where(nonzero, sums, torch.ones_like(sums)), torch.zeros_like(values))


def _sigma_level_m(level: int, radius: float = 6371000.0) -> float:
    # sigma = sqrt(4*pi / (12*4**level)) * R
    return math.sqrt(4.0 * math.pi / (12.0 * (4.0 ** level))) * radius
    
# Cells at the same distance as the Npt-th nearest one, up to this tolerance
# (metres), are all linked to the sample (see `_nearest_with_ties`).
TIE_TOLERANCE_M = 1e-6
# at most this many tied cells beyond the Npt-th one
MAX_TIED_EXTRA = 3


def _arc_distance(xyz_a: torch.Tensor, xyz_b: torch.Tensor, radius: float) -> torch.Tensor:
    """Great-circle distance from the chord. For short distances it resolves
    about 1e-9 m, where ``radius * acos(dot)`` resolves only ~0.1 mm (the dot
    product is then within 1e-11 of 1)."""
    chord = torch.linalg.vector_norm(xyz_a - xyz_b, dim=-1)
    return radius * 2.0 * torch.asin(torch.clamp(chord / 2.0, max=1.0))


def _nearest_with_ties(dist: torch.Tensor, cand: torch.Tensor, k: int):
    """The ``k`` nearest candidates of each row (as ``torch.topk``), plus the
    candidates at the same distance as the ``k``-th one, within
    ``TIE_TOLERANCE_M`` (at most ``MAX_TIED_EXTRA`` of them).

    On some HEALPix rings the cell spacing is commensurate with the spacing of
    a regular lon/lat input (e.g. 1/2700 degree cells and 1 arc-second
    inputs at 45.65 N, level 18): whole columns of samples then have their
    ``k``-th and ``(k+1)``-th nearest cells at exactly the same distance.
    Keeping only one of them would leave the choice to the order of
    ``torch.topk`` for equal values or to the last bits of the distances,
    which differ between platforms. All tied cells are linked instead, so the
    operator is symmetric and the same everywhere.

    Returns ``(d_k, i_k, d_x, i_x)``: distances and candidate entries of the
    ``k`` nearest, then of the tied extra ones (``inf`` / ``-1`` where none),
    sorted by distance.
    """
    kk = min(k + MAX_TIED_EXTRA, dist.shape[1])
    d_s, p_s = torch.topk(dist, k=kk, dim=1, largest=False, sorted=True)
    i_s = torch.gather(cand, 1, p_s)
    d_k, i_k = d_s[:, :k], i_s[:, :k]
    d_x, i_x = d_s[:, k:], i_s[:, k:]
    tied = torch.isfinite(d_x) & (d_x <= d_k[:, -1:] + TIE_TOLERANCE_M)
    d_x = torch.where(tied, d_x, torch.full_like(d_x, float("inf")))
    i_x = torch.where(tied, i_x, torch.full_like(i_x, -1))
    return d_k, i_k, d_x, i_x


def _append_tied(idx_out, dist_out, tied_rows, tied_idx, tied_dist):
    """Add the tied extra links (lists of row indices and ``(n, E)`` blocks
    from `_nearest_with_ties`) as extra columns of ``idx_out`` / ``dist_out``
    (``-1`` / ``inf`` in the rows without ties). Without ties the arrays are
    returned unchanged, ``(N, Npt)``."""
    if not tied_rows:
        return idx_out, dist_out
    rows = torch.cat(tied_rows)
    i_x = torch.cat(tied_idx)
    d_x = torch.cat(tied_dist)
    n_extra = int((i_x >= 0).sum(dim=1).max())
    N, Npt = idx_out.shape
    idx_new = torch.full((N, Npt + n_extra), -1, device=idx_out.device, dtype=idx_out.dtype)
    dist_new = torch.full((N, Npt + n_extra), float("inf"), device=dist_out.device, dtype=dist_out.dtype)
    idx_new[:, :Npt] = idx_out
    dist_new[:, :Npt] = dist_out
    idx_new[rows, Npt:] = i_x[:, :n_extra]
    dist_new[rows, Npt:] = d_x[:, :n_extra]
    return idx_new, dist_new


@torch.no_grad()
def healpix_weighted_nearest(
    longitude1: torch.Tensor,          # (N,) degrees
    latitude1: torch.Tensor,           # (N,) degrees
    level: int,
    Npt: int,
    *,
    nest: bool = True,
    threshold: float = 0.1,
    radius: float = 6371000.0,
    ellipsoid: str = "WGS84",
    sigma: float | None = None,
    # allowed subset of output pixels (HEALPix ids at the same "level")
    out_cell_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
    # neighbourhood used to estimate the weights (builds cell_ids)
    ring_weight: Optional[int] = None,
    # neighbourhood used to find Npt neighbours among the kept pixels (can grow automatically)
    ring_search_init: Optional[int] = None,
    ring_search_max: int = 20,
    num_threads: int = 0,
    device_for_dist: Optional[torch.device] = None,
    # process the samples in batches of this size (None: all at once)
    batch_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      cell_ids: (K,) HEALPix pixels (at level) kept by the weight threshold (and intersected with out_cell_ids, if given)
      idx_k   : (N, Npt + E) indices into cell_ids (0..K-1), -1 if not enough neighbours;
                the E extra columns hold the cells at the same distance as the
                Npt-th one (ties, see ``_nearest_with_ties``), -1 elsewhere; E = 0
                when no sample has a tie
      dist_k  : (N, Npt + E) distances (m) to the centres of those pixels, inf if not enough neighbours

    Notes:
      - Geodesic distance from angle = acos(dot(xyz)); dist = R * angle.
      - healpix_geo is used for lonlat_to_healpix, kth_neighbourhood, healpix_to_lonlat.
      - The accuracy is good enough for "pixel centre matching".
      - ``batch_size``: the weight accumulation and the Npt-nearest search
        allocate about ``(2*ring+1)**2 * 100`` bytes per sample (the
        neighbourhood ids, centre coordinates, distances and weights of every
        sample at once). With ``batch_size`` set, the samples are processed in
        slices of that size, so this temporary memory is bounded by the batch
        and not by N; the per-cell weight sums and the per-sample results are
        the same (see ``_healpix_weighted_nearest_batched``).
    """
    assert longitude1.ndim == latitude1.ndim == 1
    N = int(longitude1.numel())
    assert Npt >= 1

    # The geometry runs in float64: healpix_geo returns float64 cell centres,
    # and float32 coordinates would mix dtypes in the distance buffers (#73).
    longitude1 = longitude1.to(torch.float64)
    latitude1 = latitude1.to(torch.float64)

    # --- default rings
    # smallest ring giving >= Npt candidates in a (2r+1)^2 square, plus a margin
    r_min = int(math.ceil((math.sqrt(Npt) - 1.0) / 2.0))
    if ring_weight is None:
        ring_weight = max(2, r_min + 2)  # wider: better estimate of the global weights
    if ring_search_init is None:
        ring_search_init = max(1, r_min + 1)
    # the search below must run at least once
    ring_search_max = max(ring_search_max, ring_search_init)

    # --- distances: computed on the GPU when available
    dev = device_for_dist if device_for_dist is not None else longitude1.device

    if batch_size is not None and Npt > 1:
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        return _healpix_weighted_nearest_batched(
            longitude1,
            latitude1,
            level=level,
            Npt=Npt,
            nest=nest,
            threshold=threshold,
            radius=radius,
            ellipsoid=ellipsoid,
            sigma=sigma,
            out_cell_ids=out_cell_ids,
            ring_weight=ring_weight,
            ring_search_init=ring_search_init,
            ring_search_max=ring_search_max,
            num_threads=num_threads,
            dev=dev,
            batch_size=int(batch_size),
        )

    # --- CPU numpy for healpix_geo
    lon_np = longitude1.detach().cpu().numpy().astype(np.float64)
    lat_np = latitude1.detach().cpu().numpy().astype(np.float64)

    if nest:
        ipix1 = healpix_geo.nested.lonlat_to_healpix(lon_np, lat_np, level, num_threads=num_threads,ellipsoid=ellipsoid)
        ipix_u, inv = np.unique(ipix1.astype(np.uint64), return_inverse=True)
        if Npt==1:
            return torch.from_numpy(ipix_u.astype(np.int64)).to(dev), torch.from_numpy(inv.astype(np.int64)).to(dev),0            
        neigh_u_w = healpix_geo.nested.kth_neighbourhood(ipix_u, level, ring_weight, num_threads=num_threads)
    else:
        ipix1 = healpix_geo.ring.lonlat_to_healpix(lon_np, lat_np, level, num_threads=num_threads,ellipsoid=ellipsoid)
        ipix_u, inv = np.unique(ipix1.astype(np.uint64), return_inverse=True)
        if Npt==1:
            return torch.from_numpy(ipix_u.astype(np.int64)).to(dev), torch.from_numpy(inv.astype(np.int64)).to(dev),0
        neigh_u_w = healpix_geo.ring.kth_neighbourhood(ipix_u, level, ring_weight, num_threads=num_threads)

    # healpix_geo.kth_neighbourhood may return -1 for "invalid" neighbours.
    # We replace invalid entries by a valid neighbour (the last valid in the row; fallback to the center pixel).
    # This keeps arrays dense (no masking) and avoids uint64 overflow issues.
    # Duplicates are later handled naturally by weight accumulation + normalization.

    neigh_u_w = neigh_u_w.astype(np.int64, copy=False)

    valid = neigh_u_w >= 0

    # position of the last valid entry in each row
    last_valid_pos = valid[:, ::-1].argmax(axis=1)
    last_valid_pos = (neigh_u_w.shape[1] - 1) - last_valid_pos
    last_valid = neigh_u_w[np.arange(neigh_u_w.shape[0]), last_valid_pos]

    # fallback when the whole row is invalid -> centre pixel
    all_invalid = ~valid.any(axis=1)
    last_valid[all_invalid] = ipix_u[all_invalid].astype(np.int64, copy=False)

    # replace the -1 entries (explicit broadcast)
    mask = neigh_u_w < 0
    neigh_u_w[mask] = np.broadcast_to(last_valid[:, None], neigh_u_w.shape)[mask]

    # neigh_w : (N, Kw)
    neigh_w = neigh_u_w[inv]
    Kw = neigh_w.shape[1]

    # --- lon/lat centres of the neighbourhood pixels (Kw*N can be large) => computed on unique pixels
    neigh_w_flat = neigh_w.reshape(-1).astype(np.uint64)
    neigh_w_uniq, back = np.unique(neigh_w_flat, return_inverse=True)

    if nest:
        lon_c_deg, lat_c_deg = healpix_geo.nested.healpix_to_lonlat(neigh_w_uniq, level,ellipsoid=ellipsoid)
    else:
        lon_c_deg, lat_c_deg = healpix_geo.ring.healpix_to_lonlat(neigh_w_uniq, level,ellipsoid=ellipsoid)

    lon1 = torch.deg2rad(longitude1.to(dev))
    lat1 = torch.deg2rad(latitude1.to(dev))
    xyz1 = _lonlat_to_xyz(lon1, lat1)  # (N,3)

    lon_c = torch.deg2rad(torch.from_numpy(np.asarray(lon_c_deg)).to(dev))
    lat_c = torch.deg2rad(torch.from_numpy(np.asarray(lat_c_deg)).to(dev))
    xyz_c = _lonlat_to_xyz(lon_c, lat_c)  # (Kuniq,3)

    # Remap neighbours to unique indices (Kuniq)
    back_t = torch.from_numpy(back.astype(np.int64)).to(dev)         # (N*Kw,)
    back_t = back_t.view(N, Kw)                                      # (N,Kw)
    xyz_c_n = xyz_c[back_t]                                          # (N,Kw,3)

    dot = (xyz_c_n * xyz1[:, None, :]).sum(dim=-1)                   # (N,Kw)
    dot = torch.clamp(dot, -1.0, 1.0)
    ang = torch.acos(dot)
    dist = radius * ang                                              # (N,Kw)

    # --- weights and per-pixel sum
    # Same default as KNeighborsResampler.__init__: when sigma is not supplied,
    # fall back to the HEALPix pixel scale sqrt(4*pi/(12*4**level))*R.
    if sigma is None:
        sigma = _sigma_level_m(level, radius=radius)
    # w = exp(-2*d^2/sigma^2)
    w = torch.exp((-2.0) * (dist * dist) / (sigma * sigma))          # (N,Kw)

    # scatter_add over unique pixels
    # w is accumulated on each pixel (Kuniq)
    sums = torch.zeros((xyz_c.shape[0],), device=dev, dtype=w.dtype)
    # Thresholding stage:
    # We compute Gaussian weights from each sample to its neighbourhood pixel centers,
    # then accumulate a global weight sum per HEALPix cell. Only cells whose total
    # influence exceeds 'threshold' are kept in cell_ids_keep (size K).

    sums.scatter_add_(0, back_t.reshape(-1), w.reshape(-1))

    keep = sums >= threshold
    if not torch.any(keep):
        # no pixel passes the threshold -> return empty + -1/inf
        cell_ids = torch.empty((0,), device=dev, dtype=torch.long)
        idx_k = torch.full((N, Npt), -1, device=dev, dtype=torch.long)
        dist_k = torch.full((N, Npt), float("inf"), device=dev, dtype=lon1.dtype)
        return cell_ids, idx_k, dist_k

    # kept cell_ids (HEALPix ids)
    cell_ids_np = neigh_w_uniq  # np.uint64, size Kuniq
    keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)         # indices in [0..Kuniq-1]
    cell_ids_keep = torch.from_numpy(cell_ids_np.astype(np.int64)).to(dev)[keep_idx]  # (K,)

    # xyz of the kept pixels (K,3)
    xyz_keep = xyz_c[keep_idx]


    # --- optional: explicitly restrict the output pixels
    # out_cell_ids can be a list/array/torch tensor of HEALPix ids (at the same level).
    if out_cell_ids is not None:
        out_t = out_cell_ids if isinstance(out_cell_ids, torch.Tensor) else torch.as_tensor(out_cell_ids)
        out_t = out_t.to(device=dev, dtype=torch.long).reshape(-1)
        if out_t.numel() == 0:
            # empty subset requested -> nothing to regrid
            cell_ids = torch.empty((0,), device=dev, dtype=torch.long)
            idx_k = torch.full((N, Npt), -1, device=dev, dtype=torch.long)
            dist_k = torch.full((N, Npt), float("inf"), device=dev, dtype=lon1.dtype)
            return cell_ids, idx_k, dist_k

        out_sorted = torch.unique(out_t)
        out_sorted, _ = torch.sort(out_sorted)

        # robust membership test via searchsorted (avoids depending on torch.isin)
        pos_out = torch.searchsorted(out_sorted, cell_ids_keep)
        pos_out = torch.clamp(pos_out, 0, out_sorted.numel() - 1)
        in_mask = (out_sorted[pos_out] == cell_ids_keep)

        if not torch.any(in_mask):
            cell_ids = torch.empty((0,), device=dev, dtype=torch.long)
            idx_k = torch.full((N, Npt), -1, device=dev, dtype=torch.long)
            dist_k = torch.full((N, Npt), float("inf"), device=dev, dtype=lon1.dtype)
            return cell_ids, idx_k, dist_k

        keep_idx = keep_idx[in_mask]
        cell_ids_keep = cell_ids_keep[in_mask]
        xyz_keep = xyz_c[keep_idx]

    # --- for each point: find the Npt nearest kept pixels
    # Strategy: HEALPix neighbourhood around the pixel containing the point, with a growing ring.
    # Neighbourhood pixels are mapped to indices in cell_ids_keep via sort + searchsorted.
    cell_sorted, order = torch.sort(cell_ids_keep)                   # (K,), (K,)
    K = int(cell_ids_keep.numel())

    idx_out = torch.full((N, Npt), -1, device=dev, dtype=torch.long)
    dist_out = torch.full((N, Npt), float("inf"), device=dev, dtype=lon1.dtype)

    r = ring_search_init
    done = torch.zeros((N,), device=dev, dtype=torch.bool)
    tied_rows, tied_idx, tied_dist = [], [], []

    # kth_neighbourhood runs on the CPU, but only for the rings that are needed.
    # It is computed on unique pixels to limit the number of calls.
    while r <= ring_search_max and not bool(torch.all(done).item()):
        if nest:
            neigh_u = healpix_geo.nested.kth_neighbourhood(ipix_u, level, r, num_threads=num_threads)
        else:
            neigh_u = healpix_geo.ring.kth_neighbourhood(ipix_u, level, r, num_threads=num_threads)

        neigh = neigh_u[inv]  # (N, Ks)
        Ks = neigh.shape[1]
        neigh_t = torch.from_numpy(neigh.astype(np.int64)).to(dev)

        # map neigh_t -> index in cell_ids_keep
        pos = torch.searchsorted(cell_sorted, neigh_t)
        pos = torch.clamp(pos, 0, K - 1)
        hit = (cell_sorted[pos] == neigh_t)
        cand_idx_keep = torch.where(hit, order[pos], torch.full_like(pos, -1))  # indices in cell_ids_keep

        # distances for the valid candidates
        safe = torch.clamp(cand_idx_keep, 0, K - 1)
        xyz2 = xyz_keep[safe]                                  # (N,Ks,3)

        dist2 = _arc_distance(xyz2, xyz1[:, None, :], radius)
        dist2 = torch.where(cand_idx_keep >= 0, dist2, torch.full_like(dist2, float("inf")))

        # the k nearest among these candidates, plus the cells tied with the k-th
        k = min(Npt, Ks)
        d_k, i_k, d_x, i_x = _nearest_with_ties(dist2, cand_idx_keep, k)

        # points not "done" yet are accepted once they have at least Npt valid neighbours (i.e. d_k[:, Npt-1] < inf)
        valid_enough = (d_k[:, -1] < float("inf"))
        update = (~done) & valid_enough

        if torch.any(update):
            idx_out[update] = i_k[update]
            dist_out[update] = d_k[update]
            done[update] = True
            rows = torch.nonzero(update & (i_x >= 0).any(dim=1)).squeeze(1)
            if rows.numel():
                tied_rows.append(rows)
                tied_idx.append(i_x[rows])
                tied_dist.append(d_x[rows])

        r += 1

    # Points that never found Npt kept pixels within ring_search_max rings keep
    # the ones they found at the last (widest) ring, padded with -1/inf. Dropping
    # them all instead left every sample without a link when the samples are
    # much sparser than the cells, and the operator then came out empty (#74).
    partial = ~done
    if torch.any(partial):
        found = torch.isfinite(d_k)
        idx_out[partial, :k] = torch.where(found, i_k, torch.full_like(i_k, -1))[partial]
        dist_out[partial, :k] = d_k[partial]
    idx_out, dist_out = _append_tied(idx_out, dist_out, tied_rows, tied_idx, tied_dist)
    return cell_ids_keep, idx_out, dist_out


def _fill_invalid_neighbours(neigh: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Replace the -1 entries of a ``kth_neighbourhood`` result (rows = pixels
    ``centre``) by the last valid neighbour of the row, or by the centre pixel
    when the whole row is invalid: the same rule as the unbatched search, so
    the arrays stay dense and duplicates are handled by the weight sums."""
    neigh = neigh.astype(np.int64, copy=False)
    valid = neigh >= 0
    last_valid_pos = valid[:, ::-1].argmax(axis=1)
    last_valid_pos = (neigh.shape[1] - 1) - last_valid_pos
    last_valid = neigh[np.arange(neigh.shape[0]), last_valid_pos]
    all_invalid = ~valid.any(axis=1)
    last_valid[all_invalid] = centre[all_invalid].astype(np.int64, copy=False)
    mask = neigh < 0
    neigh[mask] = np.broadcast_to(last_valid[:, None], neigh.shape)[mask]
    return neigh


@torch.no_grad()
def _healpix_weighted_nearest_batched(
    longitude1: torch.Tensor,
    latitude1: torch.Tensor,
    *,
    level: int,
    Npt: int,
    nest: bool,
    threshold: float,
    radius: float,
    ellipsoid: str,
    sigma: float | None,
    out_cell_ids,
    ring_weight: int,
    ring_search_init: int,
    ring_search_max: int,
    num_threads: int,
    dev: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched version of :func:`healpix_weighted_nearest` (same arguments,
    after the defaults have been resolved; ``Npt > 1``).

    The unbatched function holds, for every sample at once, the
    ``(2*ring_weight+1)**2`` neighbourhood pixels with their centre
    coordinates, distances and weights, and then the ``(2*r+1)**2``
    candidates of the nearest search: about 1.5-2 kB per sample, so a
    healpix-convert chunk of 32 million samples needs more than 100 GiB.
    Here the samples are processed in slices of ``batch_size``:

    1. the set of neighbourhood pixels (the candidates for ``cell_ids``) is
       built from the unique sample pixels, in slices;
    2. the per-pixel weight sums used by ``threshold`` are accumulated slice
       by slice into one global array (``scatter_add_`` in sample order, so
       each pixel receives the same contributions in the same order as in
       the unbatched function);
    3. the ``Npt``-nearest search runs slice by slice, with the ring growing
       until every sample of the slice is served (as before, a sample is
       served at the first ring where it finds ``Npt`` kept pixels, so the
       result of a sample does not depend on the other samples of its
       slice).

    Steps 2 and 3 therefore give the same ``cell_ids``, ``idx_k`` and
    ``dist_k`` as the unbatched function; the only possible difference is the
    floating point summation order inside ``scatter_add_``, which does not
    change the ``threshold`` decision except for a pixel whose weight sum is
    within rounding of ``threshold``. The memory of the temporaries is then
    about ``2 kB * batch_size`` plus the global arrays (``O(N * Npt)`` for the
    results and ``O(K)`` for the candidate pixels).
    """
    N = int(longitude1.numel())
    hp = healpix_geo.nested if nest else healpix_geo.ring

    # --- CPU numpy for healpix_geo
    lon_np = longitude1.detach().cpu().numpy().astype(np.float64)
    lat_np = latitude1.detach().cpu().numpy().astype(np.float64)

    ipix1 = hp.lonlat_to_healpix(lon_np, lat_np, level, num_threads=num_threads, ellipsoid=ellipsoid)
    ipix1 = ipix1.astype(np.uint64, copy=False)
    ipix_u = np.unique(ipix1)

    # --- 1. candidate pixels: the union of the ring_weight neighbourhoods of the
    # unique sample pixels, built in slices of the (sorted, hence spatially
    # grouped) unique pixels so that no (U, Kw) array is ever allocated.
    parts = []
    for s in range(0, ipix_u.size, batch_size):
        u = ipix_u[s : s + batch_size]
        neigh = hp.kth_neighbourhood(u, level, ring_weight, num_threads=num_threads)
        neigh = neigh.astype(np.int64, copy=False)
        parts.append(np.unique(neigh[neigh >= 0]).astype(np.uint64))
        parts.append(u)  # fallback of all-invalid rows
    neigh_w_uniq = np.unique(np.concatenate(parts))  # sorted uint64, size Kuniq
    del parts
    Kuniq = int(neigh_w_uniq.size)

    lon_c_deg, lat_c_deg = hp.healpix_to_lonlat(neigh_w_uniq, level, ellipsoid=ellipsoid)
    lon_c = torch.deg2rad(torch.from_numpy(np.asarray(lon_c_deg)).to(dev))
    lat_c = torch.deg2rad(torch.from_numpy(np.asarray(lat_c_deg)).to(dev))
    xyz_c = _lonlat_to_xyz(lon_c, lat_c)  # (Kuniq,3)

    if sigma is None:
        sigma = _sigma_level_m(level, radius=radius)

    lon1_all = torch.deg2rad(longitude1.to(dev))
    lat1_all = torch.deg2rad(latitude1.to(dev))

    # --- 2. per-pixel weight sums, accumulated over the slices
    sums = torch.zeros((Kuniq,), device=dev, dtype=torch.float64)
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        ipu_b, inv_b = np.unique(ipix1[s:e], return_inverse=True)
        neigh_u_w = _fill_invalid_neighbours(
            hp.kth_neighbourhood(ipu_b, level, ring_weight, num_threads=num_threads), ipu_b
        )
        neigh_w = neigh_u_w[inv_b]  # (b, Kw)
        Kw = neigh_w.shape[1]
        back = np.searchsorted(neigh_w_uniq, neigh_w.reshape(-1).astype(np.uint64))
        back_t = torch.from_numpy(back.astype(np.int64)).to(dev).view(e - s, Kw)
        del neigh_u_w, neigh_w, back

        xyz1 = _lonlat_to_xyz(lon1_all[s:e], lat1_all[s:e])  # (b,3)
        dot = (xyz_c[back_t] * xyz1[:, None, :]).sum(dim=-1)  # (b,Kw)
        dot = torch.clamp(dot, -1.0, 1.0)
        dist = radius * torch.acos(dot)
        w = torch.exp((-2.0) * (dist * dist) / (sigma * sigma))
        del dot, dist
        sums.scatter_add_(0, back_t.reshape(-1), w.reshape(-1))
        del back_t, w, xyz1

    keep = sums >= threshold
    if not torch.any(keep):
        cell_ids = torch.empty((0,), device=dev, dtype=torch.long)
        idx_k = torch.full((N, Npt), -1, device=dev, dtype=torch.long)
        dist_k = torch.full((N, Npt), float("inf"), device=dev, dtype=lon1_all.dtype)
        return cell_ids, idx_k, dist_k

    keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    cell_ids_keep = torch.from_numpy(neigh_w_uniq.astype(np.int64)).to(dev)[keep_idx]
    xyz_keep = xyz_c[keep_idx]
    del sums, keep

    # --- optional: explicitly restrict the output pixels (same as unbatched)
    if out_cell_ids is not None:
        out_t = out_cell_ids if isinstance(out_cell_ids, torch.Tensor) else torch.as_tensor(out_cell_ids)
        out_t = out_t.to(device=dev, dtype=torch.long).reshape(-1)
        if out_t.numel() == 0:
            cell_ids = torch.empty((0,), device=dev, dtype=torch.long)
            idx_k = torch.full((N, Npt), -1, device=dev, dtype=torch.long)
            dist_k = torch.full((N, Npt), float("inf"), device=dev, dtype=lon1_all.dtype)
            return cell_ids, idx_k, dist_k

        out_sorted = torch.unique(out_t)
        out_sorted, _ = torch.sort(out_sorted)
        pos_out = torch.searchsorted(out_sorted, cell_ids_keep)
        pos_out = torch.clamp(pos_out, 0, out_sorted.numel() - 1)
        in_mask = (out_sorted[pos_out] == cell_ids_keep)
        if not torch.any(in_mask):
            cell_ids = torch.empty((0,), device=dev, dtype=torch.long)
            idx_k = torch.full((N, Npt), -1, device=dev, dtype=torch.long)
            dist_k = torch.full((N, Npt), float("inf"), device=dev, dtype=lon1_all.dtype)
            return cell_ids, idx_k, dist_k
        keep_idx = keep_idx[in_mask]
        cell_ids_keep = cell_ids_keep[in_mask]
        xyz_keep = xyz_c[keep_idx]
    del xyz_c

    # --- 3. for each sample: the Npt nearest kept pixels, slice by slice
    cell_sorted, order = torch.sort(cell_ids_keep)
    K = int(cell_ids_keep.numel())

    idx_out = torch.full((N, Npt), -1, device=dev, dtype=torch.long)
    dist_out = torch.full((N, Npt), float("inf"), device=dev, dtype=lon1_all.dtype)
    tied_rows, tied_idx, tied_dist = [], [], []

    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        b = e - s
        ipu_b, inv_b = np.unique(ipix1[s:e], return_inverse=True)
        xyz1 = _lonlat_to_xyz(lon1_all[s:e], lat1_all[s:e])  # (b,3)
        idx_b = idx_out[s:e]
        dist_b = dist_out[s:e]
        done = torch.zeros((b,), device=dev, dtype=torch.bool)

        r = ring_search_init
        while r <= ring_search_max and not bool(torch.all(done).item()):
            neigh_u = hp.kth_neighbourhood(ipu_b, level, r, num_threads=num_threads)
            neigh_t = torch.from_numpy(neigh_u[inv_b].astype(np.int64)).to(dev)  # (b, Ks)
            Ks = neigh_t.shape[1]

            pos = torch.searchsorted(cell_sorted, neigh_t)
            pos = torch.clamp(pos, 0, K - 1)
            hit = (cell_sorted[pos] == neigh_t)
            cand_idx_keep = torch.where(hit, order[pos], torch.full_like(pos, -1))
            del neigh_t, pos, hit

            safe = torch.clamp(cand_idx_keep, 0, K - 1)
            dist2 = _arc_distance(xyz_keep[safe], xyz1[:, None, :], radius)
            del safe
            dist2 = torch.where(cand_idx_keep >= 0, dist2, torch.full_like(dist2, float("inf")))

            k = min(Npt, Ks)
            d_k, i_k, d_x, i_x = _nearest_with_ties(dist2, cand_idx_keep, k)
            del dist2, cand_idx_keep

            valid_enough = (d_k[:, -1] < float("inf"))
            update = (~done) & valid_enough
            if torch.any(update):
                idx_b[update] = i_k[update]
                dist_b[update] = d_k[update]
                done[update] = True
                rows = torch.nonzero(update & (i_x >= 0).any(dim=1)).squeeze(1)
                if rows.numel():
                    tied_rows.append(rows + s)
                    tied_idx.append(i_x[rows])
                    tied_dist.append(d_x[rows])
            r += 1

        partial = ~done
        if torch.any(partial):
            found = torch.isfinite(d_k)
            idx_b[partial, :k] = torch.where(found, i_k, torch.full_like(i_k, -1))[partial]
            dist_b[partial, :k] = d_k[partial]

    idx_out, dist_out = _append_tied(idx_out, dist_out, tied_rows, tied_idx, tied_dist)
    return cell_ids_keep, idx_out, dist_out


class KNeighborsResampler(Generic[T_Array]):
    """GPU-friendly sparse HEALPix regridding via local Gaussian weights + CG deconvolution.

    This class builds two sparse operators from unstructured lon/lat samples to a subset
    of HEALPix pixels at a target resolution (nside = 2**level).

    Notation (matching your notebook):
      - N: number of samples (lon/lat)
      - K: number of kept HEALPix cells (cell_ids)
      - M: operator of shape (N, K)   (named ``M`` here)
      - MT: operator of shape (K, N)  (named ``MT`` here)

    The solver estimates ``hval`` (B,K) such that
    ``M @ hval.T`` matches ``val`` (B,N),
    by solving a damped normal equation around a reference field ``x_ref = val @ M``.
    """

    def __init__(
        self,
        lon_deg: T_Array,
        lat_deg: T_Array,
        Npt: int,
        level: int,
        *,
        nest: bool = True,
        radius: float = 6371000.0,
        ellipsoid: str = "WGS84",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        ring_weight: Optional[int] = None,
        ring_search_init: Optional[int] = None,
        ring_search_max: int = 2,
        num_threads: int = 0,
        threshold: float = 0.1,
        sigma_m: float | None = None,
        verbose: bool = True,
        out_cell_ids: T_Array | None = None,
        group_by: bool = False,
        batch_size: int | None = None,
    ) -> None:
        """Pre-compute sparse operators.

        Args:
            lon_deg, lat_deg: unstructured sample coordinates in degrees, shape (N,);
                float32 or float64, the geometry is computed in float64
            Npt: number of nearest HEALPix cells used per sample. A sample that
                does not find Npt kept cells within ``ring_search_max`` rings
                (samples much sparser than the cells) keeps the ones it found.
            ring_search_init, ring_search_max: first and last HEALPix ring of
                the per-sample neighbour search; a ``ring_search_max`` below
                ``ring_search_init`` is raised to it
            level: HEALPix level, nside = 2**level
            sigma_m: Gaussian length scale (meters). If None, uses the HEALPix pixel scale
                     sigma = sqrt(4*pi/(12*4**level))*R.
            threshold: keep only HEALPix cells whose global weight sum >= threshold
            nest: HEALPix indexing scheme
            dtype/device: torch dtype/device for all matrices and computations
            out_cell_ids: optional subset of ``level`` cells to restrict the
                output to (e.g. one chunk or one parent cell). If none of
                these cells has enough support among the samples, the
                operator is built with ``K = 0`` instead of raising:
                ``resample()`` then returns ``cell_ids`` of size 0 and
                ``cell_data`` of shape ``(B, 0)``, except
                ``NearestResampler``, whose fallback fills the requested
                cells with their nearest sample (see its ``max_distance_m``).
            batch_size: process the samples in slices of this size when
                building the operators (see
                ``healpix_weighted_nearest``): the temporary memory of
                the neighbour search is then about 2 kB per sample of a
                slice instead of 2 kB per sample. ``None`` (default)
                processes all samples at once. The result is the same.

        Raises:
            RuntimeError: without ``out_cell_ids``, when no cell passes
                ``threshold`` (lower ``threshold`` or widen the neighbourhood
                rings), or when cells pass it but no sample reaches any of
                them within ``ring_search_max`` rings.
        """
        self.level = int(level)
        self.nside = 2 ** int(level)
        self.group_by = bool(group_by)
        if not self.group_by:
            self.Npt = int(Npt)
            
        self.nest = bool(nest)
        self.radius = float(radius)
        self.ellipsoid = str(ellipsoid)
        self.dtype = dtype
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            # device may already be a torch.device (e.g. passed through from
            # another resampler's .device attribute) rather than a string --
            # only str has .startswith(). Same guard already used in
            # clough_tocher.py's equivalent constructor.
            if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")

        self.device = torch.device(device)
        self.threshold = float(threshold)
        self.out_cell_ids = out_cell_ids
        self.batch_size = None if batch_size is None else int(batch_size)
        
        # --- sigma in meters (controls the Gaussian weights used for thresholding)
        sigma = float(_sigma_level_m(level, radius=radius) if sigma_m is None else sigma_m)
        self.sigma_m = sigma
        self.verbose = verbose

        # --- move lon/lat to torch on target device (but healpix_geo needs CPU numpy internally)
        lon_t = lon_deg if isinstance(lon_deg, torch.Tensor) else torch.as_tensor(lon_deg)
        lat_t = lat_deg if isinstance(lat_deg, torch.Tensor) else torch.as_tensor(lat_deg)
                
        if self.group_by:    
            if self.nest:
                cell_ids = healpix_geo.nested.lonlat_to_healpix(lon_t,lat_t, 
                                                                self.level,
                                                                ellipsoid=self.ellipsoid)
            else:
                cell_ids = healpix_geo.ring.lonlat_to_healpix(lon_t,lat_t, 
                                                                self.level,
                                                                ellipsoid=self.ellipsoid)
                                                                
            cell_ids,hi = torch.unique(torch.tensor(cell_ids,dtype=torch.long,device=self.device),
                                        return_inverse=True)
                                        
            self.cell_ids = cell_ids
            self.hi = hi
            
        else:
            # float64 geometry, whatever the dtype of the input coordinates (#73)
            lon_t = lon_t.to(self.device, dtype=torch.float64)
            lat_t = lat_t.to(self.device, dtype=torch.float64)
            
            if self.out_cell_ids is not None:
                self.xyz_samples = _lonlat_to_xyz(torch.deg2rad(lon_t),torch.deg2rad(lat_t))  # (N,3)
            
            # --- get kept healpix cells + per-sample nearest indices + distances
            cell_ids, hi, d = healpix_weighted_nearest(
                lon_t,
                lat_t,
                level=self.level,
                Npt=self.Npt,
                nest=self.nest,
                threshold=self.threshold,
                radius=self.radius,
                ellipsoid=self.ellipsoid,
                sigma=self.sigma_m,
                out_cell_ids=self.out_cell_ids,
                ring_weight=ring_weight,
                ring_search_init=ring_search_init,
                ring_search_max=ring_search_max,
                num_threads=num_threads,
                device_for_dist=self.device,
                batch_size=self.batch_size,
            )
            

            # With `out_cell_ids`, an empty result is a legitimate outcome: none of
            # the requested cells has enough support among the samples (e.g. a
            # chunk at the edge of the input whose nearby samples all lie just
            # outside it). The operator is then built with K = 0 and every
            # resampler handles the requested cells as it does any unsupported
            # cell. Without `out_cell_ids` every sample supports its own cell, so
            # an empty result means the threshold or the rings are wrong.
            if cell_ids.numel() == 0 and self.out_cell_ids is None:
                raise RuntimeError(
                    "No HEALPix cell passed the threshold. "
                    "Lower 'threshold' or increase neighbourhood rings."
                )

            # Store geometry outputs
            self.cell_ids = cell_ids.to(torch.long).to(self.device)   # (K,)
            self.hi = hi.to(torch.long).to(self.device)               # (N,Npt+E) indices into cell_ids (E: tied cells)
            if Npt>1:
                self.d_m = d.to(self.dtype).to(self.device)               # (N,Npt) meters

                # --- prune retained cells with zero actual sample links -----
                # `healpix_weighted_nearest` decides which cells to *retain*
                # (self.cell_ids) via a wide-radius (`ring_weight`) accumulated
                # weight-sum threshold test, but separately searches each
                # sample's own nearest `Npt` cells *among the retained set*
                # using a different, narrower `ring_search_init`/
                # `ring_search_max` radius. These two criteria are not
                # guaranteed consistent: a cell can pass the wide-radius
                # threshold test via many samples' small individual
                # contributions while never actually being one of any single
                # sample's own `Npt`-nearest retained cells. Such a cell ends
                # up with zero (i, k) links in `self.hi` -- an entirely empty
                # column once `comp_matrix()` builds the sparse `M`/`MT` -- so
                # every resampler built on this class silently returns exactly
                # `0.0` for it, for *any* input value (not just NaN). This was
                # tracked down from the "all-NaN input produces some non-NaN
                # cells" report in `planning/03_bilinear_nan_investigation.md`
                # -- confirmed by direct investigation to be this, not a
                # CSR/COO sparse-matmul NaN-propagation bug (both formats
                # agree exactly, since neither has any stored entry for these
                # columns to begin with).
                #
                # Only prune in the default (`out_cell_ids=None`)
                # auto-discovery mode: if the caller explicitly requested a
                # specific cell subset via `out_cell_ids`, don't silently drop
                # any of it here. `PSFResampler` already has its own
                # fallback-fill mechanism for weakly-supported `out_cell_ids`
                # columns (see `psf.py:comp_matrix`'s `bad_k` handling); other
                # resamplers' `out_cell_ids` gap is a separate, pre-existing
                # limitation out of scope for this fix.
                if self.out_cell_ids is None:
                    K_before = int(self.cell_ids.numel())
                    reachable = torch.zeros(K_before, dtype=torch.bool, device=self.device)
                    valid_hi = self.hi[self.hi >= 0]
                    if valid_hi.numel() > 0:
                        reachable[valid_hi] = True

                    n_orphaned = int((~reachable).sum())
                    if n_orphaned > 0:
                        if self.verbose:
                            print(
                                f"[KNeighborsResampler] dropping {n_orphaned:,} retained "
                                f"cell(s) that passed the wide-neighbourhood 'threshold' "
                                f"test but were never selected by any sample's own "
                                f"Npt-nearest search (would otherwise silently read back "
                                f"as 0.0 for any input)."
                            )
                        new_index = torch.full((K_before,), -1, dtype=torch.long, device=self.device)
                        new_index[reachable] = torch.arange(
                            int(reachable.sum()), device=self.device
                        )

                        self.cell_ids = self.cell_ids[reachable]
                        self.hi = torch.where(
                            self.hi >= 0, new_index[self.hi.clamp(min=0)], self.hi
                        )

                    # Cells passed the threshold, but no sample reached any of
                    # them: an empty operator would be returned silently (#74).
                    if self.cell_ids.numel() == 0:
                        raise RuntimeError(
                            f"{K_before:,} HEALPix cell(s) passed the threshold, but no "
                            f"sample found any of them within ring_search_max rings. "
                            f"Increase 'ring_search_max' or use a coarser 'level'."
                        )

        self.K = int(self.cell_ids.numel())
                
        self.N = int(lon_t.numel())
 
        if self.out_cell_ids is not None:
            
            # --- geometry buffers for optional fallbacks (e.g. when out_cell_ids forces empty columns)

            # unit vectors for output HEALPix cell centers (K,3)
            cell_np = self.cell_ids.detach().cpu().numpy().astype(np.uint64)
            
            if self.nest:
                lon_c_deg, lat_c_deg = healpix_geo.nested.healpix_to_lonlat(cell_np, self.level, ellipsoid=self.ellipsoid)
            else:
                lon_c_deg, lat_c_deg = healpix_geo.ring.healpix_to_lonlat(cell_np, self.level, ellipsoid=self.ellipsoid)

            lon_c = torch.deg2rad(torch.as_tensor(lon_c_deg, device=self.device, dtype=self.xyz_samples.dtype))
            lat_c = torch.deg2rad(torch.as_tensor(lat_c_deg, device=self.device, dtype=self.xyz_samples.dtype))
            
            self.xyz_cells = _lonlat_to_xyz(lon_c, lat_c)  # (K,3)
        
        if not self.group_by:
            self.comp_matrix()
        
    # ------------------------------------------------------------------
    # Sparse operators M (N,K), MT (K,N), M_cons (N,K)
    #
    # `comp_matrix()` only builds `M`, the operator `resample()` uses, and
    # builds it in CSC layout: `val @ M` runs over the cells, so torch
    # transposes a CSR `M` to CSC on every call (a full sort of the N*Npt
    # links, ~200 B per sample of temporaries, several seconds); stored in
    # CSC it is a plain spMM. The links come out of the neighbour search
    # one row per sample, so the per-cell normalisation is a bincount and
    # the CSC order is one stable argsort of the cell indices -- no
    # (2, N*Npt) COO index tensor and no `coalesce()`. When every sample has
    # Npt links (the dense case) the sorted arrays are reused as views and
    # normalised in place, so the construction holds ~4 arrays of N*Npt.
    #
    # `MT` (used by `invert()` and the CG solvers) and `M_cons` (used by
    # `resample(conservative=True)`) are built on first access and cached,
    # so a caller that only interpolates never pays for them. All three are
    # identical, entry by entry, to the former eager COO -> coalesce -> CSR
    # construction (same links, same values; `M` differs only in layout).
    #
    # Subclasses customise the weights through `_link_weights()`,
    # `_norm_col()` and `_norm_row()` rather than overriding `comp_matrix()`.
    # ------------------------------------------------------------------

    def _link_weights(self) -> torch.Tensor:
        """Raw (unnormalised) weight of every (sample, cell) link, ``(N, Npt)``.

        Entries where ``self.hi < 0`` (sample found fewer than Npt cells)
        are ignored by the callers. Gaussian ``exp(-2 d^2 / sigma^2)`` here;
        ``BilinearResampler`` and ``BicubicResampler`` override it.
        """
        return torch.exp((-2.0) * (self.d_m * self.d_m) / (self.sigma_m * self.sigma_m))

    def _norm_col(self, cols: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Per-cell weight sum ``(K,)`` that `M`'s columns are divided by."""
        return torch.bincount(cols, weights=w, minlength=self.K).to(self.dtype)

    def _norm_row(self, rows: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Per-sample weight sum ``(N,)`` that `MT`'s / `M_cons`'s rows are divided by."""
        return torch.bincount(rows, weights=w, minlength=self.N).to(self.dtype)

    def _sorted_links(self):
        """Valid links in row-major order (by sample, cells sorted within a sample).

        Returns ``(cols, w, counts, duplicated)``: cell index and raw weight
        of every valid link (flat, ``N*Npt`` long when every sample has Npt
        links -- then views of the sorted ``(N, Npt)`` arrays), the number
        of valid links per sample ``(N,)`` or ``None`` when all equal Npt,
        and whether some sample links the same cell twice (then the order is
        not coalesced and the caller falls back to COO).
        """
        hi_s, perm = torch.sort(self.hi, dim=1)
        w_s = torch.gather(self._link_weights(), 1, perm)
        del perm
        valid = hi_s >= 0
        duplicated = bool(((hi_s[:, 1:] == hi_s[:, :-1]) & valid[:, 1:]).any())
        if bool(valid.all()):
            return hi_s.reshape(-1), w_s.reshape(-1), None, duplicated
        counts = valid.sum(dim=1)
        cols = hi_s[valid]
        del hi_s
        w = w_s[valid]
        return cols, w, counts, duplicated

    def _flat_links(self):
        """Valid links in neighbour-search order (``hi.reshape(-1)``), as the
        eager construction enumerated them: ``(rows, cols, w)``."""
        flat_hi = self.hi.reshape(-1)
        valid = flat_hi >= 0
        rows = torch.arange(self.N, device=self.device, dtype=torch.long)[:, None].expand(self.hi.shape).reshape(-1)[valid]
        return rows, flat_hi[valid], self._link_weights().reshape(-1)[valid]

    def _csr_from_coo(self, rows: torch.Tensor, cols: torch.Tensor, vals: torch.Tensor, shape) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            torch.stack([rows, cols], dim=0), vals.to(self.dtype),
            size=shape, device=self.device, dtype=self.dtype,
        ).coalesce().to_sparse_csr()

    def comp_matrix(self):
        """Build ``self.M`` (N,K), sparse CSC: links weighted by `_link_weights()`,
        normalised per cell (column). ``MT`` and ``M_cons`` are built lazily."""
        self._MT = None
        self._M_cons = None
        cols, w, counts, duplicated = self._sorted_links()
        if duplicated:
            # A sample linking the same cell twice: `coalesce()` adds the two
            # weights; keep that behaviour on this (unseen in practice) path.
            del cols, w, counts
            rows, cols, w = self._flat_links()
            norm_col = self._norm_col(cols, w)
            self.M = self._csr_from_coo(rows, cols, _safe_normalize(w, norm_col[cols]), (self.N, self.K)).to_sparse_csc()
            return
        norm_col = self._norm_col(cols, w)
        # per-cell normalisation (0 where every weight of the cell underflowed)
        w = _safe_normalize(w.to(self.dtype), norm_col[cols])
        del norm_col
        ccol = torch.zeros(self.K + 1, dtype=torch.long, device=self.device)
        torch.cumsum(torch.bincount(cols, minlength=self.K), 0, out=ccol[1:])
        perm = torch.argsort(cols, stable=True)   # row-major -> column-major
        del cols
        vals = w[perm]
        del w
        if counts is None:
            # every sample has Npt links: flat link l belongs to sample l // Npt
            perm.floor_divide_(self.hi.shape[1])  # links per sample
            rows = perm
        else:
            rows = torch.repeat_interleave(
                torch.arange(self.N, device=self.device, dtype=torch.long), counts
            )[perm]
            del perm
        self.M = torch.sparse_csc_tensor(
            ccol, rows, vals, size=(self.N, self.K), device=self.device, dtype=self.dtype,
            check_invariants=False,  # sorted and in range by construction
        )

    def _build_MT(self) -> None:
        # (K,N), normalised per sample (row of M): weights and sums are
        # computed in the search order like the former eager construction.
        # A sample whose weights all underflowed (float32, far from every
        # kept cell: #77) has norm_row 0; its links get 0, not 0/0 = NaN.
        rows, cols, w = self._flat_links()
        norm_row = self._norm_row(rows, w)
        self._MT = self._csr_from_coo(cols, rows, _safe_normalize(w, norm_row[rows]), (self.K, self.N))

    def _build_M_cons(self) -> None:
        # (N,K) CSR, same links as M, normalised per sample (issue #44).
        rows, cols, w = self._flat_links()
        norm_row = self._norm_row(rows, w)
        self._M_cons = self._csr_from_coo(rows, cols, _safe_normalize(w, norm_row[rows]), (self.N, self.K))

    @property
    def MT(self) -> torch.Tensor:
        """Sparse CSR ``(K, N)`` operator, cells -> samples (``invert()``); built on first access."""
        if getattr(self, "_MT", None) is None:
            self._build_MT()
        return self._MT

    @MT.setter
    def MT(self, value: torch.Tensor) -> None:
        self._MT = value

    @property
    def M_cons(self) -> torch.Tensor:
        """Sparse CSR ``(N, K)`` conservative operator (``resample(conservative=True)``,
        same links as ``M``, each sample's weights summing to 1); built on first access."""
        if getattr(self, "_M_cons", None) is None:
            self._build_M_cons()
        return self._M_cons

    @M_cons.setter
    def M_cons(self, value: torch.Tensor) -> None:
        self._M_cons = value

    @torch.no_grad()
    def invert(self, hval: T_Array) -> T_Array:
        """Project HEALPix field back to the sample locations.

        Args:
            hval: (B,K) or (K,) on same device
        Returns:
            val_hat: (B,N) or (N,)
        """
        y = hval if isinstance(hval, torch.Tensor) else torch.as_tensor(hval)
        y = y.to(self.device, dtype=self.dtype)
        if hval.ndim == 1:
            res = (y[None, :] @ self.MT)[0]
        else:
            res =  y @ self.MT
        
        if not isinstance(hval, torch.Tensor):
            res=res.cpu().numpy()
        
        return res

    @torch.no_grad()
    def resample(
        self,
        val: T_Array,
        *,
        lam: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-8,
        x0: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ) -> ResampleResults[T_Array]:
        """Estimate the HEALPix field from unstructured samples.

        Args:
            val: (B,N) or (N,) values at lon/lat sample points
            lam: Tikhonov regularization strength (damping) used in CG
            max_iter, tol: CG parameters
            x0: optional initial guess for the *delta* around x_ref, shape (B,K)
            return_info: whether to return CG diagnostics

        Returns:
            hval: (B,K) or (K,)
            (optional) info: CG information dict
        """
        y = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
        y = y.to(self.device, dtype=self.dtype)
        clean_shape=False
        if y.ndim == 1:
            clean_shape=True
            y = y[None, :]

        # reference field (B,K)
        hval = y @ self.M
        cell_ids = self.cell_ids
        
        if not isinstance(val, torch.Tensor):
            hval= hval.cpu().numpy()
            cell_ids = cell_ids.cpu().numpy()
        if clean_shape:
            hval = hval[0]
        
        return ResampleResults(cell_data=hval, cell_ids=cell_ids)
        
    def get_cell_ids(self):
        return self.cell_ids.cpu().numpy()


@torch.no_grad()
def _conservative_resample(op: "KNeighborsResampler", val: T_Array, area: torch.Tensor) -> ResampleResults:
    """Shared conservative-mode forward pass for KNN-based interpolating
    resamplers (`BilinearResampler`, `BicubicResampler`) that build an
    `M_cons` operator in their own `comp_matrix()` -- see either class's
    `resample(conservative=True)` docstring for the full derivation and
    guarantee. `op.M_cons` is `(N, K)`, with each row (source sample) summing
    to exactly 1 (a partition of unity) for well-conditioned samples -- see
    each caller's own comp_matrix()/docstring for when that can be only
    approximate (e.g. BicubicResampler's signed-kernel floor).

    Because `M_cons`'s per-sample normalization is fixed at construction
    time (unlike `M`'s per-cell normalization, which implicitly assumes
    every linked sample is present), zeroing a NaN sample's value *and*
    excluding its area here is enough on its own to keep the exact identity

        sum_k hval[k] == sum_i (valid i) val[i] * area[i]

    with no separate renormalization step needed -- a cleaner NaN story
    than the interpolation path's (see `KNeighborsResampler.resample()`'s
    docstring for that caveat).
    """
    y = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
    y = y.to(op.device, dtype=op.dtype)
    squeezed = y.ndim == 1
    if squeezed:
        y = y.unsqueeze(0)  # (1, N)

    nan_mask = torch.isnan(y)
    valid = ~nan_mask
    y_filled = torch.where(valid, y, torch.zeros_like(y))
    row_has_data = valid.any(dim=-1)  # (B,)

    weighted = y_filled * area  # (B, N), area (N,) broadcasts over rows
    hval = weighted @ op.M_cons  # (B, K)
    hval[~row_has_data] = float("nan")

    cell_ids = op.cell_ids
    if squeezed:
        hval = hval.squeeze(0)
    if not isinstance(val, torch.Tensor):
        hval = hval.cpu().numpy()
        cell_ids = cell_ids.cpu().numpy()

    return ResampleResults(cell_data=hval, cell_ids=cell_ids)

