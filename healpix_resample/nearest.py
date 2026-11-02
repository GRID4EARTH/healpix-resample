"""
nearest.py

Nearest-neighbour HEALPix resampler built on top of KNeighborsResampler.

Strategy
--------
1. Run the standard KNN (Npt neighbours per source sample) — this gives
   ``hi (N, Npt)`` (cell indices per source) and ``d_m (N, Npt)`` (distances).
2. Compute Gaussian weights  w[n,j] = exp(-2 d²/σ²).
3. For each HEALPix cell k, keep only the source n that maximises w  →  hi_k (K,).
   This is a single ``scatter_reduce('amax')`` pass — O(N·Npt), no sparse matrix.
4. ``resample`` : hval[:, k] = val[:, hi_k[k]]            (direct index)
   ``invert``   : val[:, n]  = mean of hval[:, k] ∀k s.t. hi_k[k]==n  (scatter-mean)

When ``out_cell_ids`` is provided
----------------------------------
``healpix_weighted_nearest`` already intersects the KNN result with ``out_cell_ids``,
but cells beyond ``ring_search_max`` rings may be absent.  After the KNN step,
missing cells are filled from their nearest sample, found with a KD-tree.
"""
from __future__ import annotations

from typing import Optional

import healpix_geo
import numpy as np
import torch
from scipy.spatial import cKDTree

from healpix_resample.base import ResampleResults, T_Array
from healpix_resample.knn import KNeighborsResampler, _lonlat_to_xyz


# ─────────────────────────────────────────────────────────────────────────────
# Helper: scatter-mean
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_mean(src: torch.Tensor, idx: torch.Tensor, T: int) -> torch.Tensor:
    """out[:, t] = mean of src[:, s] where idx[s] == t.  Shape (B, S) → (B, T)."""
    B   = src.shape[0]
    out = torch.zeros(B, T, device=src.device, dtype=src.dtype)
    out.scatter_add_(1, idx.unsqueeze(0).expand(B, -1), src)
    count = torch.bincount(idx, minlength=T).to(src.dtype).clamp(min=1)
    out /= count.unsqueeze(0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# NearestResampler
# ─────────────────────────────────────────────────────────────────────────────

class NearestResampler(KNeighborsResampler):
    """Nearest-neighbour HEALPix resampler — no sparse matrices.

    Uses ``Npt`` KNN neighbours (default 9) to robustly find the nearest source
    for every HEALPix cell, even when the output grid is finer than the input.

    Parameters
    ----------
    Npt : int
        Number of HEALPix neighbours per source sample used by the KNN.
        Larger values cover more cells at the cost of more memory during
        construction.  Default 9 is a good trade-off for most use cases.
    max_distance_m : float or None
        Only used with ``out_cell_ids``. A requested cell that the KNN rings
        did not reach is filled with its nearest sample only if that sample
        lies within ``max_distance_m`` metres of the cell centre; otherwise
        the cell is left out of the result, like the cells the interpolating
        resamplers cannot compute. ``None`` (default) fills every requested
        cell, however far its nearest sample is.
    search : {"knn", "kdtree"}
        How the nearest sample of each cell is found. ``"knn"`` (default):
        through the ``Npt`` nearest cells of every sample (the KNN links of
        ``KNeighborsResampler``), the cells they miss from a KD-tree.
        ``"kdtree"`` (with ``out_cell_ids`` only): every requested cell takes
        its nearest sample from a KD-tree of the samples, directly: the same
        sample (the KNN path keeps, per cell, the linked sample of largest
        Gaussian weight, i.e. the nearest), several times faster on dense
        inputs, and no KNN links (``neighbours``) to search or keep. Samples
        exactly as near as each other may be picked differently. On a CUDA
        device (nested cells), the search runs on the GPU from HEALPix
        buckets of the samples (see :meth:`_nearest_source_buckets`): the
        same samples as the KD-tree, ties apart.
        ``max_distance_m`` then applies to every requested cell (the KNN
        path applies it only to the cells its rings did not reach).
    All other parameters are forwarded to ``KNeighborsResampler``.
    """

    def __init__(self, *args, Npt: int = 9, max_distance_m: Optional[float] = None,
                 search: str = "knn", **kwargs):
        if max_distance_m is not None and not max_distance_m > 0:
            raise ValueError(f"max_distance_m must be None or positive, got {max_distance_m!r}.")
        if search not in ("knn", "kdtree"):
            raise ValueError(f"search must be 'knn' or 'kdtree', got {search!r}.")
        self.max_distance_m = max_distance_m
        if search == "kdtree":
            self._init_kdtree(*args, Npt=Npt, **kwargs)
            return
        # Ensure ring_search_max >= ring_search_init(Npt) so the KNN search
        # loop in healpix_weighted_nearest actually executes.
        #
        # healpix_weighted_nearest computes:
        #   r_min            = ceil((sqrt(Npt) - 1) / 2)
        #   ring_search_init = max(1, r_min + 1)
        #
        # KNeighborsResampler default is ring_search_max=2, which is too small
        # for Npt >= 16 (needs ring_search_init=3).  We auto-correct here only
        # when the caller has not supplied ring_search_max explicitly.
        if "ring_search_max" not in kwargs:
            import math as _math
            r_min = int(_math.ceil((_math.sqrt(Npt) - 1.0) / 2.0))
            ring_search_init_needed = max(1, r_min + 1)
            # +2 margin so the loop has room to grow and find Npt candidates
            kwargs["ring_search_max"] = ring_search_init_needed + 2

        # super().__init__ calls self.comp_matrix() at the end — our override
        # will be called there, building hi_k instead of sparse M/MT.
        super().__init__(*args, Npt=Npt, **kwargs)

        # If out_cell_ids was requested, fill any cells the KNN rings missed.
        if self.out_cell_ids is not None:
            self._fill_missing_out_cells()
            # Geometry buffers only needed during construction — free them.
            for attr in ("xyz_samples", "xyz_cells"):
                if hasattr(self, attr):
                    delattr(self, attr)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _init_kdtree(self, *args, **kwargs) -> None:
        """``search="kdtree"``: the requested cells and the nearest sample of
        each, from a KD-tree of the samples (no KNN links, no operator)."""
        import inspect

        bound = inspect.signature(KNeighborsResampler.__init__).bind(self, *args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        if a["out_cell_ids"] is None:
            raise ValueError("search='kdtree' needs out_cell_ids (the cells to resample onto).")
        self.level, self.Npt = int(a["level"]), int(a["Npt"])
        self.nest, self.radius = bool(a["nest"]), float(a["radius"])
        self.ellipsoid, self.dtype = str(a["ellipsoid"]), a["dtype"]
        device = a["device"]
        self.device = torch.device(device if device is not None else "cpu")
        self.verbose, self.group_by = a["verbose"], False
        self.out_cell_ids = a["out_cell_ids"]
        self.threshold = float(a["threshold"])
        self.neighbours = None

        lon = torch.as_tensor(a["lon_deg"]).to(dtype=torch.float64).reshape(-1)
        lat = torch.as_tensor(a["lat_deg"]).to(dtype=torch.float64).reshape(-1)
        self.N = int(lon.numel())
        out = torch.as_tensor(self.out_cell_ids).to(torch.long).reshape(-1)
        cells = torch.unique(out)                                   # ascending
        if self.N == 0 or cells.numel() == 0:
            self.cell_ids = cells[:0].to(self.device)
            self.hi_k = torch.empty((0,), dtype=torch.long, device=self.device)
            self.K = 0
            return

        hp = healpix_geo.nested if self.nest else healpix_geo.ring
        lon_c, lat_c = hp.healpix_to_lonlat(cells.numpy().astype(np.uint64), self.level,
                                            ellipsoid=self.ellipsoid)
        xyz_cells = _lonlat_to_xyz(
            torch.deg2rad(torch.as_tensor(np.asarray(lon_c), dtype=torch.float64)),
            torch.deg2rad(torch.as_tensor(np.asarray(lat_c), dtype=torch.float64)),
        )
        xyz_samples = _lonlat_to_xyz(torch.deg2rad(lon), torch.deg2rad(lat))
        if self.device.type == "cuda" and self.nest:
            hi, dot = self._nearest_source_buckets(
                cells, self.level, lon, lat, xyz_cells, xyz_samples,
                ellipsoid=self.ellipsoid, device=self.device)
        else:
            hi, dot = self._nearest_source(xyz_cells, xyz_samples)
        if self.max_distance_m is not None:
            near = self.radius * torch.acos(dot.clamp(-1.0, 1.0)) <= self.max_distance_m
            cells, hi = cells[near], hi[near]
        self.cell_ids = cells.to(self.device)
        self.hi_k = hi.to(self.device)
        self.K = int(self.cell_ids.numel())

    # ── comp_matrix: scatter_reduce instead of sparse M / MT ─────────────────

    def comp_matrix(self) -> None:
        """Build hi_k (K,) — nearest source index per HEALPix cell.

        Algorithm
        ---------
        1. Compute Gaussian weights w[n,j] from the KNN distances d_m[n,j].
        2. Flatten (N, Npt) → (N*Npt,) pairs (source_n, cell_k, weight).
        3. scatter_reduce('amax') over cell_k → max weight per cell  O(N·Npt).
        4. Keep the winner per cell (deduplicate ties with stable sort).

        Result: ``self.hi_k`` (K,) long — index into [0..N-1] for each cell.
        """
        # w[n,j] = exp(-2 d²/σ²)  ─ same formula as KNeighborsResampler
        w = torch.exp(
            (-2.0) * (self.d_m * self.d_m) / (self.sigma_m * self.sigma_m)
        )                                                           # (N, Npt)

        flat_hi = self.hi.reshape(-1)                               # (N*Npt,)
        flat_w  = w.reshape(-1)                                     # (N*Npt,)
        flat_n  = (
            torch.arange(self.N, device=self.device, dtype=torch.long)
            .unsqueeze(1).expand(self.N, self.Npt).reshape(-1)
        )                                                           # (N*Npt,)
        del w

        # Discard invalid links (hi == -1 means no cell was found)
        valid     = flat_hi >= 0
        flat_hi_v = flat_hi[valid]
        flat_w_v  = flat_w[valid]
        flat_n_v  = flat_n[valid]
        del flat_hi, flat_w, flat_n

        # the links to the cells tied with a sample's Npt-th nearest one
        if self.tie_rows.numel():
            flat_hi_v = torch.cat([flat_hi_v, self.tie_hi])
            flat_w_v  = torch.cat([flat_w_v, torch.exp(
                (-2.0) * (self.tie_d * self.tie_d) / (self.sigma_m * self.sigma_m))])
            flat_n_v  = torch.cat([flat_n_v, self.tie_rows])

        # ── Max weight per cell ────────────────────────────────────────────
        max_w = torch.full(
            (self.K,), float("-inf"), device=self.device, dtype=flat_w_v.dtype
        )
        max_w.scatter_reduce_(
            0, flat_hi_v, flat_w_v, reduce="amax", include_self=True
        )

        # ── Keep winner pairs (weight == max for their cell) ──────────────
        is_winner = flat_w_v >= max_w[flat_hi_v]
        del max_w, flat_w_v

        win_cell = flat_hi_v[is_winner]
        win_src  = flat_n_v[is_winner]
        del flat_hi_v, flat_n_v, is_winner

        # ── Deduplicate ties: stable sort by cell, first occurrence wins ──
        ord_w    = torch.argsort(win_cell, stable=True)
        win_cell = win_cell[ord_w]
        win_src  = win_src[ord_w]
        del ord_w

        is_first      = torch.ones(len(win_cell), dtype=torch.bool, device=self.device)
        is_first[1:]  = win_cell[1:] != win_cell[:-1]

        # ── Write result: hi_k[k] = nearest source index for cell k ───────
        # -1 means no source reached the cell (shouldn't happen for cells that
        # passed the threshold, but kept as a defensive sentinel).
        hi_k = torch.full((self.K,), -1, dtype=torch.long, device=self.device)
        hi_k[win_cell[is_first]] = win_src[is_first]
        del win_cell, win_src, is_first

        self.hi_k = hi_k                                            # (K,)

    # ── Fill cells from out_cell_ids missed by the KNN rings ─────────────────

    def _fill_missing_out_cells(self) -> None:
        """Extend cell_ids / hi_k to cover every cell in out_cell_ids.

        Cells in ``out_cell_ids`` not returned by ``healpix_weighted_nearest``
        (i.e., beyond ``ring_search_max`` rings of any source) get their
        nearest source from a KD-tree of ``self.xyz_samples``.
        """
        out_t = self.out_cell_ids
        if not isinstance(out_t, torch.Tensor):
            out_t = torch.as_tensor(out_t)
        out_t = out_t.to(device=self.device, dtype=torch.long).reshape(-1)

        # ── Find missing cells ─────────────────────────────────────────────
        cell_sorted, _ = torch.sort(self.cell_ids)
        out_sorted,  _ = torch.sort(out_t)

        if self.K == 0:
            # no requested cell passed the KNN threshold: all of them are missing
            missing = out_sorted
        else:
            pos     = torch.searchsorted(cell_sorted, out_sorted).clamp(0, self.K - 1)
            present = cell_sorted[pos] == out_sorted
            missing = out_sorted[~present]                          # (M,)

        if missing.numel() == 0:
            return

        if self.N == 0:
            # no sample to fill them from: they stay out of the result, like
            # the cells the interpolating resamplers cannot compute (#79)
            return

        if self.verbose:
            print(
                f"[NearestResampler] {missing.numel():,} out_cell_ids not covered "
                f"by KNN rings → nearest sample from a KD-tree"
            )

        # ── xyz of missing cell centres ────────────────────────────────────
        miss_np = missing.cpu().numpy().astype(np.uint64)
        if self.nest:
            lon_c_deg, lat_c_deg = healpix_geo.nested.healpix_to_lonlat(
                miss_np, self.level, ellipsoid=self.ellipsoid
            )
        else:
            lon_c_deg, lat_c_deg = healpix_geo.ring.healpix_to_lonlat(
                miss_np, self.level, ellipsoid=self.ellipsoid
            )

        src_dtype = self.xyz_samples.dtype
        xyz_miss = _lonlat_to_xyz(
            torch.deg2rad(torch.as_tensor(lon_c_deg, device=self.device, dtype=src_dtype)),
            torch.deg2rad(torch.as_tensor(lat_c_deg, device=self.device, dtype=src_dtype)),
        )                                                           # (M, 3)

        # ── Nearest source per missing cell — chunked, O(chunk × N) memory ─
        hi_miss, dot_miss = self._nearest_source(xyz_miss, self.xyz_samples)  # (M,), (M,)
        del xyz_miss

        # ── Leave out the cells whose nearest sample is too far (#75) ─────
        if self.max_distance_m is not None:
            dist_miss = self.radius * torch.acos(dot_miss.clamp(-1.0, 1.0))
            near = dist_miss <= self.max_distance_m
            if self.verbose:
                print(
                    f"[NearestResampler] {int((~near).sum()):,} of them left out: "
                    f"nearest sample farther than max_distance_m={self.max_distance_m:g} m"
                )
            missing, hi_miss = missing[near], hi_miss[near]
            if missing.numel() == 0:
                return

        # ── Extend cell_ids and hi_k, then re-sort to match out_cell_ids ──
        self.cell_ids = torch.cat([self.cell_ids, missing])
        self.hi_k     = torch.cat([self.hi_k,     hi_miss])
        self.K        = int(self.cell_ids.numel())

        order         = torch.argsort(self.cell_ids)
        self.cell_ids = self.cell_ids[order]
        self.hi_k     = self.hi_k[order]

    @staticmethod
    def _nearest_source(
        xyz_query: torch.Tensor,          # (Q, 3)
        xyz_src:   torch.Tensor,          # (N, 3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest source per query point, from a KD-tree of the sources.

        Returns hi (Q,) — index of the nearest source for each query cell —
        and the matching dot product (Q,), the cosine of the angular
        distance. On the unit sphere the chord ``c`` grows with the angle, so
        the nearest source by chord is the one of largest dot product, and
        ``dot = 1 - c**2 / 2``. O((N + Q) log N), where comparing every query
        to every source took hours for a chunk with millions of cells and
        samples (Sentinel-2 chunks at the edge of the swath).
        """
        src = xyz_src.detach().cpu().numpy().astype(np.float64)
        query = xyz_query.detach().cpu().numpy().astype(np.float64)
        chord, hi = cKDTree(src).query(query, k=1)
        dot = 1.0 - 0.5 * chord * chord
        return (
            torch.as_tensor(hi, dtype=torch.long, device=xyz_src.device),
            torch.as_tensor(dot, dtype=xyz_src.dtype, device=xyz_src.device),
        )

    @staticmethod
    def _nearest_source_buckets(
        cells: torch.Tensor,              # (Q,) nested ids at ``level``, ascending
        level: int,
        lon_deg: torch.Tensor,            # (N,) samples
        lat_deg: torch.Tensor,
        xyz_query: torch.Tensor,          # (Q, 3) centres of ``cells``
        xyz_src: torch.Tensor,            # (N, 3)
        *,
        ellipsoid: str,
        device: torch.device,
        delta: int = 2,
        batch: int = 1 << 18,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`_nearest_source` on ``device`` (a GPU), for nested cells.

        The samples go into HEALPix buckets ``delta`` levels coarser than the
        cells. Each cell compares the samples of its bucket and of the 8
        around it, with the KD-tree's float64 squared chord
        ``(dx*dx + dy*dy) + dz*dz``, and keeps the nearest (the smallest sample
        index among equal distances). Any sample outside these 9 buckets is
        at least about one bucket width away; a cell whose nearest candidate
        is not closer than 0.4 of the nominal bucket width (or that has no
        candidate, e.g. outside the swath) takes its nearest sample from the
        KD-tree instead. So the picks are the KD-tree's, ties apart.

        Returns hi (Q,) and the dot product (Q,), on the CPU, like
        :meth:`_nearest_source`.
        """
        import math

        dev = torch.device(device)
        bl = level - delta
        q_bucket = cells.to(dev, torch.long) >> (2 * delta)
        s_bucket = torch.as_tensor(healpix_geo.nested.lonlat_to_healpix(
            lon_deg.numpy(), lat_deg.numpy(), bl, ellipsoid=ellipsoid).astype(np.int64)).to(dev)
        ub, q_inv = torch.unique(q_bucket, return_inverse=True)
        nb = torch.as_tensor(healpix_geo.nested.kth_neighbourhood(
            ub.cpu().numpy().astype(np.uint64), bl, 1).astype(np.int64)).to(dev)   # (U, 9), -1 invalid
        sb, order = torch.sort(s_bucket, stable=True)
        start = torch.searchsorted(sb, nb.clamp(min=0))
        count = torch.searchsorted(sb, nb.clamp(min=0), right=True) - start
        count[nb < 0] = 0
        del q_bucket, s_bucket, ub, nb, sb
        mmax = int(count.max()) if count.numel() else 0

        xq, xs = xyz_query.to(dev, torch.float64), xyz_src.to(dev, torch.float64)
        Q = xq.shape[0]
        hi = torch.full((Q,), -1, dtype=torch.long, device=dev)
        best = torch.full((Q,), math.inf, dtype=torch.float64, device=dev)
        if mmax:
            # about 16 M candidate pairs per batch (~0.4 GB of float64 differences)
            batch = max(1024, min(batch, (1 << 24) // (9 * mmax)))
            ar = torch.arange(mmax, device=dev)
            none = torch.iinfo(torch.long).max
            for b0 in range(0, Q, batch):
                b1 = min(Q, b0 + batch)
                u = q_inv[b0:b1]
                pos = start[u][:, :, None] + ar                             # (B, 9, m)
                ok = (ar < count[u][:, :, None]).reshape(b1 - b0, -1)
                cand = order[pos.clamp(max=order.numel() - 1)].reshape(b1 - b0, -1)
                d = xs[cand] - xq[b0:b1, None, :]
                d2 = (d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1]) + d[..., 2] * d[..., 2]
                d2 = torch.where(ok, d2, torch.full_like(d2, math.inf))
                m = d2.min(dim=1).values
                pick = torch.where((d2 == m[:, None]) & ok, cand, torch.full_like(cand, none))
                hi[b0:b1] = pick.min(dim=1).values
                best[b0:b1] = m
        chord = best.sqrt()
        guard = 0.4 * math.sqrt(4 * math.pi / (12 * 4**bl))
        unsure = ~(chord < guard)
        hi, chord = hi.cpu(), chord.cpu()
        if bool(unsure.any()):
            uq = unsure.nonzero().reshape(-1).cpu()
            c2, h2 = cKDTree(xyz_src.cpu().numpy().astype(np.float64)).query(
                xyz_query[uq].cpu().numpy().astype(np.float64), k=1)
            hi[uq] = torch.as_tensor(h2, dtype=torch.long)
            chord[uq] = torch.as_tensor(c2, dtype=torch.float64)
        dot = 1.0 - 0.5 * chord * chord
        return hi, dot.to(xyz_src.dtype)

    # ── resample ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def resample(self, val: T_Array, **_kwargs) -> ResampleResults:
        """Source samples → HEALPix cells.

        hval[:, k] = val[:, hi_k[k]]  — direct index, zero allocation overhead.
        """
        y = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
        y = y.to(self.device, dtype=self.dtype)

        squeezed = y.ndim == 1
        if squeezed:
            y = y.unsqueeze(0)                                      # (1, N)

        # Clamp -1 sentinels (defensive; should not occur after _fill_missing)
        safe_hi = self.hi_k.clamp(min=0)
        hval     = y[:, safe_hi]                                    # (B, K)

        cell_ids = self.cell_ids
        if squeezed:
            hval = hval.squeeze(0)
        if not isinstance(val, torch.Tensor):
            hval     = hval.cpu().numpy()
            cell_ids = cell_ids.cpu().numpy()

        return ResampleResults(cell_data=hval, cell_ids=cell_ids)

    # ── invert ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def invert(self, hval: T_Array) -> T_Array:
        """HEALPix cells → source samples.

        val[:, n] = mean of hval[:, k]  for all k s.t. hi_k[k] == n.
        (scatter-mean; multiple cells can share the same nearest source)
        """
        y = hval if isinstance(hval, torch.Tensor) else torch.as_tensor(hval)
        y = y.to(self.device, dtype=self.dtype)

        squeezed = y.ndim == 1
        if squeezed:
            y = y.unsqueeze(0)                                      # (1, K)

        safe_hi = self.hi_k.clamp(min=0)
        res = _scatter_mean(y, safe_hi, self.N)                     # (B, N)

        if squeezed:
            res = res.squeeze(0)
        if not isinstance(hval, torch.Tensor):
            res = res.cpu().numpy()

        return res
