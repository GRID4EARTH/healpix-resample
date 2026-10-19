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
missing cells are filled via a memory-bounded chunked dot-product fallback.
"""
from __future__ import annotations

from typing import Optional

import healpix_geo
import numpy as np
import torch

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
    All other parameters are forwarded to ``KNeighborsResampler``.
    """

    def __init__(self, *args, Npt: int = 9, max_distance_m: Optional[float] = None, **kwargs):
        if max_distance_m is not None and not max_distance_m > 0:
            raise ValueError(f"max_distance_m must be None or positive, got {max_distance_m!r}.")
        self.max_distance_m = max_distance_m
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
        (i.e., beyond ``ring_search_max`` rings of any source) are handled via
        a memory-bounded chunked dot-product: the nearest source is found
        directly from ``self.xyz_samples`` without pair explosion.
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
                f"by KNN rings → chunked dot-product fallback"
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
        hi_miss, dot_miss = self._chunked_nearest(xyz_miss, self.xyz_samples)  # (M,), (M,)
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
    def _chunked_nearest(
        xyz_query: torch.Tensor,          # (Q, 3)
        xyz_src:   torch.Tensor,          # (N, 3)
        mem_budget_bytes: int = 512 * 1024 * 1024,   # 512 MB
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """argmax dot(query, src) per query row, memory-bounded by budget.

        Returns hi (Q,) — index of nearest source for each query cell — and
        the matching dot product (Q,), the cosine of the angular distance.
        Memory peak = chunk_q × N × bytes_per_element.
        """
        N    = xyz_src.shape[0]
        Q    = xyz_query.shape[0]
        bpe  = xyz_src.element_size()
        chunk_q = max(1, mem_budget_bytes // (N * bpe))

        hi_out  = torch.empty(Q, dtype=torch.long, device=xyz_src.device)
        dot_out = torch.empty(Q, dtype=xyz_src.dtype, device=xyz_src.device)
        for start in range(0, Q, chunk_q):
            end                = min(start + chunk_q, Q)
            dots               = xyz_query[start:end] @ xyz_src.T    # (chunk_q, N)
            dot_out[start:end], hi_out[start:end] = dots.max(dim=1)
            del dots
        return hi_out, dot_out

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
