"""
bilinear.py

GPU-friendly sparse HEALPix regridding from unstructured lon/lat samples
to a subset of HEALPix pixels at a target resolution (nside = 2**level).

Core ideas:
- Use npt=4.

This module is designed for large N and batched values (B,N) on CUDA.
"""
from typing import Optional

import math
import numpy as np
import torch

from healpix_resample.base import ResampleResults, T_Array
from healpix_resample.knn import KNeighborsResampler, _conservative_resample


class BilinearResampler(KNeighborsResampler):
    """Bilinear (Npt=4, inverse-distance weighted) HEALPix resampler.

    Parameters
    ----------
    area : array-like or None
        Per-sample pixel area/weight, shape ``(N,)``. Only used by
        ``resample(conservative=True)`` (see below) -- ignored by the
        default interpolation path. Defaults to ``1.0`` for every sample
        (equal-area pixels / already-extensive quantities), the same
        convention as :class:`~healpix_resample.conservative.ConservativeResampler`.
    All other parameters are forwarded to ``KNeighborsResampler``.
    """

    def __init__(self, *args, area: Optional[T_Array] = None, **kwargs):
        super().__init__(Npt=4, *args, **kwargs)

        if area is None:
            area_t = torch.ones(self.N, dtype=self.dtype, device=self.device)
        else:
            area_t = area if isinstance(area, torch.Tensor) else torch.as_tensor(area)
            area_t = area_t.to(self.device, dtype=self.dtype).reshape(-1)
            if area_t.numel() != self.N:
                raise ValueError(
                    f"area must have {self.N} elements (one per sample), got {area_t.numel()}"
                )
            if torch.any(area_t < 0):
                raise ValueError("area must be non-negative")
        self.area = area_t

    def comp_matrix(self):

        # --- weights per sample->cell link
        # w = exp(-2*d^2/sigma^2)
        w = 1/( 1e-6 + self.d_m/self.sigma_m)

        # Build (N,K) operator M and (K,N) operator MT.
        # We avoid numpy bincount; use torch.bincount on GPU.

        # idx: (N,Npt) row indices 0..N-1
        idx = torch.arange(self.N, device=self.device, dtype=torch.long)[:, None].expand(self.N, self.Npt)

        # -------- M : (N,K)  (normalized per column / per healpix cell)
        # norm_col[k] = sum_{i links to k} w[i,k]
        flat_hi = self.hi.reshape(-1)
        flat_w = w.reshape(-1)
        valid = flat_hi >= 0
        flat_hi_v = flat_hi[valid]
        flat_w_v = flat_w[valid]

        norm_col = torch.bincount(flat_hi_v, weights=flat_w_v, minlength=self.K).to(self.dtype)
        # weight divided by column sum
        wM = flat_w_v / norm_col[flat_hi_v]

        rowsM = idx.reshape(-1)[valid]
        colsM = flat_hi_v
        indicesM = torch.stack([rowsM, colsM], dim=0)
        M_coo = torch.sparse_coo_tensor(
            indicesM,
            wM.to(self.dtype),
            size=(self.N, self.K),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()

        # -------- MT : (K,N) (normalized per row / per input sample)
        # norm_row[i] = sum_{k links from i} w[i,k]
        flat_idx = idx.reshape(-1)
        flat_idx_v = flat_idx[valid]
        norm_row = torch.bincount(flat_idx_v, weights=flat_w_v, minlength=self.N).to(self.dtype)
        wMT = flat_w_v / norm_row[flat_idx_v]

        indicesMT = torch.stack([colsM, rowsM], dim=0)  # (hi, idx)
        MT_coo = torch.sparse_coo_tensor(
            indicesMT,
            wMT.to(self.dtype),
            size=(self.K, self.N),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()

        # -------- M_cons : (N,K) conservative-mode operator (issue #44) ----
        # Same (sample, cell) links/index layout as M (indicesM), but with
        # the *per-sample*-normalized weights already computed for MT
        # (wMT, which by construction sum to exactly 1 across the Npt cells
        # any given sample links to) instead of M's per-cell-normalized
        # weights. Used by resample(conservative=True) to redistribute each
        # sample's own (area-weighted) contribution across its nearest
        # cells without gain or loss -- see that method's docstring.
        Mcons_coo = torch.sparse_coo_tensor(
            indicesM,
            wMT.to(self.dtype),
            size=(self.N, self.K),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()

        # Convert to CSR for faster spMM (recommended on GPU)
        self.M  = M_coo.to_sparse_csr()
        self.MT = MT_coo.to_sparse_csr()
        self.M_cons = Mcons_coo.to_sparse_csr()

    @torch.no_grad()
    def resample(self, val: T_Array, *, conservative: bool = False, **kwargs) -> ResampleResults[T_Array]:
        """Estimate the HEALPix field from unstructured samples.

        Args:
            val: (N,) or (B, N) values at lon/lat sample points.
            conservative: see below. Extra ``**kwargs`` (``lam``,
                ``max_iter``, ``tol``, ``x0``, ``return_info``) are accepted
                for signature symmetry with other resamplers and forwarded
                to ``KNeighborsResampler.resample()`` when
                ``conservative=False`` -- none of them actually change
                anything for this class (no CG solve involved either way),
                and they are ignored when ``conservative=True``.

        ``conservative=False`` (default)
        ---------------------------------
        The usual bilinear interpolation: each cell's value is a weighted
        blend of its 4 nearest samples (inverse-distance weights,
        normalized per cell via ``self.M``) -- smooth, but *not* exactly
        mass-conserving (a cell's weights are renormalized against whichever
        samples happen to link to it, independent of any other cell).

        ``conservative=True`` (issue #44: "conservative bi-linear")
        --------------------------------------------------------------
        Each sample's own value (scaled by its ``area``, see ``__init__``)
        is instead redistributed across its own 4 nearest cells using
        ``self.M_cons`` -- the *same* inverse-distance weights, but
        normalized so each sample's own weights sum to exactly 1 (a
        partition of unity), rather than normalizing per output cell. No
        value is invented or lost:

            sum_k hval[k] == sum_i (valid i) val[i] * area[i]

        exactly, regardless of how many samples any given cell happens to
        receive contributions from. This trades a small amount of
        interpolation "sharpness" (identical weights to the non-conservative
        path, just redistributed the other way) for an unconditional global
        conservation guarantee -- a bilinear-weighted analogue of
        :class:`~healpix_resample.conservative.ConservativeResampler`'s exact
        area conservation, without that class's single-nearest-cell binning
        blockiness.

        NaN handling (``conservative=True`` only -- ``conservative=False``'s
        NaN behaviour is documented on the inherited
        ``KNeighborsResampler.resample()``): a NaN sample's value *and* its
        area are excluded from every cell's total, so the exact identity
        above holds over precisely the valid samples. This is a cleaner
        guarantee than the interpolation path's NaN handling, which only
        zeroes a NaN sample's contribution without renormalizing the
        *other* samples sharing a cell -- ``conservative=True`` needs no
        such caveat, because ``self.M_cons``'s per-sample normalization is
        fixed at construction time and never renormalizes per output cell.
        A batch row where every sample is NaN comes back entirely ``nan``.

        Returns:
            hval: (B, K) or (K,)
        """
        if not conservative:
            return super().resample(val, **kwargs)
        return _conservative_resample(self, val, self.area)
