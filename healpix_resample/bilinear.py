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

    def _distance_weights(self, d: torch.Tensor, rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        # inverse-distance weights per sample->cell link (Npt=4, and the
        # tied links); `M`, `MT` and `M_cons` are assembled by
        # `KNeighborsResampler` (`M` per-cell normalised; `MT` / `M_cons`
        # per-sample normalised, built on first access -- see
        # `KNeighborsResampler.comp_matrix`).
        return 1 / (1e-6 + d / self.sigma_m)

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
