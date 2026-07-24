"""
conservative.py

Area-weighted, flux-conserving HEALPix resampler built on top of the
``group_by`` binning mode of ``KNeighborsResampler`` (each source sample is
assigned to the single HEALPix cell that contains it — no distance kernel,
no sparse M/MT operators).

Strategy
--------
Every source sample ``i`` carries a scalar ``area_i`` (default 1.0, i.e.
samples are treated as equal-area/equal-weight pixels when no area is
given). The forward pass accumulates the *area-weighted sum* of the values
that fall in each HEALPix cell:

    hval[k] = sum_{i : hi[i] == k}  val[i] * area[i]

so the total integrated quantity is exactly conserved between
representations:

    sum_i val[i] * area[i]  ==  sum_k hval[k]                      (resample)

``invert`` redistributes a HEALPix field back to the sample locations
without inventing mass: each cell's total is turned into a density
(dividing by the cell's total input area) and that density is broadcast
to every sample that was binned into the cell, so that

    sum_k hval[k]  ==  sum_i invert(hval)[i] * area[i]              (invert)

Notes
-----
- If ``val`` is already an *extensive* quantity (a total already integrated
  over the sample's footprint, e.g. counts or an already-integrated flux),
  leave ``area`` at its default of 1.0 for every sample — plain summation is
  exactly conservative regardless of how the footprints vary in size.
- If ``val`` is an *intensive* quantity (a density, e.g. flux per m²,
  temperature), pass the per-sample pixel ``area`` so that samples covering
  a larger footprint are weighted proportionally more — otherwise larger
  and smaller source pixels would be conflated as if they covered the same
  physical area.
- ``out_cell_ids`` is not supported (same limitation as ``GroupByResampler``,
  which this class shares its binning strategy with): grouping produces
  exactly the cells that are hit by at least one sample, with no
  neighbourhood search to fall back on for cells outside that set.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from healpix_resample.base import ResampleResults, T_Array
from healpix_resample.knn import KNeighborsResampler


class ConservativeResampler(KNeighborsResampler):
    """Area-weighted, flux-conserving HEALPix resampler.

    Bins each source sample into its containing HEALPix cell (exact
    grouping, like :class:`~healpix_resample.groupby.GroupByResampler`) and
    accumulates area-weighted sums, so the total integrated quantity is
    exactly preserved between the sample-space and HEALPix-cell
    representations. See the module docstring for the extensive-vs-intensive
    distinction that determines whether ``area`` needs to be supplied.

    Parameters
    ----------
    area : array-like or None
        Per-sample pixel area/weight, shape ``(N,)``. Any consistent unit
        works since only ratios matter. Defaults to ``1.0`` for every
        sample (equal-area pixels / already-extensive quantities).
    All other parameters are forwarded to ``KNeighborsResampler``
    (``lon_deg``, ``lat_deg``, ``level``, ``nest``, ``device``, ``dtype``,
    ``ellipsoid``, ``verbose``, ...). ``out_cell_ids`` is not supported.
    """

    def __init__(self, *args, area: Optional[T_Array] = None, out_cell_ids=None, **kwargs):
        if out_cell_ids is not None:
            raise NotImplementedError(
                "ConservativeResampler does not support out_cell_ids: it only "
                "produces the HEALPix cells actually hit by a sample (same "
                "limitation as GroupByResampler)."
            )
        super().__init__(*args, group_by=True, Npt=1, **kwargs)

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

        # Total input area binned into each HEALPix cell — this is NOT the
        # geometric HEALPix pixel area, but the sum of the source samples'
        # areas that landed in that cell. Used to turn a cell's accumulated
        # total back into a density during `invert`.
        cell_area = torch.zeros(self.K, device=self.device, dtype=self.dtype)
        cell_area.scatter_add_(0, self.hi, self.area)
        self.cell_area = cell_area

    # ── resample ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def resample(self, val: T_Array, **_kwargs) -> ResampleResults:
        """Source samples → HEALPix cells, area-weighted sum.

        hval[:, k] = sum_{i : hi[i] == k} val[:, i] * area[i]
        """
        y = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
        y = y.to(self.device, dtype=self.dtype)

        squeezed = y.ndim == 1
        if squeezed:
            y = y.unsqueeze(0)  # (1, N)

        weighted = y * self.area  # (B, N), broadcasts (N,) over rows
        B = weighted.shape[0]
        hval = torch.zeros(B, self.K, device=self.device, dtype=self.dtype)
        hval.scatter_add_(1, self.hi.unsqueeze(0).expand(B, -1), weighted)

        cell_ids = self.cell_ids
        if squeezed:
            hval = hval.squeeze(0)
        if not isinstance(val, torch.Tensor):
            hval = hval.cpu().numpy()
            cell_ids = cell_ids.cpu().numpy()

        return ResampleResults(cell_data=hval, cell_ids=cell_ids)

    # ── invert ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def invert(self, hval: T_Array) -> T_Array:
        """HEALPix cells → source samples, mass-conserving redistribution.

        Each cell's accumulated total is turned into a density
        (``hval[k] / cell_area[k]``) and broadcast to every sample binned
        into that cell, so ``sum_k hval[k] == sum_i invert(hval)[i] * area[i]``.
        """
        y = hval if isinstance(hval, torch.Tensor) else torch.as_tensor(hval)
        y = y.to(self.device, dtype=self.dtype)

        squeezed = y.ndim == 1
        if squeezed:
            y = y.unsqueeze(0)  # (1, K)

        # dtype is always a float type in this package (float32/float64)
        safe_area = self.cell_area.clamp(min=torch.finfo(self.dtype).tiny)
        density = y / safe_area  # (B, K)
        val_hat = density[:, self.hi]  # (B, N) — direct index, like NearestResampler

        if squeezed:
            val_hat = val_hat.squeeze(0)
        if not isinstance(hval, torch.Tensor):
            val_hat = val_hat.cpu().numpy()

        return val_hat

    def get_cell_area(self) -> np.ndarray:
        """Return the total input area binned into each HEALPix cell (K,)."""
        return self.cell_area.cpu().numpy()
