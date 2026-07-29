"""
bicubic.py

Radial cubic-convolution ("bicubic") HEALPix resampler built on top of
KNeighborsResampler.

Core idea:
- Generalize Keys' cubic convolution kernel (the interpolation kernel behind
  `cv2.INTER_CUBIC` / `PIL.Image.BICUBIC` on regular pixel grids) to
  unstructured samples by evaluating it *radially*: replace the structured
  `(fx, fy)` fractional pixel offset with the existing geodesic distance
  `self.d_m` and length scale `self.sigma_m` already used by every other
  resampler in this package (see `planning/01_bicubic_resampler.md`).
- Reuse the exact same normalize-and-sparsify machinery as
  `BilinearResampler`/`KNeighborsResampler.comp_matrix()` — only the weight
  formula `w` changes.

This module is designed for large N and batched values (B,N) on CUDA.
"""
from typing import Optional

from healpix_resample.base import ResampleResults, T_Array
from healpix_resample.knn import KNeighborsResampler, _conservative_resample
import math
import numpy as np
import torch


class BicubicResampler(KNeighborsResampler):
    """Radial cubic-convolution HEALPix resampler.

    Sits between `BilinearResampler` (`Npt=4`, inverse-distance weighted) and
    `PSFResampler` (iterative CG deconvolution): a fixed, non-iterative local
    interpolation using more neighbours than bilinear, built purely from
    `comp_matrix()` — no CG solve involved.

    Weight function
    ----------------
    Keys' cubic convolution kernel with `a = -0.5` (the common default,
    matching `PIL`/`cv2`), applied to the radial distance
    `u = self.d_m / self.sigma_m` in place of a structured pixel offset::

        w(u) = (a+2)|u|^3 - (a+3)|u|^2 + 1        for |u| <= 1
             = a|u|^3 - 5a|u|^2 + 8a|u| - 4a       for 1 < |u| < 2
             = 0                                   for |u| >= 2

    Unlike the Gaussian/inverse-distance weights used by `BilinearResampler`
    and `PSFResampler`, this kernel is *signed* — it goes negative for
    `1 < |u| < 2`, which is exactly what gives cubic convolution its
    sharpening property relative to bilinear.

    Two consequences of the signed kernel, both handled in `comp_matrix()`:

    - The per-cell/per-sample weight sums used to normalize `M`/`MT` can, in
      principle, be arbitrarily close to zero from cancellation between the
      positive central lobe and the negative outer lobe — even when every
      individual link weight is well-behaved in magnitude. We guard the
      normalizing division with a small floor (relative to the *unsigned*
      accumulated weight for that row/column) rather than dropping or
      re-flagging affected cells; see the inline comments in `comp_matrix()`.
    - `invert()` (inherited from `KNeighborsResampler`, `hval @ self.MT`) can
      genuinely overshoot/ring outside the local sample-value range — this
      is expected cubic-convolution behaviour, not a bug.

    Parameters
    ----------
    Npt : int
        Number of HEALPix neighbours per source sample used by the KNN.
        Keys' kernel has support `|u| < 2`, roughly twice the reach of
        bilinear's `|u| < 1`-ish support, so the natural analogue of the
        classic 4x4 bicubic stencil on a structured grid is `Npt = 16`
        (default).
    All other parameters are forwarded to `KNeighborsResampler`.

    Parameters
    ----------
    area : array-like or None
        Per-sample pixel area/weight, shape ``(N,)``. Only used by
        ``resample(conservative=True)`` -- ignored by the default
        interpolation path. Defaults to ``1.0`` for every sample. See
        ``resample()``'s docstring for the conservation guarantee and its
        one caveat specific to this class's signed kernel.
    """

    def __init__(self, *args, Npt: int = 16, area: Optional[T_Array] = None, **kwargs):
        # Ensure ring_search_max >= ring_search_init(Npt) so the KNN search
        # loop in healpix_weighted_nearest actually executes.
        #
        # healpix_weighted_nearest computes:
        #   r_min            = ceil((sqrt(Npt) - 1) / 2)
        #   ring_search_init = max(1, r_min + 1)
        #
        # KNeighborsResampler's default ring_search_max=2 is too small for
        # Npt >= 16 (needs ring_search_init=3). Auto-correct here only when
        # the caller has not supplied ring_search_max explicitly — copied
        # from NearestResampler.__init__ (nearest.py:67-84).
        if "ring_search_max" not in kwargs:
            r_min = int(math.ceil((math.sqrt(Npt) - 1.0) / 2.0))
            ring_search_init_needed = max(1, r_min + 1)
            # +2 margin so the loop has room to grow and find Npt candidates
            kwargs["ring_search_max"] = ring_search_init_needed + 2

        super().__init__(*args, Npt=Npt, **kwargs)

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
        # --- weights per sample->cell link: Keys' cubic convolution kernel,
        # evaluated on the radial distance u = d/sigma (a=-0.5).
        a = -0.5
        u = (self.d_m / self.sigma_m).abs()
        u2 = u * u
        u3 = u2 * u

        w_inner = (a + 2.0) * u3 - (a + 3.0) * u2 + 1.0
        w_outer = a * u3 - 5.0 * a * u2 + 8.0 * a * u - 4.0 * a
        zero = torch.zeros_like(u)
        w = torch.where(u <= 1.0, w_inner, torch.where(u < 2.0, w_outer, zero))

        # Build (N,K) operator M and (K,N) operator MT.
        # We avoid numpy bincount; use torch.bincount on GPU.

        # idx: (N,Npt) row indices 0..N-1
        idx = torch.arange(self.N, device=self.device, dtype=torch.long)[:, None].expand(self.N, self.Npt)

        flat_hi = self.hi.reshape(-1)
        flat_w = w.reshape(-1)
        valid = flat_hi >= 0
        flat_hi_v = flat_hi[valid]
        flat_w_v = flat_w[valid]

        # -------- M : (N,K)  (normalized per column / per healpix cell)
        # norm_col[k] = sum_{i links to k} w[i,k]
        #
        # Unlike the nonnegative Gaussian/IDW weights used by every other
        # resampler, Keys' kernel is signed, so norm_col[k] can land close to
        # zero purely from cancellation between the positive central lobe
        # and the negative outer lobe (rather than every contributing weight
        # being individually small). Dividing by such a norm_col would blow
        # up or flip sign unpredictably. Guard it with a floor set relative
        # to norm_col_raw -- the *unsigned* accumulated weight for that cell
        # (same area-independent pattern as `norm_col_raw` in
        # `psf.py:comp_matrix`) -- rather than an absolute epsilon, so the
        # guard scales correctly across different sigma/threshold configs.
        # This only clamps the handful of pathologically-cancelled
        # rows/columns; well-conditioned cells are unaffected (float
        # precision only matters when norm_col is already tiny relative to
        # norm_col_raw). Cells this severe are inherently borderline —
        # `threshold` (applied upstream against the unsigned Gaussian weight
        # in `healpix_weighted_nearest`) is the primary safeguard against
        # weakly-supported cells in the first place; this is a secondary,
        # purely numerical safety net.
        norm_col = torch.bincount(flat_hi_v, weights=flat_w_v, minlength=self.K).to(self.dtype)
        norm_col_raw = torch.bincount(flat_hi_v, weights=flat_w_v.abs(), minlength=self.K).to(self.dtype)
        norm_col_safe = _floor_signed(norm_col, norm_col_raw)
        wM = flat_w_v / norm_col_safe[flat_hi_v]

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
        # norm_row[i] = sum_{k links from i} w[i,k]  (same cancellation guard)
        flat_idx = idx.reshape(-1)
        flat_idx_v = flat_idx[valid]
        norm_row = torch.bincount(flat_idx_v, weights=flat_w_v, minlength=self.N).to(self.dtype)
        norm_row_raw = torch.bincount(flat_idx_v, weights=flat_w_v.abs(), minlength=self.N).to(self.dtype)
        norm_row_safe = _floor_signed(norm_row, norm_row_raw)
        wMT = flat_w_v / norm_row_safe[flat_idx_v]

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
        # the per-sample-normalized weights already computed for MT (wMT)
        # instead of M's per-cell-normalized weights -- see
        # BilinearResampler.comp_matrix() for the non-signed-kernel version
        # of this same construction. Because Keys' kernel is signed, wMT's
        # rows only sum to *exactly* 1 for samples whose norm_row wasn't
        # floored by `_floor_signed` above; see resample()'s docstring for
        # the resulting caveat on conservative=True's guarantee.
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
            conservative: see below. Extra ``**kwargs`` are accepted for
                signature symmetry with other resamplers and forwarded to
                ``KNeighborsResampler.resample()`` when
                ``conservative=False`` (no CG solve involved either way);
                ignored when ``conservative=True``.

        ``conservative=True`` (issue #44: "conservative bi-linear", applied
        here to bicubic too)
        --------------------------------------------------------------------
        Same idea as :meth:`BilinearResampler.resample`: each sample's own
        (``area``-weighted) value is redistributed across its ``Npt``
        nearest cells using ``self.M_cons`` -- Keys' kernel weights,
        normalized so each sample's own weights sum to 1 instead of being
        normalized per output cell -- guaranteeing

            sum_k hval[k] == sum_i (valid i) val[i] * area[i]

        See :meth:`BilinearResampler.resample` for the full NaN-handling
        discussion (identical here: a NaN sample's value and area are both
        excluded, and the identity above then holds over exactly the valid
        samples; an all-NaN row comes back entirely ``nan``).

        Caveat specific to this class's signed kernel
        ------------------------------------------------
        Unlike ``BilinearResampler``'s non-negative inverse-distance
        weights, Keys' cubic kernel is signed, so ``self.M_cons``'s
        per-sample rows only sum to *exactly* 1 for samples whose
        ``norm_row`` wasn't floored by ``_floor_signed`` in
        ``comp_matrix()`` -- true for the vast majority of well-conditioned
        samples. For the rare, pathologically-cancelled sample that does
        hit the floor, that sample's own contribution to the conservation
        identity above is only approximate (to the extent the floor
        perturbed its normalization), not bit-exact -- the same accepted
        trade-off ``_floor_signed`` already makes for ordinary
        (non-conservative) interpolation.

        Returns:
            hval: (B, K) or (K,)
        """
        if not conservative:
            return super().resample(val, **kwargs)
        return _conservative_resample(self, val, self.area)


def _floor_signed(norm: torch.Tensor, norm_raw: torch.Tensor, rel_floor: float = 1e-3) -> torch.Tensor:
    """Clamp ``|norm|`` away from zero (preserving sign) relative to ``norm_raw``.

    ``norm`` is a signed accumulated weight (e.g. ``norm_col``/``norm_row``
    from Keys' cubic kernel); ``norm_raw`` is the corresponding *unsigned*
    accumulated weight (sum of ``|w|``). When ``norm`` is small only because
    of sign cancellation, dividing by it directly would blow up or flip sign
    unpredictably; this floors it at ``rel_floor * norm_raw`` (with the sign
    of ``norm`` itself, defaulting to positive when ``norm`` is exactly 0)
    without otherwise altering well-conditioned entries.
    """
    floor = rel_floor * norm_raw.clamp(min=1e-12)
    sign = torch.where(norm >= 0, torch.ones_like(norm), -torch.ones_like(norm))
    return torch.where(norm.abs() < floor, sign * floor, norm)
