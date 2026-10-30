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
  formula `w` (`_link_weights()`) and the normalisation guards change.

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
    `u = self.d_m / s` in place of a structured pixel offset::

        w(u) = (a+2)|u|^3 - (a+3)|u|^2 + 1        for |u| <= 1
             = a|u|^3 - 5a|u|^2 + 8a|u| - 4a       for 1 < |u| < 2
             = 0                                   for |u| >= 2

    Unlike the Gaussian/inverse-distance weights used by `BilinearResampler`
    and `PSFResampler`, this kernel is *signed* — it goes negative for
    `1 < |u| < 2`, which is exactly what gives cubic convolution its
    sharpening property relative to bilinear.

    Kernel scale
    ------------
    Keys' kernel is defined in units of the *input* grid spacing: its
    negative lobe `1 < |u| < 2` must fall on the second ring of samples. The
    scale of each link is therefore `s = max(sigma_m, h)`, where `h` is the
    spacing of the link's sample (``sample_spacing_m`` below). When the
    samples are finer than the cells (downsampling) `sigma_m` wins, as
    before. When they are coarser (upsampling), scaling by the output cell
    size alone put the nearest samples in the negative lobe: the weights of
    a cell then cancelled, and the normalisation amplified them up to
    thousands of times, which gave values far outside the input range.

    Two further consequences of the signed kernel:

    - A cell whose links still cancel strongly (`sum|w| > 3 |sum w|`, e.g.
      on scattered samples) uses only the positive lobe of its links (their
      absolute values if it has no positive weight), so its value stays a
      weighted average of its samples. `_norm_col()` / `_norm_row()` keep a
      last floor on the normalising sums (relative to the *unsigned*
      accumulated weight), see the inline comments on `_norm_col()`.
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
    sample_spacing_m : float, array-like or None
        Spacing of the input samples in metres, a scalar or one value per
        sample ``(N,)``; sets the kernel scale ``max(sigma_m, h)`` (see
        "Kernel scale"). ``None`` (default) estimates it as the distance of
        each sample to its nearest other sample (a ``scipy`` k-d tree on the
        unit vectors, queried in slices of ``batch_size`` when given). On a
        regular grid that is the grid spacing; pass it explicitly to skip
        the estimate on very large inputs.
    """

    def __init__(
        self,
        *args,
        Npt: int = 16,
        area: Optional[T_Array] = None,
        sample_spacing_m=None,
        **kwargs,
    ):
        lon_deg = kwargs["lon_deg"] if "lon_deg" in kwargs else args[0]
        lat_deg = kwargs["lat_deg"] if "lat_deg" in kwargs else args[1]
        n = int(lon_deg.numel()) if isinstance(lon_deg, torch.Tensor) else int(np.asarray(lon_deg).size)
        if sample_spacing_m is None:
            spacing = _nearest_sample_distance_m(
                lon_deg, lat_deg,
                radius=float(kwargs.get("radius", 6371000.0)),
                batch_size=kwargs.get("batch_size"),
            )
        else:
            spacing = np.broadcast_to(np.asarray(sample_spacing_m, dtype=np.float64).reshape(-1), (n,))
            if not np.all(np.isfinite(spacing) & (spacing >= 0)):
                raise ValueError("sample_spacing_m must be finite and non-negative")
        # read by _link_weights() while super().__init__() builds M
        self._sample_spacing = np.ascontiguousarray(spacing, dtype=np.float64)
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

    def _link_weights(self) -> torch.Tensor:
        # Keys' cubic convolution kernel (a=-0.5) on the radial distance
        # u = d / s, s = max(sigma_m, spacing of the link's sample) (see
        # "Kernel scale" in the class docstring). `M`, `MT` and `M_cons` are
        # assembled by `KNeighborsResampler` (`MT` / `M_cons` on first
        # access) from these weights.
        a = -0.5
        spacing = torch.as_tensor(self._sample_spacing, device=self.d_m.device, dtype=self.d_m.dtype)
        scale = torch.clamp(spacing, min=self.sigma_m)[:, None]          # (N, 1)
        u = (self.d_m / scale).abs()
        u2 = u * u
        u3 = u2 * u

        w_inner = (a + 2.0) * u3 - (a + 3.0) * u2 + 1.0
        w_outer = a * u3 - 5.0 * a * u2 + 8.0 * a * u - 4.0 * a
        zero = torch.zeros_like(u)
        w = torch.where(u <= 1.0, w_inner, torch.where(u < 2.0, w_outer, zero))
        return self._limit_cancellation(w)

    def _limit_cancellation(self, w: torch.Tensor) -> torch.Tensor:
        # A cell whose signed weights still cancel strongly (sum|w| larger
        # than _MAX_CANCELLATION times |sum w|) would be amplified by the
        # per-cell normalisation. Its links use the positive lobe only
        # (their absolute values if the cell has no positive weight), so
        # that cell's value is a weighted average of its samples.
        valid = self.hi >= 0
        cols = self.hi[valid]
        wv = w[valid]
        signed = torch.bincount(cols, weights=wv, minlength=self.K)
        unsigned = torch.bincount(cols, weights=wv.abs(), minlength=self.K)
        positive = torch.bincount(cols, weights=wv.clamp(min=0), minlength=self.K)
        bad_cell = unsigned > _MAX_CANCELLATION * signed.abs()
        if not bool(bad_cell.any()):
            return w
        bad = torch.zeros_like(valid)
        bad[valid] = bad_cell[cols]
        has_positive = torch.zeros_like(valid)
        has_positive[valid] = (positive > 0)[cols]
        fallback = torch.where(has_positive, w.clamp(min=0), w.abs())
        return torch.where(bad, fallback, w)

    # Unlike the nonnegative Gaussian/IDW weights used by every other
    # resampler, Keys' kernel is signed, so a per-cell sum norm_col[k] (or a
    # per-sample sum norm_row[i]) can land close to zero purely from
    # cancellation between the positive central lobe and the negative outer
    # lobe (rather than every contributing weight being individually small).
    # Dividing by such a sum would blow up or flip sign unpredictably. Guard
    # it with a floor set relative to the *unsigned* accumulated weight
    # (same area-independent pattern as `norm_col_raw` in
    # `psf.py:comp_matrix`) -- rather than an absolute epsilon, so the guard
    # scales correctly across different sigma/threshold configs. This only
    # clamps the handful of pathologically-cancelled rows/columns;
    # well-conditioned cells are unaffected (float precision only matters
    # when the sum is already tiny relative to the unsigned one). Cells this
    # severe are inherently borderline -- `threshold` (applied upstream
    # against the unsigned Gaussian weight in `healpix_weighted_nearest`) is
    # the primary safeguard against weakly-supported cells in the first
    # place; this is a secondary, purely numerical safety net.
    def _norm_col(self, cols: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        norm_col = torch.bincount(cols, weights=w, minlength=self.K).to(self.dtype)
        norm_col_raw = torch.bincount(cols, weights=w.abs(), minlength=self.K).to(self.dtype)
        return _floor_signed(norm_col, norm_col_raw)

    def _norm_row(self, rows: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        norm_row = torch.bincount(rows, weights=w, minlength=self.N).to(self.dtype)
        norm_row_raw = torch.bincount(rows, weights=w.abs(), minlength=self.N).to(self.dtype)
        return _floor_signed(norm_row, norm_row_raw)

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
        Same idea as ``BilinearResampler.resample()``: each sample's own
        (``area``-weighted) value is redistributed across its ``Npt``
        nearest cells using ``self.M_cons`` -- Keys' kernel weights,
        normalized so each sample's own weights sum to 1 instead of being
        normalized per output cell -- guaranteeing

            sum_k hval[k] == sum_i (valid i) val[i] * area[i]

        See ``BilinearResampler.resample()`` for the full NaN-handling
        discussion (identical here: a NaN sample's value and area are both
        excluded, and the identity above then holds over exactly the valid
        samples; an all-NaN row comes back entirely ``nan``).

        Caveat specific to this class's signed kernel
        ------------------------------------------------
        Unlike ``BilinearResampler``'s non-negative inverse-distance
        weights, Keys' cubic kernel is signed, so ``self.M_cons``'s
        per-sample rows only sum to *exactly* 1 for samples whose
        ``norm_row`` wasn't floored by ``_floor_signed`` in
        ``_norm_row()`` -- true for the vast majority of well-conditioned
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


# Largest accepted sum|w| / |sum w| of a cell's Keys weights; see
# BicubicResampler._limit_cancellation. Well-conditioned cells on a grid stay
# below it: for Keys' kernel, sum|w| / sum w is at most 1.25 per axis (half a
# pixel offset), about 1.56 for the 4x4 stencil.
_MAX_CANCELLATION = 3.0


def _nearest_sample_distance_m(lon_deg, lat_deg, *, radius: float, batch_size=None) -> np.ndarray:
    """Distance (m) from each sample to its nearest other sample, ``(N,)``.

    Chord length between unit vectors (equal to the arc length at these
    distances) times ``radius``, from a ``scipy`` k-d tree; the query runs in
    slices of ``batch_size`` samples when given. Duplicated samples get the
    distance to the nearest *distinct* sample; a single sample gets 0.
    """
    from scipy.spatial import cKDTree

    lon = np.radians(np.asarray(lon_deg.cpu() if isinstance(lon_deg, torch.Tensor) else lon_deg, dtype=np.float64).reshape(-1))
    lat = np.radians(np.asarray(lat_deg.cpu() if isinstance(lat_deg, torch.Tensor) else lat_deg, dtype=np.float64).reshape(-1))
    n = lon.size
    out = np.zeros(n, dtype=np.float64)
    if n < 2:
        return out
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)
    tree = cKDTree(xyz)
    step = n if batch_size is None else max(1, int(batch_size))
    for s in range(0, n, step):
        q = xyz[s : s + step]
        # k=2: the first neighbour is the sample itself (or a duplicate)
        d, _ = tree.query(q, k=2, workers=-1)
        nearest = d[:, 1]
        dup = nearest == 0
        if dup.any():
            # duplicates: look further for the first non-zero distance
            dd, _ = tree.query(q[dup], k=min(n, 16), workers=-1)
            dd = np.where(dd > 0, dd, np.inf).min(axis=1)
            nearest[dup] = np.where(np.isfinite(dd), dd, 0.0)
        out[s : s + step] = nearest
    return out * radius


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
