"""
mask.py

Resamplers for categorical/mask-like data (issue #43: "make specific
resampler for mask-like data" -- e.g. a Sentinel-2 L1C cloud-mask), where
``NearestResampler``'s single-nearest-sample assignment is too blocky but the
underlying values aren't a continuous physical field either.

Both classes here follow the same idea, discussed in the issue: turn each
discrete "thing to resample" (a class, or a single bit of a bitmask) into a
0/1 *indicator* array, resample each indicator through an ordinary
interpolating resampler (any ``KNeighborsResampler`` subclass -- default
``BilinearResampler``, hence the issue's original working name
"argmax_over_bilinear"), and turn the resulting continuous-valued indicator
maps back into a discrete decision:

- :class:`BitmaskResampler` -- **independent** boolean flags packed into an
  integer (e.g. an 8-bit quality/cloud mask where several bits can be set at
  once): each bit is resampled on its own and thresholded at 50% by default,
  then the bits are reassembled. This is the "OR" case -- nothing here is
  mutually exclusive, so there's no argmax, just 1-bit decisions made and
  recombined independently.
- :class:`CategoricalResampler` -- **mutually exclusive** class labels (one
  class per sample, e.g. a land-cover classification): each class's
  indicator is resampled, and the winning class per output cell is whichever
  indicator scored highest (argmax). This is the "AND"/dominant-class case
  from the issue. Optionally also returns a softmax-normalized per-class
  score, both as a graceful tie-break and as a confidence-style diagnostic
  (see issue #4, "confidence factor").

Neither class does its own KNN/geometry work: both simply wrap an already
fully-specified interpolating resampler instance (composition, not
inheritance) and reuse its ``cell_ids``/``resample()`` machinery verbatim --
including whichever of NaN filtering, ``out_cell_ids``, and (for
``PSFResampler``) ``fill_missing_out_cells`` that wrapped instance already
implements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
import torch

from healpix_resample.base import ResampleResults, T_Array
from healpix_resample.bilinear import BilinearResampler
from healpix_resample.knn import KNeighborsResampler


@dataclass(frozen=True)
class CategoricalResampleResults(ResampleResults):
    """``ResampleResults`` returned by ``CategoricalResampler.resample()``.

    Attributes
    ----------
    classes : numpy.ndarray or torch.Tensor or None
        The distinct class labels found in the ``mask`` passed to
        ``resample()``, in the same order as ``scores``'s leading axis.
        ``None`` unless ``return_scores=True``.
    scores : numpy.ndarray or torch.Tensor or None
        Softmax-normalized per-class score, shape ``(n_classes, K)``.
        ``None`` unless ``return_scores=True`` -- see ``resample()``'s
        docstring for the softmax temperature and what it means for
        different ``kernel`` choices.
    """
    classes: Optional[T_Array] = None
    scores: Optional[T_Array] = None


def _as_kernel_device_tensor(x, op) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    return t.to(op.device)


class BitmaskResampler:
    """Resample an integer **bitmask** (independent boolean flags) to HEALPix.

    Use this when each bit of an integer mask is an independent yes/no flag
    that can co-occur with any other bit (e.g. per-pixel quality flags:
    cloud / cloud-shadow / snow / saturated / ... all packed into one
    integer). Each bit is resampled **independently** as a 0/1 indicator
    through ``kernel`` (default :class:`~healpix_resample.BilinearResampler`)
    and thresholded at ``bit_threshold`` (default 50%), then the surviving
    bits are recombined into an output bitmask -- see ``resample()``.

    Not for mutually-exclusive class labels (one class per sample) -- use
    :class:`CategoricalResampler` for that instead.

    Parameters
    ----------
    lon_deg, lat_deg : array-like, shape (N,)
        Sample coordinates in degrees.
    level : int
        HEALPix level (``nside = 2**level``).
    n_bits : int
        Number of bits to resample (bits ``0`` through ``n_bits - 1``).
        Deliberately **not** auto-detected from ``max(mask)``: a bit that
        happens to never be set in one particular ``resample()`` call's
        input (or in one parent-cell subset, if combined with
        ``subset_for_parent_cell``) would otherwise silently vanish from the
        output instead of correctly coming back as all-zero.
    bit_threshold : float
        Fraction (in ``[0, 1]``, default ``0.5``) above which a bit's
        resampled indicator is considered "on" in the output. Distinct from
        ``kernel``'s own ``threshold`` (its cell-retention support
        threshold, if you pass one via ``**kernel_kwargs``) -- the two are
        unrelated despite the similar name.
    kernel : type
        A ``KNeighborsResampler`` subclass used to interpolate each bit's
        indicator map -- default :class:`~healpix_resample.BilinearResampler`.
        See ``resample()``'s docstring for why the default is the
        best-behaved choice for this purpose.
    **kernel_kwargs
        Forwarded to ``kernel``'s constructor together with ``lon_deg``,
        ``lat_deg``, ``level`` (e.g. ``nest``, ``device``, ``dtype``,
        ``out_cell_ids``, ``threshold``, ``sigma_m``, ``verbose``, ...).
    """

    def __init__(
        self,
        lon_deg,
        lat_deg,
        level: int,
        *,
        n_bits: int,
        bit_threshold: float = 0.5,
        kernel: Type[KNeighborsResampler] = BilinearResampler,
        **kernel_kwargs,
    ):
        if n_bits <= 0:
            raise ValueError(f"n_bits must be positive, got {n_bits}")
        if not (0.0 <= bit_threshold <= 1.0):
            raise ValueError(f"bit_threshold must be in [0, 1], got {bit_threshold}")

        self._op = kernel(lon_deg=lon_deg, lat_deg=lat_deg, level=level, **kernel_kwargs)
        self.n_bits = int(n_bits)
        self.bit_threshold = float(bit_threshold)
        self.cell_ids = self._op.cell_ids
        self.device = self._op.device
        self.dtype = self._op.dtype

    @torch.no_grad()
    def resample(self, mask: T_Array, **kwargs) -> ResampleResults:
        """Resample an integer bitmask to HEALPix, bit by bit.

        Args:
            mask: (N,) integer bitmask, one value per sample. Must not
                contain NaN (a mask value has no well-defined "missing"
                indicator decomposition the way a continuous NaN sample
                does elsewhere in this package).
            **kwargs: forwarded to the wrapped ``kernel`` instance's own
                ``resample()`` (e.g. ``lam``/``tol``/``max_iter`` for a
                ``PSFResampler`` kernel).

        How it works
        ------------
        For each bit ``b`` in ``range(n_bits)``, builds the indicator
        ``((mask >> b) & 1).astype(float)`` -- 1.0 where the bit is set, 0.0
        otherwise -- and resamples all ``n_bits`` indicators in a single
        *batched* call to the wrapped kernel (``(n_bits, N) -> (n_bits, K)``,
        using the same batching every resampler in this package already
        supports). Each bit's resampled fraction is then thresholded at
        ``bit_threshold`` independently of every other bit -- unlike
        :class:`CategoricalResampler`, there is no argmax here: any
        combination of bits can end up set in the output, exactly as in the
        input.

        Choice of ``kernel`` matters for what ``bit_threshold`` means
        --------------------------------------------------------------------
        With the default ``BilinearResampler``, each bit's resampled value
        is a proper weighted fraction of "how many nearby samples have this
        bit set", bounded in ``[0, 1]`` -- so "> 0.5" cleanly means "the
        majority of nearby samples have this bit set". ``BicubicResampler``
        (signed kernel, can overshoot past ``[0, 1]``) and ``PSFResampler``
        (CG-solved, also not bounded to ``[0, 1]`` in general) can both
        produce values outside that range, which weakens (without breaking)
        the "50% majority" interpretation of ``bit_threshold`` -- the
        threshold still picks a definite bit value, just not necessarily
        exactly "more than half of nearby samples".

        Returns:
            A ``ResampleResults`` whose ``cell_data`` is the recombined
            integer bitmask per output cell, shape ``(K,)``, and
            ``cell_ids`` matching the wrapped kernel's.
        """
        is_torch_input = isinstance(mask, torch.Tensor)
        mask_t = _as_kernel_device_tensor(mask, self._op)
        if mask_t.ndim != 1:
            raise ValueError(f"mask must be 1-D (N,), got shape {tuple(mask_t.shape)}")
        if torch.is_floating_point(mask_t) and torch.isnan(mask_t).any():
            raise ValueError(
                "mask must not contain NaN -- a bitmask value has no "
                "well-defined 'missing' decomposition into per-bit indicators."
            )

        bits = torch.arange(self.n_bits, device=self._op.device)
        # indicators: (n_bits, N) float -- bit b of mask, for every sample
        indicators = ((mask_t.long().unsqueeze(0) >> bits.unsqueeze(1)) & 1).to(self._op.dtype)

        res = self._op.resample(indicators, **kwargs)  # (n_bits, K)
        bits_on = res.cell_data > self.bit_threshold  # (n_bits, K) bool

        weights = (2 ** torch.arange(self.n_bits, device=bits_on.device, dtype=torch.long))
        out = (bits_on.long() * weights.unsqueeze(1)).sum(dim=0)  # (K,)

        cell_ids = res.cell_ids
        if not is_torch_input:
            out = out.cpu().numpy()
            if isinstance(cell_ids, torch.Tensor):
                cell_ids = cell_ids.cpu().numpy()

        return ResampleResults(cell_data=out, cell_ids=cell_ids)


class CategoricalResampler:
    """Resample **mutually-exclusive** class labels to HEALPix (argmax).

    Use this when each sample carries exactly one class label out of a
    fixed set (e.g. a land-cover or scene classification) -- as opposed to
    :class:`BitmaskResampler`'s independent, co-occurring boolean flags.
    Each class's presence/absence is resampled as a 0/1 indicator through
    ``kernel`` (default :class:`~healpix_resample.BilinearResampler`), and
    the output class per cell is whichever indicator scored highest --
    "argmax_over_bilinear" in the issue's own working name.

    Parameters
    ----------
    lon_deg, lat_deg : array-like, shape (N,)
        Sample coordinates in degrees.
    level : int
        HEALPix level (``nside = 2**level``).
    kernel : type
        A ``KNeighborsResampler`` subclass used to interpolate each class's
        indicator map -- default :class:`~healpix_resample.BilinearResampler`.
        See ``resample()``'s docstring for why the default is the
        best-behaved choice for this purpose.
    **kernel_kwargs
        Forwarded to ``kernel``'s constructor together with ``lon_deg``,
        ``lat_deg``, ``level``.
    """

    def __init__(
        self,
        lon_deg,
        lat_deg,
        level: int,
        *,
        kernel: Type[KNeighborsResampler] = BilinearResampler,
        **kernel_kwargs,
    ):
        self._op = kernel(lon_deg=lon_deg, lat_deg=lat_deg, level=level, **kernel_kwargs)
        self.cell_ids = self._op.cell_ids
        self.device = self._op.device
        self.dtype = self._op.dtype

    @torch.no_grad()
    def resample(
        self,
        mask: T_Array,
        *,
        return_scores: bool = False,
        softmax_temperature: float = 0.1,
        **kwargs,
    ) -> ResampleResults:
        """Resample mutually-exclusive class labels to HEALPix by argmax.

        Args:
            mask: (N,) class label per sample (any integer dtype; need not
                be contiguous or zero-based -- the distinct values actually
                present are discovered from ``mask`` itself on every call).
                Must not contain NaN.
            return_scores: if True, also compute and return a
                softmax-normalized per-class score (see below). Costs one
                extra elementwise pass over ``(n_classes, K)``; ``False`` by
                default to keep the common case cheap.
            softmax_temperature: only used when ``return_scores=True``.
                Lower values sharpen the softmax towards the hard argmax
                decision (in the limit, a one-hot at the winning class);
                higher values spread probability mass across close
                runners-up. The default (``0.1``) is tuned for
                ``BilinearResampler``'s natural ``[0, 1]``-ish score scale
                (see the ``kernel`` caveat below) -- a bare ``softmax``
                without dividing by a temperature well below 1 would barely
                sharpen scores that are already confined to such a narrow
                range, producing near-uniform "probabilities" even for a
                clear winner. Retune if you change ``kernel`` or if your
                classes are unusually balanced/imbalanced.
            **kwargs: forwarded to the wrapped ``kernel`` instance's own
                ``resample()``.

        How it works, and how ties are broken
        --------------------------------------------------------------------
        Builds a one-hot indicator per distinct class in ``mask`` (shape
        ``(n_classes, N)``), resamples all of them in a single batched call
        to ``kernel`` (``(n_classes, N) -> (n_classes, K)``), and picks
        ``argmax`` over the class axis per output cell. Exact ties are rare
        in practice (they require perfect geometric symmetry between two
        classes' local support) but are broken deterministically: the
        lowest-valued tied class wins (``torch.argmax`` returns the first
        maximal index along the reduced axis, and classes are sorted
        ascending before the argmax).

        Choice of ``kernel`` matters for how ``return_scores`` behaves
        --------------------------------------------------------------------
        With the default ``BilinearResampler``, per-class scores are
        bounded in ``[0, 1]`` and -- because every sample belongs to
        exactly one class, so its one-hot indicator sums to 1 -- sum to
        (very close to) 1 across classes for every retained cell, by
        construction of ``BilinearResampler.M``'s per-cell normalization.
        This makes the raw scores themselves already a reasonable
        probability-like quantity even before the softmax step.
        ``BicubicResampler`` (signed kernel) and ``PSFResampler``
        (CG-solved) do **not** carry this guarantee -- their per-class
        scores can be negative or exceed 1 -- so ``argmax`` (the decision
        that matters) remains meaningful with either, but the softmax
        output is a softer, less strictly calibrated confidence signal for
        those two than for ``BilinearResampler``.

        Returns:
            If ``return_scores=False`` (default): a ``ResampleResults``
            whose ``cell_data`` is the winning class label per cell, shape
            ``(K,)``.
            If ``return_scores=True``: a ``CategoricalResampleResults`` with
            the same ``cell_data``/``cell_ids``, plus ``classes`` (the
            distinct class labels, in score-axis order) and ``scores`` (the
            softmax-normalized ``(n_classes, K)`` array).
        """
        is_torch_input = isinstance(mask, torch.Tensor)
        mask_t = _as_kernel_device_tensor(mask, self._op)
        if mask_t.ndim != 1:
            raise ValueError(f"mask must be 1-D (N,), got shape {tuple(mask_t.shape)}")
        if torch.is_floating_point(mask_t) and torch.isnan(mask_t).any():
            raise ValueError(
                "mask must not contain NaN -- a class label has no "
                "well-defined 'missing' one-hot decomposition."
            )

        classes = torch.unique(mask_t)  # sorted ascending, (n_classes,)
        # one-hot indicators: (n_classes, N)
        indicators = (mask_t.unsqueeze(0) == classes.unsqueeze(1)).to(self._op.dtype)

        res = self._op.resample(indicators, **kwargs)  # (n_classes, K)
        scores = res.cell_data  # torch tensor -- indicators was torch, so this always is too

        winner_idx = torch.argmax(scores, dim=0)  # (K,) -- ties -> lowest class value
        winner_class = classes[winner_idx]

        cell_ids = res.cell_ids
        result_classes = classes
        result_scores = None
        if return_scores:
            z = scores / float(softmax_temperature)
            z = z - z.max(dim=0, keepdim=True).values  # numerical stability
            probs = torch.exp(z)
            result_scores = probs / probs.sum(dim=0, keepdim=True)

        if not is_torch_input:
            winner_class = winner_class.cpu().numpy()
            if isinstance(cell_ids, torch.Tensor):
                cell_ids = cell_ids.cpu().numpy()
            result_classes = result_classes.cpu().numpy()
            if result_scores is not None:
                result_scores = result_scores.cpu().numpy()

        if not return_scores:
            return ResampleResults(cell_data=winner_class, cell_ids=cell_ids)
        return CategoricalResampleResults(
            cell_data=winner_class,
            cell_ids=cell_ids,
            classes=result_classes,
            scores=result_scores,
        )
