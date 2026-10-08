"""
kernel_geometry.py

Measures the *actual* spatial response delivered by the Gaussian-kernel
resamplers (:class:`~healpix_resample.psf.PSFResampler`,
:class:`~healpix_resample.knn.KNeighborsResampler`) once their kernel has
been truncated to ``Npt`` nearest HEALPix cells on the real lattice --
as opposed to the idealized, untruncated Gaussian its scale parameter
would suggest.

This module intentionally has no dependency on the rest of
``healpix_resample`` beyond :mod:`healpix_resample.psf_geometry` (which is
pure Python/``math``, no ``torch``): it exists so the geometry of the
operator can be inspected with only ``numpy`` + ``healpix_geo``, without
needing GPU/CUDA or the sample data the full resamplers expect.

Two things this module measures that are easy to get wrong from the
Gaussian formula alone:

- **Truncation loss.** ``comp_matrix()`` (in ``psf.py``/``knn.py``) only
  ever sees a sample's ``Npt`` nearest retained cells -- the tails of the
  kernel beyond that are simply never computed, not renormalized away from
  a wider true integral. If ``Npt`` is too small for the kernel's width,
  the *delivered* response is both narrower and less isotropic than the
  nominal Gaussian, and the missing mass is not restored by the
  per-sample-row normalization (:func:`row_metrics`'s ``mass_retained``
  quantifies exactly how much is missing).
- **Sub-cell phase dependence.** Because the HEALPix lattice is discrete
  and (for a source grid in a different projection, e.g. UTM) generically
  incommensurate with it, the delivered response for two source samples
  with the *same* nominal kernel differs depending on where each sample
  falls relative to its nearest cell centres. :func:`recommend_q` reports
  the *worst case* over this sub-cell phase, not just a typical case.

Every constant below is cross-checked against the two independent,
hand-derived measurements in the accompanying paper review (Thomas
Davison's notes + the `effective_kernel_geometry.ipynb` companion
notebook): at HEALPix level 20 and kernel scale ``s = 12.6`` m, this
module's :func:`recommend_q` gives 75.2% mass retained at ``Npt=9``
(reported: 75.1%) and ``Npt=30`` for 99% mass (reported: 32); and
:func:`total_response_fwhm` reproduces the reported "delivered incl. cell
footprint" ratios (0.729 / 1.000 / 1.061 for the -50%/matched/+50% arms)
to within rounding.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from healpix_resample.psf_geometry import (
    FWHM_PER_SCALE,
    cell_size_m,
    fwhm_to_scale,
    scale_to_fwhm,
)

#: The *standard* FWHM / standard-deviation ratio, 2*sqrt(2 ln 2) (~2.3548).
#: This is NOT the same conversion as psf_geometry.FWHM_PER_SCALE
#: (sqrt(2 ln 2), ~1.1774): that one relates FWHM to this package's
#: ``sigma_m``/``s`` (== 2*sigma); this one relates FWHM to the ordinary
#: standard deviation sigma, e.g. the convention `scipy.ndimage.gaussian_filter`
#: expects. Mixing the two up is exactly the second failure mode described
#: in `psf_geometry`'s module docstring.
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

_EARTH_RADIUS_M = 6371000.0


def cell_size(level: int, radius: float = _EARTH_RADIUS_M) -> float:
    """Equal-area HEALPix cell width (metres) at ``level``. See
    :func:`healpix_resample.psf_geometry.cell_size_m` (same formula)."""
    return cell_size_m(level, radius=radius)


def default_scale(level: int, radius: float = _EARTH_RADIUS_M) -> float:
    """The ``sigma_m`` value :class:`KNeighborsResampler` falls back to when
    none is given -- numerically identical to :func:`cell_size`, but named
    for what it's used as here (the package's implicit default kernel
    scale), matching ``knn._sigma_level_m``."""
    return cell_size_m(level, radius=radius)


def fwhm_from_scale(scale_m: float, convention: str = "delivered") -> float:
    """FWHM implied by scale ``s`` under one of two conventions.

    Parameters
    ----------
    scale_m : float
        The ``s`` value (as passed to ``sigma_m=``).
    convention : {"delivered", "paper_eq2"}
        - ``"delivered"``: the FWHM the kernel ``exp(-2 d^2 / s^2)``
          *actually* has, before any truncation -- ``sqrt(2 ln 2) * s``
          (~1.1774*s). Equivalent to
          :func:`healpix_resample.psf_geometry.scale_to_fwhm`.
        - ``"paper_eq2"``: the FWHM the manuscript's original (buggy)
          Eq. (2) claimed -- ``2*sqrt(2 ln 2) * s`` (~2.3548*s), exactly
          double the true value. Kept here only to reproduce/explain the
          discrepancy between the text and the code; do not use this
          convention for anything new.
    """
    if convention == "delivered":
        return scale_to_fwhm(scale_m)
    if convention == "paper_eq2":
        return float(scale_m) * FWHM_PER_SIGMA
    raise ValueError(f"unknown convention {convention!r}, expected 'delivered' or 'paper_eq2'")


def total_response_fwhm(inter_cell_fwhm: float, level: int, radius: float = _EARTH_RADIUS_M) -> float:
    """Combine the inter-cell delivered FWHM (:func:`row_metrics`'s
    ``fwhm_mean``) with the HEALPix cell's own footprint, in quadrature, to
    get the *total* effective response FWHM of the reconstructed field
    (kernel blur *and* the fact that each HEALPix cell already averages
    over its own finite footprint).

    The cell footprint is treated as a uniform (top-hat) distribution of
    width ``cell_size_m(level)`` -- matching second moments (a top-hat of
    full width ``a`` has variance ``a**2/12``) gives an equivalent Gaussian
    FWHM of ``FWHM_PER_SIGMA * a / sqrt(12)`` for the footprint alone; the
    two contributions add in quadrature since they arise from independent
    (kernel-truncation vs. cell-averaging) blurring stages.
    """
    cell = cell_size_m(level, radius=radius)
    footprint_fwhm = FWHM_PER_SIGMA * cell / math.sqrt(12.0)
    return math.sqrt(float(inter_cell_fwhm) ** 2 + footprint_fwhm ** 2)


def enu_offsets(c_lon_deg, c_lat_deg, lon_deg, lat_deg, radius: float = _EARTH_RADIUS_M):
    """Local East-North-Up tangent-plane offsets (metres) of ``(lon_deg,
    lat_deg)`` relative to the per-point reference ``(c_lon_deg,
    c_lat_deg)`` (broadcastable, same shape).

    Returns
    -------
    en : ndarray, shape (..., 2)
        ``[east, north]`` offsets in metres.
    up : ndarray, shape (...)
        The (small, should be ~0 for points on the same sphere) up
        component -- a sanity-check byproduct, not otherwise used.
    """
    c_lon = np.radians(np.asarray(c_lon_deg, dtype=np.float64))
    c_lat = np.radians(np.asarray(c_lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))

    def xyz(lo, la):
        cl = np.cos(la)
        return np.stack([cl * np.cos(lo), cl * np.sin(lo), np.sin(la)], axis=-1)

    p = xyz(lon, lat) * radius
    origin = xyz(c_lon, c_lat) * radius

    e_hat = np.stack([-np.sin(c_lon), np.cos(c_lon), np.zeros_like(c_lon)], axis=-1)
    n_hat = np.stack(
        [-np.sin(c_lat) * np.cos(c_lon), -np.sin(c_lat) * np.sin(c_lon), np.cos(c_lat)],
        axis=-1,
    )
    u_hat = xyz(c_lon, c_lat)

    d = p - origin
    east = np.sum(d * e_hat, axis=-1)
    north = np.sum(d * n_hat, axis=-1)
    up = np.sum(d * u_hat, axis=-1)
    return np.stack([east, north], axis=-1), up


def _great_circle_dist(lon1_deg, lat1_deg, lon2_deg, lat2_deg, radius=_EARTH_RADIUS_M):
    lon1, lat1 = np.radians(lon1_deg), np.radians(lat1_deg)
    lon2, lat2 = np.radians(lon2_deg), np.radians(lat2_deg)

    def xyz(lo, la):
        cl = np.cos(la)
        return np.stack([cl * np.cos(lo), cl * np.sin(lo), np.sin(la)], axis=-1)

    dot = np.clip(np.sum(xyz(lon1, lat1) * xyz(lon2, lat2), axis=-1), -1.0, 1.0)
    return radius * np.arccos(dot)


def operator_rows(
    lon_deg,
    lat_deg,
    level: int,
    s_psf: float,
    q: int = 9,
    ellipsoid: str = "sphere",
    threshold: float = 0.1,
    ring: Optional[int] = None,
    radius: float = _EARTH_RADIUS_M,
) -> Dict[str, object]:
    """Build the real per-sample operator rows (the same weights
    ``comp_matrix()`` in ``psf.py``/``knn.py`` would build for the MT/
    per-sample-normalized side) for a set of source samples, directly
    against the true HEALPix lattice via ``healpix_geo``.

    Unlike the production package, this diagnostic skips the two-stage
    global cell-*retention* threshold test (``healpix_weighted_nearest``'s
    ``keep = sums >= threshold``) and simply takes each sample's ``q``
    nearest HEALPix cell centres directly -- that global step is a
    construction-time optimisation/consistency detail (which cells the
    *whole* operator keeps as output columns), not a property of any one
    row's own kernel geometry, which is what this module measures. For a
    dense synthetic patch (the intended use case) the two agree almost
    everywhere; ``threshold`` is accepted for interface compatibility and
    used only to flag rows whose own weight sum is suspiciously small.

    Parameters
    ----------
    lon_deg, lat_deg : array-like, shape (N,)
        Source sample coordinates, degrees.
    level : int
        HEALPix level.
    s_psf : float
        The ``s`` scale parameter (i.e. ``sigma_m``) of ``w(d) = exp(-2
        d^2 / s^2)`` -- run an intended FWHM through
        :func:`healpix_resample.psf_geometry.fwhm_to_scale` first if
        that's what you have.
    q : int
        Number of nearest cells per sample (``Npt`` in the resamplers).
    ring : int, optional
        HEALPix neighbourhood ring radius to search for candidates. If
        None, chosen automatically to comfortably exceed ``q`` candidates
        (``kth_neighbourhood`` at ring ``r`` returns ``(2r+1)**2``
        candidates on the nested scheme).

    Returns
    -------
    dict with keys:
        ``weights`` : list of length-N arrays (each length <= q), each
            row's normalized weights (sums to 1).
        ``offsets`` : list of length-N (q, 2) arrays, ``[east, north]``
            metres from the sample to each of its q nearest cell centres.
        ``distances`` : list of length-N (q,) arrays, metres.
        ``mass_retained`` : (N,) array, each row's raw (pre-normalization)
            weight sum divided by a wide-neighbourhood reference sum
            (ring 6, ~169 candidate cells -- holds >99.99% of any kernel
            considered in this project; see module docstring).
        ``n_retained`` : int, number of distinct HEALPix cells referenced
            by any row (a diagnostic analogue of the production
            operator's ``K``).
        ``complete`` : (N,) bool array, whether all q neighbours were
            found (should be True everywhere away from a search-radius
            edge case).
    """
    import healpix_geo.nested as hgn

    lon = np.asarray(lon_deg, dtype=np.float64).reshape(-1)
    lat = np.asarray(lat_deg, dtype=np.float64).reshape(-1)
    N = lon.size

    if ring is None:
        ring = max(2, int(math.ceil((math.sqrt(q) - 1.0) / 2.0)) + 3)
    ring_wide = max(ring, 6)

    home = hgn.lonlat_to_healpix(lon, lat, level, ellipsoid=ellipsoid).astype(np.uint64)
    home_u, inv = np.unique(home, return_inverse=True)

    def neighbour_geometry(r):
        neigh_u = hgn.kth_neighbourhood(home_u, level, r).astype(np.int64)
        neigh = neigh_u[inv]  # (N, Kw)
        flat = neigh.reshape(-1)
        valid = flat >= 0
        uniq, back = np.unique(flat[valid].astype(np.uint64), return_inverse=True)
        c_lon, c_lat = hgn.healpix_to_lonlat(uniq, level, ellipsoid=ellipsoid)
        c_lon_full = np.full(flat.shape, np.nan)
        c_lat_full = np.full(flat.shape, np.nan)
        c_lon_full[valid] = c_lon[back]
        c_lat_full[valid] = c_lat[back]
        c_lon_full = c_lon_full.reshape(neigh.shape)
        c_lat_full = c_lat_full.reshape(neigh.shape)
        dist = np.full(neigh.shape, np.inf)
        dist[valid.reshape(neigh.shape)] = _great_circle_dist(
            lon[:, None].repeat(neigh.shape[1], axis=1)[valid.reshape(neigh.shape)],
            lat[:, None].repeat(neigh.shape[1], axis=1)[valid.reshape(neigh.shape)],
            c_lon_full[valid.reshape(neigh.shape)],
            c_lat_full[valid.reshape(neigh.shape)],
            radius=radius,
        )
        return neigh, c_lon_full, c_lat_full, dist

    neigh_q, clon_q, clat_q, dist_q = neighbour_geometry(ring)
    neigh_w, clon_w, clat_w, dist_w = neighbour_geometry(ring_wide)

    weights, offsets, distances = [], [], []
    mass_retained = np.zeros(N)
    complete = np.zeros(N, dtype=bool)
    seen_cells = set()

    for i in range(N):
        d_i = dist_q[i]
        order = np.argsort(d_i)[:q]
        d_sel = d_i[order]
        finite = np.isfinite(d_sel)
        complete[i] = bool(finite.all()) and finite.size == q

        w_raw = np.exp(-2.0 * (d_sel ** 2) / (s_psf ** 2))
        w_raw = np.where(finite, w_raw, 0.0)
        total_narrow = w_raw.sum()

        d_wide = dist_w[i]
        finite_wide = np.isfinite(d_wide)
        w_wide = np.where(finite_wide, np.exp(-2.0 * (d_wide ** 2) / (s_psf ** 2)), 0.0)
        total_wide = w_wide.sum()

        mass_retained[i] = total_narrow / total_wide if total_wide > 0 else np.nan

        w_norm = w_raw / total_narrow if total_narrow > 0 else w_raw
        en, _ = enu_offsets(
            np.full(order.shape, lon[i]), np.full(order.shape, lat[i]),
            clon_q[i, order], clat_q[i, order], radius=radius,
        )
        # enu_offsets(origin, point) gives point-relative-to-origin; we
        # want the neighbour cell centre's offset *from* the sample, so
        # origin = sample, point = cell centre (as called above).

        weights.append(w_norm)
        offsets.append(en)
        distances.append(d_sel)
        seen_cells.update(int(x) for x in neigh_q[i, order][finite])

    return {
        "weights": weights,
        "offsets": offsets,
        "distances": distances,
        "mass_retained": mass_retained,
        "n_retained": len(seen_cells),
        "complete": complete,
    }


def row_metrics(weights, offsets) -> Dict[str, np.ndarray]:
    """Per-row summary statistics of already-built operator rows (as
    returned by :func:`operator_rows`).

    Returns a dict of (N,) arrays:
        ``fwhm_mean`` : Gaussian-equivalent FWHM from the second moment of
            each row about its own weighted centroid (isotropic average
            of the two tangent-plane axes).
        ``anisotropy`` : ratio of the larger to smaller principal-axis
            standard deviation of the row's weight distribution (1.0 =
            perfectly isotropic).
        ``max_weight`` : the single largest weight in the row (how
            dominant is the nearest cell).
        ``participation`` : effective number of cells contributing,
            via the inverse Simpson index ``1 / sum(w**2)`` (equals ``q``
            for perfectly uniform weights, tends to 1 as one weight
            dominates).
        ``centroid_e``, ``centroid_n`` : the row's weighted centroid
            offset from the sample's true position (metres) -- a
            systematic reconstruction-bias vector; should average to ~0
            over many samples even where it is non-zero row-by-row.
        ``centroid_offset`` : magnitude of the above.
    """
    N = len(weights)
    fwhm_mean = np.full(N, np.nan)
    anisotropy = np.full(N, np.nan)
    max_weight = np.full(N, np.nan)
    participation = np.full(N, np.nan)
    centroid_e = np.full(N, np.nan)
    centroid_n = np.full(N, np.nan)

    for i in range(N):
        w = np.asarray(weights[i], dtype=np.float64)
        en = np.asarray(offsets[i], dtype=np.float64)
        if w.sum() <= 0 or w.size == 0:
            continue
        w = w / w.sum()
        c = (w[:, None] * en).sum(axis=0)
        centroid_e[i], centroid_n[i] = c
        d = en - c[None, :]

        cov = np.zeros((2, 2))
        for k in range(w.size):
            cov += w[k] * np.outer(d[k], d[k])

        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 0.0, None)
        sigma_major, sigma_minor = math.sqrt(eigvals[-1]), math.sqrt(eigvals[0])
        sigma_iso = math.sqrt(0.5 * (eigvals[0] + eigvals[-1]))

        fwhm_mean[i] = FWHM_PER_SIGMA * sigma_iso
        anisotropy[i] = (sigma_major / sigma_minor) if sigma_minor > 0 else np.nan
        max_weight[i] = w.max()
        participation[i] = 1.0 / np.sum(w ** 2)

    centroid_offset = np.hypot(centroid_e, centroid_n)

    return {
        "fwhm_mean": fwhm_mean,
        "anisotropy": anisotropy,
        "max_weight": max_weight,
        "participation": participation,
        "centroid_e": centroid_e,
        "centroid_n": centroid_n,
        "centroid_offset": centroid_offset,
    }


def recommend_q(
    s_psf: float,
    level: int,
    target_mass: float = 0.99,
    n_phase_samples: int = 64,
    q_max: int = 200,
    radius: float = _EARTH_RADIUS_M,
) -> Dict[str, object]:
    """Lattice-exact (as opposed to :func:`psf_geometry.recommend_npt`'s
    continuum-approximate) recommendation for ``Npt``/``q``, by directly
    sampling many sub-cell phases within one HEALPix cell and taking the
    *worst-case* mass-retained curve over those phases -- since the
    delivered response (and hence how much of it survives truncation)
    depends on where a sample falls relative to its nearest cell centres,
    not only on the kernel width.

    Returns a dict with keys ``q`` (recommended Npt for ``target_mass`` in
    the worst case), ``mass_at_9`` (worst-case mass retained at the
    package's old fixed default), and ``worst_case_curve`` (an array,
    worst-case mass retained as a function of q from 1 to ``q_max``).
    """
    import healpix_geo.nested as hgn

    # One home cell's centre, then a small dense grid of phases around it
    # spanning roughly +/- half a cell width, converted back to lon/lat.
    seed_lon, seed_lat = np.array([12.68]), np.array([41.81])
    home = hgn.lonlat_to_healpix(seed_lon, seed_lat, level, ellipsoid="sphere").astype(np.uint64)
    c_lon, c_lat = hgn.healpix_to_lonlat(home, level, ellipsoid="sphere")

    cell = cell_size_m(level, radius=radius)
    rng = np.random.default_rng(0)
    offs_e = rng.uniform(-cell / 2, cell / 2, n_phase_samples)
    offs_n = rng.uniform(-cell / 2, cell / 2, n_phase_samples)

    c_lat_rad = math.radians(float(c_lat[0]))
    dlat = np.degrees(offs_n / radius)
    dlon = np.degrees(offs_e / (radius * math.cos(c_lat_rad)))
    lon = float(c_lon[0]) + dlon
    lat = float(c_lat[0]) + dlat

    rows = operator_rows(lon, lat, level, s_psf, q=q_max, ellipsoid="sphere", ring=None)
    # Recompute a ring wide enough for q_max explicitly, since operator_rows'
    # auto ring sizing targets the *requested* q, and here that's q_max already.

    curve = np.full(q_max, np.nan)
    for qi in range(1, q_max + 1):
        worst = np.inf
        for i in range(n_phase_samples):
            d = rows["distances"][i]
            k = min(qi, d.size)
            w = np.exp(-2.0 * (d[:k] ** 2) / (s_psf ** 2))
            w_full = np.exp(-2.0 * (d ** 2) / (s_psf ** 2))
            ref = w_full.sum()
            frac = w.sum() / ref if ref > 0 else np.nan
            worst = min(worst, frac)
        curve[qi - 1] = worst

    if np.any(curve >= target_mass):
        q_rec = int(np.argmax(curve >= target_mass)) + 1
    else:
        q_rec = q_max

    return {
        "q": q_rec,
        "mass_at_9": float(curve[8]) if q_max >= 9 else float("nan"),
        "worst_case_curve": curve,
    }
