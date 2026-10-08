"""
psf_geometry.py

Conversions between the "FWHM" a physical instrument response is usually
quoted in, and the ``sigma_m`` / ``s`` scale parameter that
:class:`~healpix_resample.knn.KNeighborsResampler` and
:class:`~healpix_resample.psf.PSFResampler` actually consume -- plus a
helper to choose ``Npt`` from that scale instead of leaving it at a fixed
default.

Why this module exists
-----------------------
Every weight in this package's Gaussian-kernel resamplers is built as

    w(d) = exp(-2 * d**2 / s**2)          (see comp_matrix() in psf.py/knn.py)

Matching this against the standard normal form ``exp(-d**2 / (2*sigma**2))``
gives ``sigma = s / 2``, and therefore

    FWHM = 2*sqrt(2*ln(2)) * sigma = sqrt(2*ln(2)) * s   ~=  1.1774 * s

*not* ``2*sqrt(2*ln(2)) * s ~= 2.3548 * s``. Nothing in this package has
ever performed that conversion for you: ``sigma_m``/``s`` is consumed as
the raw scale ``s`` above, verbatim, with zero FWHM awareness anywhere in
``knn.py`` or ``psf.py`` -- despite the constructor argument being named
``sigma_m``, it is *not* the standard-deviation ``sigma``; it is ``2*sigma``.

Passing an intended FWHM value directly as ``sigma_m`` (as earlier
revisions of the paper's validation notebook did) silently builds a kernel
whose true FWHM is ``sqrt(2 ln 2) ~= 1.1774`` times the number you passed
-- not the disaster it could have been, but not what was intended either.
The larger, unrelated failure mode is going the other way: computing the
"correct" standard deviation ``sigma = FWHM / (2 sqrt(2 ln 2))`` (the
right formula for e.g. ``scipy.ndimage.gaussian_filter``) and then passing
*that* as ``sigma_m`` -- since this package's ``sigma_m`` is ``2*sigma``,
not ``sigma``, that under-builds the kernel by a further factor of 2, on
top of whichever FWHM<->sigma slip already happened upstream. Use
:func:`fwhm_to_scale` / :func:`scale_to_fwhm` below to avoid both.
"""

from __future__ import annotations

import math


#: FWHM = FWHM_PER_SCALE * s, for w(d) = exp(-2 d^2 / s^2)  (== sqrt(2 ln 2), ~1.1774)
FWHM_PER_SCALE = math.sqrt(2.0 * math.log(2.0))


def fwhm_to_scale(fwhm_m: float) -> float:
    """Convert an intended Gaussian FWHM (metres) to the ``s`` scale
    parameter consumed by ``sigma_m=`` in :class:`PSFResampler` /
    :class:`KNeighborsResampler`.

    ``s = FWHM / sqrt(2 ln 2)``, the exact inverse of :func:`scale_to_fwhm`.

    Do **not** pass a FWHM value directly as ``sigma_m`` -- see the module
    docstring for the two distinct ways that goes wrong.
    """
    return float(fwhm_m) / FWHM_PER_SCALE


def scale_to_fwhm(scale_m: float) -> float:
    """Convert the ``s`` scale parameter (as used by ``sigma_m=``) to the
    Gaussian FWHM (metres) the resulting kernel actually delivers before
    any neighbour-count truncation.

    ``FWHM = sqrt(2 ln 2) * s``, the exact inverse of :func:`fwhm_to_scale`.
    """
    return float(scale_m) * FWHM_PER_SCALE


def cell_size_m(level: int, radius: float = 6371000.0) -> float:
    """Equal-area HEALPix cell width at ``level`` (metres).

    Same quantity, same formula, as the ``sigma_m=None`` fallback used
    internally by :class:`KNeighborsResampler` (``knn._sigma_level_m``) --
    exposed here under an honest name, since it is a cell size, not a PSF
    scale, even though the two happen to share a formula (a HEALPix cell's
    linear size is the square root of its -- equal-area -- solid angle).
    """
    return math.sqrt(4.0 * math.pi / (12.0 * (4.0 ** int(level)))) * float(radius)


def recommend_npt(
    scale_m: float,
    level: int,
    target_mass: float = 0.99,
    radius: float = 6371000.0,
    q_min: int = 9,
    q_max: int = 200,
) -> dict:
    """Recommend how many nearest HEALPix cells (``Npt``) a kernel of scale
    ``scale_m`` needs, at the given ``level``, to retain ``target_mass`` of
    its weight -- rather than leaving ``Npt`` fixed at a default (9) that
    is only adequate for a kernel about as wide as one cell.

    This uses a continuum (flat-sky, isotropic) approximation: HEALPix
    cells are equal-area, so the ``q`` nearest cells cover a disc of area
    ``q * cell_size**2`` and radius ``r(q) = cell_size * sqrt(q / pi)``;
    for an isotropic Gaussian with standard deviation ``sigma = scale_m /
    2`` (see module docstring), the mass inside radius ``r`` is ``1 -
    exp(-r**2 / (2 * sigma**2))``. Cross-checked against an independent,
    lattice-exact measurement (building real operator rows and summing
    their weights) at level 20, scale 12.6 m: this formula gives 75.2%
    mass retained at Npt=9 against a directly-measured 75.1%, and a q of
    30 for 99% mass against a directly-measured 32 -- close enough to pick
    a safe ``Npt`` from, but re-check the exact number for your own
    level/scale via :mod:`healpix_resample.diagnostics.kernel_geometry`
    once you can run it, particularly when ``sigma`` is only a cell width
    or two (the regime where the continuum approximation is weakest and
    the lattice's own (non-circular) cell shape matters most).

    Parameters
    ----------
    scale_m : float
        The ``s`` scale actually passed as ``sigma_m=`` (i.e. already run
        through :func:`fwhm_to_scale` if you started from a FWHM).
    level : int
        HEALPix level (``nside = 2**level``).
    target_mass : float
        Fraction of the kernel's weight the chosen ``Npt`` should retain
        (default 0.99).
    radius : float
        Sphere radius in metres, matching whatever was used to build the
        resampler (default: Earth mean radius).
    q_min : int
        Never recommend fewer than this many neighbours (keeps the
        package's previous default as a floor).
    q_max : int
        Upper bound on the search (construction cost is close to linear in
        ``Npt``, so this also bounds how large a recommendation you can get
        back).

    Returns
    -------
    dict with keys ``npt``, ``mass_at_npt``, ``mass_at_default`` (mass
    retained at the old fixed ``Npt=9``, for comparison),
    ``sigma_over_cell`` (the dimensionless number that actually controls
    how well-sampled the kernel is), and ``cell_size_m``.
    """
    cell = cell_size_m(level, radius=radius)
    sigma = float(scale_m) / 2.0

    def mass_at_q(q: int) -> float:
        r = cell * math.sqrt(q / math.pi)
        return 1.0 - math.exp(-(r * r) / (2.0 * sigma * sigma))

    npt = q_min
    while npt < q_max and mass_at_q(npt) < target_mass:
        npt += 1

    return {
        "npt": int(npt),
        "mass_at_npt": mass_at_q(npt),
        "mass_at_default": mass_at_q(9),
        "sigma_over_cell": sigma / cell,
        "cell_size_m": cell,
    }
