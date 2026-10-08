"""
tests/test_kernel_geometry.py

Test suite for `healpix_resample.diagnostics.kernel_geometry`, the module
that measures what the Gaussian-kernel resamplers *actually* deliver once
truncated to `Npt` nearest cells on the real HEALPix lattice -- as opposed
to the idealized, untruncated Gaussian a scale parameter alone would
suggest. Written to reproduce, from a completely independent
implementation, the operator-geometry measurements in Thomas Davison's
paper review (and its companion `effective_kernel_geometry.ipynb`), which
were the basis for concluding that `Npt=9` silently truncates any kernel
wider than about one cell.

Only needs `numpy` + `healpix_geo` -- no `torch`, no GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from healpix_resample.diagnostics.kernel_geometry import (
    FWHM_PER_SIGMA,
    cell_size,
    default_scale,
    fwhm_from_scale,
    operator_rows,
    recommend_q,
    row_metrics,
    total_response_fwhm,
)

LEVEL = 20


def _patch(n=32, gsd=10.0, lon_c=12.68, lat_c=41.81):
    from pyproj import CRS, Transformer

    zone = int((lon_c + 180.0) // 6.0) + 1
    crs = CRS.from_dict({"proj": "utm", "zone": zone, "south": lat_c < 0})
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x_c, y_c = fwd.transform(lon_c, lat_c)
    half = (n - 1) * gsd / 2.0
    xx, yy = np.meshgrid(x_c + np.linspace(-half, half, n), y_c + np.linspace(-half, half, n))
    lon, lat = inv.transform(xx.ravel(), yy.ravel())
    return np.asarray(lon), np.asarray(lat)


def test_fwhm_from_scale_conventions():
    s = 12.6
    delivered = fwhm_from_scale(s, "delivered")
    paper_eq2 = fwhm_from_scale(s, "paper_eq2")
    assert delivered == pytest.approx(14.835, abs=1e-2)
    assert paper_eq2 == pytest.approx(29.671, abs=1e-2)
    # The old manuscript bug: paper_eq2 is exactly double delivered.
    assert paper_eq2 == pytest.approx(2.0 * delivered)


def test_fwhm_from_scale_invalid_convention():
    with pytest.raises(ValueError):
        fwhm_from_scale(12.6, "nonsense")


def test_cell_size_matches_default_scale():
    assert cell_size(LEVEL) == pytest.approx(default_scale(LEVEL))
    assert cell_size(LEVEL) == pytest.approx(6.2176, abs=1e-3)


@pytest.mark.parametrize(
    "s_psf,expected_mass_at_9",
    [
        # Cross-checked against Thomas Davison's review letter (0.751 /
        # 0.996 / 0.461 -- these are the exact "code ran with the FWHM
        # value used unconverted as scale" configurations from the bug).
        (12.6, 0.751),
        (6.3, 0.996),
        (18.9, 0.461),
    ],
)
def test_operator_rows_mass_retained_matches_review(s_psf, expected_mass_at_9):
    lon, lat = _patch(n=32)
    rows = operator_rows(lon, lat, LEVEL, s_psf, q=9)
    assert np.nanmedian(rows["mass_retained"]) == pytest.approx(expected_mass_at_9, abs=0.03)
    assert np.all(rows["complete"])


def test_row_metrics_matched_arm_matches_review():
    lon, lat = _patch(n=32)
    rows = operator_rows(lon, lat, LEVEL, 12.6, q=9)
    m = row_metrics(rows["weights"], rows["offsets"])
    # Reported: fwhm_mean~10.83, anisotropy~1.09, max_weight~0.19,
    # participation~7.76 (median, over a 256x256 patch in the original
    # review -- a smaller patch here, so allow more slack).
    assert np.nanmedian(m["fwhm_mean"]) == pytest.approx(10.83, rel=0.1)
    assert np.nanmedian(m["anisotropy"]) == pytest.approx(1.09, rel=0.15)
    assert np.nanmedian(m["max_weight"]) == pytest.approx(0.19, rel=0.2)
    assert np.nanmedian(m["participation"]) == pytest.approx(7.76, rel=0.15)


def test_total_response_fwhm_reproduces_review_ratios():
    # Reported "delivered incl. cell footprint" ratios, -50%/matched/+50%:
    # 0.729 / 1.000 / 1.061, from inter-cell FWHMs 7.354 / 10.834 / 11.598.
    base = total_response_fwhm(10.834, LEVEL)
    narrow = total_response_fwhm(7.354, LEVEL)
    wide = total_response_fwhm(11.598, LEVEL)
    assert narrow / base == pytest.approx(0.729, abs=0.01)
    assert wide / base == pytest.approx(1.061, abs=0.01)


def test_recommend_q_increases_with_scale():
    q_narrow = recommend_q(6.3, LEVEL, n_phase_samples=16, q_max=40)["q"]
    q_matched = recommend_q(12.6, LEVEL, n_phase_samples=16, q_max=60)["q"]
    q_wide = recommend_q(18.9, LEVEL, n_phase_samples=16, q_max=90)["q"]
    assert q_narrow <= q_matched <= q_wide
    # Reported: 32 and 69 for matched/wide (this module's continuum sibling,
    # psf_geometry.recommend_npt, independently gives 30/67 -- both within
    # ~10% of the reported values; the lattice-exact search here should
    # land in the same neighbourhood).
    assert q_matched == pytest.approx(32, abs=6)
    assert q_wide == pytest.approx(69, abs=10)
