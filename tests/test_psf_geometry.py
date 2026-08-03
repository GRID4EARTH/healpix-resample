"""
tests/test_psf_geometry.py

Test suite for `healpix_resample.psf_geometry`: the FWHM<->scale conversion
helpers added in response to the paper review (see planning notes / Thomas
Davison's review + `effective_kernel_geometry.ipynb`). Before this module
existed, `sigma_m` (as consumed by `KNeighborsResampler`/`PSFResampler`) was
fed an intended FWHM value directly, with no conversion to the `s` scale the
kernel `w(d) = exp(-2 d^2 / s^2)` actually expects -- these tests pin down
the correct relationship (`s = FWHM / sqrt(2 ln 2)`) and the `Npt`
recommendation derived from it, against numbers independently measured on
the real HEALPix lattice (Thomas Davison's review letter + the companion
`kernel_geometry` diagnostics -- see `test_kernel_geometry.py`).
"""
from __future__ import annotations

import math

import pytest

from healpix_resample.psf_geometry import (
    FWHM_PER_SCALE,
    cell_size_m,
    fwhm_to_scale,
    recommend_npt,
    scale_to_fwhm,
)


def test_fwhm_per_scale_constant():
    # sqrt(2 ln 2), not 2*sqrt(2 ln 2) -- the latter is the pre-fix, buggy
    # Eq. (2) relationship this module exists to not reproduce.
    assert FWHM_PER_SCALE == pytest.approx(math.sqrt(2.0 * math.log(2.0)))
    assert FWHM_PER_SCALE == pytest.approx(1.1774, abs=1e-4)


@pytest.mark.parametrize("fwhm", [1.0, 5.31, 10.0, 12.5, 12.6, 100.0])
def test_fwhm_scale_roundtrip(fwhm):
    s = fwhm_to_scale(fwhm)
    assert scale_to_fwhm(s) == pytest.approx(fwhm, rel=1e-12)


def test_fwhm_to_scale_is_not_the_naive_factor_of_two():
    # The bug this module fixes: passing an intended FWHM straight through
    # as sigma_m is *not* the same as fwhm_to_scale(fwhm) -- they differ by
    # exactly FWHM_PER_SCALE (~1.1774), not by 1.0.
    fwhm = 12.6
    naive = fwhm  # what earlier notebook revisions did
    correct = fwhm_to_scale(fwhm)
    assert correct == pytest.approx(fwhm / FWHM_PER_SCALE)
    assert correct != pytest.approx(naive)


def test_target_fwhm_12p5_gives_scale_near_10p6():
    # The paper's revised target: FWHM = 12.5 m (see review). The scale
    # actually passed to sigma_m should be ~10.6 m, NOT 12.5 m (the naive
    # bug) and NOT 5.31 m (the old, double-converted manuscript value:
    # 5.31 == 12.5 / (2*sqrt(2 ln 2)), i.e. the buggy Eq. (2)).
    s = fwhm_to_scale(12.5)
    assert s == pytest.approx(10.6165, abs=1e-3)
    assert s != pytest.approx(12.5, abs=0.5)
    assert s != pytest.approx(5.31, abs=0.5)


def test_cell_size_level_20():
    # Cross-checked directly against healpix_geo / Thomas's review at level 20.
    assert cell_size_m(20) == pytest.approx(6.2176, abs=1e-3)


@pytest.mark.parametrize(
    "level,expected_ratio",
    [(18, 4.0), (19, 2.0), (21, 0.5)],
)
def test_cell_size_scales_with_level(level, expected_ratio):
    # Equal-area cells: halving nside quadruples cell area, doubles cell width.
    assert cell_size_m(level) == pytest.approx(cell_size_m(20) * expected_ratio, rel=1e-9)


@pytest.mark.parametrize(
    "raw_scale,expected_mass_at_9",
    [
        # These three are the exact "what the code used to run with" scales
        # (FWHM passed straight through, unconverted) from the old buggy
        # notebook -- independently measured on the real lattice in Thomas
        # Davison's review letter (0.751 / 0.996 / 0.461) and re-confirmed by
        # a from-scratch reimplementation in
        # healpix_resample.diagnostics.kernel_geometry (0.7506 / 0.9958 /
        # 0.4614). This continuum approximation should land close to both.
        (12.6, 0.751),
        (6.3, 0.996),
        (18.9, 0.461),
    ],
)
def test_recommend_npt_matches_independently_measured_mass(raw_scale, expected_mass_at_9):
    result = recommend_npt(raw_scale, level=20)
    assert result["mass_at_default"] == pytest.approx(expected_mass_at_9, abs=0.02)


def test_recommend_npt_increases_with_scale():
    npt_narrow = recommend_npt(6.3, level=20)["npt"]
    npt_matched = recommend_npt(12.6, level=20)["npt"]
    npt_wide = recommend_npt(18.9, level=20)["npt"]
    assert npt_narrow <= npt_matched <= npt_wide


def test_recommend_npt_never_below_floor():
    # Even a kernel narrower than one cell shouldn't recommend fewer
    # neighbours than the package's original default.
    result = recommend_npt(0.5, level=20, q_min=9)
    assert result["npt"] >= 9
