"""BicubicResampler: kernel scale and cancellation of its signed weights.

Keys' kernel is defined in units of the input sample spacing. Scaled by the
output cell size only, it put the nearest samples in its negative lobe when
the samples are coarser than the cells: the signed weights of a cell then
cancelled, and the per-cell normalisation amplified them by up to ~1e5,
which gave values tens of times outside the input range.
"""

import healpix_geo.nested as hn
import numpy as np
import pytest
import torch

import healpix_resample as hr
import healpix_resample.bicubic as bc

LEVEL = 18  # cells of ~25 m
LON0, LAT0 = 2.35, 48.85
KW = {"level": LEVEL, "nest": True, "ellipsoid": "WGS84", "verbose": False}


def _grid(step_m, half_m=1500.0):
    lon = np.arange(LON0 - half_m / 73000, LON0 + half_m / 73000, step_m / 73000)
    lat = np.arange(LAT0 - half_m / 111000, LAT0 + half_m / 111000, step_m / 111000)
    lon, lat = np.meshgrid(lon, lat)
    return lon.ravel(), lat.ravel()


def _field(lon, lat, wavelength_m=3000.0):
    k = 2 * np.pi / wavelength_m
    return 0.1 + 0.05 * np.sin((lon - LON0) * 73000 * k) * np.cos((lat - LAT0) * 111000 * k)


def _interior_error(op, lon, lat, wavelength_m=3000.0):
    res = op.resample(_field(lon, lat, wavelength_m))
    ids, data = np.asarray(res.cell_ids), np.asarray(res.cell_data, dtype=np.float64)
    clon, clat = hn.healpix_to_lonlat(ids.astype(np.uint64), LEVEL, ellipsoid="WGS84")
    clon, clat = np.asarray(clon), np.asarray(clat)
    inside = (np.abs(clon - LON0) * 73000 < 1300) & (np.abs(clat - LAT0) * 111000 < 1300)
    return data, data[inside] - _field(clon[inside], clat[inside], wavelength_m)


@pytest.fixture(scope="module")
def coarse_samples():
    """Samples every 40 m on ~25 m cells: the case that went wrong."""
    return _grid(40.0)


def test_upsampling_stays_in_the_input_range(coarse_samples):
    lon, lat = coarse_samples
    data, _ = _interior_error(hr.BicubicResampler(lon_deg=lon, lat_deg=lat, **KW), lon, lat)
    assert data.min() >= 0.05 - 1e-3
    assert data.max() <= 0.15 + 1e-3


def test_upsampling_is_more_accurate_than_bilinear(coarse_samples):
    lon, lat = coarse_samples
    _, err_cubic = _interior_error(hr.BicubicResampler(lon_deg=lon, lat_deg=lat, **KW), lon, lat)
    _, err_linear = _interior_error(hr.BilinearResampler(lon_deg=lon, lat_deg=lat, **KW), lon, lat)
    assert np.sqrt(np.mean(err_cubic**2)) < np.sqrt(np.mean(err_linear**2))


@pytest.mark.parametrize("step_m", [40.0, 10.0])
def test_constant_field_is_reproduced(step_m):
    lon, lat = _grid(step_m, half_m=800.0)
    op = hr.BicubicResampler(lon_deg=lon, lat_deg=lat, **KW)
    data = np.asarray(op.resample(np.full(lon.size, 0.1)).cell_data)
    np.testing.assert_allclose(data, 0.1, rtol=1e-9)


def test_no_cell_amplifies_its_weights(coarse_samples):
    lon, lat = coarse_samples
    op = hr.BicubicResampler(lon_deg=lon, lat_deg=lat, **KW)
    m = op.M.to_sparse_coo().coalesce()
    cols, vals = m.indices()[1], m.values()
    unsigned = torch.zeros(op.K, dtype=vals.dtype).index_add_(0, cols, vals.abs())
    assert float(unsigned.max()) <= bc._MAX_CANCELLATION * (1 + 1e-9)


def test_float32_matches_float64(coarse_samples):
    lon, lat = coarse_samples
    vals = _field(lon, lat).astype(np.float32)
    d64 = np.asarray(hr.BicubicResampler(lon_deg=lon, lat_deg=lat, dtype=torch.float64, **KW).resample(vals).cell_data)
    d32 = np.asarray(hr.BicubicResampler(lon_deg=lon, lat_deg=lat, dtype=torch.float32, **KW).resample(vals).cell_data)
    np.testing.assert_allclose(d32, d64, atol=1e-5)


# ── sample spacing ───────────────────────────────────────────────────────────


def test_nearest_sample_distance_on_a_regular_grid(coarse_samples):
    lon, lat = coarse_samples
    d = bc._nearest_sample_distance_m(lon, lat, radius=6371000.0)
    # the grid is 40 m east-west and north-south (approximately, on the ellipsoid)
    assert np.median(d) == pytest.approx(40.0, rel=0.05)
    np.testing.assert_array_equal(
        d, bc._nearest_sample_distance_m(lon, lat, radius=6371000.0, batch_size=1000)
    )


def test_nearest_sample_distance_skips_duplicates():
    lon = np.array([2.0, 2.0, 2.001])
    lat = np.array([48.0, 48.0, 48.0])
    d = bc._nearest_sample_distance_m(lon, lat, radius=6371000.0)
    assert (d > 0).all()
    assert bc._nearest_sample_distance_m(lon[:1], lat[:1], radius=6371000.0).tolist() == [0.0]


def test_explicit_sample_spacing_matches_the_estimate(coarse_samples):
    lon, lat = coarse_samples
    auto = np.asarray(hr.BicubicResampler(lon_deg=lon, lat_deg=lat, **KW).resample(_field(lon, lat)).cell_data)
    given = np.asarray(
        hr.BicubicResampler(lon_deg=lon, lat_deg=lat, sample_spacing_m=40.0, **KW).resample(_field(lon, lat)).cell_data
    )
    np.testing.assert_allclose(given, auto, atol=2e-3)


@pytest.mark.parametrize("bad", [-1.0, float("nan")])
def test_sample_spacing_must_be_non_negative(coarse_samples, bad):
    lon, lat = coarse_samples
    with pytest.raises(ValueError, match="sample_spacing_m"):
        hr.BicubicResampler(lon_deg=lon, lat_deg=lat, sample_spacing_m=bad, **KW)


# ── MT of the base class in float32 (same 0/0 as #77) ────────────────────────


def test_nearest_mt_has_no_nan_in_float32():
    # samples ~1 km around one level-11 chunk: the margin ones are far from
    # every requested cell and their float32 weights underflow to 0
    chunk = int(hn.lonlat_to_healpix(np.array([LON0]), np.array([LAT0]), 11, ellipsoid="WGS84")[0])
    n = 4 ** (LEVEL - 11)
    out_cells = np.arange(chunk * n, (chunk + 1) * n, dtype=np.uint64)
    vlon, vlat = hn.vertices(np.array([chunk], dtype=np.uint64), 11, ellipsoid="WGS84")
    vlon, vlat = np.asarray(vlon[0]), np.asarray(vlat[0])
    lon = np.arange(vlon.min() - 0.0137, vlon.max() + 0.0137, 40.0 / 73000)
    lat = np.arange(vlat.min() - 0.009, vlat.max() + 0.009, 40.0 / 111000)
    lon, lat = (a.ravel() for a in np.meshgrid(lon, lat))
    op = hr.NearestResampler(
        lon_deg=lon, lat_deg=lat, out_cell_ids=out_cells, dtype=torch.float32, **KW
    )
    assert torch.isfinite(op.MT.to_sparse_coo().coalesce().values()).all()
