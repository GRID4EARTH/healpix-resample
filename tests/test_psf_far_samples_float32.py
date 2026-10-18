"""PSFResampler with out_cell_ids and samples far from every requested cell (#77).

A caller that resamples chunk by chunk passes the chunk's cells as
``out_cell_ids`` together with the samples found around the chunk, margin
included. The samples of that margin can be hundreds of metres from the
nearest *requested* cell: their Gaussian weights ``exp(-2 d^2 / sigma^2)``
all underflow to 0 in float32, so the per-sample normalisation of ``MT``
used to compute ``0 / 0`` for them. The resulting NaN entries poisoned every
Conjugate Gradient iteration and the whole output came back NaN, while the
same call in float64 (where the weights are tiny but non-zero) was fine.
"""

import healpix_geo.nested as hn
import numpy as np
import pytest
import torch

import healpix_resample as hr

LEVEL, CHUNK_LEVEL = 18, 11


@pytest.fixture(scope="module")
def chunk_with_wide_margin():
    """One level-11 chunk's cells and a 40 m sample grid 1 km beyond it."""
    n = 4 ** (LEVEL - CHUNK_LEVEL)
    chunk = int(
        hn.lonlat_to_healpix(
            np.array([2.35]), np.array([48.85]), CHUNK_LEVEL, ellipsoid="WGS84"
        )[0]
    )
    out_cells = np.arange(chunk * n, (chunk + 1) * n, dtype=np.uint64)
    vlon, vlat = hn.vertices(
        np.array([chunk], dtype=np.uint64), CHUNK_LEVEL, ellipsoid="WGS84"
    )
    vlon, vlat = np.asarray(vlon[0]), np.asarray(vlat[0])
    margin_lon, margin_lat = 1000.0 / 73000.0, 1000.0 / 111000.0  # ~1 km at 48.85N
    step_lon, step_lat = 40.0 / 73000.0, 40.0 / 111000.0  # ~40 m
    lon = np.arange(vlon.min() - margin_lon, vlon.max() + margin_lon, step_lon)
    lat = np.arange(vlat.min() - margin_lat, vlat.max() + margin_lat, step_lat)
    lon, lat = (a.reshape(-1) for a in np.meshgrid(lon, lat))
    return lon, lat, out_cells


def _resample(lon, lat, out_cells, values, dtype):
    op = hr.PSFResampler(
        lon_deg=lon,
        lat_deg=lat,
        level=LEVEL,
        out_cell_ids=out_cells,
        nest=True,
        ellipsoid="WGS84",
        verbose=False,
        Npt=9,
        threshold=0.1,
        dtype=dtype,
    )
    assert op.uncomputable_out_cells is None  # every requested cell is covered
    res = op.resample(values, lam=0.1, max_iter=100, tol=1e-8)
    return np.asarray(res.cell_ids), np.asarray(res.cell_data, dtype=np.float64)


def test_float32_matches_float64_with_far_samples(chunk_with_wide_margin):
    lon, lat, out_cells = chunk_with_wide_margin
    rng = np.random.default_rng(0)
    values = (0.1 + 0.05 * np.sin(lon * 3000.0) * np.cos(lat * 3000.0)).astype(np.float32)
    values += rng.normal(scale=0.002, size=values.shape).astype(np.float32)

    ids64, data64 = _resample(lon, lat, out_cells, values, torch.float64)
    ids32, data32 = _resample(lon, lat, out_cells, values, torch.float32)

    np.testing.assert_array_equal(ids32, ids64)
    np.testing.assert_array_equal(ids32, out_cells.astype(np.int64))
    assert np.isfinite(data64).all()
    assert np.isfinite(data32).all()
    np.testing.assert_allclose(data32, data64, atol=2e-5)


def test_far_samples_get_zero_weight_not_nan(chunk_with_wide_margin):
    lon, lat, out_cells = chunk_with_wide_margin
    op = hr.PSFResampler(
        lon_deg=lon,
        lat_deg=lat,
        level=LEVEL,
        out_cell_ids=out_cells,
        nest=True,
        ellipsoid="WGS84",
        verbose=False,
        Npt=9,
        threshold=0.1,
        dtype=torch.float32,
    )
    for name in ("M", "MT"):
        values = getattr(op, name).to_sparse_coo().values()
        assert torch.isfinite(values).all(), f"{name} has non-finite weights"
    # the premise of the test: some margin samples do link to requested cells
    # (the KNN found them) but all their float32 weights underflowed to 0
    linked = torch.bincount(op.MT.to_sparse_coo().indices()[1], minlength=op.N) > 0
    row_sum = torch.zeros(op.N, dtype=torch.float32).index_add_(
        0, op.MT.to_sparse_coo().indices()[1], op.MT.to_sparse_coo().values()
    )
    assert bool((linked & (row_sum == 0)).any())
