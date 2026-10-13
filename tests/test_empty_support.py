"""Requested output cells without any support among the samples.

A caller that resamples chunk by chunk passes the chunk's cells as
``out_cell_ids`` and the samples found around the chunk. At the edge of the
input those samples can all lie just outside the chunk, so that no requested
cell passes the KNN threshold. That is a legitimate outcome (nothing to fill
there), not a configuration error, and must not stop the caller.
"""

import healpix_geo.nested as hn
import numpy as np
import pytest

import healpix_resample as hr

LEVEL, CHUNK_LEVEL = 18, 11


@pytest.fixture(scope="module")
def outside_samples():
    """The cells of one level-11 chunk and three samples about 1 km east of it."""
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
    east = int(np.argmax(vlon))
    lon = vlon[east] + 0.015 + np.array([0.0, 0.0002, 0.0004])
    lat = vlat[east] + np.array([0.0, 0.0002, -0.0002])
    inside = hn.lonlat_to_healpix(lon, lat, CHUNK_LEVEL, ellipsoid="WGS84")
    assert not np.isin(inside, [chunk]).any()
    return lon, lat, out_cells


@pytest.mark.parametrize(
    ["cls", "kwargs"],
    [
        pytest.param(hr.BilinearResampler, {"threshold": 0.05}, id="bilinear"),
        pytest.param(hr.PSFResampler, {"Npt": 9, "threshold": 0.1}, id="psf"),
        pytest.param(hr.BicubicResampler, {}, id="bicubic"),
    ],
)
def test_interpolating_resamplers_return_no_cell(outside_samples, cls, kwargs):
    lon, lat, out_cells = outside_samples
    op = cls(
        lon_deg=lon,
        lat_deg=lat,
        level=LEVEL,
        out_cell_ids=out_cells,
        nest=True,
        ellipsoid="WGS84",
        verbose=False,
        **kwargs,
    )
    res = op.resample(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    cell_ids, cell_data = np.asarray(res.cell_ids), np.asarray(res.cell_data)
    assert np.isin(cell_ids, out_cells.astype(np.int64)).all()
    assert cell_data.shape == (2, cell_ids.size)
    assert not np.isfinite(cell_data).any()


def test_nearest_fills_the_requested_cells(outside_samples):
    # NearestResampler fills requested cells the KNN rings missed with their
    # nearest sample (its documented out_cell_ids fallback); with no cell found
    # by the KNN at all, that fallback now covers every requested cell.
    lon, lat, out_cells = outside_samples
    op = hr.NearestResampler(
        lon_deg=lon,
        lat_deg=lat,
        level=LEVEL,
        out_cell_ids=out_cells,
        nest=True,
        ellipsoid="WGS84",
        verbose=False,
    )
    res = op.resample(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_array_equal(
        np.sort(np.asarray(res.cell_ids)), out_cells.astype(np.int64)
    )
    assert np.isin(np.asarray(res.cell_data), [1.0, 2.0, 3.0]).all()


def test_without_out_cell_ids_an_impossible_threshold_still_raises(outside_samples):
    lon, lat, _ = outside_samples
    with pytest.raises(RuntimeError, match="No HEALPix cell passed the threshold"):
        hr.PSFResampler(
            lon_deg=lon,
            lat_deg=lat,
            level=LEVEL,
            nest=True,
            ellipsoid="WGS84",
            verbose=False,
            threshold=1e9,
        )
