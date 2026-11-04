"""
tests/test_nearest_fill_kdtree.py

`NearestResampler` with ``out_cell_ids`` fills the requested cells beyond the
KNN rings from their nearest sample. That sample comes from a KD-tree
(`_nearest_source`): same sample and distance as comparing every cell with
every sample, which took hours on a chunk at the edge of a satellite swath
(millions of empty cells and of samples).

CPU-only.
"""
from __future__ import annotations

import healpix_geo
import numpy as np
import pytest
import torch

from healpix_resample import NearestResampler
from healpix_resample.knn import _lonlat_to_xyz


def _xyz(lon, lat, dtype=torch.float64):
    return _lonlat_to_xyz(torch.deg2rad(torch.as_tensor(lon, dtype=dtype)),
                          torch.deg2rad(torch.as_tensor(lat, dtype=dtype)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nearest_source_is_the_largest_dot(dtype):
    rng = np.random.default_rng(1)
    src = _xyz(rng.uniform(0, 1, 2000), rng.uniform(40, 41, 2000), dtype)
    query = _xyz(rng.uniform(-0.5, 1.5, 500), rng.uniform(39.5, 41.5, 500), dtype)
    hi, dot = NearestResampler._nearest_source(query, src)
    # the nearest sample (or one at the same distance) and its dot product
    chord = torch.cdist(query.double(), src.double())
    best = chord.min(dim=1).values
    assert torch.all(chord[torch.arange(len(hi)), hi] <= best + 1e-12)
    dots = query.double() @ src.double().T
    np.testing.assert_allclose(dot.double().numpy(), dots.max(dim=1).values.numpy(),
                               rtol=0, atol=1e-6 if dtype == torch.float32 else 1e-12)
    assert hi.dtype == torch.long and dot.dtype == dtype


def test_partial_chunk_fills_the_far_cells_from_the_nearest_sample():
    """Samples on the west corner of a level-8 cell, all its level-14 cells requested."""
    level = 14
    chunk = healpix_geo.nested.lonlat_to_healpix(np.array([0.5]), np.array([45.0]), 8)
    out = healpix_geo.nested.zoom_to(chunk.astype(np.uint64), 8, level)[0]
    clon, clat = healpix_geo.nested.healpix_to_lonlat(out, level)
    clon, clat = np.asarray(clon), np.asarray(clat)
    corner = clon < np.quantile(clon, 0.1)      # the west corner of the cell
    rng = np.random.default_rng(0)
    lon = clon[corner] + rng.uniform(-1e-3, 1e-3, corner.sum())
    lat = clat[corner] + rng.uniform(-1e-3, 1e-3, corner.sum())
    val = np.arange(lon.size, dtype=np.float64)

    op = NearestResampler(lon, lat, level=level, out_cell_ids=out, ellipsoid="sphere")
    res = op.resample(val)
    np.testing.assert_array_equal(np.asarray(res.cell_ids), np.sort(out.astype(np.int64)))
    # the far cells (no sample within the KNN rings) get their nearest sample
    xyz_c = _xyz(*healpix_geo.nested.healpix_to_lonlat(np.asarray(res.cell_ids).astype(np.uint64), level))
    nearest = (xyz_c @ _xyz(lon, lat).T).argmax(dim=1).numpy()
    far = ~np.isin(np.asarray(res.cell_ids), out[corner].astype(np.int64))
    assert far.sum() > out.size // 2
    np.testing.assert_array_equal(np.asarray(res.cell_data)[far], val[nearest[far]])
