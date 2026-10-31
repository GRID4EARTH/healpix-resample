"""`comp_matrix()` builds `M` directly (CSC layout, no COO / coalesce) and
`MT` / `M_cons` only on first access. These tests check that the operators
are identical to the former eager COO -> coalesce -> CSR construction, and
that the laziness is not observable through the public attributes."""
import numpy as np
import pytest
import torch

import healpix_resample as hr

KW = dict(level=11, threshold=0.05, ring_weight=2, verbose=False, nest=True, ellipsoid="WGS84")
CLASSES = {
    "bilinear": (hr.BilinearResampler, {}),
    "bicubic": (hr.BicubicResampler, {}),
    "knn": (hr.KNeighborsResampler, {"Npt": 4}),
}


@pytest.fixture(scope="module")
def grid():
    lon, lat = np.meshgrid(np.linspace(6.0, 6.3, 160), np.linspace(45.0, 45.2, 120))
    return lon.ravel(), lat.ravel()


def _eager(op, transposed=False, per_sample=False):
    """The former construction: raw weights in search order, COO, coalesce, CSR
    (plus the links to tied cells, kept apart from `hi`)."""
    flat_hi = op.hi.reshape(-1)
    valid = flat_hi >= 0
    rows = torch.arange(op.N, dtype=torch.long)[:, None].expand(op.N, op.Npt).reshape(-1)[valid]
    cols = flat_hi[valid]
    w = op._link_weights().reshape(-1)[valid]
    rows = torch.cat([rows, op.tie_rows])
    cols = torch.cat([cols, op.tie_hi])
    w = torch.cat([w, op._tie_weights()])
    norm = op._norm_row(rows, w)[rows] if per_sample else op._norm_col(cols, w)[cols]
    idx = torch.stack([cols, rows] if transposed else [rows, cols], dim=0)
    size = (op.K, op.N) if transposed else (op.N, op.K)
    return torch.sparse_coo_tensor(idx, (w / norm).to(op.dtype), size=size).coalesce().to_sparse_csr()


def _same_csr(a, b):
    """Same sparse matrix, entry by entry (layouts may differ: `M` is CSC)."""
    a, b = a.to_sparse_coo().coalesce(), b.to_sparse_coo().coalesce()
    return (
        tuple(a.shape) == tuple(b.shape)
        and torch.equal(a.indices(), b.indices())
        and torch.equal(a.values(), b.values())
    )


@pytest.mark.parametrize("name", list(CLASSES))
def test_operators_identical_to_eager_construction(grid, name):
    cls, extra = CLASSES[name]
    op = cls(lon_deg=grid[0], lat_deg=grid[1], **extra, **KW)
    assert op.M.layout == torch.sparse_csc
    assert _same_csr(op.M, _eager(op))
    assert _same_csr(op.MT, _eager(op, transposed=True, per_sample=True))
    assert _same_csr(op.M_cons, _eager(op, per_sample=True))


@pytest.mark.parametrize("name", list(CLASSES))
def test_MT_and_M_cons_are_built_on_first_access(grid, name):
    cls, extra = CLASSES[name]
    op = cls(lon_deg=grid[0], lat_deg=grid[1], **extra, **KW)
    assert op._MT is None and op._M_cons is None
    val = np.sin(grid[0] * 40).astype(np.float32)
    op.resample(val)
    assert op._MT is None and op._M_cons is None, "interpolation must not build MT / M_cons"
    hval = op.resample(val).cell_data
    op.invert(hval)
    assert op._MT is not None and tuple(op._MT.shape) == (op.K, op.N)
    assert op.MT is op._MT  # cached
    if hasattr(op, "area"):
        op.resample(val, conservative=True)
        assert op._M_cons is not None and tuple(op._M_cons.shape) == (op.N, op.K)
        assert op.M_cons is op._M_cons


def test_setters_keep_subclass_assignment_working(grid):
    op = hr.BilinearResampler(lon_deg=grid[0], lat_deg=grid[1], **KW)
    other = torch.eye(3).to_sparse_csr()
    op.MT = other
    op.M_cons = other
    assert op.MT is other and op.M_cons is other


def test_M_columns_sum_to_one_and_cons_rows_sum_to_one(grid):
    op = hr.BilinearResampler(lon_deg=grid[0], lat_deg=grid[1], **KW)
    ones = torch.ones(1, op.N, dtype=op.dtype)
    torch.testing.assert_close(ones @ op.M, torch.ones(1, op.K, dtype=op.dtype))
    row_sums = torch.ones(1, op.K, dtype=op.dtype) @ op.M_cons.to_dense().T
    served = (op.hi >= 0).any(dim=1)
    torch.testing.assert_close(row_sums[0][served], torch.ones(int(served.sum()), dtype=op.dtype))


def test_duplicate_link_falls_back_to_coalesced_sum(grid):
    """A sample linking the same cell twice is not produced by the search,
    but the direct CSR build must still match coalesce (weights added)."""
    op = hr.BilinearResampler(lon_deg=grid[0], lat_deg=grid[1], **KW)
    hi = op.hi.clone()
    hi[0, 1] = hi[0, 0]  # duplicate the first link of sample 0
    op.hi = hi
    op.comp_matrix()
    assert _same_csr(op.M, _eager(op))
    assert _same_csr(op.MT, _eager(op, transposed=True, per_sample=True))
    assert _same_csr(op.M_cons, _eager(op, per_sample=True))
    # sample 0 has one merged entry
    assert int((op.M.row_indices() == 0).sum()) == op.Npt - 1 + int((op.tie_rows == 0).sum())


def test_partial_rows_take_the_masked_path(grid):
    """Samples with fewer than Npt links (hi == -1) use the masked (counts)
    path of `comp_matrix()`; the result must still equal the eager build."""
    op = hr.BilinearResampler(lon_deg=grid[0], lat_deg=grid[1], **KW)
    hi = op.hi.clone()
    hi[::7, 3] = -1
    hi[5, 1:] = -1
    op.hi = hi
    op.comp_matrix()
    assert _same_csr(op.M, _eager(op))
    assert int(op.M.row_indices().numel()) == int((hi >= 0).sum()) + op.tie_rows.numel()
    assert _same_csr(op.M_cons, _eager(op, per_sample=True))
    val = np.sin(grid[0] * 40)
    out = op.resample(val).cell_data
    ref = (torch.as_tensor(val, dtype=op.dtype)[None] @ _eager(op))[0].numpy()
    np.testing.assert_array_equal(out, ref)
