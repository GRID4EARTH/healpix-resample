"""
clough_tocher.py

Delaunay / Clough-Tocher C1 cubic HEALPix resampler, implemented in torch.

See `planning/05_clough_tocher_resampler.md` for the full design discussion
this module implements. One-paragraph summary of *why* this exists: on
fields with real curvature, `scipy.interpolate.griddata(method='cubic')`
(Delaunay triangulation + a Clough-Tocher C1 macro-element per triangle)
shows visibly fewer small-scale artifacts than `BicubicResampler`
(`bicubic.py`) -- a radial generalization of Keys' kernel over a
*discrete* KNN neighbour set, whose neighbour-set membership can flip
between adjacent output cells even though the weight formula itself is
continuous for a fixed neighbour set. A genuine Delaunay/CT construction
has no such failure mode: adjacent triangles share two vertices, and the
Clough-Tocher macro-element is built, by construction, to be C1 (value
*and* gradient) continuous across shared edges.

Design decisions (do not re-litigate here -- see the planning doc):

- **Full custom torch implementation, not a scipy wrapper.** Gradient
  estimation (`_build_gradient_operators`), the per-triangle Bezier
  control-net assembly (`_triangle_ct_coefficients`), and the final sparse
  `(N, K)` operator assembly (`_assemble_M`) are all torch tensor
  operations, capable of running on GPU (`device="cuda"`) exactly like
  every other resampler's `comp_matrix()`. `self.M` is built once in
  `__init__`, so `resample(val)` is a single batched sparse matmul, not a
  per-call computation. The **only** unavoidable CPU/NumPy steps are the
  Delaunay triangulation *topology* itself (`scipy.spatial.Delaunay`,
  Qhull-backed -- no mature GPU/torch-native Delaunay library exists) and
  locating which triangle each candidate HEALPix cell center falls in
  (`Delaunay.find_simplex`), plus the `healpix_geo` calls used to enumerate
  candidate cells -- exactly analogous to how this package already treats
  `healpix_geo` elsewhere (see `knn.py`'s
  ``lon_np = longitude1.detach().cpu().numpy()`` pattern). Everything
  downstream of "which triangle, which micro-triangle" -- i.e. everything
  whose cost scales with `N` (samples), the number of Delaunay edges, or
  `K` (output cells) -- runs as torch tensor ops on `self.device`.
- **Gradient estimation: local least-squares plane fit** at each
  triangulation vertex, using that vertex's direct Delaunay 1-ring
  neighbours. This is precomputed as two sparse ``(N, N)`` matrices
  ``self.Gx``, ``self.Gy`` (geometry only, independent of `val`), so that
  for any sample-space field `f`: ``grad_x = Gx @ f``, ``grad_y = Gy @ f``.
  This is **not** claimed to be bit-identical to
  ``scipy.interpolate.CloughTocher2DInterpolator`` (which uses a different,
  Nielson-style discrete-curvature gradient estimator) -- it is *a*
  correct, standard Clough-Tocher construction, in the same spirit as
  `bicubic.py`'s docstring calling out `"bicubic"` in quotes for the same
  kind of honesty about a from-first-principles reimplementation.
- **Local gnomonic (central) projection** about the centroid of the sample
  set, because gnomonic projection maps great-circle geodesics to straight
  lines -- so Delaunay triangle edges in the projected plane correspond to
  true geodesics on the sphere. This distorts badly for large angular
  extents (much beyond a single regional patch), so this resampler is
  intended for **regional/local** input extents, not global datasets in one
  construction call -- ``__init__`` raises if any sample lands outside the
  gnomonic projection's well-behaved hemisphere around the centroid.
  `planning/04_parent_cell_subsetting.md`'s `subset_for_parent_cell` is the
  natural way to apply this resampler to a large dataset one local patch at
  a time.
- **Output validity = inside the convex hull of the triangulation.** Like
  `scipy.interpolate.griddata`, Clough-Tocher does not extrapolate: a
  candidate HEALPix cell is only kept in ``self.cell_ids`` if its projected
  center falls inside the Delaunay triangulation
  (``Delaunay.find_simplex(...) != -1``).
- **``invert()`` is not implemented.** Unlike the KNN-based resamplers
  (symmetric `M`/`MT` pair), Delaunay/CT has no equally natural closed form
  from HEALPix cells back to scattered sample locations. See
  ``invert()``'s docstring.

NOTE on implementation risk (transparency, not boilerplate)
-------------------------------------------------------------
This module was written in a session where the sandboxed shell used to
develop and test this package was unavailable (infra failure), so **none**
of the validation the planning doc calls for (affine-exactness check,
edge-continuity check, comparison against
``scipy.interpolate.CloughTocher2DInterpolator``, `pytest`) has actually
been run against this code. The Clough-Tocher macro-element formulas below
(the single highest-risk block -- see `_triangle_ct_coefficients`) were not
reconstructed from memory: they were looked up from a peer-reviewed,
citable source (Kosinka, J. and Cashman, T.J., "Watertight conversion of
trimmed CAD surfaces to Clough-Tocher splines", Computer Aided Geometric
Design 37 (2015) 25-41, Section 3.1 "CTo: the Clough-Tocher construction",
explicitly attributed there to Clough and Tocher (1965) and to the C1
conditions in Farin (1985, 1986)) and transcribed with each step checked by
hand for internal consistency (in particular, an affine-exactness argument:
every quantity in the construction is an affine/convex combination of
previously affine-exact quantities, so the whole pipeline reproduces an
affine field exactly by induction -- this is a structural argument, not a
substitute for actually running the numeric check in the test suite).
**Before trusting this module, run `tests/test_clough_tocher.py` and, in
particular, `test_affine_field_is_exact` and
`test_edge_value_continuity_from_both_sides` -- these are the two checks
`planning/05_clough_tocher_resampler.md` says must pass before anything
built on top of this is trustworthy.** The gradient-estimation and
Bezier-assembly code was subsequently ported from NumPy to torch tensor
ops (`_build_gradient_operators`, `_triangle_ct_coefficients`,
`_barycentric_2d`, `_assemble_M`) so construction can run on GPU -- this is
a mechanical, op-by-op port of the same already-reasoned-through formulas
(no new math), but it is itself equally unexecuted; the NumPy/torch op
correspondence (`torch.bincount` for `np.bincount`, `torch.repeat_interleave`
for `np.repeat`, the `cumsum`-based ragged-gather index trick, etc.) should
be spot-checked against the test suite too.

This module is designed for large N and batched values (B,N), matching the
rest of the package, though the Delaunay/assembly step itself is CPU-only.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from scipy.spatial import Delaunay

try:
    from scipy.spatial import QhullError
except ImportError:  # pragma: no cover - older scipy versions
    from scipy.spatial.qhull import QhullError

import healpix_geo

from healpix_resample.base import ResampleResults, T_Array


# ─────────────────────────────────────────────────────────────────────────
# Gnomonic projection
# ─────────────────────────────────────────────────────────────────────────

def _lonlat_to_xyz_np(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon_rad = np.radians(lon_deg)
    lat_rad = np.radians(lat_deg)
    clat = np.cos(lat_rad)
    return np.stack(
        [clat * np.cos(lon_rad), clat * np.sin(lon_rad), np.sin(lat_rad)], axis=-1
    )


def _gnomonic_tangent_frame(lon_deg: np.ndarray, lat_deg: np.ndarray):
    """Tangent point (mean unit vector, renormalized) + local (east, north) basis."""
    xyz = _lonlat_to_xyz_np(lon_deg, lat_deg)
    t = xyz.mean(axis=0)
    norm = np.linalg.norm(t)
    if norm < 1e-8:
        raise ValueError(
            "CloughTocherResampler: the sample centroid is (near) degenerate "
            "(e.g. antipodally spread samples) -- no well-defined tangent "
            "point for a local gnomonic projection. This resampler is only "
            "intended for regional/local input extents; see the module "
            "docstring and planning/04_parent_cell_subsetting.md."
        )
    t = t / norm

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(t[2]) > 0.999999:
        world_up = np.array([1.0, 0.0, 0.0])
    east = np.cross(world_up, t)
    east = east / np.linalg.norm(east)
    north = np.cross(t, east)
    return t, east, north


def _gnomonic_project(lon_deg, lat_deg, t, east, north, radius: float):
    """Project (lon_deg, lat_deg) about tangent point `t` (east/north basis).

    Returns (x, y, denom) -- `denom = dot(unit_vector, t)`, the cosine of the
    angular distance from the tangent point. `denom <= 0` means "on or past
    the far hemisphere" (projection undefined/unstable); callers decide
    whether to raise (input samples) or mask out (candidate output cells,
    which are simply never going to land inside the local hull anyway).
    """
    xyz = _lonlat_to_xyz_np(lon_deg, lat_deg)
    denom = xyz @ t
    x = radius * (xyz @ east) / denom
    y = radius * (xyz @ north) / denom
    return x, y, denom


# ─────────────────────────────────────────────────────────────────────────
# Gradient estimation: local least-squares plane fit -> sparse Gx, Gy
# ─────────────────────────────────────────────────────────────────────────

def _build_gradient_operators(
    points2d: torch.Tensor,
    tri: Delaunay,
    device: torch.device,
    dtype: torch.dtype,
    det_floor_rel: float = 1e-9,
):
    """Build sparse (N,N) Gx, Gy s.t. grad_x = Gx @ f, grad_y = Gy @ f.

    Torch tensor ops throughout (GPU-capable) -- the only NumPy/CPU input is
    `tri.vertex_neighbor_vertices` (scipy's Delaunay adjacency, cheap
    bookkeeping), converted to torch immediately. `points2d` must already be
    a torch tensor on `device`.

    For vertex i with 1-ring Delaunay neighbours j, fit a local plane
    (gx_i, gy_i) by least squares on value differences:

        minimize sum_j (f_j - f_i - gx*dx_ij - gy*dy_ij)^2

    Solving the 2x2 normal equations gives gx_i, gy_i as a fixed linear
    combination of {f_j - f_i}, hence of f itself -- see the module
    docstring and `planning/05_clough_tocher_resampler.md` step 3.
    Correctness invariant used by the tests: since a constant field (all
    f_j = f_i) must give zero gradient, each row of Gx/Gy sums to exactly
    zero by construction (the diagonal is set to minus the row's
    off-diagonal sum, not fit independently).

    Returns
    -------
    Gx, Gy : sparse (N,N) torch tensors.
    edges_j, coef_gx_edge, coef_gy_edge : (E,) torch tensors -- per-edge
        neighbour index / gradient-estimator coefficients, reused directly
        by `_assemble_M`'s ragged gather (avoids re-deriving them from Gx/Gy
        by sparse-row extraction).
    diag_gx, diag_gy : (N,) torch tensors -- per-vertex diagonal terms.
    indptr : (N+1,) torch long tensor -- `tri`'s own CSR neighbour-list
        offsets, reused by `_assemble_M`.
    """
    N = points2d.shape[0]
    tiny = torch.finfo(dtype).tiny

    indptr_np, indices_np = tri.vertex_neighbor_vertices
    indptr = torch.as_tensor(indptr_np.astype(np.int64), device=device)
    edges_j = torch.as_tensor(indices_np.astype(np.int64), device=device)
    degree = indptr[1:] - indptr[:-1]

    edges_i = torch.repeat_interleave(torch.arange(N, device=device, dtype=torch.long), degree)

    dx = points2d[edges_j, 0] - points2d[edges_i, 0]
    dy = points2d[edges_j, 1] - points2d[edges_i, 1]

    Sxx = torch.bincount(edges_i, weights=dx * dx, minlength=N).to(dtype)
    Sxy = torch.bincount(edges_i, weights=dx * dy, minlength=N).to(dtype)
    Syy = torch.bincount(edges_i, weights=dy * dy, minlength=N).to(dtype)
    det = Sxx * Syy - Sxy * Sxy

    # Guard near-singular 1-rings (e.g. a vertex with < 2 non-collinear
    # neighbours) -- floor |det| away from zero relative to the 1-ring's own
    # squared length scale, rather than dividing by ~0. This only perturbs
    # pathological vertices; well-conditioned ones (the vast majority of any
    # real Delaunay triangulation) are unaffected. `tiny` (not a hardcoded
    # 1e-300) so this stays sane under float32 too, not just the package's
    # float64 default.
    scale2 = (Sxx + Syy) ** 2
    floor = det_floor_rel * torch.clamp(scale2, min=tiny)
    det_sign = torch.where(det >= 0, torch.ones_like(det), -torch.ones_like(det))
    det_safe = torch.where(det.abs() < floor, det_sign * torch.clamp(floor, min=tiny), det)

    coef_gx_edge = (Syy[edges_i] * dx - Sxy[edges_i] * dy) / det_safe[edges_i]
    coef_gy_edge = (-Sxy[edges_i] * dx + Sxx[edges_i] * dy) / det_safe[edges_i]

    diag_gx = -torch.bincount(edges_i, weights=coef_gx_edge, minlength=N).to(dtype)
    diag_gy = -torch.bincount(edges_i, weights=coef_gy_edge, minlength=N).to(dtype)

    idx_all = torch.arange(N, device=device, dtype=torch.long)
    rows_all = torch.cat([edges_i, idx_all])
    cols_all = torch.cat([edges_j, idx_all])
    gx_vals_all = torch.cat([coef_gx_edge, diag_gx])
    gy_vals_all = torch.cat([coef_gy_edge, diag_gy])

    indices_t = torch.stack([rows_all, cols_all], dim=0)
    Gx = torch.sparse_coo_tensor(indices_t, gx_vals_all, size=(N, N), device=device, dtype=dtype).coalesce()
    Gy = torch.sparse_coo_tensor(indices_t, gy_vals_all, size=(N, N), device=device, dtype=dtype).coalesce()

    return Gx, Gy, edges_j, coef_gx_edge, coef_gy_edge, diag_gx, diag_gy, indptr


# ─────────────────────────────────────────────────────────────────────────
# Candidate output cells: input-sample footprint, expanded by a few rings
# ─────────────────────────────────────────────────────────────────────────

def _candidate_output_cells(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    level: int,
    nest: bool,
    ellipsoid: str,
    ring_expand: int,
    num_threads: int,
) -> np.ndarray:
    """HEALPix cells covering the sample footprint, expanded by a few rings.

    Reuses the same `kth_neighbourhood`-based ring-expansion pattern already
    used by `subsetting.py` for margin filtering, rather than inventing new
    bounding-box/grid-enumeration geometry: the exact candidate set doesn't
    matter for correctness (candidates outside the convex hull are dropped
    by `find_simplex` regardless -- see step 4 in the planning doc), only
    that it's generous enough to cover the full hull, which a few rings
    around the sample footprint comfortably is for any reasonably-dense
    point cloud.
    """
    hp = healpix_geo.nested if nest else healpix_geo.ring
    sample_cells = np.unique(
        np.asarray(
            hp.lonlat_to_healpix(lon_deg, lat_deg, level, num_threads=num_threads, ellipsoid=ellipsoid)
        ).astype(np.int64)
    )
    if ring_expand > 0:
        neigh = np.asarray(
            hp.kth_neighbourhood(sample_cells.astype(np.uint64), level, ring_expand, num_threads=num_threads)
        ).reshape(-1).astype(np.int64)
        neigh = neigh[neigh >= 0]
        cand = np.unique(np.concatenate([sample_cells, neigh]))
    else:
        cand = sample_cells
    return cand


# ─────────────────────────────────────────────────────────────────────────
# Barycentric coordinates (2D)
# ─────────────────────────────────────────────────────────────────────────

def _barycentric_2d(px: torch.Tensor, py: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Barycentric coords (u, v, w) of (px, py) w.r.t. triangle (A, B, C).

    Torch tensors throughout (GPU-capable). P = u*A + v*B + w*C,
    u + v + w = 1. Standard 2x2-solve formula (e.g. Ericson, "Real-Time
    Collision Detection"); well-defined (up to the `denom` floor below) for
    any non-degenerate triangle, including for points outside it (u/v/w can
    go negative -- the Clough-Tocher patch is a polynomial, well-defined
    there too, which is exactly what's wanted at a shared-edge continuity
    check evaluated "from both sides").
    """
    tiny = torch.finfo(px.dtype).tiny
    v0 = B - A
    v1 = C - A
    v2 = torch.stack([px, py], dim=-1) - A
    d00 = (v0 * v0).sum(dim=-1)
    d01 = (v0 * v1).sum(dim=-1)
    d11 = (v1 * v1).sum(dim=-1)
    d20 = (v2 * v0).sum(dim=-1)
    d21 = (v2 * v1).sum(dim=-1)
    denom = d00 * d11 - d01 * d01
    denom_safe = torch.where(denom.abs() < tiny, torch.full_like(denom, tiny), denom)
    v = (d11 * d20 - d01 * d21) / denom_safe
    w = (d00 * d21 - d01 * d20) / denom_safe
    u = 1.0 - v - w
    return torch.stack([u, v, w], dim=-1)


# ─────────────────────────────────────────────────────────────────────────
# Clough-Tocher macro-element: per-(used-)triangle Bezier ordinate
# coefficients, expressed as linear functionals of the 9 local unknowns
# (f0, f1, f2, gx0, gy0, gx1, gy1, gx2, gy2) at the triangle's 3 vertices.
#
# Formulas: Kosinka & Cashman (2015), Computer Aided Geometric Design 37,
# Section 3.1 "CTo: the Clough-Tocher construction" (split point Z = the
# centroid, tau = (1/3, 1/3, 1/3); the projection direction v_eps used to
# fix C0/C1/C2 is orthogonal to each macro-edge -- the "o" in CTo -- which
# is the *original* Clough & Tocher (1965) construction, matching what
# planning/05 asks for).
# ─────────────────────────────────────────────────────────────────────────

def _triangle_ct_coefficients(U: torch.Tensor):
    """9-dim linear-functional coefficients for every named CT quantity.

    Torch tensors throughout (GPU-capable). `U` has shape (T, 3, 2): the 3
    (projected) vertex positions of each of T triangles, in the triangle's
    own local order 0, 1, 2 (matching `Delaunay.simplices`' own vertex order
    -- this function does not care which vertex ends up being an "apex";
    that's resolved later, per query point, in `_assemble_M`).

    Every returned tensor has shape (T, 9); column order is
    [f0, f1, f2, gx0, gy0, gx1, gy1, gx2, gy2]. A quantity Q's coefficient
    row `c` means `Q = c[0]*f0 + c[1]*f1 + ... + c[8]*gy2`.

    Returns a dict with keys:
      "V"  : list of 3 tensors, coef(V_i) = coef(f_i) (trivial, one-hot)
      "T"  : dict {(i, j): coef(T_ij)} for i != j in {0, 1, 2}
      "I1" : list of 3 tensors, coef(I_{i,1})  (Step 1, eq. 12)
      "C"  : list of 3 tensors, coef(C_c)      (Step 2, eq. 15/16, CTo)
      "I2" : list of 3 tensors, coef(I_{i,2})  (Step 3, eq. 13)
      "S"  : tensor, coef(S) = coef(f(Z))      (Step 3, eq. 14)
    """
    Tn = U.shape[0]
    device, dtype = U.device, U.dtype
    tiny = torch.finfo(dtype).tiny

    def e(i):
        c = torch.zeros((Tn, 9), dtype=dtype, device=device)
        c[:, i] = 1.0
        return c

    def grad_dot(i, d):
        c = torch.zeros((Tn, 9), dtype=dtype, device=device)
        c[:, 3 + 2 * i] = d[:, 0]
        c[:, 3 + 2 * i + 1] = d[:, 1]
        return c

    V = [e(0), e(1), e(2)]

    # Step 1 (eq. 11): Hermite edge control points on the 3 macro-edges.
    T_coef = {}
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            d = U[:, j, :] - U[:, i, :]
            T_coef[(i, j)] = V[i] + (1.0 / 3.0) * grad_dot(i, d)

    # Step 1 (eq. 12): I_{i,1} = tau0*T_i0 + tau1*T_i1 + tau2*T_i2 (T_ii := V_i),
    # tau = (1/3, 1/3, 1/3) for the centroid split.
    I1 = []
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        I1.append((V[i] + T_coef[(i, j)] + T_coef[(i, k)]) / 3.0)

    Z = U.mean(dim=1)  # (T, 2) centroid

    # Step 2 (eq. 15/16, CTo -- orthogonal projection direction): for each
    # macro-edge (a, b) opposite vertex c, C_c is the interior ("b111")
    # control point of the micro-triangle (Ua, Ub, Z).
    C = [None, None, None]
    for c in range(3):
        a = (c + 1) % 3
        b = (c + 2) % 3
        Ua = U[:, a, :]
        Ub = U[:, b, :]
        edge = Ub - Ua
        edge_len2 = (edge * edge).sum(dim=1)
        edge_len2_safe = torch.where(edge_len2 < tiny, torch.full_like(edge_len2, tiny), edge_len2)
        lam_b = ((Z - Ua) * edge).sum(dim=1) / edge_len2_safe
        lam_a = 1.0 - lam_b
        C[c] = (
            lam_a[:, None] * T_coef[(a, b)]
            + lam_b[:, None] * T_coef[(b, a)]
            + 0.5
            * (
                I1[a]
                + I1[b]
                - lam_a[:, None] * (V[a] + T_coef[(b, a)])
                - lam_b[:, None] * (V[b] + T_coef[(a, b)])
            )
        )

    # Step 3 (eq. 13): I_{i,2} -- the spoke control point nearer Z -- needs
    # the two OTHER micro-triangles' interior points (both of which touch
    # the spoke i-Z), not C_i itself.
    I2 = []
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        I2.append((I1[i] + C[j] + C[k]) / 3.0)

    # Step 3 (eq. 14): value at the split point Z, shared by all 3 micro-triangles.
    S = (I2[0] + I2[1] + I2[2]) / 3.0

    return {"V": V, "T": T_coef, "I1": I1, "C": C, "I2": I2, "S": S}


# ─────────────────────────────────────────────────────────────────────────
# Sparse (N, K) operator assembly
# ─────────────────────────────────────────────────────────────────────────

def _assemble_M(
    points2d: torch.Tensor,
    tri: Delaunay,
    simplex_idx: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    N: int,
    K: int,
    edges_j: torch.Tensor,
    coef_gx_edge: torch.Tensor,
    coef_gy_edge: torch.Tensor,
    diag_gx: torch.Tensor,
    diag_gy: torch.Tensor,
    nbr_indptr: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build sparse CSR (N, K) `M` such that `hval = y @ M` evaluates the CT
    interpolant at the K (already hull-filtered) query points `(qx, qy)`.

    Torch tensor ops throughout (GPU-capable). `points2d`, `edges_j`,
    `coef_gx_edge`, `coef_gy_edge`, `diag_gx`, `diag_gy`, `nbr_indptr` must
    already be torch tensors on `device` (as returned by
    `_build_gradient_operators`). `simplex_idx`/`qx`/`qy` are still NumPy at
    the call site (they come straight out of `scipy.spatial.Delaunay.
    find_simplex`, which is CPU-only) and are converted to torch immediately
    below -- everything scaling with `K` (query points) or the Delaunay edge
    count from that point on runs as batched tensor ops.

    For each query point: locate its macro-triangle (`simplex_idx`), compute
    barycentric coordinates w.r.t. that triangle to pick the micro-triangle
    (apex = smallest barycentric coordinate -- see
    `planning/05_clough_tocher_resampler.md` step 5), evaluate the cubic
    Bernstein-Bezier basis against that micro-triangle's 10 control points,
    and expand every control point's 9-dim (f, gradient) linear-functional
    coefficients (from `_triangle_ct_coefficients`) through the per-vertex
    Delaunay edge lists into sparse (sample, query-point) entries.
    """
    # Triangle bookkeeping (which of the K query points share a triangle) is
    # cheap, small (<= K), and scipy-adjacent -- do it in NumPy, then move
    # only the resulting compact per-triangle geometry to torch.
    used_simplices, simplex_local = np.unique(simplex_idx, return_inverse=True)
    verts_np = tri.simplices[used_simplices].astype(np.int64)  # (Tloc, 3)
    verts = torch.as_tensor(verts_np, device=device)  # (Tloc, 3) long

    U = points2d[verts]  # (Tloc, 3, 2) torch, gathers directly from points2d

    ct = _triangle_ct_coefficients(U)
    V, T_coef, I1, C, I2, S = ct["V"], ct["T"], ct["I1"], ct["C"], ct["I2"], ct["S"]
    Z = U.mean(dim=1)

    tl = torch.as_tensor(simplex_local, dtype=torch.long, device=device)  # (K,)
    qx_t = torch.as_tensor(qx, dtype=dtype, device=device)
    qy_t = torch.as_tensor(qy, dtype=dtype, device=device)

    beta = _barycentric_2d(qx_t, qy_t, U[tl, 0, :], U[tl, 1, :], U[tl, 2, :])
    c_local = torch.argmin(beta, dim=1)  # apex = smallest barycentric coordinate

    rows_list = []
    cols_list = []
    vals_list = []

    for c in range(3):
        sel = torch.nonzero(c_local == c, as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        a = (c + 1) % 3
        b = (c + 2) % 3
        tl_sel = tl[sel]

        gamma = _barycentric_2d(qx_t[sel], qy_t[sel], U[tl_sel, a, :], U[tl_sel, b, :], Z[tl_sel])
        ga, gb, gz = gamma[:, 0], gamma[:, 1], gamma[:, 2]

        w_va = ga ** 3
        w_vb = gb ** 3
        w_s = gz ** 3
        w_tab = 3 * ga * ga * gb
        w_tba = 3 * ga * gb * gb
        w_iaz = 3 * ga * ga * gz  # near a, spoke a-Z -> I_{a,1}
        w_iza = 3 * ga * gz * gz  # near Z, spoke a-Z -> I_{a,2}
        w_ibz = 3 * gb * gb * gz  # near b, spoke b-Z -> I_{b,1}
        w_izb = 3 * gb * gz * gz  # near Z, spoke b-Z -> I_{b,2}
        w_c = 6 * ga * gb * gz

        coef = (
            w_va[:, None] * V[a][tl_sel]
            + w_vb[:, None] * V[b][tl_sel]
            + w_s[:, None] * S[tl_sel]
            + w_tab[:, None] * T_coef[(a, b)][tl_sel]
            + w_tba[:, None] * T_coef[(b, a)][tl_sel]
            + w_iaz[:, None] * I1[a][tl_sel]
            + w_iza[:, None] * I2[a][tl_sel]
            + w_ibz[:, None] * I1[b][tl_sel]
            + w_izb[:, None] * I2[b][tl_sel]
            + w_c[:, None] * C[c][tl_sel]
        )  # (len(sel), 9)

        vglobal = [verts[tl_sel, 0], verts[tl_sel, 1], verts[tl_sel, 2]]
        cf = [coef[:, 0], coef[:, 1], coef[:, 2]]
        cgx = [coef[:, 3], coef[:, 5], coef[:, 7]]
        cgy = [coef[:, 4], coef[:, 6], coef[:, 8]]

        for slot in range(3):
            v = vglobal[slot]
            c_f = cf[slot]
            c_gx = cgx[slot]
            c_gy = cgy[slot]

            # direct (f_v) + diagonal-gradient contribution -> one entry per
            # (query point, slot), landing on sample v itself.
            diag_val = c_f + c_gx * diag_gx[v] + c_gy * diag_gy[v]
            rows_list.append(v)
            cols_list.append(sel)
            vals_list.append(diag_val)

            # off-diagonal neighbour-gradient contribution: ragged gather
            # over v's 1-ring, via the same CSR-offset trick used to build
            # edges_i in _build_gradient_operators (torch.repeat_interleave
            # is the exact analogue of NumPy's per-element np.repeat here).
            deg = nbr_indptr[v + 1] - nbr_indptr[v]
            total = int(deg.sum().item())  # one small GPU->CPU sync, unavoidable for a ragged gather
            if total > 0:
                k_rep = torch.repeat_interleave(sel, deg)
                starts = nbr_indptr[v]
                group_start = torch.cumsum(deg, dim=0) - deg
                pos = torch.arange(total, device=device, dtype=torch.long) - torch.repeat_interleave(group_start, deg)
                src_idx = torch.repeat_interleave(starts, deg) + pos
                rows_off = edges_j[src_idx]
                cgx_rep = torch.repeat_interleave(c_gx, deg)
                cgy_rep = torch.repeat_interleave(c_gy, deg)
                vals_off = cgx_rep * coef_gx_edge[src_idx] + cgy_rep * coef_gy_edge[src_idx]
                rows_list.append(rows_off)
                cols_list.append(k_rep)
                vals_list.append(vals_off)

    rows_all = torch.cat(rows_list)
    cols_all = torch.cat(cols_list)
    vals_all = torch.cat(vals_list)

    indices_t = torch.stack([rows_all, cols_all], dim=0)
    M_coo = torch.sparse_coo_tensor(indices_t, vals_all, size=(N, K), device=device, dtype=dtype).coalesce()
    return M_coo.to_sparse_csr()


# ─────────────────────────────────────────────────────────────────────────
# Public resampler
# ─────────────────────────────────────────────────────────────────────────

class CloughTocherResampler:
    """Delaunay triangulation + Clough-Tocher C1 cubic HEALPix resampler.

    Unlike every other resampler in this package, this class does **not**
    subclass `KNeighborsResampler`: that base class's `__init__`/
    `comp_matrix()` orchestration is built around the KNN/Gaussian-threshold
    geometry model (`healpix_weighted_nearest`, `Npt`, `sigma_m`,
    `threshold`), none of which apply to a triangulation-based construction
    -- there is no fixed neighbour count, no Gaussian weight, and no
    distance threshold here. This is a deliberate design choice, not an
    oversight; see `planning/05_clough_tocher_resampler.md`.

    It still follows every other shared convention from
    `planning/00_init.md`: generic NumPy/Torch in/out symmetry, `(N,)`/
    `(B, N)` batching, `@torch.no_grad()` on `resample()`, returning
    `ResampleResults` (`cg_residual_norms`/`cg_niters` are always `None`
    here -- no CG solve is involved), plus `self.cell_ids`, `self.K`,
    `self.N`, `get_cell_ids()`.

    What it computes
    -----------------
    A genuine bivariate Delaunay/Clough-Tocher interpolant (exact at input
    sample points, C1 across triangle edges), not a radial kernel sum --
    see the module docstring for the full comparison against
    `BicubicResampler`. Vertex gradients are estimated by a local
    least-squares plane fit over each vertex's Delaunay 1-ring (see
    `_build_gradient_operators`); this is a standard, but not scipy-
    identical, Clough-Tocher construction (see module docstring).

    Output validity: a candidate HEALPix cell is only kept in
    `self.cell_ids` if its projected center falls **inside the convex hull**
    of the (projected) input samples -- Clough-Tocher, like
    `scipy.interpolate.griddata`, does not extrapolate.

    Parameters
    ----------
    lon_deg, lat_deg : array-like, shape (N,)
        Unstructured sample coordinates in degrees. Must span a
        regional/local extent (see the module docstring's gnomonic-
        projection caveat) -- `__init__` raises if any sample is not well
        inside the projection's valid hemisphere around the sample
        centroid.
    level : int
        HEALPix level (`nside = 2**level`) for candidate/output cells.
    nest : bool
        HEALPix indexing scheme.
    radius : float
        Sphere radius (meters); only affects the projected-plane length
        scale (`sigma_m`-style quantities are not used by this resampler).
    ellipsoid : str
        Passed through to `healpix_geo`.
    dtype, device : torch dtype/device for `self.Gx`, `self.Gy`, `self.M`.
    verbose : bool
        Print a one-line construction summary.
    out_cell_ids : array-like or None
        Optional caller-supplied subset of HEALPix cell ids (at `level`) to
        further restrict the output to, intersected with the convex-hull
        criterion above.
    candidate_ring_expand : int
        Number of fine-`level` HEALPix rings to expand the sample footprint
        by when enumerating *candidate* output cells, before filtering to
        "inside the convex hull" -- see `_candidate_output_cells`. The exact
        value doesn't affect correctness (over-generous candidates are just
        dropped by the hull test), only whether the hull is fully covered;
        the default is comfortably generous for any reasonably-dense point
        cloud relative to `level`.
    grad_det_floor_rel : float
        Relative floor applied to the 2x2 normal-equation determinant in
        `_build_gradient_operators`, guarding near-singular vertex 1-rings.

    Attributes (after construction)
    ---------------------------------
    N, K, cell_ids : as in every other resampler.
    points2d : (N, 2) numpy array -- samples projected to the local
        gnomonic tangent plane.
    tri : the `scipy.spatial.Delaunay` triangulation object.
    Gx, Gy : sparse (N, N) torch tensors -- ``grad_x = Gx @ f``,
        ``grad_y = Gy @ f`` for any sample-space field `f`.
    M : sparse CSR (N, K) torch tensor -- ``hval = y @ M``.
    """

    def __init__(
        self,
        lon_deg: T_Array,
        lat_deg: T_Array,
        level: int,
        *,
        nest: bool = True,
        radius: float = 6371000.0,
        ellipsoid: str = "WGS84",
        dtype: torch.dtype = torch.float64,
        device: Optional[Union[torch.device, str]] = None,
        verbose: bool = True,
        out_cell_ids: Optional[T_Array] = None,
        candidate_ring_expand: int = 3,
        grad_det_floor_rel: float = 1e-9,
        num_threads: int = 0,
    ) -> None:
        self.level = int(level)
        self.nside = 2 ** int(level)
        self.nest = bool(nest)
        self.radius = float(radius)
        self.ellipsoid = str(ellipsoid)
        self.dtype = dtype
        self.verbose = verbose
        self.out_cell_ids = out_cell_ids

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")
        self.device = torch.device(device)

        lon_np = _to_numpy_f64(lon_deg).reshape(-1)
        lat_np = _to_numpy_f64(lat_deg).reshape(-1)
        if lon_np.shape != lat_np.shape:
            raise ValueError("lon_deg and lat_deg must have the same shape.")
        self.N = int(lon_np.size)
        if self.N < 3:
            raise ValueError("CloughTocherResampler needs at least 3 samples to triangulate.")

        # ---- 1. local gnomonic projection -----------------------------
        t, east, north = _gnomonic_tangent_frame(lon_np, lat_np)
        self._tangent_point = t
        self._east = east
        self._north = north

        x, y, denom = _gnomonic_project(lon_np, lat_np, t, east, north, radius=self.radius)
        if np.any(denom <= 0.05):
            raise ValueError(
                "CloughTocherResampler: at least one input sample is too far "
                "(>~87 degrees) from the sample centroid for a stable local "
                "gnomonic projection. This resampler is intended for "
                "regional/local input extents, not global datasets in one "
                "construction call -- see the module docstring and "
                "planning/04_parent_cell_subsetting.md for processing a "
                "large dataset one local patch at a time."
            )
        self.points2d = np.stack([x, y], axis=-1)
        # Torch copy on `self.device`, used for every downstream computation
        # whose cost scales with N/edges/K (gradient operators, CT
        # coefficients, sparse M assembly) -- `self.points2d` itself stays
        # NumPy (documented public attribute, and scipy.spatial.Delaunay
        # needs a NumPy array anyway).
        points2d_t = torch.as_tensor(self.points2d, dtype=self.dtype, device=self.device)

        # ---- 2. Delaunay triangulation ---------------------------------
        try:
            self.tri = Delaunay(self.points2d)
        except QhullError as exc:
            raise RuntimeError(
                "CloughTocherResampler: scipy.spatial.Delaunay failed to "
                "triangulate the projected input samples (e.g. all samples "
                "collinear/degenerate in the projected plane)."
            ) from exc

        # ---- 3. gradient-estimation operators Gx, Gy -------------------
        (
            self.Gx,
            self.Gy,
            edges_j,
            coef_gx_edge,
            coef_gy_edge,
            diag_gx,
            diag_gy,
            nbr_indptr,
        ) = _build_gradient_operators(
            points2d_t, self.tri, self.device, self.dtype, det_floor_rel=grad_det_floor_rel
        )

        # ---- 4. candidate output cells, filtered to the convex hull ----
        cand_ids = _candidate_output_cells(
            lon_np, lat_np, self.level, self.nest, self.ellipsoid,
            ring_expand=candidate_ring_expand, num_threads=num_threads,
        )
        if self.out_cell_ids is not None:
            out_np = _to_numpy_int64(self.out_cell_ids).reshape(-1)
            cand_ids = np.intersect1d(cand_ids, out_np)
        if cand_ids.size == 0:
            raise RuntimeError(
                "CloughTocherResampler: no candidate HEALPix cells near the "
                "input samples (empty out_cell_ids intersection, or the "
                "sample footprint is empty)."
            )

        hp = healpix_geo.nested if self.nest else healpix_geo.ring
        cand_lon, cand_lat = hp.healpix_to_lonlat(cand_ids.astype(np.uint64), self.level, ellipsoid=self.ellipsoid)
        cand_lon = np.asarray(cand_lon, dtype=np.float64)
        cand_lat = np.asarray(cand_lat, dtype=np.float64)
        qx, qy, qdenom = _gnomonic_project(cand_lon, cand_lat, t, east, north, radius=self.radius)

        simplex_idx = self.tri.find_simplex(np.stack([qx, qy], axis=-1))
        # Belt-and-braces: candidates far enough from the tangent point that
        # the projection itself is unreliable can never legitimately be
        # inside the (local) input hull -- force them out explicitly rather
        # than trust find_simplex on a possibly-garbled projected position.
        simplex_idx = np.where(qdenom <= 0.05, -1, simplex_idx)

        keep = simplex_idx >= 0
        if not np.any(keep):
            raise RuntimeError(
                "CloughTocherResampler: no HEALPix cell centers fall inside "
                "the Delaunay convex hull of the input samples -- "
                "Clough-Tocher does not extrapolate. Check `level` / sample "
                "density, or widen `out_cell_ids` if it was supplied."
            )

        self.cell_ids = torch.as_tensor(cand_ids[keep].astype(np.int64), device=self.device)
        self.K = int(self.cell_ids.numel())
        simplex_idx = simplex_idx[keep]
        qx_k = qx[keep]
        qy_k = qy[keep]

        # ---- 5. assemble sparse M ---------------------------------------
        self.M = _assemble_M(
            points2d=points2d_t,
            tri=self.tri,
            simplex_idx=simplex_idx,
            qx=qx_k,
            qy=qy_k,
            N=self.N,
            K=self.K,
            edges_j=edges_j,
            coef_gx_edge=coef_gx_edge,
            coef_gy_edge=coef_gy_edge,
            diag_gx=diag_gx,
            diag_gy=diag_gy,
            nbr_indptr=nbr_indptr,
            device=self.device,
            dtype=self.dtype,
        )

        if self.verbose:
            print(
                f"[CloughTocherResampler] N={self.N} samples, "
                f"{self.tri.simplices.shape[0]} Delaunay triangles, "
                f"K={self.K} output cells retained (inside convex hull)."
            )

    @torch.no_grad()
    def resample(self, val: T_Array) -> ResampleResults[T_Array]:
        """Estimate the HEALPix field from unstructured samples.

        Args:
            val: (N,) or (B, N) values at lon/lat sample points.

        `hval = y @ self.M`, exactly like every other resampler's
        non-conservative path. NaN handling: `self.M` is a fixed,
        geometry-only sparse operator (never touches `val`), so NaN in `y`
        propagates through the matmul following ordinary IEEE 754 rules
        (confirmed elsewhere in this package -- see
        `planning/03_bilinear_nan_investigation.md` -- that CSR sparse
        matmul in this codebase's torch version does *not* have the
        CSR-specific NaN-dropping bug that investigation set out to rule
        out; only the separate "orphaned column" mechanism documented in
        `knn.py` was confirmed, which does not apply here since `self.M`'s
        columns are built directly from located triangles, not from a
        wide-then-narrow two-stage KNN search).

        Returns:
            hval: (B, K) or (K,)
        """
        y = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
        y = y.to(self.device, dtype=self.dtype)
        clean_shape = False
        if y.ndim == 1:
            clean_shape = True
            y = y[None, :]

        hval = y @ self.M
        cell_ids = self.cell_ids

        if not isinstance(val, torch.Tensor):
            hval = hval.cpu().numpy()
            cell_ids = cell_ids.cpu().numpy()
        if clean_shape:
            hval = hval[0]

        return ResampleResults(cell_data=hval, cell_ids=cell_ids)

    def invert(self, hval: T_Array) -> T_Array:
        """Not implemented -- see `planning/05_clough_tocher_resampler.md`,
        "Composing everything into one sparse (N,K) matrix M".

        Unlike the KNN-based resamplers (which get a natural `MT` "for
        free" from the same symmetric neighbour search that builds `M`),
        Delaunay/Clough-Tocher only defines a mapping from scattered samples
        to arbitrary query points, not the reverse -- a second, independent
        Delaunay/CT operator built the other way round (retained HEALPix
        cells as the "scattered samples") would roughly double the
        implementation for a direction the planning doc explicitly says not
        to build unless asked. Raises unconditionally.
        """
        raise NotImplementedError(
            "CloughTocherResampler.invert() is not implemented: Delaunay/"
            "Clough-Tocher has no natural, cheap reverse operator the way "
            "KNeighborsResampler's M/MT pair does. See "
            "planning/05_clough_tocher_resampler.md, 'Composing everything "
            "into one sparse (N,K) matrix M', for why this was deliberately "
            "left unimplemented rather than building a second, independent "
            "CT operator in the reverse direction."
        )

    def get_cell_ids(self) -> np.ndarray:
        return self.cell_ids.cpu().numpy()


def _to_numpy_f64(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy().astype(np.float64)
    return np.asarray(a, dtype=np.float64)


def _to_numpy_int64(a) -> np.ndarray:
    # Kept separate from _to_numpy_f64: HEALPix cell ids at high levels can
    # exceed float64's 2**53 exact-integer range, so this must never round-
    # trip through float64 (unlike lon/lat, where float64 is exact enough).
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy().astype(np.int64)
    return np.asarray(a).astype(np.int64)
