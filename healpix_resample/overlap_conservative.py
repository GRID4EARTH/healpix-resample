"""
overlap_conservative.py

First-order overlap-area conservative remapping onto HEALPix.

This resampler implements the classical first-order conservative
formulation used in Earth-system remapping (ESMF/xESMF ``conservative`` /
``conservative_normed``): the value of each source cell is redistributed
over every HEALPix cell it overlaps, with weights proportional to the
source/target intersection areas

    O_ji = |S_i ∩ D_j|,

rather than assigned to the single cell containing the source centre as
the hard-binning :class:`~healpix_resample.conservative.ConservativeResampler`
does. Both conserve the global integral; only the overlap method also
reproduces the local geometric redistribution (and hence preserves a
constant field, see below).

Exactness
---------
Overlap areas are computed **exactly on the unit sphere** (up to floating
point), with no polygon densification and no small-angle approximation.
The construction exploits two structural facts about HEALPix
[Gorski et al. 2005]:

1. In the HEALPix planar projection (x, y) the mapping from (lon, z=sin lat)
   has a *constant* Jacobian (equal-area projection), so plane areas are
   spherical areas times a constant.
2. Level-``L`` HEALPix cells are exact squares (diamonds rotated 45
   degrees) of half-diagonal ``s = pi / (4 * 2**L)`` in that plane, and
   the images of parallels are horizontal straight lines everywhere, while
   the images of meridians are straight lines within the equatorial belt
   (|z| <= 2/3) and within each polar quadrant (lon in [k*pi/2, (k+1)*pi/2)).

Therefore, after splitting every source lat/lon rectangle at the two
transition latitudes (z = ±2/3) and at the polar-quadrant boundary
meridians (lon = k*pi/2), each piece maps to a *straight-edged convex
quadrilateral* in the plane, and each source/target intersection is an
exact convex polygon clipping problem. In rotated coordinates
(u, v) = (x+y, x-y) the target diamonds become axis-aligned squares, so
the clipping reduces to four axis-aligned half-plane cuts
(Sutherland-Hodgman), which is numerically robust.

Consequences: the antimeridian and the poles need no special casing
beyond the splitting above (a pole-touching piece simply maps to a
triangle), and the only approximation in the whole computation is
floating-point rounding.

Earth model: the current implementation is **spherical** (``ellipsoid
="sphere"``); areas are returned in steradians (multiply by R^2 for m^2).
The conservation statement is exact for the spherical cell geometry; an
authalic-ellipsoid variant would only change the lat -> z mapping.

Normalizations (xESMF correspondence)
-------------------------------------
``normalization="destination"`` divides by the full target-cell area
|D_j| (xESMF ``conservative``): uncovered target area contributes zero.
``normalization="covered"`` divides by the actually covered area
sum_i O_ji (xESMF ``conservative_normed``): a constant source field is
reproduced exactly on every partially covered cell.

Intensive vs extensive fields
-----------------------------
``quantity="intensive"`` (default; e.g. W m^-2, K): first-order
remapping of a density, x_j = sum_i O_ji y_i / A_j with A_j as above.
``quantity="extensive"`` (e.g. counts, cell-integrated energy): the
source total is distributed by overlap fraction of the *source* cell,
q_j = sum_i q_i O_ji / |S_i|; the ``normalization`` parameter does not
apply and the destination values are again extensive.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

import healpix_geo

__all__ = ["OverlapConservativeResampler"]

_Z_TRANSITION = 2.0 / 3.0  # |z| boundary between equatorial belt and polar caps
_JACOBIAN = 3.0 * np.pi / 8.0  # d(x,y) = _JACOBIAN * d(lon, z): plane area -> sphere area / _JACOBIAN... (see _plane_to_sphere_area)


def _project(lon, z):
    """HEALPix planar projection of (lon [rad, in [0, 2pi)], z=sin(lat)).

    Vectorized; exact standard formulas (Gorski et al. 2005, Sec. 4.4).
    """
    lon = np.asarray(lon, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x = np.array(lon, copy=True)
    y = _JACOBIAN * z  # equatorial belt
    cap = np.abs(z) > _Z_TRANSITION
    if np.any(cap):
        zc = z[cap]
        lc = lon[cap]
        sigma = np.sqrt(3.0 * (1.0 - np.abs(zc)))
        phi_c = (np.floor(lc / (0.5 * np.pi)) + 0.5) * (0.5 * np.pi)
        x[cap] = phi_c + (lc - phi_c) * sigma
        y[cap] = np.sign(zc) * (np.pi / 4.0) * (2.0 - sigma)
    return x, y


def _unproject(x, y):
    """Inverse of :func:`_project`. Returns (lon, z, valid).

    ``valid`` is False where (x, y) lies outside the image of the sphere
    (the notches between polar-cap triangles).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lon = np.array(x, copy=True)
    z = y / _JACOBIAN
    valid = np.abs(y) <= 0.5 * np.pi
    cap = (np.abs(y) > 0.25 * np.pi) & valid
    if np.any(cap):
        yc = y[cap]
        xc = x[cap]
        sigma = 2.0 - 4.0 * np.abs(yc) / np.pi
        z[cap] = np.sign(yc) * (1.0 - sigma * sigma / 3.0)
        phi_c = (np.floor(xc / (0.5 * np.pi)) + 0.5) * (0.5 * np.pi)
        with np.errstate(divide="ignore", invalid="ignore"):
            lon_cap = phi_c + (xc - phi_c) / np.where(sigma > 0, sigma, np.inf)
        lon[cap] = np.where(sigma > 0, lon_cap, phi_c)
        # outside the triangular cap faces: |x - phi_c| > (pi/4) * sigma
        bad = np.abs(xc - phi_c) > (0.25 * np.pi) * sigma + 1e-12
        v = valid[cap]
        v[bad] = False
        valid[cap] = v
    return lon, z, valid


def _plane_to_sphere_area(a_xy):
    """Convert an area in the projection plane to steradians."""
    return a_xy / _JACOBIAN


def _split_edges(bounds, cuts):
    """Insert the values of ``cuts`` falling strictly inside [b0, b1] into a
    sorted 1D bounds array, returning the refined array."""
    inside = cuts[(cuts > bounds[0] + 1e-14) & (cuts < bounds[-1] - 1e-14)]
    return np.unique(np.concatenate([bounds, inside]))


def _clip_axis(poly, count, axis, bound, keep_below):
    """Vectorized Sutherland-Hodgman cut of many convex polygons against an
    axis-aligned half-plane.

    poly  : (M, K, 2) padded vertex arrays (padding = repeat of last vertex)
    count : (M,) number of meaningful vertices
    Returns (poly2, count2) with K2 = K + 1.
    """
    M, K, _ = poly.shape
    bound = np.asarray(bound, dtype=np.float64)
    bcol = bound[:, None] if bound.ndim == 1 else bound
    coord = poly[:, :, axis]
    if keep_below:
        inside = coord <= bcol + 1e-15
    else:
        inside = coord >= bcol - 1e-15
    nxt = np.roll(np.arange(K), -1)
    # edge k: from vertex k to vertex nxt[k]; only edges k < count are real,
    # and the closing edge is (count-1) -> 0, which roll handles only for
    # k == K-1. Build explicit next indices honoring count.
    idx = np.arange(K)[None, :].repeat(M, axis=0)
    nxt_idx = idx + 1
    nxt_idx[nxt_idx >= count[:, None]] = 0
    nxt_idx[idx >= count[:, None]] = 0  # padded slots: degenerate self-edges
    p0 = poly
    p1 = np.take_along_axis(poly, nxt_idx[:, :, None], axis=1)
    in0 = np.take_along_axis(inside, idx, axis=1)
    in1 = np.take_along_axis(inside, nxt_idx, axis=1)
    real = idx < count[:, None]

    # intersection points along each edge with the cut line
    c0 = p0[:, :, axis]
    c1 = p1[:, :, axis]
    denom = c1 - c0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(denom) > 0, (bcol - c0) / np.where(denom == 0, 1, denom), 0.0)
    t = np.clip(t, 0.0, 1.0)
    inter = p0 + t[:, :, None] * (p1 - p0)

    emit_vertex = in0 & real
    emit_inter = (in0 != in1) & real

    K2 = K + 1
    out = np.zeros((M, K2, 2), dtype=np.float64)
    out_count = np.zeros(M, dtype=np.int64)
    # slot layout: for edge k, up to two emissions (vertex then intersection);
    # compact with cumulative counts.
    emits = np.concatenate([emit_vertex[:, :, None], emit_inter[:, :, None]], axis=2).reshape(M, 2 * K)
    pts = np.concatenate([p0[:, :, None, :], inter[:, :, None, :]], axis=2).reshape(M, 2 * K, 2)
    pos = np.cumsum(emits, axis=1) - 1
    rows, slots = np.nonzero(emits)
    target = pos[rows, slots]
    keep = target < K2  # convex input: never exceeded, guard anyway
    out[rows[keep], target[keep]] = pts[rows[keep], slots[keep]]
    out_count = emits.sum(axis=1).astype(np.int64)
    np.clip(out_count, 0, K2, out=out_count)
    # re-pad with the last valid vertex so downstream rolls stay harmless
    has = out_count > 0
    last = np.clip(out_count - 1, 0, K2 - 1)
    pad_src = out[np.arange(M), last]
    pad_mask = np.arange(K2)[None, :] >= out_count[:, None]
    out[pad_mask] = pad_src.repeat(pad_mask.sum(axis=1), axis=0)
    out[~has] = 0.0
    return out, out_count


def _polygon_area(poly, count):
    """Shoelace area of padded polygons (padding-safe: repeated vertices
    contribute zero)."""
    M, K, _ = poly.shape
    idx = np.arange(K)[None, :].repeat(M, axis=0)
    nxt_idx = idx + 1
    nxt_idx[nxt_idx >= count[:, None]] = 0
    nxt_idx[idx >= count[:, None]] = 0
    p1 = np.take_along_axis(poly, nxt_idx[:, :, None], axis=1)
    cross = poly[:, :, 0] * p1[:, :, 1] - p1[:, :, 0] * poly[:, :, 1]
    real = idx < count[:, None]
    return 0.5 * np.abs(np.sum(np.where(real, cross, 0.0), axis=1))


class OverlapConservativeResampler:
    """First-order overlap-area conservative remapping onto HEALPix.

    Parameters
    ----------
    lon_bounds, lat_bounds : 1D array-like, degrees
        Cell *boundaries* of a (possibly irregular) rectilinear lat/lon
        source grid: ``nlon = len(lon_bounds) - 1`` columns and
        ``nlat = len(lat_bounds) - 1`` rows. ``lat_bounds`` must be
        strictly monotonic in [-90, 90]; ``lon_bounds`` strictly monotonic
        with total span <= 360 degrees (antimeridian crossing is allowed).
    level : int
        Target HEALPix level (nside = 2**level).
    nest : bool
        Nested (default) or ring target indexing.
    normalization : {"destination", "covered"}
        Weight normalization for intensive fields; see the module
        docstring for the xESMF correspondence.

    Attributes
    ----------
    cell_ids : (M,) int64 — HEALPix cells receiving nonzero overlap.
    weights : scipy.sparse.csr_matrix, shape (M, nlat*nlon) — the
        normalization-dependent intensive-remapping matrix W (x = W y).
    overlap : scipy.sparse.csr_matrix — raw overlap areas O_ji [sr].
    target_area : float — exact HEALPix cell area 4*pi/(12*nside^2) [sr].
    covered_area : (M,) float — sum_i O_ji [sr] per target cell.
    source_area : (nlat*nlon,) float — |S_i| [sr].
    """

    def __init__(self, lon_bounds, lat_bounds, level, nest=True,
                 normalization="destination"):
        if normalization not in ("destination", "covered"):
            raise ValueError("normalization must be 'destination' or 'covered'")
        self.level = int(level)
        self.nest = bool(nest)
        self.normalization = normalization
        nside = 2 ** self.level
        self._s = np.pi / (4.0 * nside)  # diamond half-diagonal in the plane
        self.target_area = 4.0 * np.pi / (12.0 * nside * nside)

        lonb = np.asarray(lon_bounds, dtype=np.float64)
        latb = np.asarray(lat_bounds, dtype=np.float64)
        if lonb.ndim != 1 or latb.ndim != 1 or lonb.size < 2 or latb.size < 2:
            raise ValueError("lon_bounds and lat_bounds must be 1D with >= 2 entries")
        if np.any(np.diff(latb) <= 0):
            latb = latb[::-1]
            self._lat_flipped = True
            if np.any(np.diff(latb) <= 0):
                raise ValueError("lat_bounds must be strictly monotonic")
        else:
            self._lat_flipped = False
        if np.any(np.diff(lonb) <= 0):
            raise ValueError("lon_bounds must be strictly increasing")
        if lonb[-1] - lonb[0] > 360.0 + 1e-9:
            raise ValueError("lon_bounds span exceeds 360 degrees")
        if latb[0] < -90.0 - 1e-9 or latb[-1] > 90.0 + 1e-9:
            raise ValueError("lat_bounds outside [-90, 90]")

        self.nlon = lonb.size - 1
        self.nlat = latb.size - 1
        self._build(np.radians(lonb), np.sin(np.radians(np.clip(latb, -90, 90))))

    # ------------------------------------------------------------------ build
    def _build(self, lonb, zb):
        s = self._s
        pieces_poly = []   # (P, 4, 2) plane quads in (u, v) coords
        pieces_src = []    # source flat index per piece

        quad_cuts = np.arange(-8.0, 9.0) * (0.5 * np.pi)
        z_cuts = np.array([-_Z_TRANSITION, _Z_TRANSITION])

        for irow in range(self.nlat):
            z0, z1 = zb[irow], zb[irow + 1]
            z_edges = _split_edges(np.array([z0, z1]), z_cuts)
            for icol in range(self.nlon):
                l0, l1 = lonb[icol], lonb[icol + 1]
                # normalize into [0, 2pi) and split a wrap into two spans
                off = np.floor(l0 / (2 * np.pi)) * 2 * np.pi
                l0n, l1n = l0 - off, l1 - off
                spans = ([(l0n, l1n)] if l1n <= 2 * np.pi + 1e-14
                         else [(l0n, 2 * np.pi), (0.0, l1n - 2 * np.pi)])
                src = irow * self.nlon + icol
                for (a, b) in spans:
                    lon_edges = _split_edges(np.array([a, b]), quad_cuts)
                    for kz in range(len(z_edges) - 1):
                        za, zc = z_edges[kz], z_edges[kz + 1]
                        for kl in range(len(lon_edges) - 1):
                            la, lb = lon_edges[kl], lon_edges[kl + 1]
                            corner_lon = np.array([la, lb, lb, la])
                            corner_z = np.array([za, za, zc, zc])
                            px, py = _project(corner_lon, corner_z)
                            pieces_poly.append(np.stack([px + py, px - py], axis=1))
                            pieces_src.append(src)

        pieces_poly = np.asarray(pieces_poly)              # (P, 4, 2) in (u,v)
        pieces_src = np.asarray(pieces_src, dtype=np.int64)
        P = pieces_poly.shape[0]

        # --- candidate diamond centres per piece (lattice enumeration) -----
        # In (u, v) = (x+y, x-y), level-L cell centres lie on the integer
        # lattice u = p*s, v = q*s with p ≡ q ≡ (nside+1) (mod 2); diamonds
        # are axis-aligned squares of half-side s.
        nside = 2 ** self.level
        parity = (nside + 1) % 2
        umin = pieces_poly[:, :, 0].min(axis=1) - s
        umax = pieces_poly[:, :, 0].max(axis=1) + s
        vmin = pieces_poly[:, :, 1].min(axis=1) - s
        vmax = pieces_poly[:, :, 1].max(axis=1) + s

        pair_piece = []
        pair_pc = []
        pair_qc = []
        for i in range(P):
            plo = int(np.ceil(umin[i] / s))
            phi = int(np.floor(umax[i] / s))
            qlo = int(np.ceil(vmin[i] / s))
            qhi = int(np.floor(vmax[i] / s))
            ps = np.arange(plo, phi + 1)
            qs = np.arange(qlo, qhi + 1)
            ps = ps[(ps % 2) == parity]
            qs = qs[(qs % 2) == parity]
            if ps.size == 0 or qs.size == 0:
                continue
            PP, QQ = np.meshgrid(ps, qs, indexing="ij")
            pair_piece.append(np.full(PP.size, i, dtype=np.int64))
            pair_pc.append(PP.ravel())
            pair_qc.append(QQ.ravel())
        if not pair_piece:
            raise ValueError("no candidate overlaps found (empty grid?)")
        pair_piece = np.concatenate(pair_piece)
        pc = np.concatenate(pair_pc).astype(np.float64) * s
        qc = np.concatenate(pair_qc).astype(np.float64) * s

        # --- validity of candidate centres (must be a real HEALPix cell) ---
        xc = 0.5 * (pc + qc)
        yc = 0.5 * (pc - qc)
        lon_c, z_c, ok = _unproject(xc, yc)
        keep = ok & (np.abs(z_c) <= 1.0 + 1e-12)
        pair_piece, pc, qc = pair_piece[keep], pc[keep], qc[keep]
        lon_c, z_c = lon_c[keep], np.clip(z_c[keep], -1, 1)

        # --- clip each piece polygon against its candidate diamond ---------
        poly = pieces_poly[pair_piece]                     # (M, 4, 2)
        count = np.full(poly.shape[0], 4, dtype=np.int64)
        poly, count = _clip_axis(poly, count, 0, pc + s, True)
        poly, count = _clip_axis(poly, count, 0, pc - s, False)
        poly, count = _clip_axis(poly, count, 1, qc + s, True)
        poly, count = _clip_axis(poly, count, 1, qc - s, False)
        a_uv = _polygon_area(poly, count)
        area = _plane_to_sphere_area(0.5 * a_uv)           # du dv = 2 dx dy
        nz = area > self.target_area * 1e-14
        pair_piece, area = pair_piece[nz], area[nz]
        lon_c, z_c = lon_c[nz], z_c[nz]

        # --- map candidate centres to HEALPix ids --------------------------
        lat_c_deg = np.degrees(np.arcsin(z_c))
        lon_c_deg = np.degrees(np.mod(lon_c, 2 * np.pi))
        hp = healpix_geo.nested if self.nest else healpix_geo.ring
        ids = hp.lonlat_to_healpix(lon_c_deg, lat_c_deg, self.level,
                                   ellipsoid="sphere")

        src = np.asarray(pieces_src)[pair_piece]
        cell_u, inv = np.unique(ids, return_inverse=True)
        n_src = self.nlat * self.nlon
        overlap = sparse.coo_matrix(
            (area, (inv, src)), shape=(cell_u.size, n_src)).tocsr()
        # merge duplicate (cell, src) entries produced by piece splitting
        overlap.sum_duplicates()

        self.cell_ids = cell_u.astype(np.int64)
        self.overlap = overlap
        self.covered_area = np.asarray(overlap.sum(axis=1)).ravel()
        dphi = np.diff(lonb)
        dz = np.abs(np.diff(zb))
        self.source_area = (dz[:, None] * dphi[None, :]).ravel()

        denom = (np.full(cell_u.size, self.target_area)
                 if self.normalization == "destination" else self.covered_area)
        inv_d = sparse.diags(1.0 / denom)
        self.weights = (inv_d @ overlap).tocsr()

    # -------------------------------------------------------------- resample
    def resample(self, values, quantity="intensive"):
        """Remap a source field. ``values`` has shape (nlat, nlon) (or the
        flattened equivalent), ordered like the bounds arrays as passed
        (a descending ``lat_bounds`` input is handled transparently)."""
        v = np.asarray(values, dtype=np.float64)
        if v.ndim == 2:
            if v.shape != (self.nlat, self.nlon):
                raise ValueError(f"expected shape {(self.nlat, self.nlon)}, got {v.shape}")
            if self._lat_flipped:
                v = v[::-1]
            v = v.ravel()
        elif v.size != self.nlat * self.nlon:
            raise ValueError("flattened values have the wrong size")
        if quantity == "intensive":
            data = self.weights @ v
        elif quantity == "extensive":
            frac = self.overlap @ sparse.diags(1.0 / self.source_area)
            data = frac @ v
        else:
            raise ValueError("quantity must be 'intensive' or 'extensive'")
        from healpix_resample.base import ResampleResults
        return ResampleResults(cell_data=data, cell_ids=self.cell_ids)
