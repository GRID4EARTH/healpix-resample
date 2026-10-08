#!/usr/bin/env python3
"""Validation and benchmark of OverlapConservativeResampler.

Writes ``notebooks/tables/overlap_conservative_validation.csv`` (property
and cross-validation metrics) and
``notebooks/tables/overlap_conservative_benchmark.csv`` (weight-construction
and application timings), the two tables backing the supplement's
overlap-conservative appendix.

Run from the repository root or notebooks/::

    python notebooks/overlap_conservative_validation.py

Validation strategy
-------------------
The resampler's overlap areas are computed exactly on the sphere by
convex clipping in the HEALPix equal-area plane; the tests below check
the *mathematical properties* of the result (conservation, constant
preservation, bounds) at floating-point tolerance, and cross-validate
the geometry against an **independent oracle**: a dense equal-area
quadrature where every quadrature point is assigned to its source cell
and to its HEALPix cell by ``healpix_geo.lonlat_to_healpix`` — a code
path that plays no role in the clipping-based area computation. The
quadrature estimate converges to the same overlap integrals, so
agreement at the expected quadrature-error level validates the clipping
geometry end to end.

An ESMF/xESMF numerical cross-comparison is deliberately *not* run
here: ESMF's structured API cannot represent HEALPix (whose cells do
not form a logically rectangular mesh), so a cell-by-cell comparison is
not expressible; the correspondence with ESMF/xESMF ``conservative`` /
``conservative_normed`` is at the level of the first-order overlap-area
formulation, which the quadrature oracle validates numerically.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from healpix_resample.overlap_conservative import OverlapConservativeResampler

REPO = Path(__file__).resolve().parents[1]
TABLE_DIR = REPO / "notebooks" / "tables"

FOUR_PI = 4.0 * np.pi


def smooth_field(lon_c, lat_c):
    """Deterministic smooth test field on cell centres (degrees)."""
    lam = np.radians(lon_c)
    phi = np.radians(lat_c)
    return (2.0 + np.sin(2 * lam) * np.cos(phi) ** 2
            + 0.5 * np.sin(3 * phi))


def property_metrics(dlon, dlat, level, normalization):
    lonb = np.arange(0.0, 360.0 + 1e-9, dlon)
    latb = np.arange(-90.0, 90.0 + 1e-9, dlat)
    r = OverlapConservativeResampler(lonb, latb, level=level,
                                     normalization=normalization)
    lon_c = 0.5 * (lonb[:-1] + lonb[1:])
    lat_c = 0.5 * (latb[:-1] + latb[1:])
    y = smooth_field(*np.meshgrid(lon_c, lat_c))
    x = r.resample(y).cell_data
    i_src = float(np.sum(r.source_area * y.ravel()))
    i_dst = float(np.sum(r.target_area * x))
    const = r.resample(np.ones_like(y)).cell_data
    return r, {
        "eps_global": abs(i_dst - i_src) / abs(i_src),
        "eps_const": float(np.abs(const - 1.0).max()),
        "coverage_err": float(np.abs(r.covered_area.sum() / FOUR_PI - 1.0)),
        # tolerance commensurate with eps_const: per-cell weight sums are
        # 1 +/- O(1e-11) roundoff, so bound slack scales with |y|*1e-11
        "within_bounds": bool(x.min() >= y.min() - 1e-9
                              and x.max() <= y.max() + 1e-9),
        "n_cells": int(r.cell_ids.size),
        "nnz": int(r.weights.nnz),
    }


def quadrature_crosscheck(dlon, level, n_quad_lon=5400):
    """Compare clipped covered-norm values against an equal-area dense
    quadrature classified by lonlat_to_healpix (independent oracle)."""
    import healpix_geo

    lonb = np.arange(0.0, 360.0 + 1e-9, dlon)
    latb = np.arange(-90.0, 90.0 + 1e-9, dlon)
    r = OverlapConservativeResampler(lonb, latb, level=level,
                                     normalization="covered")
    lon_c = 0.5 * (lonb[:-1] + lonb[1:])
    lat_c = 0.5 * (latb[:-1] + latb[1:])
    y = smooth_field(*np.meshgrid(lon_c, lat_c))
    x = r.resample(y).cell_data

    # equal-area quadrature lattice: uniform in lon and in z = sin(lat)
    n_lon = n_quad_lon
    n_z = n_quad_lon // 2
    qlon = (np.arange(n_lon) + 0.5) * (360.0 / n_lon)
    qz = -1.0 + (np.arange(n_z) + 0.5) * (2.0 / n_z)
    QL, QZ = np.meshgrid(qlon, qz)
    qlat = np.degrees(np.arcsin(QZ)).ravel()
    qlon = QL.ravel()
    src_col = np.minimum((qlon / dlon).astype(np.int64), lonb.size - 2)
    src_row = np.minimum(((np.sin(np.radians(qlat)) + 1.0) * 0
                          + (qlat + 90.0) / dlon).astype(np.int64),
                         latb.size - 2)
    yq = y[src_row, src_col]
    ids = healpix_geo.nested.lonlat_to_healpix(qlon, qlat, level,
                                               ellipsoid="sphere")
    order = np.argsort(r.cell_ids)
    pos = order[np.searchsorted(r.cell_ids[order], ids.astype(np.int64))]
    sums = np.bincount(pos, weights=yq, minlength=r.cell_ids.size)
    counts = np.bincount(pos, minlength=r.cell_ids.size)
    ok = counts > 0
    x_quad = sums[ok] / counts[ok]
    diff = np.abs(x[ok] - x_quad)
    field_range = float(y.max() - y.min())
    return {
        "quad_points": int(qlon.size),
        "quad_max_absdiff": float(diff.max()),
        "quad_rms_absdiff": float(np.sqrt(np.mean(diff ** 2))),
        "quad_rel_to_range": float(diff.max() / field_range),
    }


def benchmark(dlon, level, n_repeat_apply=20):
    lonb = np.arange(0.0, 360.0 + 1e-9, dlon)
    latb = np.arange(-90.0, 90.0 + 1e-9, dlon)
    t0 = time.perf_counter()
    r = OverlapConservativeResampler(lonb, latb, level=level)
    t_build = time.perf_counter() - t0
    y = np.random.default_rng(0).uniform(0.0, 1.0,
                                         (latb.size - 1, lonb.size - 1))
    r.resample(y)  # warm
    t0 = time.perf_counter()
    for _ in range(n_repeat_apply):
        r.resample(y)
    t_apply = (time.perf_counter() - t0) / n_repeat_apply
    return {
        "source_grid": f"{dlon}deg",
        "n_source": (lonb.size - 1) * (latb.size - 1),
        "healpix_level": level,
        "n_target": int(r.cell_ids.size),
        "nnz_weights": int(r.weights.nnz),
        "weights_mib": round(
            (r.weights.data.nbytes + r.weights.indices.nbytes
             + r.weights.indptr.nbytes) / 2 ** 20, 2),
        "build_s": round(t_build, 3),
        "apply_s": round(t_apply, 5),
    }


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for dlon, level, norm in [(1.0, 6, "destination"), (1.0, 7, "destination"),
                              (0.5, 7, "destination"), (1.0, 6, "covered")]:
        _, m = property_metrics(dlon, dlon, level, norm)
        rows.append({"case": f"global_{dlon}deg_L{level}_{norm}", **m,
                     "quad_points": "", "quad_max_absdiff": "",
                     "quad_rms_absdiff": "", "quad_rel_to_range": ""})
        print(rows[-1])

    q = quadrature_crosscheck(2.0, 4)
    rows.append({"case": "quadrature_oracle_2deg_L4", "eps_global": "",
                 "eps_const": "", "coverage_err": "", "within_bounds": "",
                 "n_cells": "", "nnz": "", **q})
    print(rows[-1])

    fields = list(rows[0].keys())
    with (TABLE_DIR / "overlap_conservative_validation.csv").open(
            "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    bench = [benchmark(1.0, 6), benchmark(1.0, 7), benchmark(0.5, 7)]
    with (TABLE_DIR / "overlap_conservative_benchmark.csv").open(
            "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bench[0].keys()))
        w.writeheader()
        w.writerows(bench)
    for b in bench:
        print(b)
    print(f"\nWrote {TABLE_DIR / 'overlap_conservative_validation.csv'}")
    print(f"Wrote {TABLE_DIR / 'overlap_conservative_benchmark.csv'}")


if __name__ == "__main__":
    main()
