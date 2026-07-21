from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

import numpy as np
import torch


T_Array = TypeVar("T_Array", np.ndarray, torch.Tensor)


def estimate_pixel_area(
    lon_deg,
    lat_deg,
    radius: float = 6371000.0,
    min_points_per_ring: float = 3.0,
) -> Optional[np.ndarray]:
    """Estimate each sample's pixel area from shared-latitude "ring" structure.

    Many native lon/lat grids -- regular lat/lon grids, and reduced Gaussian
    grids such as ECMWF's N-grids (see the ERA5 tutorial) -- have samples that
    share *exact* latitude values, with the number of longitude points per
    ring possibly varying (e.g. shrinking towards the poles). When that
    structure is present, each sample's cell area can be computed exactly as
    a spherical zone: the latitudinal width from the midpoints with
    neighbouring rings, times the longitudinal width implied by how many
    samples share that ring.

    This does **not** detect grids that are regular in a different projection
    (e.g. a UTM pixel grid): after reprojection to lon/lat, such grids
    generally have no two samples sharing an exact latitude, so no ring
    structure is found.

    Parameters
    ----------
    lon_deg, lat_deg : array-like, shape (N,)
        Sample coordinates in degrees.
    radius : float
        Sphere radius; the returned area is in the same squared units
        (default: metres, giving an area in m^2).
    min_points_per_ring : float
        Minimum *average* number of samples per unique latitude value for the
        ring structure to be considered genuine rather than incidental
        (default 3 -- comfortably above 1, which would just mean every
        sample has a numerically-unique latitude).

    Returns
    -------
    numpy.ndarray, shape (N,), or None
        Per-sample area estimate, or ``None`` if no ring structure was
        detected -- callers should then fall back to a uniform weight.
    """
    lat = np.asarray(lat_deg, dtype=np.float64).reshape(-1)
    lon = np.asarray(lon_deg, dtype=np.float64).reshape(-1) % 360.0
    N = lat.size
    if N == 0:
        return None

    lat_u, lat_inv, lat_counts = np.unique(lat, return_inverse=True, return_counts=True)
    if lat_u.size == 0 or N / lat_u.size < min_points_per_ring:
        return None

    n_lon_per_point = lat_counts[lat_inv]
    dlon_rad = 2.0 * np.pi / n_lon_per_point

    J = lat_u.size
    bounds = np.empty(J + 1)
    bounds[0], bounds[-1] = -90.0, 90.0
    bounds[1:-1] = (lat_u[:-1] + lat_u[1:]) / 2.0
    sin_bounds = np.sin(np.radians(bounds))
    dsin = (sin_bounds[1:] - sin_bounds[:-1])[lat_inv]

    return (radius ** 2) * dsin * dlon_rad


@dataclass(frozen=True)
class ResampleResults(Generic[T_Array]):
    """Proxy to resampling results.

    Attributes
    ----------
    cell_data : numpy.ndarray or torch.Tensor
        Data values resampled on HEALPix cells
    cell_ids : numpy.ndarray or torch.Tensor
        HEALPix cell ids.
    cg_residual_norms : numpy.ndarray or torch.Tensor or None
        Conjugate gradient residual norms (if any).
    cg_niters : numpy.ndarray or torch.Tensor or None
        Conjugate gradient number of iterations (if any).
    
    """
    cell_data: T_Array
    cell_ids: T_Array
    cg_residual_norms: T_Array | None = None
    cg_niters: T_Array | None = None
