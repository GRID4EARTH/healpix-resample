"""
utm.py

HEALPix -> UTM raster resampling.

Every resampler in this package is built *from* a set of lon/lat samples
*to* HEALPix cells (``resample()``), and exposes the opposite direction as
``invert()`` (HEALPix cells -> those same sample locations). Going from a
HEALPix field to a regular UTM raster is therefore just a matter of choosing
the samples to be the pixel centers of the target UTM grid:

1. build the target grid (`UTMGrid`): pixel-center eastings/northings in a
   projected CRS;
2. reproject the pixel centers to lon/lat (``pyproj``);
3. construct a resampler on those lon/lat points, restricted to the HEALPix
   cells that actually carry data (``out_cell_ids=cell_ids``);
4. call ``resampler.invert(cell_data)`` and reshape to ``(ny, nx)``.

`HealpixToUTM` packages steps 2-4 (the operator is built once and can be
applied to any number of fields / batches defined on the same cells), and
`healpix_to_utm` is the one-call convenience wrapper.

Which ``invert()`` is used
--------------------------
======================  ==================================================
``method``              operator
======================  ==================================================
``"nearest"``           `GroupByResampler.invert` -- each pixel takes the
                        value of the HEALPix cell that contains it.
``"bilinear"``          `BilinearResampler.invert`
``"bicubic"``           `BicubicResampler.invert`
``"clough_tocher"``     `CloughTocherResampler.invert` -- Delaunay /
                        Clough-Tocher C1 cubic interpolant of the cell
                        centers, evaluated at the pixel centers (exact for
                        an affine field, no extrapolation).
a resampler class       instantiated as ``cls(lon_deg=, lat_deg=, level=,
                        out_cell_ids=, ...)`` and used through ``invert()``.
======================  ==================================================

``"nearest"`` deliberately does **not** use `NearestResampler.invert`: that
method scatters each cell to its single nearest sample, so on a raster finer
than the HEALPix cells most pixels would receive nothing.

Pixels whose center does not fall in one of the cells of `cell_ids` (no
extrapolation beyond the data footprint), and, for Clough-Tocher, pixels
outside the convex hull of the cell centers, are returned as NaN.

``pyproj`` is an optional dependency, only needed by this module
(``pip install healpix-resample[utm]``).

Nothing here is specific to UTM beyond the helpers that pick a UTM zone:
any projected CRS accepted by ``pyproj`` can be used for the target grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch

import healpix_geo

from healpix_resample.base import T_Array
from healpix_resample.bicubic import BicubicResampler
from healpix_resample.bilinear import BilinearResampler
from healpix_resample.clough_tocher import CloughTocherResampler
from healpix_resample.groupby import GroupByResampler


def _pyproj():
    try:
        import pyproj
    except ImportError as exc:  # pragma: no cover - exercised only without pyproj
        raise ImportError(
            "healpix_resample.utm needs the optional dependency `pyproj` "
            "(pip install healpix-resample[utm], or `pyproj` from conda-forge "
            "in a conda/pixi environment)."
        ) from exc
    return pyproj


def _as_crs(crs):
    pyproj = _pyproj()
    if isinstance(crs, pyproj.CRS):
        return crs
    if isinstance(crs, (int, np.integer)):
        return pyproj.CRS.from_epsg(int(crs))
    return pyproj.CRS.from_user_input(crs)


def _to_numpy(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _cell_size_m(level: int, radius: float = 6371000.0) -> float:
    """Side of a square with the area of one HEALPix cell at `level` (m)."""
    return float(np.sqrt(4.0 * np.pi / (12.0 * 4.0 ** int(level))) * radius)


def utm_crs_from_lonlat(lon_deg, lat_deg):
    """WGS84 / UTM CRS of the zone containing the centroid of the points.

    The centroid is taken on the sphere (mean unit vector), so points
    straddling the antimeridian are handled. The standard 6-degree zones
    are used; the Norway / Svalbard exceptions are not applied.

    Returns
    -------
    pyproj.CRS
        ``EPSG:326zz`` (northern hemisphere) or ``EPSG:327zz`` (southern).
    """
    pyproj = _pyproj()
    lon = np.radians(np.asarray(_to_numpy(lon_deg), dtype=np.float64).reshape(-1))
    lat = np.radians(np.asarray(_to_numpy(lat_deg), dtype=np.float64).reshape(-1))
    if lon.size == 0:
        raise ValueError("utm_crs_from_lonlat: empty input.")
    clat = np.cos(lat)
    v = np.array([(clat * np.cos(lon)).mean(), (clat * np.sin(lon)).mean(), np.sin(lat).mean()])
    lon_c = np.degrees(np.arctan2(v[1], v[0]))
    lat_c = np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))
    zone = int(np.floor((lon_c + 180.0) / 6.0)) % 60 + 1
    return pyproj.CRS.from_epsg((32600 if lat_c >= 0.0 else 32700) + zone)


@dataclass(frozen=True)
class UTMGrid:
    """A regular raster in a projected (typically UTM) CRS.

    Attributes
    ----------
    crs : pyproj.CRS
        Projected CRS of the grid.
    x : numpy.ndarray, shape (nx,)
        Pixel-center eastings (CRS units, metres for UTM), ascending.
    y : numpy.ndarray, shape (ny,)
        Pixel-center northings. Descending (north-up, the usual raster
        convention) when built by `from_bounds` / `from_cell_ids`.
    """

    crs: Any
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "crs", _as_crs(self.crs))
        x = np.asarray(self.x, dtype=np.float64).reshape(-1)
        y = np.asarray(self.y, dtype=np.float64).reshape(-1)
        if x.size == 0 or y.size == 0:
            raise ValueError("UTMGrid: x and y must be non-empty.")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    @property
    def shape(self) -> Tuple[int, int]:
        """``(ny, nx)``."""
        return (int(self.y.size), int(self.x.size))

    @classmethod
    def from_bounds(cls, crs, bounds, resolution: float) -> "UTMGrid":
        """Grid covering ``bounds = (xmin, ymin, xmax, ymax)`` (outer pixel
        *edges*) with square pixels of side `resolution`. The upper-left
        corner is kept exactly; the extent is rounded up to a whole number
        of pixels."""
        xmin, ymin, xmax, ymax = (float(b) for b in bounds)
        res = float(resolution)
        if res <= 0:
            raise ValueError("UTMGrid.from_bounds: resolution must be > 0.")
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("UTMGrid.from_bounds: empty bounds.")
        nx = max(1, int(np.ceil((xmax - xmin) / res - 1e-9)))
        ny = max(1, int(np.ceil((ymax - ymin) / res - 1e-9)))
        x = xmin + (np.arange(nx) + 0.5) * res
        y = ymax - (np.arange(ny) + 0.5) * res
        return cls(crs=crs, x=x, y=y)

    @classmethod
    def from_cell_ids(
        cls,
        cell_ids,
        level: int,
        resolution: Optional[float] = None,
        *,
        crs=None,
        nest: bool = True,
        ellipsoid: str = "WGS84",
    ) -> "UTMGrid":
        """Grid covering the bounding box of a set of HEALPix cells.

        Parameters
        ----------
        cell_ids : array-like
            HEALPix cell ids at `level`.
        resolution : float, optional
            Pixel size in CRS units. Defaults to the HEALPix cell size at
            `level` (square root of the cell area).
        crs : optional
            Target CRS (EPSG code, string or ``pyproj.CRS``). Defaults to
            the UTM zone of the cells' centroid (`utm_crs_from_lonlat`).

        The bounds are those of the cell *centers*, snapped outwards to a
        multiple of `resolution`.
        """
        pyproj = _pyproj()
        ids = _to_numpy(cell_ids).astype(np.uint64).reshape(-1)
        hp = healpix_geo.nested if nest else healpix_geo.ring
        lon, lat = hp.healpix_to_lonlat(ids, int(level), ellipsoid=ellipsoid)
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        crs = utm_crs_from_lonlat(lon, lat) if crs is None else _as_crs(crs)
        res = _cell_size_m(level) if resolution is None else float(resolution)
        fwd = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs, always_xy=True)
        x, y = fwd.transform(lon, lat)
        xmin = np.floor(np.min(x) / res) * res
        ymin = np.floor(np.min(y) / res) * res
        xmax = np.ceil(np.max(x) / res) * res
        ymax = np.ceil(np.max(y) / res) * res
        return cls.from_bounds(crs, (xmin, ymin, max(xmax, xmin + res), max(ymax, ymin + res)), res)

    def lonlat(self) -> Tuple[np.ndarray, np.ndarray]:
        """WGS84 longitude / latitude of the pixel centers, each ``(ny, nx)``
        (degrees, longitude in [-180, 180])."""
        pyproj = _pyproj()
        inv = pyproj.Transformer.from_crs(self.crs, pyproj.CRS.from_epsg(4326), always_xy=True)
        X, Y = np.meshgrid(self.x, self.y)
        lon, lat = inv.transform(X, Y)
        return np.asarray(lon), np.asarray(lat)

    def padded(self, pad: int) -> "UTMGrid":
        """Same grid extended by `pad` pixels on every side (needs a
        regularly spaced axis of at least two pixels to extrapolate)."""
        pad = int(pad)
        if pad <= 0:
            return self

        def _extend(a, name):
            if a.size < 2:
                raise ValueError(f"UTMGrid.padded: cannot pad a single-pixel {name} axis.")
            step = a[1] - a[0]
            if not np.allclose(np.diff(a), step, rtol=1e-6, atol=0.0):
                raise ValueError(f"UTMGrid.padded: the {name} axis is not regularly spaced.")
            before = a[0] - step * np.arange(pad, 0, -1)
            after = a[-1] + step * np.arange(1, pad + 1)
            return np.concatenate([before, a, after])

        return UTMGrid(crs=self.crs, x=_extend(self.x, "x"), y=_extend(self.y, "y"))


_METHODS = {
    "nearest": GroupByResampler,
    "bilinear": BilinearResampler,
    "bicubic": BicubicResampler,
    "clough_tocher": CloughTocherResampler,
}


class HealpixToUTM:
    """Resample HEALPix cell data onto a UTM raster through ``invert()``.

    The operator is geometry-only: it is built once from the HEALPix cells
    and the target grid, and `resample()` can then be applied to any field
    (or batch of fields) defined on those cells.

    Parameters
    ----------
    cell_ids : array-like, shape (K,)
        HEALPix cell ids (at `level`) on which the data are defined. Must be
        unique; any order.
    level : int
        HEALPix level of `cell_ids`.
    grid : UTMGrid
        Target raster.
    method : {"bilinear", "nearest", "bicubic", "clough_tocher"} or class
        Which resampler's ``invert()`` to use -- see the module docstring.
        A resampler class can be passed directly; it must accept
        ``lon_deg, lat_deg, level, out_cell_ids`` and implement ``invert()``.
    nest : bool
        HEALPix indexing scheme of `cell_ids`.
    ellipsoid : str
        Passed through to the resampler (``healpix_geo``).
    pad : int, optional
        Number of extra pixels added around the grid while building the
        operator, cropped from the output. Only useful for Clough-Tocher,
        whose forward construction keeps a cell only if it lies inside the
        convex hull of the *samples* (here: the pixel centers): without a
        margin the cells surrounding the raster would be discarded and the
        border pixels could not be interpolated. Default: two HEALPix cell
        widths for ``"clough_tocher"``, 0 otherwise.
    dtype, device, verbose
        Passed to the resampler.
    **resampler_kwargs
        Extra keyword arguments for the resampler constructor (e.g.
        ``Npt``, ``threshold``).

    Attributes
    ----------
    grid : UTMGrid
    resampler
        The underlying resampler (built on the padded pixel centers).
    valid : numpy.ndarray, bool, shape (ny, nx)
        Pixels whose center lies in a cell of `cell_ids` and that the
        operator can interpolate; the others are NaN in every output.
    """

    def __init__(
        self,
        cell_ids: T_Array,
        level: int,
        grid: UTMGrid,
        *,
        method: Union[str, type] = "bilinear",
        nest: bool = True,
        ellipsoid: str = "WGS84",
        pad: Optional[int] = None,
        dtype: torch.dtype = torch.float64,
        device: Optional[Union[torch.device, str]] = None,
        verbose: bool = False,
        **resampler_kwargs,
    ) -> None:
        if not isinstance(grid, UTMGrid):
            raise TypeError("HealpixToUTM: `grid` must be a UTMGrid.")
        self.grid = grid
        self.level = int(level)
        self.nest = bool(nest)

        if isinstance(method, str):
            if method not in _METHODS:
                raise ValueError(
                    f"HealpixToUTM: unknown method {method!r}; expected one of "
                    f"{sorted(_METHODS)} or a resampler class."
                )
            resampler_cls = _METHODS[method]
        else:
            resampler_cls = method
        self.method = method

        ids = _to_numpy(cell_ids).astype(np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("HealpixToUTM: `cell_ids` is empty.")
        order = np.argsort(ids, kind="stable")
        ids_sorted = ids[order]
        if np.any(ids_sorted[1:] == ids_sorted[:-1]):
            raise ValueError("HealpixToUTM: `cell_ids` must be unique.")
        self.K = int(ids.size)

        # ---- pixel centers (optionally padded) -> lon/lat ------------------
        if pad is None:
            if resampler_cls is CloughTocherResampler and min(grid.shape) >= 2:
                step = min(abs(grid.x[1] - grid.x[0]), abs(grid.y[1] - grid.y[0]))
                pad = int(np.ceil(2.0 * _cell_size_m(self.level) / step))
            else:
                pad = 0
        self.pad = int(pad)
        work = grid.padded(self.pad)
        self._work_shape = work.shape
        lon, lat = work.lonlat()

        # ---- resampler on the pixel centers --------------------------------
        kwargs = dict(
            lon_deg=lon.reshape(-1),
            lat_deg=lat.reshape(-1),
            level=self.level,
            nest=self.nest,
            ellipsoid=ellipsoid,
            dtype=dtype,
            device=device,
            verbose=verbose,
        )
        kwargs.update(resampler_kwargs)
        if not (isinstance(resampler_cls, type) and issubclass(resampler_cls, GroupByResampler)):
            # Group-by resamplers bin each pixel into its own cell and do not
            # take `out_cell_ids`; cells without data are masked below.
            kwargs.setdefault("out_cell_ids", ids)
        self.resampler = resampler_cls(**kwargs)

        # ---- align the caller's cell order with the resampler's ------------
        rs_ids = np.asarray(self.resampler.get_cell_ids()).astype(np.int64).reshape(-1)
        pos = np.clip(np.searchsorted(ids_sorted, rs_ids), 0, self.K - 1)
        found = ids_sorted[pos] == rs_ids
        dev = self.resampler.device
        self._sel = torch.as_tensor(order[pos], dtype=torch.long, device=dev)
        self._missing = torch.as_tensor(~found, device=dev)

        # ---- pixels actually supported by input cells -----------------------
        # invert(1) is the per-pixel weight sum: ~1 where the pixel is
        # interpolated from data cells, 0 where no cell reaches it, NaN where
        # the operator itself declines (Clough-Tocher outside the hull, or a
        # cell absent from `cell_ids`).
        ones = torch.ones(rs_ids.size, dtype=self.resampler.dtype, device=dev)
        ones[self._missing] = float("nan")
        cov = self.resampler.invert(ones)
        # On top of that, a pixel is only kept if its center lies in a cell
        # that carries data: the KNN-based operators would otherwise
        # extrapolate a few cells beyond the footprint of `cell_ids`.
        hp = healpix_geo.nested if self.nest else healpix_geo.ring
        pix_cell = np.asarray(
            hp.lonlat_to_healpix(lon.reshape(-1), lat.reshape(-1), self.level, ellipsoid=ellipsoid)
        ).astype(np.int64)
        ppos = np.clip(np.searchsorted(ids_sorted, pix_cell), 0, self.K - 1)
        in_footprint = torch.as_tensor(ids_sorted[ppos] == pix_cell, device=dev)
        self._valid_flat = torch.isfinite(cov) & (cov > 0.5) & in_footprint
        self.valid = self._crop(self._valid_flat.reshape(self._work_shape)).cpu().numpy()

    def _crop(self, a):
        if self.pad == 0:
            return a
        p = self.pad
        return a[..., p:-p, p:-p]

    @torch.no_grad()
    def resample(self, cell_data: T_Array) -> T_Array:
        """HEALPix cells -> UTM raster.

        Args:
            cell_data: (K,) or (B, K) values, in the order of the `cell_ids`
                given at construction. NumPy array or torch tensor.

        Returns:
            (ny, nx) or (B, ny, nx), same array type as `cell_data`, in the
            resampler's floating dtype. Unsupported pixels are NaN (see
            `valid`). NaN cells propagate to the pixels they contribute to.
        """
        rs = self.resampler
        y = cell_data if isinstance(cell_data, torch.Tensor) else torch.as_tensor(np.asarray(cell_data))
        y = y.to(rs.device, dtype=rs.dtype)
        squeezed = y.ndim == 1
        if squeezed:
            y = y[None, :]
        if y.ndim != 2 or y.shape[-1] != self.K:
            raise ValueError(
                f"HealpixToUTM.resample: expected cell_data of shape (K,) or "
                f"(B, K) with K={self.K}, got {tuple(cell_data.shape)}."
            )

        aligned = y[:, self._sel]
        aligned[:, self._missing] = float("nan")

        out = rs.invert(aligned)  # (B, ny_work * nx_work)
        out = torch.where(self._valid_flat[None, :], out, torch.full_like(out, float("nan")))
        out = self._crop(out.reshape((out.shape[0],) + self._work_shape))

        if squeezed:
            out = out[0]
        if not isinstance(cell_data, torch.Tensor):
            out = out.cpu().numpy()
        return out


def healpix_to_utm(
    cell_data: T_Array,
    cell_ids: T_Array,
    level: int,
    *,
    grid: Optional[UTMGrid] = None,
    resolution: Optional[float] = None,
    crs=None,
    method: Union[str, type] = "bilinear",
    nest: bool = True,
    ellipsoid: str = "WGS84",
    **kwargs,
) -> Tuple[T_Array, UTMGrid]:
    """Resample a HEALPix field onto a UTM raster in one call.

    If `grid` is not given, it is built with `UTMGrid.from_cell_ids` from
    the bounding box of the cells, using `resolution` (default: the HEALPix
    cell size) and `crs` (default: the UTM zone of the cells' centroid).

    Returns
    -------
    data : (ny, nx) or (B, ny, nx)
        Raster values, same array type as `cell_data`.
    grid : UTMGrid
        The grid the data are defined on (``grid.x``, ``grid.y``,
        ``grid.crs``).

    To apply the same geometry to several fields, build a `HealpixToUTM`
    once and call its ``resample()`` instead.
    """
    if grid is None:
        grid = UTMGrid.from_cell_ids(
            cell_ids, level, resolution, crs=crs, nest=nest, ellipsoid=ellipsoid
        )
    elif resolution is not None or crs is not None:
        raise ValueError("healpix_to_utm: pass either `grid` or `resolution`/`crs`, not both.")
    op = HealpixToUTM(
        cell_ids, level, grid, method=method, nest=nest, ellipsoid=ellipsoid, **kwargs
    )
    return op.resample(cell_data), grid
