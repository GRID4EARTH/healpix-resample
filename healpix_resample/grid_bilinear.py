"""
grid_bilinear.py

Bilinear interpolation of regular lon/lat grids at HEALPix cell centres
("gather" bilinear): each output cell takes the value of the bilinear surface
through the 4 input grid nodes around its centre, as
``scipy.interpolate.RegularGridInterpolator(method="linear")``.

Unlike :class:`~healpix_resample.bilinear.BilinearResampler` (each sample
linked to its 4 nearest cells, inverse-distance weights normalised per cell),
the result is exact for a field that is linear in lon/lat, involves no
nearest-neighbour search (no distance ties), and costs O(K) for K cells.

The operator is a sparse ``(N, K)`` matrix with at most 4 entries per cell,
so ``resample`` is one sparse matmul, as for the other resamplers.
"""
from __future__ import annotations

from typing import Optional, Sequence

import healpix_geo
import numpy as np
import torch

from healpix_resample.base import ResampleResults, T_Array


def _as_bands(axis) -> list[np.ndarray]:
    if isinstance(axis, (list, tuple)):
        return [np.asarray(a, dtype=np.float64).reshape(-1) for a in axis]
    return [np.asarray(axis, dtype=np.float64).reshape(-1)]


def _lon_axis(lon: np.ndarray):
    """Sort a longitude axis into an ascending, unwrapped one.

    Returns ``(lon_sorted, columns, periodic)``: ``columns`` maps sorted
    positions to the columns of ``lon``; the axis starts after its largest gap,
    so windows crossing the antimeridian (e.g. -180 .. -179.7, 179.7 .. 180)
    become one ascending axis. ``periodic`` is True when the axis covers the
    whole circle (largest gap not wider than 1.5 steps).
    """
    wrapped = lon % 360.0
    order = np.argsort(wrapped, kind="stable")
    ws = wrapped[order]
    if lon.size == 1:
        return ws, order, False
    dup = np.flatnonzero(np.diff(ws) <= 0)
    if dup.size:
        # a global axis that lists both ends (-180 and 180, or 0 and 360):
        # the two columns are the same meridian, the later one is not used
        a, b = order[dup], order[dup + 1]
        if dup.size > 1 or not np.isclose(abs(lon[a[0]] - lon[b[0]]), 360.0):
            raise ValueError("longitude axis has duplicate values")
        keep = np.ones(ws.size, dtype=bool)
        keep[dup + (b > a)] = False
        order, ws = order[keep], ws[keep]
    gaps = np.diff(ws, append=ws[0] + 360.0)
    g = int(np.argmax(gaps))
    order = np.roll(order, -(g + 1))
    ws = np.roll(ws, -(g + 1))
    ws = ws[0] + (ws - ws[0]) % 360.0
    step = np.median(np.diff(ws))
    periodic = gaps[g] <= 1.5 * step
    return ws, order, bool(periodic)


class GridBilinearResampler:
    """Bilinear interpolation of lon/lat grids at HEALPix cell centres.

    Parameters
    ----------
    lon_deg, lat_deg : array-like
        1-D longitude and latitude axes of the input grid, in any order
        (ascending, descending, or crossing the antimeridian). Several grids
        ("bands" of rows with their own longitude axis, e.g. the 1" and 1.5"
        tiles of the Copernicus DEM on both sides of 50°N) are given as two
        lists of 1-D axes of the same length; they must not overlap in latitude.
        Cells between the last row of a band and the first row of the next
        one are interpolated between those two rows, each along its own
        longitude axis, when the gap is not wider than 1.5 row steps of the
        two bands; cells in a wider gap are NaN.
    level : int
        HEALPix level of the output. Meant for levels whose cells are not
        larger than the grid step: each cell takes the value at its centre, so
        at coarser levels the result aliases (use block means of a finer
        level, or a conservative resampler, there).
    out_cell_ids : array-like, optional
        Output cells. Default: the cells around the grid nodes, which include
        every cell whose centre is inside the grid (the others are NaN).
    nest : bool
        Nested (default) or ring indexing, as for the other resamplers.
    ellipsoid : str
        Ellipsoid of the cell centres, as for the other resamplers.
    min_weight : float
        Cells whose valid (non-NaN, inside the grid) corner weights sum to
        less than this are NaN; the others are renormalised over their valid
        corners. ``1.0`` gives NaN as soon as one corner is NaN (as scipy);
        the default 0.5 extends valid areas by up to half a pixel.

    The values given to :meth:`resample` are, for each band in the order
    given, the 2-D ``(lat, lon)`` array of that band (axes in the order
    given) flattened in C order, concatenated: ``N = sum(nlat * nlon)``.
    """

    def __init__(
        self,
        lon_deg,
        lat_deg,
        level: int,
        *,
        out_cell_ids: Optional[T_Array] = None,
        nest: bool = True,
        ellipsoid: str = "WGS84",
        min_weight: float = 0.5,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        verbose: bool = False,
    ):
        lons, lats = _as_bands(lon_deg), _as_bands(lat_deg)
        if len(lons) != len(lats):
            raise ValueError("lon_deg and lat_deg must give the same number of bands")
        self.level = int(level)
        self.nest = bool(nest)
        self.ellipsoid = str(ellipsoid)
        self.min_weight = float(min_weight)
        self.device = torch.device(device)
        self.dtype = dtype
        hp = healpix_geo.nested if self.nest else healpix_geo.ring

        # --- rows of every band, sorted by latitude
        offsets = np.cumsum([0] + [la.size * lo.size for la, lo in zip(lats, lons)])
        self.N = int(offsets[-1])
        row_lat, row_band, row_index = [], [], []
        self._lon = []
        band_step = np.zeros(len(lats))  # largest row step of each band
        for b, (la, lo) in enumerate(zip(lats, lons)):
            order = np.argsort(la, kind="stable")
            if np.any(np.diff(la[order]) <= 0):
                raise ValueError("latitude axis has duplicate values")
            band_step[b] = np.diff(la[order]).max(initial=0.0)
            row_lat.append(la[order])
            row_band.append(np.full(la.size, b))
            row_index.append(order)
            self._lon.append(_lon_axis(lo))
        row_lat = np.concatenate(row_lat)
        order = np.argsort(row_lat, kind="stable")
        row_lat = row_lat[order]
        if np.any(np.diff(row_lat) <= 0):
            raise ValueError("bands overlap in latitude")
        row_band = np.concatenate(row_band)[order]
        row_index = np.concatenate(row_index)[order]

        # --- output cells and their centres
        if out_cell_ids is None:
            cell_ids = self._cells_in_grid(lats, lons, band_step)
        else:
            cell_ids = np.asarray(
                out_cell_ids.cpu().numpy() if isinstance(out_cell_ids, torch.Tensor) else out_cell_ids
            ).astype(np.int64).reshape(-1)
        clon, clat = hp.healpix_to_lonlat(cell_ids.astype(np.uint64), self.level, ellipsoid=self.ellipsoid)
        clon, clat = np.asarray(clon, np.float64), np.asarray(clat, np.float64)
        K = cell_ids.size

        # --- the 2 rows around each centre, the 2 columns on each row
        i = np.searchsorted(row_lat, clat, side="right") - 1
        inside = ((i >= 0) & (i < row_lat.size - 1)) | (clat == row_lat[-1])
        i = np.clip(i, 0, max(row_lat.size - 2, 0))
        if row_lat.size > 1:
            # between two bands only across a gap of about one row (the 1" and
            # 1.5" tiles at 50 N): a wider gap is missing data, not a row step
            a, b = row_band[i], row_band[i + 1]
            gap = row_lat[i + 1] - row_lat[i]
            inside &= (a == b) | (gap <= 1.5 * np.maximum(band_step[a], band_step[b]))
        if row_lat.size > 1:
            s = (clat - row_lat[i]) / (row_lat[i + 1] - row_lat[i])
        else:
            s = np.zeros(K)
        # (K, 4) node index and weight of each corner (-1 / 0: no link)
        node = np.full((K, 4), -1, dtype=np.int64)
        weight = np.zeros((K, 4))
        for p, (rows, ws) in enumerate(((i, 1.0 - s), (np.minimum(i + 1, row_lat.size - 1), s))):
            for b in range(len(lons)):
                m = inside & (row_band[rows] == b)
                if not m.any():
                    continue
                lon_s, cols, periodic = self._lon[b]
                if periodic:
                    lon_s = np.append(lon_s, lon_s[0] + 360.0)
                    cols = np.append(cols, cols[0])
                x = lon_s[0] + (clon[m] - lon_s[0]) % 360.0
                j = np.searchsorted(lon_s, x, side="right") - 1
                ok = ((j >= 0) & (j < lon_s.size - 1)) | (x == lon_s[-1])
                j = np.clip(j, 0, max(lon_s.size - 2, 0))
                t = (x - lon_s[j]) / (lon_s[j + 1] - lon_s[j]) if lon_s.size > 1 else np.zeros(j.size)
                base = offsets[b] + row_index[rows[m]] * lons[b].size
                for q, (col, w) in enumerate(((cols[j], 1.0 - t), (cols[np.minimum(j + 1, lon_s.size - 1)], t))):
                    w = np.where(ok, w * ws[m], 0.0)
                    node[m, 2 * p + q] = np.where(w > 0, base + col, -1)
                    weight[m, 2 * p + q] = w
        # CSC directly: one column per cell, rows (nodes) sorted in each column
        node = np.where(weight > 0, node, -1)
        order = np.argsort(np.where(node >= 0, node, np.iinfo(np.int64).max), axis=1)
        node = np.take_along_axis(node, order, axis=1)
        weight = np.take_along_axis(weight, order, axis=1)
        keep = node >= 0
        ccol = np.zeros(K + 1, dtype=np.int64)
        np.cumsum(keep.sum(1), out=ccol[1:])
        self.n_links = int(ccol[-1])
        self.M = torch.sparse_csc_tensor(
            torch.as_tensor(ccol), torch.as_tensor(node[keep]), torch.as_tensor(weight[keep]),
            size=(self.N, K),
        ).to(self.device, dtype=self.dtype)
        self.cell_ids = torch.as_tensor(cell_ids, dtype=torch.long, device=self.device)
        if verbose:
            print(f"[GridBilinearResampler] {self.N:,} grid nodes -> {K:,} cells, {self.n_links:,} links")

    def _cells_in_grid(self, lats, lons, band_step) -> np.ndarray:
        """Cells whose centre may be inside the grid (the operator gives NaN to
        those outside): the cells of the grid nodes and their neighbours, at a
        level whose cells are not smaller than the grid step, then their
        children at ``level``. Taking them at ``level`` directly would leave
        holes where the cells are much smaller than the grid step."""
        step = max(
            [band_step.max(initial=0.0)]
            + [np.diff(self._lon[b][0]).max(initial=0.0) for b in range(len(lons))]
        )
        coarse = self.level
        while coarse > 0 and np.degrees(np.sqrt(4 * np.pi / (12 * 4.0**coarse))) < step:
            coarse -= 1
        nested = healpix_geo.nested
        ids = []
        for la, lo in zip(lats, lons):
            LA, LO = np.meshgrid(la, lo, indexing="ij")
            c = np.unique(nested.lonlat_to_healpix(LO.ravel(), LA.ravel(), coarse, ellipsoid=self.ellipsoid))
            ids.append(np.asarray(nested.kth_neighbourhood(c, coarse, ring=1)).ravel())
        ids = np.unique(np.concatenate(ids))
        ids = ids[ids >= 0].astype(np.uint64)
        if coarse < self.level:
            ids = np.asarray(nested.zoom_to(ids, coarse, self.level)).ravel()
        if not self.nest:
            ids = np.asarray(nested.to_ring(ids, self.level))
        return np.unique(ids).astype(np.int64)

    @torch.no_grad()
    def resample(self, val: T_Array, **unused) -> ResampleResults[T_Array]:
        """Interpolate the grid values at the cell centres.

        Args:
            val: (N,) or (B, N) grid values (see the class docstring for the order).

        Returns:
            cell values (K,) or (B, K), NaN where the valid weight is below
            ``min_weight``.
        """
        y = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
        y = y.to(self.device, dtype=self.dtype)
        clean_shape = y.ndim == 1
        if clean_shape:
            y = y[None, :]
        valid = torch.isfinite(y)
        num = torch.where(valid, y, torch.zeros_like(y)) @ self.M
        den = valid.to(self.dtype) @ self.M
        # the corner weights of a cell sum to 1 up to the rounding of dtype
        tol = 8 * torch.finfo(self.dtype).eps
        hval = torch.where(den >= self.min_weight - tol, num / den, torch.full_like(num, float("nan")))
        cell_ids = self.cell_ids
        if not isinstance(val, torch.Tensor):
            hval = hval.cpu().numpy()
            cell_ids = cell_ids.cpu().numpy()
        if clean_shape:
            hval = hval[0]
        return ResampleResults(cell_data=hval, cell_ids=cell_ids)

    def get_cell_ids(self):
        return self.cell_ids.cpu().numpy()
