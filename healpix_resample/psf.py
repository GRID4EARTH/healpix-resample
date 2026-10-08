"""
psf.py

GPU-friendly sparse HEALPix regridding from unstructured lon/lat samples
to a subset of HEALPix pixels at a target resolution (nside = 2**level).

Core ideas:
- Use HEALPix local neighbourhoods (healpix_geo.kth_neighbourhood) to avoid N×npix distance matrices.
- Build sparse operators M (samples -> grid) and MT (grid -> samples) with Gaussian weights.
- Solve a damped least-squares problem with Conjugate Gradient (CG) on normal equations.

This module is designed for large N and batched values (B,N) on CUDA.
"""

from typing import Callable, Generic, Optional, Tuple, Dict

import math
import numpy as np
import torch

from healpix_resample.base import ResampleResults, T_Array, estimate_pixel_area
from healpix_resample.knn import KNeighborsResampler, _sigma_level_m, _lonlat_to_xyz


@torch.no_grad()
def conjugate_gradient(
    A_mv: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    verbose: bool = True,
    weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Solve A x = b with Conjugate Gradient, using only matvec A_mv(v).
    No autograd (uses torch.no_grad).

    ``A_mv`` (as built by :func:`least_squares_cg` for :class:`PSFResampler`) is
    self-adjoint and positive-definite with respect to the *weighted* inner
    product ``<u, v>_w = sum(u * v * weight)`` on the HEALPix-cell space
    (weight = per-cell column-weight ``Dx`` used to normalize ``M``), **not**
    with respect to the plain Euclidean inner product. Pass ``weight`` so CG's
    own dot products use the inner product the operator is actually SPD in —
    with ``weight=None`` (Euclidean), CG's classical convergence guarantees do
    not formally apply to this operator, even though it often still behaves
    reasonably in practice.

    Returns:
        x: solution
        info: dict with residual norms history, iterations
    """
    def _wdot(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # <u, v>_weight = sum_k u_k v_k weight_k, reduced over the last (K) axis, kept per batch row
        uv = u * v if weight is None else u * v * weight
        return torch.sum(uv, dim=-1)

    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    r = b - A_mv(x)          # residual
    p = r.clone()
    rs_old = _wdot(r, r)

    b_norm = torch.sqrt(torch.sum(_wdot(b, b)))
    if b_norm == 0:
        return x, {"residual_norms": torch.tensor([0.0], device=b.device, dtype=b.dtype),
                   "niters": torch.tensor(0, device=b.device)}

    residual_norms = [torch.sqrt(rs_old)]

    for k in range(max_iter):
        Ap = A_mv(p)
        denom = _wdot(p, Ap)
        if torch.max(denom.abs()) < 1e-30:
            break  # breakdown (shouldn't happen for SPD unless numerical issues)

        alpha = rs_old / denom
        x = x + torch.einsum('k,ki->ki',alpha,p)
        r = r - torch.einsum('k,ki->ki',alpha,Ap)
        rs_new = _wdot(r, r)

        residual_norms.append(torch.sqrt(rs_new))

        # stopping criterion: relative residual (in the same weighted norm as rs_new)
        if torch.max(torch.sqrt(rs_new)) <= tol * b_norm:
            rs_old = rs_new
            break

        beta = rs_new / rs_old
        p = r + torch.einsum('k,ki->ki',beta,p)
        rs_old = rs_new
        if k%4==0 and verbose:
            # rs_old has one entry per batch row (shape (B,), or 0-d when B==1);
            # print the worst-case (max) row so this works for both unbatched and
            # batched (B>1) inputs.
            print('Itt %d : %.4g'%(k, float(rs_old.max())))

    info = {
        "residual_norms": torch.stack(residual_norms),
        "niters": torch.tensor(len(residual_norms) - 1, device=b.device),
    }
    if verbose:
        print('Final Itt %d : %.4g'%(k, float(rs_old.max())))
    return x, info


@torch.no_grad()
def least_squares_cg(M,
        MT,
        y,
        x_ref,
        x0,
        max_iter = 200,
        tol = 1e-6,
        damp = 0.0,
        verbose: bool = True,
        weight: Optional[torch.Tensor] = None,
        ):
    """
    Solve for delta in a damped least-squares problem without forming dense matrices.

    ``M`` and ``MT`` are *not* Euclidean transposes of one another (each is
    normalized against a different axis of the raw weight matrix), but ``MT``
    is exactly the adjoint of ``M`` with respect to a pair of weighted inner
    products: ``<.,.>_Dy`` on the sample space (weight = per-sample row-sum
    used to normalize ``MT``) and ``<.,.>_Dx`` on the HEALPix-cell space
    (weight = per-cell column-sum used to normalize ``M``, passed here as
    ``weight``). Concretely ``MT = Dx @ M.T @ Dy^-1`` (in this module's
    row-vector convention). This solves the stationarity condition of the
    *weighted* least-squares problem

        delta_hat = argmin_delta || delta @ MT - r_ref ||^2_Dy + damp * || delta ||^2_Dx

    where ``r_ref = y - x_ref @ MT`` is the sample-space residual, which is
    exactly:

        (MT-then-M + damp*I) delta = (y - x_ref @ MT) @ M

    i.e. the same linear system as a naive (unweighted) Tikhonov normal
    equation, but its correct interpretation -- and the correct inner product
    for the Conjugate Gradient solver below -- uses ``Dx = weight`` (see
    :func:`conjugate_gradient`).

    Shapes:
        M      : (N, K) sparse CSR
        MT     : (K, N) sparse CSR
        y      : (B, N)
        x_ref  : (B, K)
        delta  : (B, K)
        weight : (K,) or None -- Dx, the per-cell weight columns of M were
        normalized by; None falls back to the (formally unjustified)
        Euclidean inner product.
    """

    # b = M^T y
    b = (y - x_ref@MT) @ M
    def A_mv(v: torch.Tensor) -> torch.Tensor:
        # (M^T M + damp I) v
        return (v@MT) @ M + damp * v

    x, info = conjugate_gradient(
        A_mv=A_mv, b=b, x0=x0, max_iter=max_iter, tol=tol, verbose=verbose, weight=weight,
    )
    return x, info


class PSFResampler(KNeighborsResampler, Generic[T_Array]):
    def __init__(
        self,
        lon_deg,
        lat_deg,
        level: int,
        *,
        out_cell_ids=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        verbose: bool = False,
        ellipsoid: str = "WGS84",
        Npt: int = 9,
        sigma_m=None,
        threshold: float = 0.1,
        area: Optional[T_Array] = None,
        fill_missing_out_cells: bool = False,
        **kwargs,
    ):
        """
        PSF regridding Set.

        Parameters
        ----------
        fill_missing_out_cells : bool
            Only relevant when ``out_cell_ids`` is supplied. Some requested
            output cells can end up with too little real kernel support (an
            empty or near-empty column of ``M``, or -- separately -- some
            input samples with too little real link weight in ``MT``) --
            e.g. when ``out_cell_ids`` force-includes cells the KNN search
            wouldn't have retained on its own. This originated as a targeted
            patch for a specific use case (river mouths in ocean models,
            where an approximate value was preferred over a gap), but the
            fallback it triggers is expensive: an unvectorized Python loop
            computing a full distance search per affected cell/sample
            (``comp_matrix``'s ``bad_k`` handling), which can be very slow
            for more than a handful of affected cells -- exactly the
            situation you can hit when combining ``out_cell_ids`` with
            :func:`~healpix_resample.subset_for_parent_cell`.

            - ``False`` (the default): skip that fallback entirely. Output
              cells that don't have enough real support are left as ``nan``
              in ``resample()``'s result rather than filled with an
              approximate nearest-single-sample value -- correct, and fast.
            - ``True``: restore the original fallback-fill behaviour exactly
              (approximate but non-NaN values for weakly-supported cells).
              Opt into this only if you specifically need a value rather
              than a gap for those cells, and can tolerate the extra
              construction cost when many cells/samples are affected.
        area : array-like, "auto", or None
            Per-sample pixel area/weight of the *native* (source) grid, shape
            ``(N,)``. This is baked directly into the source-to-HEALPix
            operator ``M`` (a.k.a. ``B`` in the accompanying paper) -- larger
            source pixels contribute proportionally more to a HEALPix cell's
            reconstructed value -- making the reconstruction a *conservative
            rebinning* rather than a plain local average. The HEALPix side
            needs no such weight since HEALPix cells are equal-area
            (iso-surface) by construction. This does not by itself guarantee
            *exact* global conservation (see ``resample(..., conservative=True)``
            for that); it removes the local bias a plain unweighted average
            would otherwise introduce.

            - If omitted (``None``, the default) or ``"auto"``: the area is
              estimated automatically from the grid's geometry, assuming
              samples share latitude "rings" as in regular lat/lon grids or
              reduced Gaussian grids (e.g. ECMWF's N-grids -- see
              :func:`~healpix_resample.base.estimate_pixel_area`). If no such
              structure is detected (e.g. a grid regular in a different
              projection such as UTM, or scattered points), silently falls
              back to a uniform weight of ``1.0`` per sample -- the same as
              the current, unweighted behaviour.
            - If an explicit array: used as-is (own convention/units; only
              ratios matter).
        """
        # Ensure ring_search_max >= ring_search_init(Npt) so the KNN search
        # loop in healpix_weighted_nearest actually executes.
        #
        # healpix_weighted_nearest computes:
        #   r_min            = ceil((sqrt(Npt) - 1) / 2)
        #   ring_search_init = max(1, r_min + 1)
        #
        # KNeighborsResampler's default ring_search_max=2 is too small for
        # Npt >= 16 (needs ring_search_init=3) -- when that happens the
        # search loop's `while r <= ring_search_max` condition is false
        # before its first iteration, every sample's `hi` stays at -1 for
        # every slot, and *every* retained cell then looks "never selected
        # by any sample's own Npt-nearest search" and gets pruned --
        # surfacing as KNeighborsResampler's "dropping N retained cell(s)"
        # warning at N == essentially the whole operator. BicubicResampler
        # and NearestResampler already auto-correct this (see their
        # __init__); PSFResampler didn't, which went unnoticed while every
        # caller used the historical default Npt=9 (small enough that
        # ring_search_init=2 == the old ring_search_max=2 by coincidence).
        # It starts mattering as soon as Npt grows past ~9 -- e.g. via
        # healpix_resample.recommend_npt() for a PSF wider than one cell.
        # Auto-correct here only when the caller hasn't supplied
        # ring_search_max explicitly -- copied from
        # NearestResampler.__init__ (nearest.py:67-84).
        if "ring_search_max" not in kwargs:
            r_min = int(math.ceil((math.sqrt(Npt) - 1.0) / 2.0))
            ring_search_init_needed = max(1, r_min + 1)
            # +2 margin so the loop has room to grow and find Npt candidates
            kwargs["ring_search_max"] = ring_search_init_needed + 2

        N = lon_deg.numel() if isinstance(lon_deg, torch.Tensor) else len(lon_deg)
        if device is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)
        radius = float(kwargs.get("radius", 6371000.0))

        if area is None or (isinstance(area, str) and area == "auto"):
            estimated = estimate_pixel_area(lon_deg, lat_deg, radius=radius)
            if estimated is not None:
                if verbose:
                    print(
                        "[PSFResampler] area not provided: auto-estimated from "
                        "shared-latitude-ring grid structure."
                    )
                area_t = torch.as_tensor(estimated, dtype=dtype, device=dev)
            else:
                if verbose:
                    print(
                        "[PSFResampler] area not provided and no shared-latitude-ring "
                        "structure detected: falling back to a uniform weight of 1.0 "
                        "per sample."
                    )
                area_t = torch.ones(N, dtype=dtype, device=dev)
        else:
            area_t = area if isinstance(area, torch.Tensor) else torch.as_tensor(area)
            area_t = area_t.to(dev, dtype=dtype).reshape(-1)
            if area_t.numel() != N:
                raise ValueError(
                    f"area must have {N} elements (one per sample), got {area_t.numel()}"
                )
        # Set before super().__init__() -- KNeighborsResampler.__init__ calls
        # comp_matrix() internally at construction time, and comp_matrix()
        # (below) needs self.area and self.fill_missing_out_cells to already
        # exist.
        self.area = area_t
        self.fill_missing_out_cells = bool(fill_missing_out_cells)

        super().__init__(
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            level=level,
            out_cell_ids=out_cell_ids,
            device=device,
            dtype=dtype,
            verbose=verbose,
            ellipsoid=ellipsoid,
            Npt=Npt,
            sigma_m=sigma_m,
            threshold=threshold,
            **kwargs,
        )

    def comp_matrix(self):
        # --- weights per sample->cell link
        # w = exp(-2*d^2/sigma^2)
        w_raw = torch.exp((-2.0) * (self.d_m * self.d_m) / (self.sigma_m * self.sigma_m))

        # Conservative rebinning: bake the native-grid (source) pixel area into
        # the raw weights before normalization. Multiplying by self.area here
        # (uniform 1.0 if not supplied -- see __init__) and letting M and MT
        # both be built from these area-weighted raw weights, rather than only
        # weighting M, is what keeps MT == the *unweighted* adjoint of M w.r.t.
        # the plain (non-area) construction: the per-sample area factor cancels
        # out of MT's own per-sample normalization algebraically (norm_row[i]
        # gains a factor of area[i], which then divides back out), leaving MT
        # unchanged while M becomes properly area-weighted -- exactly matching
        # "the output HEALPix grid is iso-surface, only the input needs
        # weighting". This also preserves the Dx-weighted adjoint relation
        # MT = Dx @ M.T @ Dy^-1 (see conjugate_gradient()/least_squares_cg()),
        # since that relation holds for *any* nonnegative raw weight matrix,
        # area-weighted or not.
        w = w_raw * self.area.unsqueeze(1)

        # Build (N,K) operator M and (K,N) operator MT.
        # We avoid numpy bincount; use torch.bincount on GPU.

        # idx: (N,Npt) row indices 0..N-1
        idx = torch.arange(self.N, device=self.device, dtype=torch.long)[:, None].expand(self.N, self.Npt)

        # -------- M : (N,K)  (normalized per column / per healpix cell)
        # norm_col[k] = sum_{i links to k} w[i,k]
        flat_hi = self.hi.reshape(-1)
        flat_w = w.reshape(-1)
        flat_w_raw = w_raw.reshape(-1)
        valid = flat_hi >= 0
        flat_hi_v = flat_hi[valid]
        flat_w_v = flat_w[valid]
        flat_w_raw_v = flat_w_raw[valid]

        norm_col = torch.bincount(flat_hi_v, weights=flat_w_v, minlength=self.K).to(self.dtype)
        # norm_col_raw: the *area-independent* accumulated weight per cell --
        # used below (not norm_col) for the out_cell_ids support threshold, so
        # `threshold`'s meaning ("enough kernel support") doesn't silently
        # rescale with the units/magnitude of `area`.
        norm_col_raw = torch.bincount(flat_hi_v, weights=flat_w_raw_v, minlength=self.K).to(self.dtype)
        # weight divided by column sum
        wM = flat_w_v / norm_col[flat_hi_v]

        rowsM = idx.reshape(-1)[valid]
        colsM = flat_hi_v
        indicesM = torch.stack([rowsM, colsM], dim=0)
        M_coo = torch.sparse_coo_tensor(
            indicesM,
            wM.to(self.dtype),
            size=(self.N, self.K),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()

        # --- after initial M_coo = ... .coalesce()

        # -------- MT : (K,N) (normalized per row / per input sample)
        # norm_row[i] = sum_{k links from i} w[i,k]
        flat_idx = idx.reshape(-1)
        flat_idx_v = flat_idx[valid]
        norm_row = torch.bincount(flat_idx_v, weights=flat_w_v, minlength=self.N).to(self.dtype)
        norm_row_raw = torch.bincount(flat_idx_v, weights=flat_w_raw_v, minlength=self.N).to(self.dtype)
        wMT = flat_w_v / norm_row[flat_idx_v]
            
        indicesMT = torch.stack([colsM, rowsM], dim=0)  # (hi, idx)
        MT_coo = torch.sparse_coo_tensor(
                indicesMT,
                wMT.to(self.dtype),
                size=(self.K, self.N),
                device=self.device,
                dtype=self.dtype,
            ).coalesce()
            
        
        cell_out_ids = getattr(self, "cell_out_ids", None)
        if cell_out_ids is None:
            cell_out_ids = getattr(self, "out_cell_ids", None)

        # Uncomputable output cells (requested via out_cell_ids but with too
        # little real kernel support to trust): resample() forces these to
        # NaN, unless fill_missing_out_cells=True restores the old
        # approximate-fallback behaviour instead. None when there's nothing
        # to flag (no out_cell_ids, or every requested cell has enough
        # support).
        self.uncomputable_out_cells = None

        if cell_out_ids is not None:
            # weak/empty columns in M (per output healpix cell k) -- compared
            # against the area-independent raw weight sum, so `threshold`
            # keeps its original meaning ("enough kernel support") regardless
            # of the magnitude/units of `area`.
            bad_k = torch.nonzero(norm_col_raw <= self.threshold).reshape(-1)

            if bad_k.numel() > 0 and not self.fill_missing_out_cells:
                # Fast path (default): don't run the expensive per-cell
                # fallback below at all. Leave these columns exactly as
                # computed (weak or empty) and record them so resample()
                # can force their output to nan -- see
                # fill_missing_out_cells's docstring in __init__.
                self.uncomputable_out_cells = bad_k.clone()

            if bad_k.numel() > 0 and self.fill_missing_out_cells:

                # Require geometry buffers (unit vectors)
                if (not hasattr(self, "xyz_samples")) or (not hasattr(self, "xyz_cells")):
                    raise RuntimeError(
                        "Fallback for missing out_cell_ids columns requires "
                        "self.xyz_samples (N,3) and self.xyz_cells (K,3)."
                    )

                # We'll REPLACE these columns: remove their current entries first
                I = M_coo.indices()
                V = M_coo.values()
                rows0 = I[0]
                cols0 = I[1]

                bad_set = set(int(x) for x in bad_k.detach().cpu().numpy().astype(np.int64))
                keep_mask = torch.ones_like(cols0, dtype=torch.bool)
                for kb in bad_set:
                    keep_mask &= (cols0 != int(kb))

                base_rows = rows0[keep_mask]
                base_cols = cols0[keep_mask]
                base_vals = V[keep_mask]

                # Fallback parameters (bilinear spirit)
                Npt_fallback = 1          # like bilinear
                eps = 1e-6
                sigma = float(self.sigma_m) if hasattr(self, "sigma_m") else 1.0

                add_rows, add_cols, add_vals = [], [], []
                
                # For each bad column, pick the closest source sample
                fallback_area = []
                for kb in range(len(bad_k)):
                    kb = int(kb)
                    # cosine similarity between all samples and the cell center
                    # (N,) = (N,3) @ (3,)
                    u = self.xyz_samples              # (N,3)
                    v = self.xyz_cells[bad_k[kb]]            # (3,)

                    dots = torch.sum((u - v)*(u - v), dim=1)    # (N,)


                    # take top-Npt_fallback closest (largest dot = smallest angular distance)
                    topv, topi = torch.topk(dots, k=min(Npt_fallback, self.N), largest=False)

                    add_rows.append(topi.to(torch.long))
                    add_cols.append(torch.tensor(bad_k[kb:kb+1], dtype=torch.long))
                    add_vals.append(torch.ones([1], dtype=self.dtype,device=self.device))
                    # this fallback column's raw weight is that single sample's
                    # own area (consistent units with area-weighted norm_col
                    # elsewhere), rather than a flat 1.0 -- see below.
                    fallback_area.append(self.area[topi[0]])

                add_rows = torch.cat(add_rows, dim=0)
                add_cols = torch.cat(add_cols, dim=0)
                add_vals = torch.cat(add_vals, dim=0)

                # rebuild M and coalesce
                new_rows = torch.cat([base_rows, add_rows], dim=0)
                new_cols = torch.cat([base_cols, add_cols], dim=0)
                new_vals = torch.cat([base_vals, add_vals], dim=0)

                M_coo = torch.sparse_coo_tensor(
                    torch.stack([new_rows, new_cols], dim=0),
                    new_vals,
                    size=(self.N, self.K),
                    device=self.device,
                    dtype=self.dtype,
                ).coalesce()

                # These columns now carry a single raw (unnormalized) fallback
                # link instead of their original (sub-threshold) weight sum --
                # correct Dx (norm_col) to match: the area of the single
                # sample used for the fallback link (matching the units of
                # area-weighted norm_col elsewhere; falls back to 1.0 when
                # area is uniform, same as before). See conjugate_gradient /
                # least_squares_cg docstrings for how Dx is used.
                norm_col = norm_col.clone()
                norm_col[bad_k] = torch.stack(fallback_area)

            # do the same for the transpose
            # weak/empty rows in MT (per input sample) -- again compared
            # against the area-independent raw weight sum (norm_row and
            # norm_row_raw only differ by each sample's own area factor,
            # which cancels out of MT itself, but not out of a raw threshold
            # comparison). This is a *sample*-side concern (affects
            # invert()), unrelated to which output cells resample() can
            # compute -- so it's gated by the same fill_missing_out_cells
            # flag (same expensive-fallback tradeoff) but doesn't feed
            # uncomputable_out_cells.
            bad_k = torch.nonzero(norm_row_raw <= self.threshold).reshape(-1)

            if bad_k.numel() > 0 and self.fill_missing_out_cells:

                # Require geometry buffers (unit vectors)
                if (not hasattr(self, "xyz_samples")) or (not hasattr(self, "xyz_cells")):
                    raise RuntimeError(
                        "Fallback for missing out_cell_ids columns requires "
                        "self.xyz_samples (N,3) and self.xyz_cells (K,3)."
                    )

                # We'll REPLACE these columns: remove their current entries first
                I = MT_coo.indices()
                V = MT_coo.values()
                rows0 = I[0]
                cols0 = I[1]

                bad_set = set(int(x) for x in bad_k.detach().cpu().numpy().astype(np.int64))
                keep_mask = torch.ones_like(cols0, dtype=torch.bool)
                for kb in bad_set:
                    keep_mask &= (cols0 != int(kb))

                base_rows = rows0[keep_mask]
                base_cols = cols0[keep_mask]
                base_vals = V[keep_mask]

                # Fallback parameters (bilinear spirit)
                Npt_fallback = 1          # like bilinear
                eps = 1e-6
                sigma = float(self.sigma_m) if hasattr(self, "sigma_m") else 1.0

                add_rows, add_cols, add_vals = [], [], []
                
                # For each bad column, pick the closest source sample
                for kb in range(len(bad_k)):
                    kb = int(kb)
                    # cosine similarity between all samples and the cell center
                    # (N,) = (N,3) @ (3,)
                    u = self.xyz_samples[bad_k[kb]]      # (3)
                    v = self.xyz_cells            # (K,3)

                    dots = torch.sum((u - v)*(u - v), dim=1)    # (N,)


                    # take top-Npt_fallback closest (largest dot = smallest angular distance)
                    topv, topi = torch.topk(dots, k=min(Npt_fallback, self.K), largest=False)
                    
                    add_rows.append(topi.to(torch.long))
                    add_cols.append(torch.tensor(bad_k[kb:kb+1], dtype=torch.long))
                    add_vals.append(torch.ones([1], dtype=self.dtype,device=self.device))

                add_rows = torch.cat(add_rows, dim=0)
                add_cols = torch.cat(add_cols, dim=0)
                add_vals = torch.cat(add_vals, dim=0)
                
                # rebuild M and coalesce
                new_rows = torch.cat([base_rows, add_rows], dim=0)
                new_cols = torch.cat([base_cols, add_cols], dim=0)
                new_vals = torch.cat([base_vals, add_vals], dim=0)
                
                MT_coo = torch.sparse_coo_tensor(
                    torch.stack([new_rows, new_cols], dim=0),
                    new_vals,
                    size=(self.K, self.N),
                    device=self.device,
                    dtype=self.dtype,
                ).coalesce() 
                
        # Dx: the raw (pre-normalization) per-cell weight columns of M were
        # divided by -- i.e. the weighted inner product <.,.>_Dx that MT is
        # the adjoint of M for (MT = Dx @ M.T @ Dy^-1). Needed by resample()
        # to run Conjugate Gradient in the inner product it is actually SPD
        # in -- see conjugate_gradient()/least_squares_cg() docstrings.
        self.cell_weight = norm_col

        # Convert to CSR for faster spMM (recommended on GPU)
        self.M  = M_coo #.to_sparse_csr()
        del M_coo
        self.MT = MT_coo.to_sparse_csr()
        del MT_coo

    @torch.no_grad()
    def resample(
        self,
        val: T_Array,
        *,
        lam: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-8,
        x0: Optional[torch.Tensor] = None,
        return_info: bool = False,
        conservative: bool = False,
    ) -> ResampleResults[T_Array]:
        """Estimate the HEALPix field from unstructured samples.

        Args:
            val: (B,N) or (N,) values at lon/lat sample points
            lam: Tikhonov regularization strength (damping) used in CG
            max_iter, tol: CG parameters
            x0: optional initial guess for the *delta* around x_ref, shape (B,K)
            return_info: whether to return CG diagnostics
            conservative: if True, apply an exact linear-equality correction
                (minimum-distortion in the same Dx-weighted metric CG uses) so
                that the unweighted mean of the output HEALPix field exactly
                matches the area-weighted mean of the input samples:
                ``mean(hval) == sum(val*area)/sum(area)``. Because HEALPix
                cells are equal-area, this is equivalent to conserving the
                total area-integrated quantity. See ``area`` in ``__init__``.
                Off by default -- does not change existing behaviour.

        NaN handling
        ------------
        NaN-valued samples in ``val`` are filtered out *before* the CG solve,
        not masked from the output afterwards. This matters specifically for
        ``PSFResampler``: ``least_squares_cg``/``conjugate_gradient`` compute
        ``alpha``/``beta`` from ``_wdot``, a sum over *all* K cells for each
        batch row (see ``conjugate_gradient`` docstring). A single unfiltered
        NaN sample would make ``x_ref = y @ M`` (and everything derived from
        it) NaN in every cell whose support includes that sample, and the
        very first ``_wdot`` reduction would then turn the CG update for the
        *entire* batch row into NaN -- not just the cells near the bad
        sample, unlike ``BilinearResampler``/``NearestResampler``.

        Concretely: NaN samples are replaced by ``0`` (per batch row) before
        computing ``x_ref`` and before entering the CG solve. This correctly
        removes the sample's *value* contribution, but -- because ``self.M``'s
        columns are pre-normalized at construction time against *all* samples
        that can reach a cell (``comp_matrix``'s ``norm_col``, fixed at
        construction, independent of any particular call's NaNs) -- a
        zeroed-but-still-linked NaN sample still counts in that
        normalization's denominator. The surviving samples' effective weight
        in an affected cell is therefore slightly diluted rather than the
        NaN sample being truly excluded from the local average. This is a
        deliberate trade-off (see ``planning/02_psf_nan_filtering.md``):
        cheap and requires no changes to the precomputed sparse operators,
        at the cost of a small residual bias that only matters for large,
        spatially-contiguous NaN regions rather than scattered/occasional
        NaNs. If that bias turns out to matter for your data, restricting
        the input to the valid samples' own KNN operator (rebuilding
        ``self.M``/``self.MT``) would be the exact fix, at the cost of
        rebuilding the sparse operators per call.

        Batch rows where *every* sample is NaN are handled separately: they
        are excluded from the CG solve entirely (rather than relying on the
        ``0/0`` that an all-zero row would otherwise produce inside
        ``conjugate_gradient``, which is not guaranteed to resolve to NaN
        without corrupting the shared, all-rows-at-once stopping criteria
        computed via ``torch.max(...)`` there) and their ``hval`` row is set
        directly to ``nan``. ``cg_residual_norms``/``cg_niters`` reflect only
        the rows that actually went through a CG solve; if every row is
        all-NaN, ``cg_niters`` is ``0`` and ``cg_residual_norms`` is ``nan``.

        Uncomputable ``out_cell_ids`` cells also read as NaN by default
        --------------------------------------------------------------------
        Separately from sample-side NaN handling above: if this instance was
        constructed with ``out_cell_ids`` and ``fill_missing_out_cells=False``
        (the default), any requested cell with too little real kernel
        support is forced to ``nan`` here, for every batch row, rather than
        filled with an approximate value -- see ``fill_missing_out_cells`` in
        ``__init__``.

        Returns:
            hval: (B,K) or (K,)
            (optional) info: CG information dict
        """
        y = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
        y = y.to(self.device, dtype=self.dtype)
        clean_shape=False
        if y.ndim == 1:
            clean_shape=True
            y = y[None, :]

        # --- NaN filtering (see "NaN handling" in the docstring above) -----
        nan_mask = torch.isnan(y)
        valid = ~nan_mask
        y_filled = torch.where(valid, y, torch.zeros_like(y))

        # Rows with no valid sample at all can't go through the shared CG
        # loop (its stopping criteria are computed across all batch rows at
        # once -- see conjugate_gradient()); exclude them up front and patch
        # NaN back in afterwards instead.
        row_has_data = valid.any(dim=-1)                      # (B,)
        B = y.shape[0]

        # reference field (B,K) -- from the NaN-filled input, so a bad
        # sample only zeroes its own (normalized) contribution instead of
        # poisoning x_ref with NaN.
        x_ref = y_filled @ self.M

        if x0 is None:
            x0 = torch.zeros_like(x_ref)
        else:
            x0 = x0.to(self.device, dtype=self.dtype)

        hval = torch.full_like(x_ref, float("nan"))
        cg_residual_norms = torch.full((1, B), float("nan"), device=self.device, dtype=self.dtype)
        cg_niters = torch.tensor(0, device=self.device)

        if bool(row_has_data.any()):
            sel = torch.nonzero(row_has_data, as_tuple=False).reshape(-1)
            y_sel = y_filled[sel]
            x_ref_sel = x_ref[sel]
            x0_sel = x0[sel]

            delta, info = least_squares_cg(
                M=self.M,
                MT=self.MT,
                y=y_sel,
                x_ref=x_ref_sel,
                x0=x0_sel,
                max_iter=max_iter,
                tol=tol,
                damp=float(lam),
                verbose=self.verbose,
                weight=self.cell_weight,
            )

            hval[sel] = delta + x_ref_sel
            cg_residual_norms = info["residual_norms"]  # (iters+1, len(sel)) -- valid rows only
            cg_niters = info["niters"]

            if conservative:
                # Single scalar equality constraint sum_k(hval_k) == F_target, with
                # F_target chosen so the constraint is equivalent to conserving
                # sum(val*area) (HEALPix cells being equal-area, this reduces to
                # matching the area-weighted input mean -- see resample() docstring).
                # NaN samples are excluded from both the numerator and the
                # area normalization -- otherwise a NaN sample's area would
                # still count towards "total area" while contributing 0 to
                # the weighted sum, silently biasing target_mean downward.
                #
                # Minimum-Dx-distortion correction from the KKT system of the
                # *constrained* weighted least-squares problem (constraint
                # appended to Eq. for delta, not a post-hoc projection of the
                # unconstrained delta): with c = 1_K and Dx = diag(cell_weight),
                # the correction direction is w = H^-1 (Dx^-1 c), NOT H^-1 c --
                # i.e. the extra CG solve below must use a Dx^-1-weighted
                # right-hand side, or the result satisfies the constraint but is
                # *not* the minimum-distortion correction the derivation claims.
                area = self.area
                valid_sel = valid[sel]                                                     # (S,N)
                area_b = area[None, :].expand(valid_sel.shape[0], -1)
                area_valid_sum = (area_b * valid_sel).sum(dim=-1, keepdim=True)             # (S,1)
                target_mean = (y_sel * area_b).sum(dim=-1, keepdim=True) / area_valid_sum   # (S,1)
                F_target = target_mean * self.K                                             # (S,1)

                def _Hmv(v: torch.Tensor) -> torch.Tensor:
                    return (v @ self.MT) @ self.M + float(lam) * v

                dxinv_c = (1.0 / self.cell_weight).unsqueeze(0)  # (1,K) = Dx^-1 @ 1_K
                w, _ = conjugate_gradient(
                    A_mv=_Hmv, b=dxinv_c, max_iter=max_iter, tol=tol,
                    verbose=False, weight=self.cell_weight,
                )                                                                          # (1,K)

                mu = (F_target - hval[sel].sum(dim=-1, keepdim=True)) / w.sum()             # (S,1)
                hval[sel] = hval[sel] + mu * w                                              # (S,K)

        # Force nan for output cells flagged as uncomputable at construction
        # time (out_cell_ids-requested cells with too little real kernel
        # support -- see fill_missing_out_cells in __init__). Applied last,
        # after the CG solve and any conservative correction: those cells'
        # CG-produced values (if any) are not meaningful regardless of what
        # the solve happened to converge to, and applying this before the
        # conservative correction would let a handful of nan cells poison
        # every other cell's correction via hval[sel].sum(). Note this means
        # conservative=True's global constraint is still computed *including*
        # these cells' (pre-nan) contribution -- a known, minor interaction
        # this flag doesn't attempt to fully resolve.
        if self.uncomputable_out_cells is not None and self.uncomputable_out_cells.numel() > 0:
            hval[:, self.uncomputable_out_cells] = float("nan")

        if val is not None and val.ndim == 1:
            hval = hval[0]

        cell_ids = self.cell_ids

        if not isinstance(val, torch.Tensor):
            hval= hval.cpu().numpy()
            cell_ids = cell_ids.cpu().numpy()
            cg_residual_norms = cg_residual_norms.cpu().numpy()
            cg_niters = cg_niters.cpu().numpy()

        return ResampleResults(
            cell_data=hval,
            cell_ids=cell_ids,
            cg_residual_norms=cg_residual_norms,
            cg_niters=cg_niters
        )
  
