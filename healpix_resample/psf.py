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

from healpix_resample.base import ResampleResults, T_Array
from healpix_resample.knn import KNeighborsResampler, _sigma_level_m, _lonlat_to_xyz


@torch.no_grad()
def conjugate_gradient(
    A_mv: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    verbose: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Solve A x = b with Conjugate Gradient where A is SPD, using only matvec A_mv(v).
    No autograd (uses torch.no_grad).

    Returns:
        x: solution
        info: dict with residual norms history, iterations
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    r = b - A_mv(x)          # residual
    p = r.clone()
    rs_old = torch.einsum('ik,ik->i',r,r)

    b_norm = torch.linalg.norm(b)
    if b_norm == 0:
        return x, {"residual_norms": torch.tensor([0.0], device=b.device, dtype=b.dtype),
                   "niters": torch.tensor(0, device=b.device)}

    residual_norms = [torch.sqrt(rs_old)]

    for k in range(max_iter):
        Ap = A_mv(p)
        denom = torch.einsum('ik,ik->i',p,Ap)
        if torch.max(denom.abs()) < 1e-30:
            break  # breakdown (shouldn't happen for SPD unless numerical issues)

        alpha = rs_old / denom
        x = x + torch.einsum('k,ki->ki',alpha,p)
        r = r - torch.einsum('k,ki->ki',alpha,Ap)
        rs_new = torch.einsum('ik,ik->i',r,r)

        residual_norms.append(torch.sqrt(rs_new))

        # stopping criterion: relative residual
        if torch.max(torch.sqrt(rs_new)) <= tol * b_norm:
            rs_old = rs_new
            break

        beta = rs_new / rs_old
        p = r + torch.einsum('k,ki->ki',beta,p)
        rs_old = rs_new
        if k%4==0 and verbose:
            print('Itt %d : %.4g'%(k,rs_old))

    info = {
        "residual_norms": torch.stack(residual_norms),
        "niters": torch.tensor(len(residual_norms) - 1, device=b.device),
    }
    if verbose:
        print('Final Itt %d : %.4g'%(k,rs_old))
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
        ):
    """
    Solve for delta in a damped least-squares problem without forming dense matrices.

    We solve:
        (MT @ M + damp*I) delta = (y - x_ref @ MT) @ M

    Shapes:
        M  : (N, K) sparse CSR
        MT : (K, N) sparse CSR
        y  : (B, N)
        x_ref : (B, K)
        delta : (B, K)
    """

    # b = M^T y
    b = (y - x_ref@MT) @ M
    def A_mv(v: torch.Tensor) -> torch.Tensor:
        # (M^T M + damp I) v
        return (v@MT) @ M + damp * v

    x, info = conjugate_gradient(A_mv=A_mv, b=b, x0=x0, max_iter=max_iter, tol=tol,verbose=verbose)
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
        **kwargs,
    ):
        """
        PSF regridding Set.
        """
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
        w = torch.exp((-2.0) * (self.d_m * self.d_m) / (self.sigma_m * self.sigma_m))
        
        # Build (N,K) operator M and (K,N) operator MT.
        # We avoid numpy bincount; use torch.bincount on GPU.

        # idx: (N,Npt) row indices 0..N-1
        idx = torch.arange(self.N, device=self.device, dtype=torch.long)[:, None].expand(self.N, self.Npt)

        # -------- M : (N,K)  (normalized per column / per healpix cell)
        # norm_col[k] = sum_{i links to k} w[i,k]
        flat_hi = self.hi.reshape(-1)
        flat_w = w.reshape(-1)
        valid = flat_hi >= 0
        flat_hi_v = flat_hi[valid]
        flat_w_v = flat_w[valid]

        norm_col = torch.bincount(flat_hi_v, weights=flat_w_v, minlength=self.K).to(self.dtype)
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

        if cell_out_ids is not None:
            # weak/empty columns in M (per output healpix cell k)
            bad_k = torch.nonzero(norm_col <= self.threshold).reshape(-1)
              
            if bad_k.numel() > 0:
                
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
                
                
            # do the same fo the transpose
            # weak/empty columns in M (per output healpix cell k)
            bad_k = torch.nonzero(norm_row <= self.threshold).reshape(-1)
              
            if bad_k.numel() > 0:
                
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
    ) -> ResampleResults[T_Array]:
        """Estimate the HEALPix field from unstructured samples.

        Args:
            val: (B,N) or (N,) values at lon/lat sample points
            lam: Tikhonov regularization strength (damping) used in CG
            max_iter, tol: CG parameters
            x0: optional initial guess for the *delta* around x_ref, shape (B,K)
            return_info: whether to return CG diagnostics

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

        # reference field (B,K)
        x_ref = y @ self.M
        
        if x0 is None:
            x0 = torch.zeros_like(x_ref)
        else:
            x0 = x0.to(self.device, dtype=self.dtype)

        delta, info = least_squares_cg(
            M=self.M,
            MT=self.MT,
            y=y,
            x_ref=x_ref,
            x0=x0,
            max_iter=max_iter,
            tol=tol,
            damp=float(lam),
            verbose=self.verbose,
        )
        
        hval = delta + x_ref 
        if val is not None and val.ndim == 1:
            hval = hval[0]

        cell_ids = self.cell_ids
        cg_residual_norms = info["residual_norms"]
        cg_niters = info["niters"]

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
  
