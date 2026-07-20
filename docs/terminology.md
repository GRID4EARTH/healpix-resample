# Terminology

A quick reference for the key concepts used throughout this documentation.

---

## HEALPix

Key properties:
- All cells have the **same area** (equal-area projection).
- Cells are organized in a **hierarchical** structure: each cell at level `l` contains exactly 4 sub-cells at level `l+1`.
- Resolution is controlled by `level`: `nside = 2**level`, total cell count = `12 * nside²`.

:::{seealso}
See [healpix-geo](https://healpix-geo.readthedocs.io) for more informations.
:::

---

## Nested vs. Ring indexing

HEALPix cells can be indexed in two schemes:

- **Nested** (`nest=True`, default): cells with similar indices are spatially close. Best for hierarchical operations.
- **Ring**: cells are indexed by iso-latitude rings. Used in some harmonic analysis tools.

`healpix-resample` uses **nested** indexing by default.

---

## Sparse operator M and MT

The library represents the mapping between sample points and HEALPix cells as two sparse matrices:

- **`M`** of shape `(N, K)`: maps a HEALPix field (on K kept cells) → values at N sample locations.
- **`MT`** of shape `(K, N)`: maps sample values (N) → accumulated values on K HEALPix cells.

These operators are built once during initialization and reused across all calls to `resample()`.

---

## N, K, B

Throughout the documentation:

- **N** — number of input sample points (lon/lat measurements).
- **K** — number of HEALPix cells that receive at least one contribution (a subset of all cells at the given level).
- **B** — batch size (number of fields to process at once, e.g. time steps or variables).

---

## Threshold

The `threshold` parameter controls which HEALPix cells are kept. After computing accumulated Gaussian weights from all samples, only cells whose total weight exceeds `threshold` are retained. This keeps the operator compact and avoids nearly-empty cells polluting results.

- Lower `threshold` → more cells kept, including weakly covered ones.
- Higher `threshold` → only well-supported cells are kept.

---

## Sigma (σ)

`sigma_m` is the Gaussian length scale in **metres** used to compute the weight of each sample-to-cell link:

```
w = exp(-2 * d² / σ²)
```

where `d` is the geodesic distance between the sample and the cell centre. If `sigma_m` is not set, the library defaults to the pixel size at the chosen level:

```
σ = sqrt(4π / (12 × 4^level)) × R_Earth
```

---

## Conservative resampling / area weighting

`ConservativeResampler` bins each sample into its containing HEALPix cell and accumulates an
**area-weighted sum**, so that a total integrated quantity is preserved exactly between representations:

```
sum(val_i * area_i) == sum(hval_k)
```

`area` is a per-sample scalar weight (default `1.0`). It matters only when `val` is an *intensive*
quantity (a density, e.g. flux per m², temperature): samples with a larger physical footprint should be
weighted proportionally more. If `val` is already an *extensive*, pre-integrated total (e.g. counts),
leave `area` at its default — plain summation is exactly conservative regardless of footprint size.

Unlike `NearestResampler`, `BilinearResampler`, and `PSFResampler` — which interpolate *values* at
points — `ConservativeResampler` preserves a *flux*, at the cost of not producing a smooth field.

---

## Conjugate Gradient (CG)

Used by `PSFResampler` to solve the damped least-squares problem:

```
minimize ‖ (x_ref + δ) @ MT - y ‖² + λ ‖δ‖²
```

The CG solver iteratively refines the solution without forming any dense matrix. `lam` (λ) is the Tikhonov regularization strength — set it higher when your data is sparse or noisy.

---

## ResampleResults

The dataclass returned by all `resample()` calls:

| Field | Shape | Description |
|---|---|---|
| `cell_data` | `(K,)` or `(B, K)` | Values on the HEALPix cells |
| `cell_ids` | `(K,)` | HEALPix cell indices (nested scheme) |
| `cg_residual_norms` | `(iters,)` | CG convergence history *(PSFResampler only)* |
| `cg_niters` | scalar | Number of CG iterations *(PSFResampler only)* |
