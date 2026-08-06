# `healpix_resample.mask` (mask-like / categorical data)

[Issue #43](https://github.com/GRID4EARTH/healpix-resample/issues/43) ("make specific resampler for
mask-like data") asked for something better than `NearestResampler`'s blocky single-nearest-sample
assignment for mask-like data — e.g. a Sentinel-2 L1C cloud-mask — without pretending the underlying
values are a continuous physical field the way `BilinearResampler`/`PSFResampler` assume.

`healpix_resample.mask` provides two resamplers for two genuinely different kinds of "mask", both built
on the same idea: turn each discrete thing being resampled into a 0/1 *indicator* array, resample every
indicator through an ordinary interpolating resampler, and turn the resulting continuous maps back into a
discrete decision.

|  | Bits/classes | Mutually exclusive? | Decision |
|---|---|---|---|
| `BitmaskResampler` | Independent boolean flags packed into an integer (e.g. an 8-bit quality mask) | No — several can co-occur | Each bit resampled and thresholded **independently** |
| `CategoricalResampler` | One class label per sample (e.g. land-cover classification) | Yes | **argmax** over all classes' resampled indicators |

Neither class does its own KNN/geometry work — both wrap an already fully-specified interpolating
resampler instance (`kernel=`, default `BilinearResampler`) and reuse its `cell_ids`/`resample()`
machinery, including whatever NaN filtering, `out_cell_ids`, or (for a `PSFResampler` kernel)
`fill_missing_out_cells` behaviour that instance already has.

---

## `BitmaskResampler` — independent flags ("OR")

```python
from healpix_resample import BitmaskResampler

op = BitmaskResampler(lon_deg=lon, lat_deg=lat, level=level, n_bits=8)
res = op.resample(quality_mask)   # (N,) int -- e.g. an 8-bit cloud/quality mask
output_mask = res.cell_data       # (K,) int -- reassembled bitmask per HEALPix cell
```

For each bit `b` in `range(n_bits)`, builds the indicator `((mask >> b) & 1)` and resamples all `n_bits`
indicators in a single batched call to `kernel` (`(n_bits, N) -> (n_bits, K)`). Each bit's resampled
fraction is thresholded at `bit_threshold` (default `0.5`) **independently of every other bit**, then the
surviving bits are recombined into an integer bitmask. There is no argmax here: any combination of bits
can end up set in the output, exactly as in the input — that's the point for flags that co-occur (cloud
*and* cloud-shadow *and* saturated, all at once).

`n_bits` is required, not auto-detected from `max(mask)`: a bit that never happens to be set in one
particular `resample()` call (or in one parent-cell subset, if combined with `subset_for_parent_cell`)
would otherwise silently vanish from the output instead of correctly coming back all-zero.

### `bit_threshold` and the choice of `kernel`

With the default `BilinearResampler`, a bit's resampled value is a proper weighted fraction of "how many
nearby samples have this bit set", bounded in `[0, 1]` — so `bit_threshold=0.5` cleanly means "the
majority of nearby samples have this bit set". `BicubicResampler` (signed kernel) and `PSFResampler`
(CG-solved) can both produce values outside `[0, 1]`, which weakens — without breaking — that "majority"
interpretation: the threshold still picks a definite bit value, just not necessarily exactly "more than
half of nearby samples".

---

## `CategoricalResampler` — mutually-exclusive classes ("AND"/dominant class)

```python
from healpix_resample import CategoricalResampler

op = CategoricalResampler(lon_deg=lon, lat_deg=lat, level=level)
res = op.resample(land_cover_class)   # (N,) int -- one class label per sample
output_class = res.cell_data          # (K,) int -- winning class per HEALPix cell
```

Distinct classes are discovered from `mask` itself on every call (`torch.unique`, sorted ascending) — a
one-hot indicator is built per class, all of them resampled in one batched call to `kernel`
(`(n_classes, N) -> (n_classes, K)`), and the winning class per cell is whichever indicator scored
highest: `argmax_over_bilinear`, the issue's own working name, with the default `kernel`.

**Ties** are broken deterministically: the lowest-valued tied class wins. Exact ties are rare in practice
(they require perfect geometric symmetry between two classes' local support).

### Optional softmax score (`return_scores=True`)

```python
res = op.resample(land_cover_class, return_scores=True, softmax_temperature=0.1)
res.cell_data   # (K,) winning class, same as without return_scores
res.classes     # (n_classes,) class labels, in res.scores's row order
res.scores      # (n_classes, K) softmax-normalized score per class per cell
```

`return_scores=True` returns a `CategoricalResampleResults` (a `ResampleResults` subclass) with two extra
fields: `classes` (the distinct labels found) and `scores` (a softmax over the raw per-class indicator
scores). This serves two purposes: a graceful, continuous alternative to the hard argmax tie-break, and a
confidence-style diagnostic per cell (related to [issue #4](https://github.com/GRID4EARTH/healpix-resample/issues/4),
"confidence factor").

`softmax_temperature` (default `0.1`) controls how sharply the softmax favours the argmax winner: lower
values sharpen towards a one-hot at the winning class, higher values spread mass across close
runners-up. The default is tuned for `BilinearResampler`'s natural `[0, 1]`-ish score scale — a bare
softmax (temperature 1) over scores already confined to such a narrow range would barely sharpen
anything, giving near-uniform "probabilities" even for a clear winner. Retune if you change `kernel` or
your classes are unusually balanced/imbalanced.

### Choice of `kernel` and what the scores mean

With the default `BilinearResampler`, per-class scores are bounded in `[0, 1]` and — because every
sample belongs to exactly one class, so its one-hot indicator sums to 1 — sum to (very close to) 1
across classes for every retained cell, by construction of `BilinearResampler.M`'s per-cell
normalization. That makes the raw scores already a reasonable probability-like quantity even before the
softmax step. `BicubicResampler` and `PSFResampler` don't carry this guarantee (scores can be negative or
exceed 1), so `argmax` — the decision that actually matters — remains meaningful with either, but
`return_scores`'s softmax output is a softer, less strictly calibrated confidence signal for those two
kernels than for `BilinearResampler`.

---

## Both classes: limitations

- `mask` must be 1-D `(N,)` — batched `(B, N)` categorical/bitmask input isn't supported (different rows
  could have different classes/bits present, which complicates the class-discovery step in ways this
  first version doesn't attempt to resolve).
- `mask` must not contain NaN — a mask value has no well-defined "missing" decomposition into indicators
  the way a continuous NaN sample does for the interpolating resamplers elsewhere in this package. If you
  have a "no-data" indicator, encode it as its own class/bit instead.
- Any extra keyword arguments passed to `resample()` (e.g. `lam`/`tol`/`max_iter` for a `PSFResampler`
  kernel) are forwarded to the wrapped kernel's own `resample()`.
