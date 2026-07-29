# healpix-resample

[![PyPI](https://img.shields.io/pypi/v/healpix-resample.svg)](https://pypi.org/project/healpix-resample/)
[![Docs](https://github.com/GRID4EARTH/healpix-resample/actions/workflows/main.yml/badge.svg)](https://github.com/GRID4EARTH/healpix-resample/actions/workflows/main.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**[`healpix-resample`](https://grid4earth.eu/healpix-resample/)** is a lightweight, GPU-friendly Python
package for regridding data from unstructured longitude/latitude samples onto a
[HEALPix](https://healpix.sourceforge.io/) grid — one sparse linear operator, several interpolation
strategies, and a consistent API across all of them.

Under the hood, every resampler builds a sparse [PyTorch](https://pytorch.org/) operator mapping input
samples to a subset of HEALPix cells at a chosen resolution level, so the same forward/inverse operator
can be reused, batched `(B, N)`, and run on CPU or CUDA. The package manages the HEALPix authalic
definition and the Earth ellipsoid using the **WGS84** reference system.

## Resamplers at a glance

| Resampler | Strategy | Good for |
|---|---|---|
| `NearestResampler` | Single nearest sample per cell | Fast, simple, exact point indexing |
| `BilinearResampler` | 4 nearest samples, inverse-distance weights | Smoother than nearest, cheap |
| `BicubicResampler` | 16 nearest samples, Keys' cubic convolution kernel | Sharper on fields with curvature, still non-iterative |
| `PSFResampler` | Gaussian PSF kernel + damped least-squares (Conjugate Gradient) | Best reconstruction quality, dense or irregular data |
| `ConservativeResampler` | Direct binning, area-weighted sum | Exact flux/mass conservation |
| `GroupByResampler` | Direct binning, configurable reduction (`mean`, `sum`, `prod`, `amax`, `amin`) | Aggregating many samples per cell |
| `CellPointResampler` | Level-29 "cell-point" encoding | Exact point indexing, no interpolation |

`BilinearResampler`, `BicubicResampler`, and `PSFResampler` also support an optional `area=` +
`conservative=True` mode that redistributes each sample's value across its cell footprint instead of
just interpolating, so the global total is preserved exactly — combining smooth interpolation with
`ConservativeResampler`'s exact-conservation guarantee. `PSFResampler` additionally supports
`out_cell_ids=` to restrict output to a specific cell subset, and `healpix_resample.subset_for_parent_cell`
lets you process one coarse HEALPix region at a time for datasets too large to load in full.

Every resampler handles NaN-valued samples consistently (excluded from the computation rather than
propagated) and accepts both NumPy arrays and PyTorch tensors, returning whichever type you passed in.

## Installation

### From PyPI

```bash
pip install healpix-resample
```

### From source (editable, for development)

```bash
git clone https://github.com/GRID4EARTH/healpix-resample.git
cd healpix-resample
pip install -e .
```

Or, using [pixi](https://pixi.sh) (the environment this package is developed and tested in):

```bash
git clone https://github.com/GRID4EARTH/healpix-resample.git
cd healpix-resample
pixi install
```

### Requirements

- Python 3.10+
- PyTorch (CPU or CUDA)
- NumPy
- [`healpix-geo`](https://github.com/healpix-geo/healpix-geo)

### Verifying the installation

```python
import healpix_resample
print(healpix_resample.__file__)
```

## Quickstart

```python
from healpix_resample import BilinearResampler

op = BilinearResampler(lon_deg=lon, lat_deg=lat, level=level, device="cuda")
result = op.resample(values)

healpix_values = result.cell_data   # (K,) or (B, K)
healpix_cells  = result.cell_ids    # (K,) HEALPix cell ids, same order as cell_data
```

Every resampler follows this same `resample(val) -> ResampleResults(cell_data, cell_ids)` pattern; swap
`BilinearResampler` for any of the other classes above to change interpolation strategy without changing
the rest of your code. See the [full documentation](https://grid4earth.eu/healpix-resample/) —
in particular the [`4resamplers` tutorial](docs/tutorials/4resamplers.md), which runs every resampler
side by side on the same dataset — for a complete tour, including conservative mode, batched inputs,
`out_cell_ids`, and large-scale parent-cell processing.

## Documentation

Full documentation, including a user guide per resampler, tutorials, and the API reference, is available
at **[grid4earth.eu/healpix-resample](https://grid4earth.eu/healpix-resample/)**.

## Development

This project uses [pixi](https://pixi.sh) to manage environments.

```bash
pixi run tests          # run the test suite
pixi run -e docs build-docs   # build the documentation locally (docs/_build)
```

## Target applications

- Earth observation data remapping (e.g. Sentinel products)
- Oceanographic or atmospheric gridding
- Astronomical sky projections
- Large-scale geospatial data harmonization

## License

Apache License 2.0 — see [LICENSE](LICENSE).
