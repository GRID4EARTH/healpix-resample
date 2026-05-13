# healpix-resample


[`healpix-resample`](https://eopf-dggs.github.io/healpix-resample) is a lightweight Python package designed to regrid
data defined on longitude--latitude coordinates onto a HEALPix
grid.

The package provides GPU-accelerated operators (via PyTorch) to
construct sparse linear mappings between input geodetic coordinates and
a target HEALPix tessellation at a chosen resolution level.

This package manages the HEALPix authalic definition and the Earth
ellipsoid using the **WGS84** reference system.

## Objectives

The main goals of the package are:

-   Provide a **generic regridding framework** from (lon, lat) to
    HEALPix.
-   Support different interpolation strategies:
    -   **Nearest-neighbor mapping**
    -   **PSF / multi-point weighted interpolation**
-   Enable efficient handling of:
    -   Large numbers of input points
    -   Batched data `(B, N)`
    -   CUDA acceleration
-   Offer a reusable linear operator that can be:
    -   Applied forward (data → HEALPix)
    -   Used inside inverse problems or iterative solvers

## Design Principles

-   Modular architecture:
    -   `knn` module: generic operator construction
    -   `nearest`: nearest-neighbor specialization
    -   `psf`: weighted multi-point interpolation
-   Sparse matrix representation for scalability
-   Torch-based implementation for CPU/GPU flexibility
-   Resolution controlled via HEALPix level parameter

## Installation (Private Repository)

This package is distributed as a **private repository** and must be installed from source.

---

### Clone the repository

```bash
git clone https://github.com/GRID4EARTH/healpix-resample.git
cd healpix-resample
```

If you use SSH access:

```bash
git clone git@github.com:GRID4EARTH/healpix-resample.git
cd healpix-resample
```

---

### Install in editable mode (recommended for development)

```bash
pip install -e .
```

Editable mode allows you to modify the source code without reinstalling the package after each change.

---

### Standard installation (optional)

If you do not need editable mode:

```bash
pip install .
```

---

### Requirements

Make sure you are using:

- Python ≥ 3.8
- A working PyTorch installation (CPU or CUDA)
- numpy
- healpix-geo

### Verifying the installation

After installation:

```python
import healpix_resample
print(healpix_resample.__file__)
```

If no error occurs, the installation is successful.

## Typical Use Case

``` python
from healpix_resample.nearest import NearestResampler

op = NearestResampler(lon_deg=lon, lat_deg=lat, level=level, device="cuda")
healpix_values = op.resample(values)
```

## Target Applications

-   Earth observation data remapping
-   Oceanographic or atmospheric gridding
-   Astronomical sky projections
-   Large-scale geospatial data harmonization
