#!/usr/bin/env python3
"""Convert every frozen-bundle Zarr store to format 2, losslessly.

Why
---
The bundle drifted into MIXED Zarr formats: the four benchmark scenes and
the Esri stores were written under zarr-python 2.x (format 2), while the 40
region patches were acquired later under zarr-python 3.x, which writes
format 3 by default. Format 3 is unreadable by zarr-python 2.x, so the mix
forced an awkward environment split and, worse, made "unsupported format"
errors indistinguishable from data corruption. Uniform format 2 is readable
by every zarr-python since 2.11 *and* by 3.x, and removes the whole class
of reviewer questions.

What it does
------------
For every ``*.zarr`` directory under the bundle paths:

1. Detect the format from the filesystem (``zarr.json`` => 3,
   ``.zgroup`` => 2) -- no zarr import needed for detection, so the report
   mode works in any environment.
2. Format-2 stores are left untouched (idempotent).
3. Format-3 stores are read with xarray, rewritten next to the original
   with ``zarr_format=2``, then VERIFIED: every group, variable, coordinate,
   dtype, value (bitwise, NaN-aware) and attribute of the copy is compared
   against the original. Only on a perfect match is the original replaced
   (atomically: original moved aside, copy moved in, backup deleted).
4. Any mismatch aborts with the original left in place.

Requirements: run it ONCE in an environment with zarr-python >= 3 (the only
zarr that reads both formats), e.g.::

    pixi exec --spec "python=3.12" --spec "zarr>=3" --spec "xarray" \\
        --spec "dask" python notebooks/convert_zarr_stores_to_v2.py

or any throwaway conda/pip env with ``zarr>=3 xarray dask``. The regular
``notebooks`` environment (zarr < 3, for xarray-eopf) cannot read the
format-3 inputs and is deliberately not suitable.

After converting, rebuild the manifest and the archives -- every converted
store changes bytes, so SHA-256 and MD5 values change, and a new Zenodo
version must be published:

    python notebooks/build_data_manifest.py --doi <new version DOI>
    python notebooks/build_zenodo_archives.py --doi <new version DOI>

Report only (no writes, works everywhere)::

    python notebooks/convert_zarr_stores_to_v2.py --report
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUNDLE_GLOBS = [
    "notebooks/data/*.zarr",
    "notebooks/data/esri_latent/*.zarr",
    "notebooks/data/multi_patch_latitude/esri_patch_cache/*.zarr",
    "notebooks/data/multi_patch_sentinel2/*.zarr",
]


def store_format(path: Path) -> str:
    if (path / "zarr.json").exists():
        return "3"
    if (path / ".zgroup").exists() or (path / ".zarray").exists():
        return "2"
    return "?"


def find_stores() -> list[Path]:
    out: list[Path] = []
    for pattern in BUNDLE_GLOBS:
        out.extend(sorted(REPO.glob(pattern)))
    return [p for p in out if p.is_dir()]


def _compare_trees(a, b, where: str, errors: list[str]) -> None:
    import numpy as np

    if dict(a.attrs) != dict(b.attrs):
        errors.append(f"{where}: root/group attrs differ")
    a_vars = set(a.ds.variables) if hasattr(a, "ds") else set(a.variables)
    b_vars = set(b.ds.variables) if hasattr(b, "ds") else set(b.variables)
    if a_vars != b_vars:
        errors.append(f"{where}: variable sets differ ({a_vars ^ b_vars})")
        return
    a_ds = a.ds if hasattr(a, "ds") else a
    b_ds = b.ds if hasattr(b, "ds") else b
    for name in a_vars:
        va, vb = a_ds[name], b_ds[name]
        if va.dtype != vb.dtype:
            errors.append(f"{where}/{name}: dtype {va.dtype} != {vb.dtype}")
            continue
        if dict(va.attrs) != dict(vb.attrs):
            errors.append(f"{where}/{name}: attrs differ")
        xa, xb = np.asarray(va.values), np.asarray(vb.values)
        if xa.shape != xb.shape:
            errors.append(f"{where}/{name}: shape {xa.shape} != {xb.shape}")
        elif xa.dtype.kind in "fc":
            if not np.array_equal(xa, xb, equal_nan=True):
                errors.append(f"{where}/{name}: values differ")
        elif not np.array_equal(xa, xb):
            errors.append(f"{where}/{name}: values differ")
    if hasattr(a, "children"):
        for key in set(a.children) | set(b.children):
            if key not in a.children or key not in b.children:
                errors.append(f"{where}: child group {key!r} missing on one side")
                continue
            _compare_trees(a.children[key], b.children[key],
                           f"{where}/{key}", errors)


def convert(path: Path) -> bool:
    import xarray as xr
    import zarr

    if int(str(zarr.__version__).split(".")[0]) < 3:
        raise SystemExit(
            f"zarr-python {zarr.__version__} cannot READ the format-3 stores "
            "to be converted. Run this script in a throwaway env with "
            "zarr>=3 (see module docstring)."
        )

    tmp = path.with_name(path.name + ".v2tmp")
    bak = path.with_name(path.name + ".v3bak")
    for stale in (tmp, bak):
        if stale.exists():
            shutil.rmtree(stale)

    dt = xr.open_datatree(path, engine="zarr", consolidated=False, chunks={})
    dt.to_zarr(tmp, mode="w", zarr_format=2, consolidated=False)

    src = xr.open_datatree(path, engine="zarr", consolidated=False, chunks={})
    dst = xr.open_datatree(tmp, engine="zarr", consolidated=False, chunks={})
    errors: list[str] = []
    _compare_trees(src, dst, path.name, errors)
    src.close(); dst.close(); dt.close()
    if errors:
        shutil.rmtree(tmp)
        print(f"  VERIFICATION FAILED for {path.name}; original untouched:")
        for e in errors[:10]:
            print("   ", e)
        return False

    path.rename(bak)
    tmp.rename(path)
    shutil.rmtree(bak)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="store_true",
                    help="list formats only; write nothing")
    args = ap.parse_args()

    stores = find_stores()
    if not stores:
        print("No bundle stores found -- install the data first "
              "(notebooks/load_data_in_zenodo.ipynb).")
        return 1

    counts = {"2": 0, "3": 0, "?": 0}
    todo = []
    for p in stores:
        fmt = store_format(p)
        counts[fmt] += 1
        if fmt == "3":
            todo.append(p)
        elif fmt == "?":
            print(f"  UNRECOGNIZED layout: {p}")
    print(f"{len(stores)} stores: {counts['2']} format-2, "
          f"{counts['3']} format-3, {counts['?']} unrecognized.")

    if args.report or not todo:
        if not todo and not args.report:
            print("Nothing to convert; bundle is uniformly format 2.")
        return 0

    ok = 0
    for i, p in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] converting {p.relative_to(REPO)}")
        if convert(p):
            ok += 1
        else:
            return 1
    print(f"\n{ok}/{len(todo)} stores converted and verified. Now rebuild "
          "the manifest and archives, and publish a new Zenodo version "
          "(the store bytes changed, so every checksum changed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
