#!/usr/bin/env python3
"""Convert every frozen-bundle Zarr store to format 2, losslessly.

Why
---
The bundle drifted into MIXED Zarr formats: most stores are format 2, but
the four ``esri_latent`` stores were (re)written on a machine whose
zarr-python 3 silently produced format 3, which zarr-python 2.x cannot
read. Uniform format 2 is readable by every zarr-python since 2.11 *and*
by 3.x, and removes the whole class of reviewer questions.

How
---
The copy uses the RAW zarr API, not xarray. This is deliberate: the
format-3 stores in this bundle were written with ``zarr.open_group`` (see
``_write_esri_latent_zarr`` in the notebooks) and carry no xarray dimension
metadata, so ``xr.open_datatree`` refuses them outright
(``KeyError: dimension_names``). A raw group-and-array copy has no such
requirement and is strictly more general. One semantic translation is
handled: if a format-3 array does carry ``dimension_names`` (the
xarray-written case), it is stored as the ``_ARRAY_DIMENSIONS`` attribute
in the format-2 copy, which is the v2 convention for the same information.

For every ``*.zarr`` directory under the bundle paths:

1. Detect the format from the filesystem (``zarr.json`` => 3,
   ``.zgroup``/``.zarray`` => 2) -- works in any environment, so
   ``--report`` needs nothing installed.
2. Format-2 stores are left untouched (idempotent).
3. Format-3 stores are copied next to the original as format 2, then
   VERIFIED: every group, array, dtype, shape, value (bitwise, NaN-aware)
   and attribute is compared. Only on a perfect match is the original
   replaced atomically. Any mismatch aborts with the original untouched.

Chunk sizes and fill values are preserved; the compressor is the format-2
default (byte-level layout is allowed to differ -- the manifest is
re-hashed afterwards anyway).

Requirements: zarr >= 3 and numpy, nothing else (zarr 3 is the only
zarr-python that READS both formats)::

    pixi exec --spec "python=3.12" --spec "zarr>=3" --spec numpy \\
        python notebooks/convert_zarr_stores_to_v2.py

After converting, re-hash and re-publish -- the converted stores' bytes
changed::

    python notebooks/build_data_manifest.py --doi <new version DOI>
    python notebooks/build_zenodo_archives.py --doi <new version DOI>

Report only (no writes)::

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


def _array_dims(arr):
    """dimension_names of a format-3 array, if any (else None)."""
    meta = getattr(arr, "metadata", None)
    return getattr(meta, "dimension_names", None)


def _copy_group(src, dst) -> None:
    import numpy as np

    dst.attrs.update(dict(src.attrs))
    for name, arr in sorted(src.arrays()):
        data = arr[...]
        create = getattr(dst, "create_array", None) or dst.create_dataset
        out = create(
            name, shape=arr.shape, dtype=arr.dtype,
            chunks=arr.chunks, fill_value=arr.fill_value,
        )
        out[...] = data
        attrs = dict(arr.attrs)
        dims = _array_dims(arr)
        if dims is not None and "_ARRAY_DIMENSIONS" not in attrs:
            # v3 keeps dimensions in array metadata; v2 keeps them in this
            # attribute. Translate so xarray-written stores stay readable.
            attrs["_ARRAY_DIMENSIONS"] = list(dims)
        out.attrs.update(attrs)
    for name, sub in sorted(src.groups()):
        _copy_group(sub, dst.create_group(name))


def _compare_groups(a, b, where: str, errors: list[str]) -> None:
    import numpy as np

    a_attrs, b_attrs = dict(a.attrs), dict(b.attrs)
    if a_attrs != b_attrs:
        errors.append(f"{where}: group attrs differ")
    a_arrays = dict(a.arrays())
    b_arrays = dict(b.arrays())
    if set(a_arrays) != set(b_arrays):
        errors.append(f"{where}: array sets differ "
                      f"({set(a_arrays) ^ set(b_arrays)})")
        return
    for name, va in a_arrays.items():
        vb = b_arrays[name]
        if va.dtype != vb.dtype:
            errors.append(f"{where}/{name}: dtype {va.dtype} != {vb.dtype}")
            continue
        if va.shape != vb.shape:
            errors.append(f"{where}/{name}: shape {va.shape} != {vb.shape}")
            continue
        xa, xb = np.asarray(va[...]), np.asarray(vb[...])
        equal = (np.array_equal(xa, xb, equal_nan=True)
                 if xa.dtype.kind in "fc" else np.array_equal(xa, xb))
        if not equal:
            errors.append(f"{where}/{name}: values differ")
        ea = dict(va.attrs)
        eb = dict(vb.attrs)
        # The copy may legitimately ADD _ARRAY_DIMENSIONS (translated from
        # v3 metadata); everything else must match exactly.
        dims = _array_dims(va)
        if dims is not None and "_ARRAY_DIMENSIONS" not in ea:
            ea["_ARRAY_DIMENSIONS"] = list(dims)
        if ea != eb:
            errors.append(f"{where}/{name}: attrs differ")
    a_groups = dict(a.groups())
    b_groups = dict(b.groups())
    for key in set(a_groups) | set(b_groups):
        if key not in a_groups or key not in b_groups:
            errors.append(f"{where}: child group {key!r} missing on one side")
            continue
        _compare_groups(a_groups[key], b_groups[key], f"{where}/{key}", errors)


def convert(path: Path) -> bool:
    import zarr

    if int(str(zarr.__version__).split(".")[0]) < 3:
        raise SystemExit(
            f"zarr-python {zarr.__version__} cannot READ the format-3 stores "
            "to be converted. Run this script with zarr>=3 (see docstring)."
        )

    tmp = path.with_name(path.name + ".v2tmp")
    bak = path.with_name(path.name + ".v3bak")
    for stale in (tmp, bak):
        if stale.exists():
            shutil.rmtree(stale)

    src = zarr.open_group(str(path), mode="r")
    dst = zarr.open_group(str(tmp), mode="w", zarr_format=2)
    _copy_group(src, dst)

    src = zarr.open_group(str(path), mode="r")
    dst = zarr.open_group(str(tmp), mode="r")
    errors: list[str] = []
    _compare_groups(src, dst, path.name, errors)
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
