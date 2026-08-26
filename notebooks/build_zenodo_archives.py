#!/usr/bin/env python3
"""Build the three ZIP archives of the frozen input bundle for Zenodo.

Run from anywhere inside the checkout, after the manifest check passes::

    python notebooks/build_zenodo_archives.py --doi 10.5281/zenodo.<reserved>

It writes `git_commit.txt`, packs the three archives, and prints the
`EXPECTED_ARCHIVES` block to paste into `notebooks/load_data_in_zenodo.ipynb`.

Why a script rather than a few `zip` commands
---------------------------------------------
Three properties matter here and none of them survives being retyped by hand.

*Path fidelity.* Every member must keep the repository-relative path
(`notebooks/data/...`), because `load_data_in_zenodo.ipynb` extracts straight
into a clean checkout. A `zip` run from the wrong directory produces an archive
that unpacks one level off, and the failure only shows up on the reproduction
machine.

*Determinism.* Members are added in sorted order with a fixed timestamp and
fixed permissions, so rebuilding the bundle from the same inputs yields the
same bytes and therefore the same MD5. Without that, the checksums recorded in
the notebook are tied to one particular run on one particular machine, and a
reviewer who rebuilds the archive gets a different digest for identical data --
which looks exactly like corruption.

*Exclusions.* Derived files (`*.idx`, `__pycache__`, `.DS_Store`) must not
enter a bundle described as immutable primary input.

The archive split keeps the 1.1 GB Esri bundle byte-identical to v1, so a new
Zenodo version can carry it over instead of re-uploading it.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Fixed DOS timestamp for every member (1980-01-01), so the archive is
# reproducible. Zarr chunk mtimes carry no information worth preserving.
FIXED_DATE = (1980, 1, 1, 0, 0, 0)

EXCLUDE_SUFFIXES = {".idx", ".tmp", ".pyc"}
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db"}
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints"}

ROOT_FILES = ["data_manifest.csv", "git_commit.txt", "ATTRIBUTION.md",
              "DATA_LICENSE.md", "DATA_DEPOSIT.md"]


def find_repo_root(start=None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "healpix_resample").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise SystemExit(f"Not inside a healpix-resample checkout: {current}")


def git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def write_git_commit(repo: Path, doi: str, allow_dirty: bool) -> Path:
    """Record exactly which code produced the archived inputs.

    The dataset is only useful if it can be paired with the code that reads
    it, and "the version on GitHub" is not an identifier -- the default branch
    moves. This pins a commit hash, and refuses to run on a dirty tree, since
    a hash that does not describe the working files is worse than no hash: it
    is a false provenance claim that looks authoritative.
    """
    try:
        commit = git(repo, "rev-parse", "HEAD")
        short = git(repo, "rev-parse", "--short", "HEAD")
        branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        described = git(repo, "describe", "--always", "--dirty", "--tags")
        commit_date = git(repo, "show", "-s", "--format=%cI", "HEAD")
        status = git(repo, "status", "--porcelain")
        try:
            remote = git(repo, "config", "--get", "remote.origin.url")
        except subprocess.CalledProcessError:
            remote = "(no origin remote)"
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(f"Cannot read git metadata: {exc}")

    if status and not allow_dirty:
        print("Working tree is not clean. The archive would claim a commit "
              "that does not describe the files being packed:\n")
        print(status)
        raise SystemExit(
            "\nCommit or stash these changes, then re-run. Use --allow-dirty "
            "only if you accept publishing an inexact provenance record."
        )

    path = repo / "git_commit.txt"
    path.write_text(
        "healpix-resample frozen input bundle\n"
        f"commit:      {commit}\n"
        f"short:       {short}\n"
        f"branch:      {branch}\n"
        f"describe:    {described}\n"
        f"commit_date: {commit_date}\n"
        f"remote:      {remote}\n"
        f"dataset_doi: {doi or '(not yet reserved)'}\n"
        f"packed_utc:  {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"clean_tree:  {'no' if status else 'yes'}\n",
        encoding="utf-8",
    )
    print(f"git_commit.txt -> {short} on {branch} "
          f"({'dirty' if status else 'clean'})")
    return path


def collect(base: Path, repo: Path) -> list[Path]:
    if not base.exists():
        return []
    if base.is_file():
        return [base]
    out = []
    for item in sorted(base.rglob("*")):
        if not item.is_file():
            continue
        if set(item.relative_to(repo).parts) & EXCLUDE_DIRS:
            continue
        if item.suffix in EXCLUDE_SUFFIXES or item.name in EXCLUDE_NAMES:
            continue
        out.append(item)
    return sorted(out)


def build(zip_path: Path, members: list[Path], repo: Path) -> tuple[int, str]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        for item in members:
            info = zipfile.ZipInfo(
                item.relative_to(repo).as_posix(), date_time=FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, item.read_bytes())
    digest = hashlib.md5()
    with zip_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return zip_path.stat().st_size, digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--doi", default="", help="reserved Zenodo DOI")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="where to write the ZIPs (default: <repo>/dist)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="pack even if the working tree has uncommitted changes")
    ap.add_argument("--skip-esri", action="store_true",
                    help="do not rebuild the 1.1 GB Esri archive (unchanged "
                         "since v1; carry the published file over instead)")
    args = ap.parse_args()

    repo = find_repo_root()
    outdir = args.outdir or (repo / "dist")
    write_git_commit(repo, args.doi, args.allow_dirty)

    nb = repo / "notebooks"
    core = []
    for name in ROOT_FILES:
        core += collect(repo / name, repo)
    core += collect(nb / "outputFLUX.grib", repo)
    for scene in ("urban", "water", "forest", "agriculture"):
        core += collect(nb / "data" / f"{scene}_data.zarr", repo)
        core += collect(nb / "data" / "esri_latent" /
                        f"{scene}__z17_n256_os4.zarr", repo)
    core += collect(nb / "data" / "README.md", repo)

    archives = {
        "healpix-resample-paper-core-data-v2.zip": core,
        "healpix-resample-paper-sentinel2-regions-v1.zip":
            collect(nb / "data" / "multi_patch_sentinel2", repo),
    }
    if not args.skip_esri:
        archives["healpix-resample-paper-esri-multipatch-v1.zip"] = collect(
            nb / "data" / "multi_patch_latitude" / "esri_patch_cache", repo)

    results = {}
    for name, members in archives.items():
        if not members:
            print(f"!! {name}: no files found -- check the data is installed")
            continue
        size, md5 = build(outdir / name, members, repo)
        results[name] = (size, md5)
        print(f"{name}: {len(members):,} files, {size / 1e6:,.1f} MB, md5 {md5}")

    print("\nPaste into notebooks/load_data_in_zenodo.ipynb:\n")
    print("EXPECTED_ARCHIVES = {")
    for name, (size, md5) in sorted(results.items()):
        print(f'    "{name}": {{')
        print(f'        "size": {size:_},')
        print(f'        "md5": "{md5}",')
        print("    },")
    if args.skip_esri:
        print("    # plus the carried-over Esri archive: keep its v1 entry.")
    print("}")
    print(f"\nArchives written to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
