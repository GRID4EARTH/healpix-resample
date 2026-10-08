#!/usr/bin/env python3
"""Publish notebook outputs into the paper's asset directories.

`tex/figures/` and `tex/tables/` are *separate copies* of what the notebooks
write to `notebooks/figures/` and `notebooks/tables/`. Nothing in the build
links the two, so every time an experiment is re-run the paper silently keeps
compiling against the previous version of the figure. That has already
happened three times in this project's history -- most recently with the
corrected spectral figures, which were regenerated on 25 August but only
reached `tex/figures/` a week later, so the submitted PDF still carried the
buggy frequency axis.

This script closes that gap and is meant to be called at the end of every
experiment notebook:

    from publish_paper_assets import publish
    publish()

Figures are discovered from `tex/main.tex` AND `tex/supplement.tex`: every
`\\includegraphics` target is resolved against `notebooks/figures/`. The two
documents are therefore the authority on what needs publishing, and a figure
that both stopped citing simply stops being copied.

Tables have no such link (the paper hard-codes its numbers rather than
`\\input`-ing them), so the CSVs that back each table are declared explicitly
below. They are archived next to the paper for provenance, not read by LaTeX.

Run standalone for a report without copying anything:

    python notebooks/publish_paper_assets.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

# CSVs backing the paper's tables. Key: the file in notebooks/tables/.
# Kept explicit because LaTeX never reads them -- they are archived for
# provenance, so an out-of-date copy is a reproducibility problem, not a
# rendering one.
TABLE_SOURCES = [
    "table_I_synthetic.csv",
    "table_II_roundtrip.csv",
    "table_ablation_rl_fair.csv",
    "estimand_point_vs_average.csv",
    "holdout_validation.csv",
    "multi_patch_latitude_summary.csv",
    "multi_patch_method_comparisons.csv",
    "multi_patch_two_stage_40_summary.csv",
    "multi_patch_two_stage_40_comparisons.csv",
    "multi_patch_run_summary.csv",
    "multi_patch_runtime_by_method.csv",
    "real_groundtruth_downscale_table.csv",
    "real_groundtruth_downscale_metrics.csv",
    "real_groundtruth_downscale_anisotropic_24x45_table.csv",
    "real_groundtruth_downscale_width_sweep.csv",
    "real_groundtruth_multiregion_xref_ablation.csv",
    "real_groundtruth_multiregion_table.csv",
    "real_groundtruth_multiregion_products.csv",
    "real_groundtruth_multiregion_quality.csv",
    "noise_sensitivity_psf_mismatch.csv",
    "geolocation_sensitivity.csv",
    "throughput_scaling_by_batch.csv",
    "throughput_scaling_environment.csv",
]


def find_repo_root(start=None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "healpix_resample").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate the healpix-resample checkout above {current}.")


def _sha256(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def figures_cited_by_paper(*tex_files: Path) -> list[str]:
    """Basenames of every figure the given documents include, in citation
    order, main paper first. Missing files are skipped (the supplement may
    not exist in every branch)."""
    seen, out = set(), []
    for tex_path in tex_files:
        if not tex_path.exists():
            continue
        tex = tex_path.read_text(encoding="utf-8")
        targets = re.findall(
            r"\\includegraphics(?:\[[^\]]*\])?\s*\{\s*([^}]+?)\s*\}", tex)
        for t in targets:
            name = Path(t.strip()).name
            if not Path(name).suffix:
                name += ".pdf"
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _sync(src: Path, dst: Path, check_only: bool, force: bool = False) -> str:
    if not src.exists():
        return "missing-source"
    if dst.exists() and _sha256(src) == _sha256(dst):
        return "up-to-date"
    if dst.exists() and not force and dst.stat().st_mtime > src.stat().st_mtime:
        # The paper's copy is NEWER than the notebook's. That is the signature
        # of a hand-maintained asset -- fig01_psf_healpix_workflow.pdf is drawn
        # from an SVG and edited directly under tex/figures/, so copying the
        # notebook's older export over it would silently regress the figure.
        # Never clobber on a guess; report and let the author decide.
        return "newer-in-tex"
    verb = "would-update" if check_only else "updated"
    if not dst.exists():
        verb = "would-add" if check_only else "added"
    if not check_only:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return verb


def publish(repo_root=None, check_only: bool = False, verbose: bool = True,
            force: bool = False) -> dict:
    """Copy every paper-cited figure and every declared table CSV into tex/.

    Returns a dict of status -> list of names. Raises nothing on a missing
    source: it is reported instead, because a figure can legitimately be
    pending a re-run.

    An asset whose copy under `tex/` is newer than the notebook's is reported
    as `newer-in-tex` and left alone, since that is how a hand-maintained
    figure looks. Pass `force=True` to overwrite those too.
    """
    root = Path(repo_root) if repo_root else find_repo_root()
    nb_fig, nb_tab = root / "notebooks" / "figures", root / "notebooks" / "tables"
    tex_fig, tex_tab = root / "tex" / "figures", root / "tex" / "tables"
    main_tex = root / "tex" / "main.tex"
    supp_tex = root / "tex" / "supplement.tex"

    report: dict[str, list[str]] = {}

    def record(status, name):
        report.setdefault(status, []).append(name)

    cited = figures_cited_by_paper(main_tex, supp_tex)
    for name in cited:
        record(_sync(nb_fig / name, tex_fig / name, check_only, force), f"figures/{name}")

    for name in TABLE_SOURCES:
        record(_sync(nb_tab / name, tex_tab / name, check_only, force), f"tables/{name}")

    if verbose:
        print(f"publish_paper_assets: {len(cited)} figures cited by "
              f"main.tex+supplement.tex, {len(TABLE_SOURCES)} declared table CSVs")
        for status in ("added", "updated", "would-add", "would-update",
                       "newer-in-tex", "up-to-date", "missing-source"):
            names = report.get(status)
            if not names:
                continue
            if status == "up-to-date":
                print(f"  up-to-date     : {len(names)}")
                continue
            print(f"  {status:<15}: {len(names)}")
            for n in names:
                print(f"      {n}")
        if report.get("newer-in-tex"):
            print("\n  NOTE: 'newer-in-tex' assets are hand-maintained under tex/ "
                  "(e.g. the workflow diagram drawn from an SVG) and were NOT "
                  "overwritten. Pass force=True only if you really mean to "
                  "replace them with the notebook export.")
        missing = report.get("missing-source", [])
        if missing:
            print("\n  NOTE: a missing source means the paper cites an asset no "
                  "notebook has produced yet (or it lives elsewhere). Re-run the "
                  "notebook that writes it, then call publish() again.")
        stale = [s for s in ("added", "updated") if report.get(s)]
        if stale and not check_only:
            print("\n  tex/ was out of date and has been refreshed -- recompile the paper.")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="report what would change without copying anything")
    ap.add_argument("--force", action="store_true",
                    help="also overwrite assets whose tex/ copy is newer")
    args = ap.parse_args(argv)
    report = publish(check_only=args.check, force=args.force)
    if args.check and (report.get("would-add") or report.get("would-update")):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
