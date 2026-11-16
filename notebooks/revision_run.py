"""Command-line driver for the JSTARS revision experiments E0, E1 and E2.

Run inside the paper environment, from anywhere in the repository::

    pixi run -e notebooks python notebooks/revision_run.py fetch-esri
    pixi run -e notebooks python notebooks/revision_run.py e0 real
    pixi run -e notebooks python notebooks/revision_run.py e0 synthetic
    pixi run -e notebooks python notebooks/revision_run.py e1e2 real
    pixi run -e notebooks python notebooks/revision_run.py e1e2 synthetic
    pixi run -e notebooks python notebooks/revision_run.py analyze real
    pixi run -e notebooks python notebooks/revision_run.py selftest

``--limit N`` keeps N regions per scene class (smoke test; ``--limit 1`` runs
four regions).  ``--shard k/n`` splits ``e1e2`` across the n tasks of a job
array.  Per-region results are written as soon as a region finishes, so an
interrupted job resumes where it stopped (``--force`` recomputes).
Protocol: review/08_protocole_preenregistre.md.  Outputs: notebooks/tables/revision/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent
os.chdir(NOTEBOOK_DIR)                     # the paper modules use relative paths
sys.path.insert(0, str(NOTEBOOK_DIR))

import revision_tools as rt  # noqa: E402


def _limit_per_class(items, class_of, limit):
    if not limit:
        return list(items)
    kept, counts = [], {}
    for item in items:
        cls = class_of(item)
        if counts.get(cls, 0) < limit:
            kept.append(item)
            counts[cls] = counts.get(cls, 0) + 1
    return kept


def _real_scenes(gt, limit):
    scenes = rt.real_usable_scenes(gt)
    return _limit_per_class(scenes, lambda s: gt.region_coordinates[s]["scene_class"], limit)


def _synthetic_patches(ns, limit):
    return _limit_per_class(rt.synthetic_selected_patches(ns), lambda r: r.scene_class, limit)


def cmd_fetch_esri(args):
    """Network step (login node): regenerate the Esri textures of the 40
    selected patches (or of all 360 with --all) from the pinned Wayback release."""
    ns = rt.load_synthetic_namespace(offline=False)
    if args.all:
        rows = list(ns["patch_manifest"].itertuples(index=False))
    else:
        rows = _synthetic_patches(ns, args.limit)
    for i, row in enumerate(rows, 1):
        grids = ns["patch_coordinate_grids"](row)
        _, info = ns["load_or_sample_esri_patch"](row, grids)
        print(f"[{i:3d}/{len(rows)}] {row.patch_id}: {info['esri_patch_cache_status']}",
              flush=True)


def cmd_e0(args):
    rt.write_provenance(f"e0_{args.bench}")
    if args.bench == "real":
        gt = rt.load_gt()
        merged = rt.e0_real(gt, _real_scenes(gt, args.limit))
    else:
        ns = rt.load_synthetic_namespace(offline=not args.online)
        merged = rt.e0_synthetic(ns, _synthetic_patches(ns, args.limit))
    bad = merged[~merged["ok"]]
    if len(bad):
        print("\nPairs above tolerance (see the CSV for all of them):")
        print(bad.sort_values("rel_diff", ascending=False).head(20).to_string(index=False))
        return 1
    return 0


def cmd_e1e2(args):
    rt.write_provenance(f"e1e2_{args.bench}", {"limit": args.limit, "no_e2": args.no_e2})
    if args.bench == "real":
        gt = rt.load_gt()
        items = _real_scenes(gt, args.limit)

        def run(item):
            return rt.run_real_region(gt, item, do_e2=not args.no_e2, force=args.force)

        def name(item):
            return item
    else:
        ns = rt.load_synthetic_namespace(offline=not args.online)
        items = _synthetic_patches(ns, args.limit)

        def run(item):
            return rt.run_synthetic_region(ns, item, do_e2=not args.no_e2, force=args.force)

        def name(item):
            return item.patch_id

    if args.shard:
        k, n = (int(v) for v in args.shard.split("/"))
        if not 0 <= k < n:
            raise SystemExit(f"--shard {args.shard}: need 0 <= k < n")
        items = items[k::n]
        print(f"shard {k}/{n}: {len(items)} region(s)")

    failures = []
    for i, item in enumerate(items, 1):
        t0 = time.perf_counter()
        try:
            frame = run(item)
            print(f"[{i:3d}/{len(items)}] {name(item)}: {len(frame)} rows "
                  f"({time.perf_counter() - t0:.1f} s)", flush=True)
        except Exception as exc:
            failures.append(name(item))
            print(f"[{i:3d}/{len(items)}] [FAIL] {name(item)}: {type(exc).__name__}: {exc}",
                  flush=True)
            traceback.print_exc()
    print(f"\n{len(items) - len(failures)}/{len(items)} regions done; failures: {failures}")
    return 1 if failures else 0


def cmd_analyze(args):
    rt.analyze(args.bench)
    return 0


def cmd_selftest(args):
    import revision_selftest
    return revision_selftest.main()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch-esri", help="download/regenerate Esri textures (network)")
    p.add_argument("--all", action="store_true", help="all 360 patches, not only the 40")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_fetch_esri)

    for name, func, help_ in (("e0", cmd_e0, "regression against the submitted CSVs"),
                              ("e1e2", cmd_e1e2, "tuned baselines (E1) and inversion ablation (E2)"),
                              ("analyze", cmd_analyze, "cross-validation, oracle, paired statistics")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("bench", choices=("real", "synthetic"))
        if name != "analyze":
            p.add_argument("--limit", type=int, default=0,
                           help="regions per scene class (0 = all)")
            p.add_argument("--online", action="store_true",
                           help="synthetic: allow fetching missing Esri tiles")
        if name == "e1e2":
            p.add_argument("--force", action="store_true", help="recompute finished regions")
            p.add_argument("--no-e2", action="store_true", help="skip the E2 inversion arms")
            p.add_argument("--shard", default="",
                           help="k/n: process regions k, k+n, k+2n, ... (job arrays)")
        p.set_defaults(func=func)

    p = sub.add_parser("selftest", help="numerical self-tests (no data needed)")
    p.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
