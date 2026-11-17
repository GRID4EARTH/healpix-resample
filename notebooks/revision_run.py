"""Command-line driver for the JSTARS revision experiments E0 to E7.

Run inside the paper environment, from anywhere in the repository::

    pixi run -e notebooks python notebooks/revision_run.py fetch-esri
    pixi run -e notebooks python notebooks/revision_run.py e0 real
    pixi run -e notebooks python notebooks/revision_run.py e0 synthetic
    pixi run -e notebooks python notebooks/revision_run.py e1e2 real
    pixi run -e notebooks python notebooks/revision_run.py e1e2 synthetic
    pixi run -e notebooks python notebooks/revision_run.py analyze real
    pixi run -e notebooks python notebooks/revision_run.py e4 real          # E4 width complement
    pixi run -e notebooks python notebooks/revision_run.py e4-analyze      # E4 tables and figures
    pixi run -e notebooks python notebooks/revision_run.py e3 S1           # E3 arms S1, S3
    pixi run -e notebooks python notebooks/revision_run.py e3-analyze      # E3 tables and figure
    pixi run -e notebooks python notebooks/revision_run.py e5-fetch        # E5 data (network)
    pixi run -e notebooks python notebooks/revision_run.py e5-manifest     # E5 filters, sample
    pixi run -e notebooks python notebooks/revision_run.py e5              # E5 runs
    pixi run -e notebooks python notebooks/revision_run.py e5 --part ext   # E5 amendment 3 grids
    pixi run -e notebooks python notebooks/revision_run.py e5-analyze      # E5 tables and figure
    pixi run -e notebooks python notebooks/revision_run.py e5-spectral     # E5-S (exploratory)
    pixi run -e notebooks python notebooks/revision_run.py e6-fetch all    # E6 data (network)
    pixi run -e notebooks python notebooks/revision_run.py e6 a_b02        # E6 arms
    pixi run -e notebooks python notebooks/revision_run.py e6-analyze      # E6 tables and figure
    pixi run -e notebooks python notebooks/revision_run.py e7              # E7 operators
    pixi run -e notebooks python notebooks/revision_run.py e7-analyze      # E7 tables and figure
    pixi run -e notebooks python notebooks/revision_run.py selftest

``--limit N`` keeps N regions per scene class (smoke test; ``--limit 1`` runs
four regions).  ``--shard k/n`` splits ``e1e2`` across the n tasks of a job
array (also ``e4`` and ``e3``).  Per-region results are written as soon as a region finishes, so an
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
    return _run_regions(args, dict(do_e2=not args.no_e2))


def cmd_e4(args):
    """E4 complement: PSF-aware and the E1 baselines at E4_EXTRA_WIDTHS."""
    rt.write_provenance(f"e4_{args.bench}", {"limit": args.limit,
                                             "widths": rt.E4_EXTRA_WIDTHS})
    return _run_regions(args, dict(do_e2=False, widths=rt.E4_EXTRA_WIDTHS, study="e4"))


def cmd_e4_analyze(args):
    for bench in ("real", "synthetic"):
        rt.e4_analyze(bench)
    rt.e4_figures(n_iter=30)
    rt.e4_figures(n_iter=14)
    return 0


def cmd_e3(args):
    """E3 arm S1 (point-sampled data) or S3 (exact Gaussian x box kernel)."""
    args.bench = "synthetic"
    study = rt.e3_study(args.arm)
    rt.write_provenance(study + "_synthetic", {"limit": args.limit, "arm": args.arm})
    return _run_regions(args, dict(do_e2=False, widths=(1.0,), study=study, arm=args.arm))


def cmd_e3_analyze(args):
    rt.e3_analyze()
    rt.e3_figure()
    return 0


def cmd_e5_fetch(args):
    import revision_e5 as e5
    e5.fetch()
    return 0


def cmd_e5_manifest(args):
    import revision_e5 as e5
    e5.build_manifest(shard=args.shard)
    return 0


def cmd_e5(args):
    """E5: every method on the sampled SEN2VENmuS patches."""
    import revision_e5 as e5
    rt.write_provenance("e5" if args.part == "main" else "e5_ext", {"limit": args.limit})
    items = e5.selected_patches()
    if args.limit:
        kept, seen = [], {}
        for it in items:
            if seen.get(it.site, 0) < args.limit:
                kept.append(it)
                seen[it.site] = seen.get(it.site, 0) + 1
        items = kept
    if args.shard:
        k, n = (int(v) for v in args.shard.split("/"))
        items = items[k::n]
        print(f"shard {k}/{n}: {len(items)} patch(es)")
    failures = []
    for i, item in enumerate(items, 1):
        t0 = time.perf_counter()
        try:
            frame = e5.run_patch(item, force=args.force, part=args.part)
            print(f"[{i:3d}/{len(items)}] {item.patch_id}: {len(frame)} rows "
                  f"({time.perf_counter() - t0:.1f} s)", flush=True)
        except Exception as exc:
            failures.append(item.patch_id)
            print(f"[{i:3d}/{len(items)}] [FAIL] {item.patch_id}: {type(exc).__name__}: {exc}",
                  flush=True)
            traceback.print_exc()
    print(f"\n{len(items) - len(failures)}/{len(items)} patches done; failures: {failures}")
    return 1 if failures else 0


def cmd_e5_analyze(args):
    import revision_e5 as e5
    e5.e5_analyze()
    return 0


def cmd_e5_spectral(args):
    """E5-S: exploratory Sentinel-2 / VENmuS spectral comparison."""
    import revision_e5_spectral as e5s
    if args.analyze_only:
        e5s.analyze()
        return 0
    rt.write_provenance("e5_spectral", {"limit": args.limit})
    e5s.run(limit=args.limit)
    return 0


def cmd_e6_fetch(args):
    """Network step (login node): data of one E6 dataset, or of all."""
    import revision_e6 as e6
    gt = rt.load_gt()
    gt.OFFLINE = False
    names = list(e6.DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        print(f"=== {name}", flush=True)
        e6.fetch(gt, name)
    return 0


def cmd_e6(args):
    import revision_e6 as e6
    arm = e6.ARMS[args.arm]
    rt.write_provenance(arm.study + "_real", {"limit": args.limit, "arm": args.arm})
    return e6.run_arm(rt.load_gt(), args.arm, shard=args.shard, force=args.force,
                      limit=args.limit)


def cmd_e6_analyze(args):
    import revision_e6 as e6
    e6.e6_analyze()
    return 0


def cmd_e7(args):
    import revision_e7 as e7
    rt.write_provenance("e7_real", {"limit": args.limit})
    return e7.run(rt.load_gt(), shard=args.shard, force=args.force, limit=args.limit)


def cmd_e7_analyze(args):
    import revision_e7 as e7
    e7.e7_analyze()
    return 0


def _run_regions(args, kwargs):
    if args.bench == "real":
        gt = rt.load_gt()
        items = _real_scenes(gt, args.limit)

        def run(item):
            return rt.run_real_region(gt, item, force=args.force, **kwargs)

        def name(item):
            return item
    else:
        ns = rt.load_synthetic_namespace(offline=not args.online)
        items = _synthetic_patches(ns, args.limit)

        def run(item):
            return rt.run_synthetic_region(ns, item, force=args.force, **kwargs)

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
                              ("e4", cmd_e4, "width complement for the width x lambda study (E4)"),
                              ("analyze", cmd_analyze, "cross-validation, oracle, paired statistics")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("bench", choices=("real", "synthetic"))
        if name != "analyze":
            p.add_argument("--limit", type=int, default=0,
                           help="regions per scene class (0 = all)")
            p.add_argument("--online", action="store_true",
                           help="synthetic: allow fetching missing Esri tiles")
        if name in ("e1e2", "e4"):
            p.add_argument("--force", action="store_true", help="recompute finished regions")
            p.add_argument("--shard", default="",
                           help="k/n: process regions k, k+n, k+2n, ... (job arrays)")
        if name == "e1e2":
            p.add_argument("--no-e2", action="store_true", help="skip the E2 inversion arms")
        p.set_defaults(func=func)

    p = sub.add_parser("e4-analyze", help="E4 tables and figures (both benchmarks)")
    p.set_defaults(func=cmd_e4_analyze)

    p = sub.add_parser("e3", help="response-mismatch arm S1 or S3 (E3, synthetic)")
    p.add_argument("arm", choices=rt.E3_NEW_ARMS)
    p.add_argument("--limit", type=int, default=0, help="regions per scene class (0 = all)")
    p.add_argument("--online", action="store_true",
                   help="allow fetching missing Esri tiles")
    p.add_argument("--force", action="store_true", help="recompute finished regions")
    p.add_argument("--shard", default="",
                   help="k/n: process regions k, k+n, k+2n, ... (job arrays)")
    p.set_defaults(func=cmd_e3)

    p = sub.add_parser("e3-analyze", help="E3 tables and figure")
    p.set_defaults(func=cmd_e3_analyze)

    p = sub.add_parser("e5-fetch", help="download the E5 SEN2VENmuS sites (network)")
    p.set_defaults(func=cmd_e5_fetch)
    p = sub.add_parser("e5-manifest", help="E5 input filters and patch sample")
    p.add_argument("--shard", default="",
                   help="k/n: only screen sites k, k+n, ... (no sample; job arrays)")
    p.set_defaults(func=cmd_e5_manifest)
    p = sub.add_parser("e5", help="E5: every method on the sampled SEN2VENmuS patches")
    p.add_argument("--limit", type=int, default=0, help="patches per site (0 = all)")
    p.add_argument("--force", action="store_true", help="recompute finished patches")
    p.add_argument("--shard", default="",
                   help="k/n: process patches k, k+n, k+2n, ... (job arrays)")
    p.add_argument("--part", choices=("main", "ext"), default="main",
                   help="main: pre-registered grid; ext: amendment 3 extension only")
    p.set_defaults(func=cmd_e5)
    p = sub.add_parser("e5-analyze", help="E5 tables and figure")
    p.set_defaults(func=cmd_e5_analyze)
    p = sub.add_parser("e5-spectral",
                       help="E5-S: exploratory S2/VENmuS spectra (not pre-registered)")
    p.add_argument("--limit", type=int, default=0, help="patches per site (0 = all)")
    p.add_argument("--analyze-only", action="store_true",
                   help="recompute tables and figure from the saved ring sums")
    p.set_defaults(func=cmd_e5_spectral)

    import revision_e6 as e6
    p = sub.add_parser("e6-fetch", help="download the data of an E6 dataset (network)")
    p.add_argument("dataset", choices=list(e6.DATASETS) + ["all"])
    p.set_defaults(func=cmd_e6_fetch)
    for name, func, help_ in (("e6", cmd_e6, "E6: one generalization arm"),
                              ("e7", cmd_e7, "E7: operators and conservation")):
        p = sub.add_parser(name, help=help_)
        if name == "e6":
            p.add_argument("arm", choices=list(e6.ARMS))
        p.add_argument("--limit", type=int, default=0, help="first N regions (0 = all)")
        p.add_argument("--force", action="store_true", help="recompute finished regions")
        p.add_argument("--shard", default="",
                       help="k/n: process regions k, k+n, k+2n, ... (job arrays)")
        p.set_defaults(func=func)
    p = sub.add_parser("e6-analyze", help="E6 tables and figure")
    p.set_defaults(func=cmd_e6_analyze)
    p = sub.add_parser("e7-analyze", help="E7 tables and figure")
    p.set_defaults(func=cmd_e7_analyze)

    p = sub.add_parser("selftest", help="numerical self-tests (no data needed)")
    p.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
