"""E7 of the JSTARS revision: which operator for which use (R2.4).

On the 40 real regions of E1 (B04, 20 m observation, L19 target), four
operators are compared on two properties: fidelity to the reference field
(RMSE, as in E1) and conservation of the area integral of the input
(protocol section E7 of ``review/08_protocole_preenregistre.md``):

* ``psf_aware``       -- the E1 configuration transferred in mode A;
* ``psf_aware_cons``  -- the same with ``conservative=True``;
* ``hard_binning``    -- ``ConservativeResampler`` (sample -> containing cell);
* ``overlap1``        -- first-order overlap weights, areas by quadrature.
"""
from __future__ import annotations

import time

import numpy as np

import revision_tools as rt

QUAD = 10                     # sub-squares per coarse pixel side (overlap quadrature)
E7_OPERATORS = ("psf_aware", "psf_aware_cons", "hard_binning", "overlap1", "geo_cubic")
E7_COLUMNS = ["scene", "scene_class", "region_id", "operator", "config", "rmse", "n_cells",
              "fill_fraction", "cons_err", "mean_err", "n_support", "runtime_s"]


def overlap_weights(x_axis, y_axis, pixel_m, to_cells, quad=QUAD):
    """First-order overlap weights of the square pixels centred on the
    (x_axis, y_axis) grid: O[j, i] = area of pixel i in cell j, in pixel
    units, from quad x quad sub-squares whose centres are assigned to their
    cell by ``to_cells(x, y) -> cell ids``.  Each pixel's weights sum to 1
    exactly, so the integral is conserved whatever the quadrature error.
    Pixels are ordered like ``img.reshape(-1)`` for img[len(y), len(x)]."""
    import scipy.sparse as sp
    off = ((np.arange(quad) + 0.5) / quad - 0.5) * pixel_m
    ox, oy = np.meshgrid(off, off)
    xx, yy = np.meshgrid(np.asarray(x_axis, float), np.asarray(y_axis, float))
    px = (xx.reshape(-1)[:, None] + ox.reshape(-1)[None, :]).reshape(-1)
    py = (yy.reshape(-1)[:, None] + oy.reshape(-1)[None, :]).reshape(-1)
    ids = np.asarray(to_cells(px, py), dtype=np.int64)
    n_pix = xx.size
    pix = np.repeat(np.arange(n_pix), quad * quad)
    cells, row = np.unique(ids, return_inverse=True)
    O = sp.csr_matrix((np.full(ids.size, 1.0 / quad ** 2), (row, pix)),
                      shape=(cells.size, n_pix))
    O.sum_duplicates()
    return cells, O


def overlap_remap(O, y):
    """Covered-area normalization: x_j = sum_i O_ji y_i / sum_i O_ji."""
    covered = np.asarray(O.sum(axis=1)).ravel()
    return (O @ y) / covered, covered


def conservation_error(weights, values, total_in):
    w = np.asarray(weights, float)
    v = np.asarray(values, float)
    m = np.isfinite(v)
    return float((np.sum(w[m] * v[m]) - total_in) / total_in)


def run_region(gt, scene, psf_cfg, force=False):
    """All E7 operators on one region; returns the region's frame."""
    import pandas as pd
    import torch  # noqa: F401  (the package's resamplers need it)
    from healpix_resample import ConservativeResampler
    out_path = rt.region_csv_path("real", scene, "e7")
    if out_path.exists() and not force:
        return pd.read_csv(out_path)
    meta_src = gt.region_coordinates[scene]
    meta = dict(scene=scene, scene_class=meta_src["scene_class"], region_id=meta_src["region_id"])
    ref = gt.build_reference(scene, force=False)
    g = gt.build_coarse_grid(scene)
    _, t_all, _ = gt.align_and_mask(ref["cell_ids_ref"], ref["cell_data_ref"], ref, g)
    n_interior = int(np.isfinite(t_all).sum())
    y = g["img_coarse"].reshape(-1).astype(np.float64)
    lon = g["lon_coarse"].reshape(-1).astype(np.float64)
    lat = g["lat_coarse"].reshape(-1).astype(np.float64)
    area = np.ones_like(y)            # 20 m UTM pixels: equal areas
    total_in = float(np.sum(area * y))
    rows = []

    def score(op, ids, vals, cons_err, n_support, runtime_s, config=""):
        e, t, _ = gt.align_and_mask(np.asarray(ids), np.asarray(vals), ref, g)
        m = gt.compute_metrics(e, t, scene, op)
        ok = np.isfinite(e) & np.isfinite(t)
        mean_err = float((e[ok].mean() - t[ok].mean()) / t[ok].mean()) if ok.any() else np.nan
        rows.append(dict(**meta, operator=op, config=config, rmse=m["rmse"],
                         n_cells=m["n_cells"], fill_fraction=m["n_cells"] / n_interior,
                         cons_err=cons_err, mean_err=mean_err, n_support=int(n_support),
                         runtime_s=runtime_s))

    # PSF-aware, unconstrained and constrained: the operator on its own support.
    lam, n_iter = psf_cfg
    scale = gt.fwhm_to_scale(gt.RECON_FWHM_M)
    npt = gt.recommend_npt(scale, gt.REF_LEVEL)["npt"]
    t0 = time.perf_counter()
    rs = gt.PSFResampler(lon_deg=lon, lat_deg=lat, level=gt.REF_LEVEL, sigma_m=scale, Npt=npt,
                         threshold=gt.THRESHOLD, device=gt.DEVICE, verbose=False, area=area)
    build_s = time.perf_counter() - t0
    for op, cons in (("psf_aware", False), ("psf_aware_cons", True)):
        t1 = time.perf_counter()
        res = rs.resample(y, lam=lam, max_iter=n_iter, tol=rt.PSF_TOL, conservative=cons)
        x = rt.to_numpy(res.cell_data).astype(np.float64)
        ids = rt.to_numpy(res.cell_ids)
        # package statement: unweighted mean of the K output cells equals the
        # area-weighted mean of the samples, i.e. weights sum(area) / K
        w = np.full(x.size, area.sum() / x.size)
        score(op, ids, x, conservation_error(w, x, total_in), x.size,
              build_s + time.perf_counter() - t1, f"lam={lam:g}|it={n_iter}")
    del rs

    # Hard binning: each sample to its containing cell, value h_k / A_k.
    t0 = time.perf_counter()
    cr = ConservativeResampler(lon_deg=lon, lat_deg=lat, level=gt.REF_LEVEL, area=area,
                               ellipsoid=gt.ELLIPSOID, verbose=False)
    res = cr.resample(y)
    h = rt.to_numpy(res.cell_data).astype(np.float64)
    A = rt.to_numpy(cr.cell_area).astype(np.float64)
    ids = rt.to_numpy(res.cell_ids)
    score("hard_binning", ids, h / A, conservation_error(A, h / A, total_in), ids.size,
          time.perf_counter() - t0)

    # First-order overlap: pixel squares on the coarse grid, quadrature areas.
    import healpix_geo
    import pyproj
    t0 = time.perf_counter()
    to_ll = pyproj.Transformer.from_crs(gt._crs_of(g["da"]), pyproj.CRS.from_epsg(4326),
                                        always_xy=True)

    def to_cells(px, py):
        lo, la = to_ll.transform(px, py)
        return healpix_geo.nested.lonlat_to_healpix(np.asarray(lo), np.asarray(la),
                                                    gt.REF_LEVEL, ellipsoid=gt.ELLIPSOID)

    cells, O = overlap_weights(g["x_coarse"], g["y_coarse"], gt.PIXEL_SIZE_COARSE_M, to_cells)
    xo, covered = overlap_remap(O, y)
    score("overlap1", cells, xo, conservation_error(covered, xo, total_in), cells.size,
          time.perf_counter() - t0, f"quad={QUAD}")

    d = gt.reconstruct_classical(scene, method="cubic", force=False)
    score("geo_cubic", d["cell_ids"], d["cell_data"], np.nan, len(d["cell_ids"]), np.nan)
    frame = pd.DataFrame(rows, columns=E7_COLUMNS)
    tmp = out_path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(out_path)
    return frame


def psf_config():
    """(lam, n_iter) of the PSF-aware configuration transferred from E1, mode A."""
    import revision_e6 as e6
    cfg = e6.transfer_configs()[("psf_aware", "A")]
    _, _, lam, n_iter, _ = cfg.split("|")
    return float(lam), int(float(n_iter))


def run(gt, *, shard="", force=False, limit=0):
    import traceback
    scenes = rt.real_usable_scenes(gt)
    if limit:
        scenes = scenes[:limit]
    if shard:
        k, n = (int(v) for v in shard.split("/"))
        scenes = scenes[k::n]
    cfg = psf_config()
    print(f"E7: PSF-aware lam={cfg[0]:g}, {cfg[1]} iterations (E1 transfer, mode A); "
          f"{len(scenes)} region(s)")
    failures = []
    for i, scene in enumerate(scenes, 1):
        t0 = time.perf_counter()
        try:
            frame = run_region(gt, scene, cfg, force=force)
            print(f"[{i:3d}/{len(scenes)}] {scene}: {len(frame)} rows "
                  f"({time.perf_counter() - t0:.1f} s)", flush=True)
        except Exception as exc:
            failures.append(scene)
            print(f"[{i:3d}/{len(scenes)}] [FAIL] {scene}: {type(exc).__name__}: {exc}",
                  flush=True)
            traceback.print_exc()
    print(f"\n{len(scenes) - len(failures)}/{len(scenes)} regions done; failures: {failures}")
    return 1 if failures else 0


def e7_analyze():
    import pandas as pd
    files = sorted(rt.ensure_out_dir("e7_real").glob("*.csv"))
    if not files:
        raise FileNotFoundError("no E7 results")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    bic = df[df.operator == "geo_cubic"].set_index("region_id")["rmse"]
    df["rel"] = df["rmse"] / df["region_id"].map(bic)
    df["abs_cons_err"] = df["cons_err"].abs()
    df["abs_mean_err"] = df["mean_err"].abs()

    def q(s, p):
        return float(s.quantile(p))
    rows = []
    for op in E7_OPERATORS:
        sub = df[df.operator == op]
        if sub.empty:
            continue
        row = dict(operator=op, n_regions=int(sub.region_id.nunique()))
        for col in ("rel", "fill_fraction", "abs_cons_err", "abs_mean_err", "runtime_s"):
            row.update({f"{col}_median": q(sub[col], 0.5), f"{col}_q25": q(sub[col], 0.25),
                        f"{col}_q75": q(sub[col], 0.75), f"{col}_max": float(sub[col].max())})
        rows.append(row)
    summary = pd.DataFrame(rows)
    out = rt.ensure_out_dir("analysis_e7")
    summary.to_csv(out / "e7_summary.csv", index=False)
    pd.set_option("display.width", 220)
    cols = ["operator", "n_regions", "rel_median", "fill_fraction_median",
            "abs_cons_err_median", "abs_cons_err_max", "abs_mean_err_median", "runtime_s_median"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    e7_figure(df)
    print(f"-> {out}")
    return summary


E7_LABELS = {"psf_aware": "PSF-aware", "psf_aware_cons": "PSF-aware, constrained",
             "hard_binning": "hard binning", "overlap1": "overlap, 1st order",
             "geo_cubic": "bicubic"}


def e7_figure(df):
    """Two strips per operator, one point per region and the median:
    (a) RMSE relative to the bicubic, (b) |conservation error| (log; values
    below 1e-16 drawn at 1e-16)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ops = ("psf_aware", "psf_aware_cons", "overlap1", "hard_binning")
    plt.rcParams.update({"axes.edgecolor": rt._E4_INK2, "axes.labelcolor": rt._E4_INK,
                         "xtick.color": rt._E4_INK2, "ytick.color": rt._E4_INK2,
                         "text.color": rt._E4_INK})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.9), sharey=True, constrained_layout=True)
    rng = np.random.default_rng(0)
    floor = 1e-16
    labels = []
    for k, op in enumerate(ops[::-1]):
        sub = df[df.operator == op]
        fill = sub["fill_fraction"].median()
        labels.append(E7_LABELS[op] + (f"\n({100 * fill:.0f} % of cells)" if fill < 0.99 else ""))
        jit = k + rng.uniform(-0.18, 0.18, len(sub))
        for ax, col, f in ((axes[0], "rel", None), (axes[1], "abs_cons_err", floor)):
            v = sub[col].to_numpy(dtype=float)
            v = np.maximum(v, f) if f else v
            ax.scatter(v, jit, s=9, color="#2a78d6", alpha=0.55, edgecolor="none", zorder=2)
            ax.plot([np.median(v)] * 2, [k - 0.3, k + 0.3], color=rt._E4_INK, lw=1.6, zorder=3)
    axes[0].axvline(1.0, color=rt._E4_INK2, lw=0.8, ls=(0, (2, 2)), zorder=1)
    axes[0].set_xlabel("RMSE / bicubic RMSE (interior cells)", fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("|conservation error| on the operator's support", fontsize=8)
    axes[0].set_yticks(range(len(ops)))
    axes[0].set_yticklabels(labels, fontsize=7)
    for ax, tag in zip(axes, "ab"):
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", color="#e4e3df", lw=0.6)
        ax.set_title(f"({tag})", fontsize=8, loc="left")
    fdir = rt.ensure_out_dir("figures")
    for ext in ("pdf", "png"):
        fig.savefig(fdir / f"e7_operators.{ext}", dpi=200)
    plt.close(fig)
