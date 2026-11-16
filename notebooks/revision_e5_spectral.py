"""Exploratory spectral comparison of Sentinel-2 and VENmuS B4 on the E5 patches.

Not pre-registered: decided after the E5 analysis (protocol, section
"Analyse exploratoire E5-S").  It asks why no method beats the bicubic
readout against VENmuS: is the reference no sharper than Sentinel-2, or are
the two sensors incoherent at high frequency?

Per patch, both images are mean-removed and apodized by the same continuous
Hann window, Fourier transformed about the common patch corner, and the
VENmuS spectrum is harmonized (gain ``fit_a``) and registered (E5 shift, as a
phase ramp).  On the common frequencies (step 1/1280 m^-1, square band up to
the Sentinel-2 Nyquist frequency 0.05 m^-1) ring sums are accumulated:

    P_S = sum |S|^2,   P_V = sum |V|^2,   C = sum S conj(V)

and, per site (sums over its patches):

    R = sqrt(P_V / P_S)            amplitude ratio VENmuS / Sentinel-2
    gamma2 = |C|^2 / (P_S P_V)     coherence
    |H| = |C| / P_S = gamma R      gain of the least-squares linear filter
                                   mapping Sentinel-2 onto VENmuS

plus the alias coherence of S(k) with V at k + p / (10 m) and an error budget.
Only these sums leave the module: the VENmuS data are CC BY-NC 4.0.
"""
from __future__ import annotations

import functools
import math
import time

import numpy as np

import revision_e5 as e5
import revision_tools as rt

PATCH_M = 1280.0                    # 128 x 10 m = 256 x 5 m
N10, N05 = 128, 256
RING = 2                            # ring width, in steps of 1 / PATCH_M
M_NYQ = 64                          # Sentinel-2 Nyquist (0.05 m^-1), in steps
CELL_SIDE_M = {20: 6.2, 19: 12.4}   # evaluation cells as squares (budget only)
BANDS = ((0.01, 0.02), (0.02, 0.03), (0.03, 0.04), (0.04, 0.05))      # m^-1
PARTNERS = {"row": (1, 0), "col": (0, 1), "diag": (1, 1)}
OUT = "analysis_e5_spectral"
WINDOW_AREA = (3.0 * PATCH_M / 8.0) ** 2     # integral of the 2-D Hann window squared


def _steps(n, pixel_m):
    """Integer frequency index m (k = m / PATCH_M) of np.fft.fftfreq(n, pixel_m)."""
    return np.rint(np.fft.fftfreq(n, pixel_m) * PATCH_M).astype(np.int64)


def spectrum(img, pixel_m):
    """Fourier transform (continuous normalization) of the mean-removed image
    apodized by sin^2(pi u / PATCH_M), about the patch corner.  Same window
    and origin on both grids: Sentinel-2 and VENmuS compare bin by bin."""
    n = img.shape[0]
    if img.shape != (n, n) or abs(n * pixel_m - PATCH_M) > 1e-9:
        raise ValueError(f"expected a {PATCH_M:.0f} m square patch, got {img.shape} x {pixel_m} m")
    u = pixel_m * (np.arange(n) + 0.5)
    w1 = np.sin(np.pi * u / PATCH_M) ** 2
    w = np.outer(w1, w1)
    a = (img - (w * img).sum() / w.sum()) * w
    ph = np.exp(-1j * np.pi * np.fft.fftfreq(n, pixel_m) * pixel_m)   # first centre at pixel/2
    return pixel_m ** 2 * np.fft.fft2(a) * ph[:, None] * ph[None, :]


@functools.lru_cache(maxsize=1)
def _geometry():
    m10, m05 = _steps(N10, 10.0), _steps(N05, 5.0)
    MR, MC = np.meshgrid(m10, m10, indexing="ij")
    band = (np.abs(MR) < M_NYQ) & (np.abs(MC) < M_NYQ)
    mr, mc = MR[band], MC[band]
    r = np.hypot(mr, mc)
    ring = (r // RING).astype(np.int64)
    M5R, M5C = np.meshgrid(m05, m05, indexing="ij")
    r5 = np.hypot(M5R, M5C)
    out5 = ~((np.abs(M5R) < M_NYQ) & (np.abs(M5C) < M_NYQ))
    ven_sel = r5 < N05 // 2
    n_ring = int((N05 // 2) // RING)
    g = dict(band=band, ring=ring, n_ring=n_ring, idx_v=(mr % N05, mc % N05), m10=m10,
             kr=mr / PATCH_M, kc=mc / PATCH_M,
             k_abs=r / PATCH_M, k5=m05 / PATCH_M, out5=out5,
             ven_ring=(r5[ven_sel] // RING).astype(np.int64), ven_sel=ven_sel,
             ven_k=r5[ven_sel] / PATCH_M)
    partners = {}
    for name, (dr, dc) in PARTNERS.items():
        ok = np.ones(mr.shape, bool)
        if dr:
            ok &= mr != 0
        if dc:
            ok &= mc != 0
        pr = mr - dr * 2 * M_NYQ * np.sign(mr)
        pc = mc - dc * 2 * M_NYQ * np.sign(mc)
        partners[name] = (ok, (pr % N05, pc % N05))
    g["partners"] = partners
    g["w_band"], g["w5"] = {}, {}
    for level, a in CELL_SIDE_M.items():
        g["w_band"][level] = (np.sinc(mr / PATCH_M * a) * np.sinc(mc / PATCH_M * a)) ** 2
        g["w5"][level] = np.outer(np.sinc(g["k5"] * a), np.sinc(g["k5"] * a)) ** 2
    return g


def residual_shift(S, V, pad=8):
    """Residual shift (m, rows / columns) of registered spectra on the
    Sentinel-2 grid (fftfreq order): V(u) ~ F(u + delta) when S shows F(u).
    Peak of the cross-correlation ifft(S conj V), zero-padded ``pad`` times,
    refined by a parabola on each axis."""
    n = S.shape[0]
    N = n * pad
    big = np.zeros((N, N), complex)
    idx = _steps(n, 10.0) % N
    big[np.ix_(idx, idx)] = S * np.conj(V)
    cc = np.fft.ifft2(big).real
    i, j = np.unravel_index(np.argmax(cc), cc.shape)

    def vertex(a, b, c):
        d = a - 2.0 * b + c
        return 0.5 * (a - c) / d if d != 0 else 0.0

    di = vertex(cc[(i - 1) % N, j], cc[i, j], cc[(i + 1) % N, j])
    dj = vertex(cc[i, (j - 1) % N], cc[i, j], cc[i, (j + 1) % N])
    pos = np.array([i + di, j + dj]) * (PATCH_M / N)
    return (pos + PATCH_M / 2) % PATCH_M - PATCH_M / 2


def patch_rings(s2, ven, shift_rc, gain):
    """Ring sums of one patch.  ``s2``: 128 x 128 at 10 m; ``ven``: 256 x 256
    at 5 m; ``shift_rc``: E5 shift (Sentinel-2 pixels, S2[r, c] ~ VEN10[r -
    drow, c - dcol]); ``gain``: harmonization slope (S2 ~ gain * VEN + b).
    Returns (dict of per-ring arrays, dict of scalars).

    Ring sums run over +k and -k alike, so C is real for real images (a
    registration error lowers |C| instead of showing as a phase).  Two
    registration variants are kept for comparison: none, and the E5 shift
    refined on the whole band (``residual_shift``)."""
    g = _geometry()
    n = g["n_ring"]

    def rsum(x, ring=g["ring"]):
        return np.bincount(ring, weights=x, minlength=n)[:n]

    def csum(x):
        return rsum(x.real) + 1j * rsum(x.imag)

    S_full = spectrum(s2, 10.0)
    S = S_full[g["band"]]
    V0 = gain * spectrum(ven, 5.0)
    d_r, d_c = 10.0 * shift_rc[0], 10.0 * shift_rc[1]          # metres, rows / columns
    k5 = g["k5"]
    V = V0 * np.exp(-2j * np.pi * (k5[:, None] * d_r + k5[None, :] * d_c))
    Vb = V[g["idx_v"]]
    m10 = g["m10"] % N05
    delta = residual_shift(S_full * g["band"], V[np.ix_(m10, m10)] * g["band"])
    Vr = Vb * np.exp(-2j * np.pi * (g["kr"] * delta[0] + g["kc"] * delta[1]))
    S2a, Vb2 = np.abs(S) ** 2, np.abs(Vb) ** 2
    rings = dict(n=rsum(np.ones_like(S2a)), k_sum=rsum(g["k_abs"]),
                 ps=rsum(S2a), pv=rsum(Vb2), c=csum(S * np.conj(Vb)),
                 c_noshift=csum(S * np.conj(V0[g["idx_v"]])),
                 c_refined=csum(S * np.conj(Vr)))
    for name, (ok, idx) in g["partners"].items():
        Vp = V[idx]
        rings[f"ps_{name}"] = rsum(np.where(ok, S2a, 0.0))
        rings[f"pv_{name}"] = rsum(np.where(ok, np.abs(Vp) ** 2, 0.0))
        rings[f"c_{name}"] = csum(np.where(ok, S * np.conj(Vp), 0.0))
    for level in CELL_SIDE_M:
        w = g["w_band"][level]
        rings[f"ps_L{level}"] = rsum(w * S2a)
        rings[f"pv_L{level}"] = rsum(w * Vb2)
        rings[f"c_L{level}"] = csum(w * S * np.conj(Vb))
    V2 = np.abs(V) ** 2
    rings["ven_n"] = np.bincount(g["ven_ring"], minlength=n)[:n].astype(float)
    rings["ven_k_sum"] = np.bincount(g["ven_ring"], weights=g["ven_k"], minlength=n)[:n]
    rings["pv_ven"] = np.bincount(g["ven_ring"], weights=V2[g["ven_sel"]], minlength=n)[:n]
    scalars = dict(e_out=float(V2[g["out5"]].sum()), e_ven=float(V2.sum()),
                   e_s2=float((np.abs(S_full) ** 2).sum()),
                   resid_shift_row_m=float(delta[0]), resid_shift_col_m=float(delta[1]))
    for level in CELL_SIDE_M:
        scalars[f"e_out_L{level}"] = float((g["w5"][level] * V2)[g["out5"]].sum())
    return rings, scalars


def _ring_frame(rings, **meta):
    import pandas as pd
    cols = {}
    for key, v in rings.items():
        if np.iscomplexobj(v):
            cols[f"{key}_re"], cols[f"{key}_im"] = v.real, v.imag
        else:
            cols[key] = v
    frame = pd.DataFrame(cols)
    frame.insert(0, "ring", np.arange(len(frame)))
    for i, (k, v) in enumerate(meta.items()):
        frame.insert(i, k, v)
    return frame


def run(limit=0):
    """Ring sums of every selected E5 patch, then the analysis and figure."""
    import pandas as pd
    items = e5.selected_patches()
    if limit:
        kept, seen = [], {}
        for it in items:
            if seen.get(it.site, 0) < limit:
                kept.append(it)
                seen[it.site] = seen.get(it.site, 0) + 1
        items = kept
    out = rt.ensure_out_dir(OUT)
    ring_frames, patch_rows = [], []
    t0 = time.time()
    archives, site = {}, None
    for i, row in enumerate(items, 1):
        if row.site != site:
            for z in archives.values():
                z.close()
            archives, site = {}, row.site
        s2_dn, ven_dn, *_ = e5.read_patch(row, archives=archives)
        rings, scalars = patch_rings(s2_dn / e5.DN_SCALE, ven_dn / e5.DN_SCALE,
                                     (row.shift_row_px, row.shift_col_px), row.fit_a)
        ring_frames.append(_ring_frame(rings, site=row.site, patch_id=row.patch_id))
        patch_rows.append(dict(site=row.site, patch_id=row.patch_id, fit_a=row.fit_a,
                               shift_row_px=row.shift_row_px, shift_col_px=row.shift_col_px,
                               **scalars))
        if i % 50 == 0:
            print(f"  {i}/{len(items)} patches ({time.time() - t0:.0f} s)", flush=True)
    for z in archives.values():
        z.close()
    rings = pd.concat(ring_frames, ignore_index=True)
    patches = pd.DataFrame(patch_rows)
    rings.to_csv(out / "e5_spectral_rings.csv", index=False)
    patches.to_csv(out / "e5_spectral_patches.csv", index=False)
    print(f"{len(patches)} patches, {patches.site.nunique()} sites ({time.time() - t0:.0f} s)")
    analyze()


# =============================================================================
# Analysis
# =============================================================================
def _c(df, key):
    return df[f"{key}_re"].to_numpy() + 1j * df[f"{key}_im"].to_numpy()


def derive(sums):
    """Ratio, coherence, optimal gain and alias coherence from ring sums
    (rows: rings, already summed over patches)."""
    import pandas as pd
    with np.errstate(invalid="ignore", divide="ignore"):
        ps, pv, C = sums["ps"].to_numpy(), sums["pv"].to_numpy(), _c(sums, "c")
        out = pd.DataFrame({"ring": sums["ring"].to_numpy(),
                            "k": (sums["k_sum"] / sums["n"]).to_numpy(),
                            "n_bins": sums["n"].to_numpy()})
        out["R"] = np.sqrt(pv / ps)
        out["gamma2"] = np.abs(C) ** 2 / (ps * pv)
        out["H"] = C.real / ps                    # C is real (sums over +k and -k)
        for var in ("noshift", "refined"):
            out[f"gamma2_{var}"] = np.abs(_c(sums, f"c_{var}")) ** 2 / (ps * pv)
        alias = np.zeros(len(out))
        for name in PARTNERS:
            cp = _c(sums, f"c_{name}")
            g2 = np.abs(cp) ** 2 / (sums[f"ps_{name}"].to_numpy() * sums[f"pv_{name}"].to_numpy())
            out[f"alias_{name}"] = g2
            alias += np.nan_to_num(g2)
        out["alias_frac"] = alias
        out["ven_k"] = (sums["ven_k_sum"] / sums["ven_n"]).to_numpy()
        out["ven_psd"] = (sums["pv_ven"] / sums["ven_n"]).to_numpy()
        out["s2_psd"] = ps / sums["n"].to_numpy()
    return out


def _sum_by(rings, keys):
    num = [c for c in rings.columns if c not in ("site", "patch_id", "ring")]
    return rings.groupby(keys + ["ring"], as_index=False)[num].sum()


def budget(site_sums, site_patches, suffix=""):
    """Error budget of one site (sums over its patches): out-of-band VENmuS
    energy, in-band error of the identity (G = 1) and of the site's optimal
    linear filter (G = H, in-sample)."""
    ps = site_sums[f"ps{suffix}"].to_numpy()
    pv = site_sums[f"pv{suffix}"].to_numpy()
    C = _c(site_sums, f"c{suffix}")
    ok = ps > 0
    e_id = float(np.sum(pv[ok] - 2.0 * C[ok].real + ps[ok]))
    e_opt = float(np.sum(pv[ok] - np.abs(C[ok]) ** 2 / ps[ok]))
    e_out = float(site_patches[f"e_out{suffix}"].sum())
    n = len(site_patches)
    return dict(e_out=e_out, e_in_identity=e_id, e_in_optimal=e_opt,
                share_out=e_out / (e_out + e_id),
                rmse_ratio_optimal=math.sqrt((e_out + e_opt) / (e_out + e_id)),
                rmse_identity_est=math.sqrt((e_out + e_id) / n / PATCH_M ** 2 / WINDOW_AREA))


def _bicubic_rmse():
    """Mean E5 bicubic RMSE per site and level (sanity check of the budget)."""
    try:
        df = e5.collect(parts=("main",))
    except Exception as exc:                     # E5 results absent: skip the check
        print(f"(no E5 bicubic RMSE: {exc})")
        return {}
    df = df[df.family == "geo_cubic"]
    return df.groupby(["region_id", "level"])["rmse"].mean().to_dict()


def analyze():
    import pandas as pd
    out = rt.ensure_out_dir(OUT)
    rings = pd.read_csv(out / "e5_spectral_rings.csv")
    patches = pd.read_csv(out / "e5_spectral_patches.csv")
    site_sums = _sum_by(rings, ["site"])
    per_site = pd.concat([derive(g).assign(site=s) for s, g in site_sums.groupby("site")],
                         ignore_index=True)
    pooled = derive(_sum_by(rings.assign(all="all"), ["all"]))
    stats = ("R", "gamma2", "gamma2_noshift", "gamma2_refined", "H", "alias_frac",
             "ven_psd", "s2_psd")
    summary = per_site.groupby("ring").agg(k=("k", "median"), ven_k=("ven_k", "median"),
                                           n_sites=("site", "nunique"))
    for s in stats:
        q = per_site.groupby("ring")[s]
        summary[f"{s}_median"] = q.median()
        summary[f"{s}_q25"] = q.quantile(0.25)
        summary[f"{s}_q75"] = q.quantile(0.75)
        summary[f"{s}_pooled"] = pooled.set_index("ring")[s]
    summary = summary.reset_index()

    # frequency bands (rings entirely inside the Sentinel-2 disc)
    inside = rings[rings.ring < M_NYQ // RING]
    k_ring = (inside["k_sum"] / inside["n"]).to_numpy()
    band_rows = []
    for lo, hi in BANDS:
        sel = inside[(k_ring >= lo) & (k_ring < hi)]
        for s, g in _sum_by(sel.assign(ring=0), ["site"]).groupby("site"):
            d = derive(g).iloc[0]
            band_rows.append(dict(site=s, k_lo=lo, k_hi=hi, R=d.R, gamma2=d.gamma2, H=d.H,
                                  gamma2_noshift=d.gamma2_noshift,
                                  gamma2_refined=d.gamma2_refined, alias_frac=d.alias_frac))
    bands = pd.DataFrame(band_rows)
    med = bands.groupby(["k_lo", "k_hi"], as_index=False)[
        ["R", "gamma2", "H", "gamma2_noshift", "gamma2_refined", "alias_frac"]
    ].median().assign(site="median")
    bands = pd.concat([bands, med], ignore_index=True)

    bic = _bicubic_rmse()
    bud_rows = []
    for s, g in site_sums.groupby("site"):
        p = patches[patches.site == s]
        for level, suffix in ((None, ""),) + tuple((lv, f"_L{lv}") for lv in CELL_SIDE_M):
            row = dict(site=s, cells="none" if level is None else f"L{level}", n_patches=len(p),
                       **budget(g, p, suffix))
            row["e5_bicubic_rmse"] = bic.get((s, level), np.nan) if level else np.nan
            bud_rows.append(row)
    bud = pd.DataFrame(bud_rows)
    bmed = bud.groupby("cells", as_index=False)[
        ["share_out", "rmse_ratio_optimal", "rmse_identity_est", "e5_bicubic_rmse"]
    ].median().assign(site="median")
    bud = pd.concat([bud, bmed], ignore_index=True)

    per_site.to_csv(out / "e5_spectral_sites.csv", index=False)
    summary.to_csv(out / "e5_spectral_summary.csv", index=False)
    bands.to_csv(out / "e5_spectral_bands.csv", index=False)
    bud.to_csv(out / "e5_spectral_budget.csv", index=False)
    pd.set_option("display.width", 200)
    print("bands (median over sites):")
    print(med.round(3).to_string(index=False))
    print("error budget (median over sites):")
    print(bmed.round(4).to_string(index=False))
    res = np.hypot(patches.resid_shift_row_m, patches.resid_shift_col_m)
    print(f"residual shift after E5 registration (m): median {res.median():.2f}, "
          f"q90 {res.quantile(0.9):.2f}, max {res.max():.2f}")
    figure()
    print(f"-> {out}")
    return summary, bands, bud


def nominal_mtf(k, f=1.0):
    """MTF of the nominal Sentinel-2 Gaussian (FWHM f x 12.5 m)."""
    fwhm = f * e5.S2_NOMINAL_FWHM_M
    return np.exp(-(math.pi * fwhm * k) ** 2 / (4.0 * math.log(2.0)))


def figure():
    """Fig. E5-S: amplitude ratio, coherence and optimal gain versus spatial
    frequency, per site (thin) and median over sites (thick)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.edgecolor": rt._E4_INK2, "axes.labelcolor": rt._E4_INK,
                         "xtick.color": rt._E4_INK2, "ytick.color": rt._E4_INK2,
                         "pdf.fonttype": 42})
    out = rt.OUT_DIR / OUT
    sites = pd.read_csv(out / "e5_spectral_sites.csv")
    summ = pd.read_csv(out / "e5_spectral_summary.csv")
    first, last = 1, M_NYQ // RING - 1          # rings 1..31: inside the Sentinel-2 disc
    summ = summ[(summ.ring >= first) & (summ.ring <= last)]
    sites = sites[(sites.ring >= first) & (sites.ring <= last)]
    blue, orange, grey = rt._E4_SERIES["psf_aware"][1], rt._E4_SERIES["rl2"][1], "#b9b8b3"
    kk = np.linspace(0.0, 0.05, 200)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), constrained_layout=True)
    panels = (("R", "(a) amplitude ratio VENµS / S2", (0.0, 4.4)),
              ("gamma2", "(b) coherence γ²", (0.0, 1.02)),
              ("H", "(c) optimal linear gain H", (0.0, 4.4)))
    n_sites = sites.site.nunique()
    for ax, (key, title, ylim) in zip(axes, panels):
        for _, g in sites.groupby("site"):
            ax.plot(1e3 * g["k"], g[key], color=grey, lw=0.6)
        ax.fill_between(1e3 * summ["k"], summ[f"{key}_q25"], summ[f"{key}_q75"],
                        color=blue, alpha=0.15, lw=0)
        ax.plot(1e3 * summ["k"], summ[f"{key}_median"], color=blue, lw=1.8,
                label=f"median of {n_sites} sites")
        if key == "gamma2":
            ax.plot(1e3 * summ["k"], summ["gamma2_refined_median"], color=blue, lw=1.0,
                    ls=":", label="γ² with refined shift (median)")
        if key in ("R", "H"):
            ax.plot(1e3 * kk, 1.0 / nominal_mtf(kk), color=rt._E4_INK2, lw=1.0,
                    ls=(0, (3, 2)), label="1 / MTF, nominal S2 (12.5 m)")
            ax.axhline(1.0, color=rt._E4_INK2, lw=0.6)
        if key == "gamma2":
            ax.fill_between(1e3 * summ["k"], summ["alias_frac_q25"], summ["alias_frac_q75"],
                            color=orange, alpha=0.15, lw=0)
            ax.plot(1e3 * summ["k"], summ["alias_frac_median"], color=orange, lw=1.4,
                    ls="--", label="alias coherence (median)")
        ax.set_xlim(0, 50)
        ax.set_ylim(*ylim)
        ax.set_xlabel("spatial frequency (cycles/km)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].plot([], [], color=grey, lw=0.6, label="one site")
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    figdir = rt.ensure_out_dir("figures")
    fig.savefig(figdir / "e5_spectral.pdf")
    fig.savefig(figdir / "e5_spectral.png", dpi=200)
    plt.close(fig)
    print(f"-> {figdir / 'e5_spectral.pdf'}")
