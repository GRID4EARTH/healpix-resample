"""Generate the LaTeX tables and figures of the JSTARS revision.

Every number printed in a generated table is read from the revision result
tables in ``notebooks/tables/revision`` (run 2 / amended grids, produced by
``revision_run.py``). Nothing is typed by hand. The tables are Tables II,
III, IV and VI of the revised paper and the tables of supplementary Sections
S11--S17. Output, under ``revision_manuscript/revised/`` (the manuscript
sources, kept outside this repository):

- ``main/tables/*.tex`` and ``main/figures/*.pdf``
- ``supplement/tables/*.tex`` and ``supplement/figures/*.pdf``

Run from the repository root::

    python notebooks/revision_tables_tex.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "notebooks" / "tables" / "revision"
OUT = ROOT / "revision_manuscript" / "revised"
MAIN_T, SUPP_T = OUT / "main" / "tables", OUT / "supplement" / "tables"
MAIN_F, SUPP_F = OUT / "main" / "figures", OUT / "supplement" / "figures"
for d in (MAIN_T, SUPP_T, MAIN_F, SUPP_F):
    d.mkdir(parents=True, exist_ok=True)

CLASSES = ["agriculture", "forest", "urban", "water"]
COMP_LABEL = {
    "rl2": "Richardson--Lucy",
    "tik_id": "Tikhonov (identity)",
    "tik_lap": "Tikhonov (Laplacian)",
    "wiener_hunt": "Wiener--Hunt",
    "geo_nearest": "Nearest neighbour",
    "geo_linear": "Bilinear",
    "geo_cubic": "Bicubic",
    "strongest": "Strongest deconvolution",
}
E2_LABEL = {
    "e2_xref": r"$\mathbf x_{\mathrm{ref}}=\mathbf B\mathbf y$ (no correction)",
    "e2_cg_euclid": "Unweighted CG, same system",
    "e2_sirt": r"SIRT/Landweber, $\omega$ and iterations tuned",
    "e2_jacobi": "Jacobi-preconditioned weighted CG",
    "e2_tik_x0": r"Euclidean Tikhonov, CG from $\mathbf x=\mathbf 0$",
    "e2_tik_dref": r"Euclidean Tikhonov around $\mathbf B\mathbf y$",
    "e2_mlem": "One-stage MLEM on \\healpix",
}


def pct(x: float, nd: int = 1, sign: bool = True) -> str:
    v = 100.0 * float(x)
    if abs(v) < 0.5 * 10 ** (-nd):
        return f"{0:.{nd}f}"
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "$-$") if not sign else s.replace("+", "$+$").replace("-", "$-$")


def cell(m: float, wins: int, n: int, nd: int = 1, show_n: bool = False) -> str:
    w = f"{int(wins)}/{int(n)}" if show_n else f"{int(wins)}"
    return f"{pct(m, nd)} ({w})"


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


def cmp_table(bench: str) -> pd.DataFrame:
    return pd.read_csv(RES / f"analysis_{bench}" / "comparisons.csv")


def pick(d: pd.DataFrame, **kw) -> pd.Series:
    q = d
    for k, v in kw.items():
        q = q[q[k] == v]
    if len(q) != 1:
        raise KeyError(f"{kw}: {len(q)} rows")
    return q.iloc[0]


# ---------------------------------------------------------------------------
# Main paper
# ---------------------------------------------------------------------------
def main_tuned() -> None:
    """Table: E1, tuned baselines, both benchmarks, modes A and B."""
    rows = []
    for comp in ["rl2", "tik_id", "tik_lap", "wiener_hunt", "geo_cubic"]:
        cells = []
        for bench in ["real", "synthetic"]:
            d = cmp_table(bench)
            d = d[(d.study == "E1") & (d.scene_class == "all")]
            for mode in ["A", "B"]:
                r = pick(d, competitor=comp, ours=f"{mode}/5fold", theirs=f"{mode}/5fold")
                cells.append(cell(r.mean_rel_margin, r.wins, r.n_regions))
        rows.append(f"{COMP_LABEL[comp]} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    write(MAIN_T / "tab_tuned.tex", rf"""\begin{{table}}[t]
\centering
\caption{{Comparison with tuned baselines (experiment E1). Mean relative
RMSE margin of \psf-aware over each competitor, in percent (positive:
\psf-aware more accurate); in brackets, regions won out of 40. Every method,
\psf-aware included, is tuned by the same stratified 5-fold
cross-validation over regions (Section~\ref{{sec:hyperparameter_selection}}).
Mode A fixes the nominal width; in mode B the width is tuned for every
\psf-based method. All 95\% region-bootstrap intervals of the paired RMSE
difference exclude zero.}}
\label{{tab:tuned}}
\setlength{{\tabcolsep}}{{2.5pt}}
\scriptsize
\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\toprule
& \multicolumn{{2}}{{c}}{{Real (reduced resolution)}} & \multicolumn{{2}}{{c}}{{Synthetic}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
Competitor & A & B & A & B \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}""")


def main_ablation() -> None:
    """Table: E2, inversion schemes with the same A and B, mode A."""
    rows = []
    for comp in ["e2_xref", "e2_cg_euclid", "e2_sirt", "e2_jacobi", "e2_tik_x0",
                 "e2_tik_dref", "e2_mlem"]:
        cells = []
        for bench in ["real", "synthetic"]:
            d = cmp_table(bench)
            d = d[(d.study == "E2") & (d.scene_class == "all")]
            r = pick(d, competitor=comp, ours="A/5fold", theirs="A/5fold")
            cells.append(cell(r.mean_rel_margin, r.wins, r.n_regions))
        rows.append(f"{E2_LABEL[comp]} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    write(MAIN_T / "tab_ablation.tex", rf"""\begin{{table}}[t]
\centering
\caption{{Ablation of the inversion scheme (experiment E2). All arms use the
same operators $\mathbf A$ and $\mathbf B$, the nominal width, the same
cells and the same 40 regions; each arm's damping and iteration count are
tuned by the cross-validation of Section~\ref{{sec:hyperparameter_selection}}.
Mean relative RMSE margin of the proposed weighted-CG solver over each arm,
in percent (positive: proposed solver more accurate); in brackets, regions
where the proposed solver is more accurate.}}
\label{{tab:ablation}}
\setlength{{\tabcolsep}}{{3pt}}
\scriptsize
\begin{{tabular}}{{@{{}}lcc@{{}}}}
\toprule
Inversion arm & Real & Synthetic \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}""")


E6_ARMS = [
    ("e1", "Original: B04, ratio 1.6"),
    ("a_b02", "Band B02"),
    ("a_b03", "Band B03"),
    ("a_b08", "Band B08"),
    ("c_random", "60 regions drawn at random on land"),
    ("b_4_19", r"Ratio 3.2 (40~m samples, L19)"),
    ("d_l19", "Landsat 8/9 OLI, ratio 2.4"),
    ("d_l18", "Landsat 8/9 OLI, ratio 1.2"),
    ("b_2_18", r"Ratio 0.8 (20~m samples, L18)"),
]


def e6_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    c = pd.read_csv(RES / "analysis_e6" / "e6_comparisons.csv")
    x = pd.read_csv(RES / "analysis_e6" / "e6x_fine_comparisons.csv")
    return c[c.scene_class == "all"], x[x.scene_class == "all"]


def main_generalization() -> None:
    c, x = e6_frames()
    rows = []
    for arm, label in E6_ARMS:
        a = c[c.arm == arm]
        t = pick(a, scheme="transfer", mode="A", competitor="strongest")
        i = pick(a, scheme="5fold", mode="B", competitor="strongest")
        rows.append(f"{label} & {int(i.n_regions)} & "
                    f"{cell(t.mean_rel_margin, t.wins, t.n_regions)} & "
                    f"{cell(i.mean_rel_margin, i.wins, i.n_regions)}" + r" \\")
    for lev in (19, 20):
        a = x[x.ours == f"psf_aware_l{lev}"]
        t = pick(a, scheme="transfer", mode="A", competitor="strongest")
        i = pick(a, scheme="5fold", mode="B", competitor="strongest")
        rows.append(rf"Ratio 0.8, solved on L{lev}, aggregated$^{{*}}$ & {int(i.n_regions)} & "
                    f"{cell(t.mean_rel_margin, t.wins, t.n_regions)} & "
                    f"{cell(i.mean_rel_margin, i.wins, i.n_regions)}" + r" \\")
    body = "\n".join(rows)
    write(MAIN_T / "tab_generalization.tex", rf"""\begin{{table}}[t]
\centering
\caption{{Generalization (experiment E6). Mean relative RMSE margin of
\psf-aware over the strongest tuned deconvolution of each region, in
percent; in brackets, regions won. \emph{{Transfer}}: configurations selected
on the original 40 regions, nominal width. \emph{{Internal}}: 5-fold
cross-validation repeated within the arm, width tuned. $^{{*}}$Exploratory,
declared after the ratio-0.8 result and before it was computed.}}
\label{{tab:generalization}}
\setlength{{\tabcolsep}}{{2pt}}
\scriptsize
\begin{{tabular}}{{@{{}}lccc@{{}}}}
\toprule
Arm & $n$ & Transfer & Internal \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}""")


def sci(v: float) -> str:
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10 ** e
    if e > -3:
        return f"{v:.1e}" if False else f"{v:.3g}"
    return rf"${m:.1f}\times10^{{{e}}}$"


def main_operators() -> None:
    s = pd.read_csv(RES / "analysis_e7" / "e7_summary.csv").set_index("operator")
    spec = [
        ("psf_aware", r"\psf-aware (main method)", "no",
         "intensive variables, spatial analysis"),
        ("psf_aware_cons", r"\psf-aware, constrained", "global integral",
         "fidelity and totals"),
        ("overlap1", "First-order overlap", "area integral",
         "fluxes, model coupling"),
        ("hard_binning", "Hard binning", "discrete total",
         "scattered points, totals"),
    ]
    rows = []
    for op, label, cons, use in spec:
        r = s.loc[op]
        rel = f"{r.rel_median:.3f}"
        if op == "hard_binning":
            rel += r"$^{\dagger}$"
        err = "exact" if r.abs_cons_err_median == 0 else sci(r.abs_cons_err_median)
        rows.append(f"{label} & {cons} & {rel} & {err} & {use}" + r" \\")
    body = "\n".join(rows)
    fill = s.loc["hard_binning", "fill_fraction_median"]
    write(MAIN_T / "tab_operators.tex", rf"""\begin{{table}}[t]
\centering
\caption{{Choice of operator (experiment E7, 40 real regions, B04 at 20~m to
level 19). Medians over regions of the RMSE relative to bicubic and of the
relative error on the area integral over each operator's own support. The
derivations and the ERA5 evaluation are in the supplement.
$^{{\dagger}}$On the {100 * fill:.0f}\% of target cells that receive a sample.}}
\label{{tab:operators}}
\setlength{{\tabcolsep}}{{2pt}}
\scriptsize
\begin{{tabular}}{{@{{}}>{{\raggedright\arraybackslash}}p{{1.6cm}}>{{\raggedright\arraybackslash}}p{{1.15cm}}cc>{{\raggedright\arraybackslash}}p{{1.95cm}}@{{}}}}
\toprule
Operator & Conserves & RMSE/bicubic & Cons.\ error & Recommended use \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}""")


# ---------------------------------------------------------------------------
# Supplement
# ---------------------------------------------------------------------------
def supp_e1_perclass() -> None:
    lines = []
    for bench, blabel in [("real", "Real (reduced resolution)"), ("synthetic", "Synthetic")]:
        d = cmp_table(bench)
        d = d[d.study == "E1"]
        lines.append(rf"\multicolumn{{9}}{{l}}{{\textit{{{blabel}}}}} \\")
        for cls in CLASSES + ["all"]:
            cells = []
            for mode in ["A", "B"]:
                for comp in ["rl2", "tik_id", "wiener_hunt", "geo_cubic"]:
                    r = pick(d, competitor=comp, ours=f"{mode}/5fold",
                             theirs=f"{mode}/5fold", scene_class=cls)
                    cells.append(cell(r.mean_rel_margin, r.wins, r.n_regions))
            name = cls.capitalize() if cls != "all" else r"\textbf{All}"
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    body = "\n".join(lines[:-1])
    write(SUPP_T / "tab_e1_perclass.tex", rf"""\begin{{table*}}[t]
\centering
\caption{{Experiment E1 by scene class. Mean relative RMSE margin of
\psf-aware over each tuned competitor, in percent (positive: \psf-aware more
accurate); in brackets, regions won (10 per class, 40 in all). Both sides are
tuned by the stratified 5-fold cross-validation. Tikhonov with a Laplacian
regularizer gives the same margins as the identity regularizer to within
0.05 point and is omitted.}}
\label{{tab:e1_perclass}}
\setlength{{\tabcolsep}}{{3pt}}
\footnotesize
\begin{{tabular}}{{@{{}}lcccccccc@{{}}}}
\toprule
& \multicolumn{{4}}{{c}}{{Mode A: nominal width}} & \multicolumn{{4}}{{c}}{{Mode B: width tuned}} \\
\cmidrule(lr){{2-5}}\cmidrule(lr){{6-9}}
Class & RL & Tikhonov & Wiener--Hunt & Bicubic & RL & Tikhonov & Wiener--Hunt & Bicubic \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}""")


def describe_config(fam: str, cfg: str) -> str:
    norm, f, lam, it, om = cfg.split("|")
    parts = []
    if f not in ("-", "1"):
        parts.append(f"$f={float(f):.3g}$")
    elif f == "1":
        parts.append("$f=1$")
    if fam == "rl2":
        parts.append(f"{norm}, {int(float(it))} it.")
    elif fam in ("tik_id", "tik_lap"):
        e = int(round(np.log10(float(lam))))
        parts.append(rf"$\lambda=10^{{{e}}}$")
    elif fam == "psf_aware":
        lv = float(lam)
        lam_s = "0" if lv == 0 else rf"10^{{{int(round(np.log10(lv)))}}}"
        parts.append(rf"$\lambda={lam_s}$, {int(float(it))} it.")
    elif fam == "e2_sirt":
        parts.append(rf"$\omega={om}$, {int(float(it))} it.")
    elif fam.startswith("e2_") and fam not in ("e2_xref",):
        if lam != "-":
            lv = float(lam)
            lam_s = "0" if lv == 0 else rf"10^{{{int(round(np.log10(lv)))}}}"
            parts.append(rf"$\lambda={lam_s}$, {int(float(it))} it.")
        else:
            parts.append(f"{int(float(it))} it.")
    return ", ".join(parts)


def supp_e1_selected() -> None:
    fams = [("psf_aware", r"\psf-aware"), ("rl2", "Richardson--Lucy"),
            ("tik_id", "Tikhonov (identity)"), ("tik_lap", "Tikhonov (Laplacian)"),
            ("wiener_hunt", "Wiener--Hunt")]
    lines = []
    for fam, label in fams:
        cells = []
        for bench in ["real", "synthetic"]:
            s = pd.read_csv(RES / f"analysis_{bench}" / "selected_configs.csv")
            for mode in ["A", "B"]:
                q = s[(s.family == fam) & (s.scheme == "5fold") & (s["mode"] == mode)]
                vc = (q.config.value_counts() / 8).astype(int)  # 8 regions per fold
                txt = "; ".join(
                    describe_config(fam, k) + ("" if v == 5 else f" ({v}/5)")
                    for k, v in vc.items())
                cells.append(txt if txt else "--")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    write(SUPP_T / "tab_e1_selected.tex", rf"""\begin{{table*}}[t]
\centering
\caption{{Configurations selected by the 5-fold cross-validation of E1.
Unless a count is given, the same configuration was selected in all five
folds. $f$ is the assumed width relative to nominal ($\sqrt{{1.25}}\approx1.118$
is the combined response of the real benchmark, 1.138 the second-moment
equivalent of the synthetic one). Wiener--Hunt estimates its own
hyperparameters; only its width is tuned. ``it.'' = iterations.}}
\label{{tab:e1_selected}}
\setlength{{\tabcolsep}}{{3pt}}
\footnotesize
\begin{{tabular}}{{@{{}}l*{{4}}{{>{{\raggedright\arraybackslash}}p{{3.3cm}}}}@{{}}}}
\toprule
& \multicolumn{{2}}{{c}}{{Real}} & \multicolumn{{2}}{{c}}{{Synthetic}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
Family & Mode A & Mode B & Mode A & Mode B \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}""")


def supp_e1_variants() -> None:
    lines = []
    for comp in ["rl2", "tik_id", "wiener_hunt"]:
        cells = []
        for bench in ["real", "synthetic"]:
            d = cmp_table(bench)
            d = d[(d.study == "E1") & (d.scene_class == "all")]
            for theirs in ["5fold", "loo_class", "oracle"]:
                r = pick(d, competitor=comp, ours="B/5fold", theirs=f"B/{theirs}")
                cells.append(cell(r.mean_rel_margin, r.wins, r.n_regions))
        lines.append(f"{COMP_LABEL[comp]} & " + " & ".join(cells) + r" \\")
    sub = []
    for comp in ["rl2", "tik_id"]:
        cells = []
        for bench in ["real", "synthetic"]:
            d = cmp_table(bench)
            d = d[(d.study == "E1") & (d.scene_class == "all")]
            try:
                r = pick(d, competitor=comp, ours="A/paper", theirs="A/paper")
                txt = cell(r.mean_rel_margin, r.wins, r.n_regions)
            except KeyError:  # Tikhonov was not part of the submitted real benchmark
                txt = "--"
            cells.append(r"\multicolumn{3}{c}{" + txt + "}")
        sub.append(f"{COMP_LABEL[comp]} & " + " & ".join(cells) + r" \\")
    write(SUPP_T / "tab_e1_variants.tex", rf"""\begin{{table}}[t]
\centering
\caption{{Robustness of the E1 comparison to the competitors' selection rule
(mode B, \psf-aware always selected by 5-fold cross-validation). LOO:
leave-one-region-out within each class. Oracle: best configuration per
region, chosen with the reference, which favours the competitor. Bottom: the
fixed configurations of the submitted version, for reference.}}
\label{{tab:e1_variants}}
\setlength{{\tabcolsep}}{{1.5pt}}
\scriptsize
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
& \multicolumn{{3}}{{c}}{{Real}} & \multicolumn{{3}}{{c}}{{Synthetic}} \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}
Competitor & 5-fold & LOO & Oracle & 5-fold & LOO & Oracle \\
\midrule
{chr(10).join(lines)}
\midrule
\multicolumn{{7}}{{l}}{{\textit{{Submitted fixed configurations (\psf-aware and competitor)}}}} \\
{chr(10).join(sub)}
\bottomrule
\end{{tabular}}
\end{{table}}""")


def supp_e2_perclass() -> None:
    lines = []
    for bench, blabel in [("real", "Real"), ("synthetic", "Synthetic")]:
        d = cmp_table(bench)
        d = d[d.study == "E2"]
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{blabel}}}}} \\")
        for comp in ["e2_xref", "e2_cg_euclid", "e2_sirt", "e2_jacobi", "e2_tik_x0",
                     "e2_tik_dref", "e2_mlem"]:
            cells = []
            for cls in CLASSES + ["all"]:
                r = pick(d, competitor=comp, ours="A/5fold", theirs="A/5fold",
                         scene_class=cls)
                cells.append(cell(r.mean_rel_margin, r.wins, r.n_regions))
            lines.append(f"{E2_LABEL[comp]} & " + " & ".join(cells) + r" \\")
        r = pick(d, competitor="e2_mlem", ours="B/5fold", theirs="B/5fold", scene_class="all")
        lines.append(r"\quad MLEM vs.\ \psf-aware with tuned width & & & & & "
                     + cell(r.mean_rel_margin, r.wins, r.n_regions) + r" \\")
        lines.append(r"\addlinespace")
    body = "\n".join(lines[:-1])
    write(SUPP_T / "tab_e2_perclass.tex", rf"""\begin{{table*}}[t]
\centering
\caption{{Experiment E2 by scene class: mean relative RMSE margin of the
proposed weighted-CG solver over each inversion arm, in percent; in
brackets, regions where it is more accurate. Nominal width for all arms. The
last row of each block compares \psf-aware with its width tuned (mode B)
against MLEM, which was run at the nominal width only.}}
\label{{tab:e2_perclass}}
\setlength{{\tabcolsep}}{{4pt}}
\footnotesize
\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\toprule
Arm & Agriculture & Forest & Urban & Water & All \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}""")


def supp_e2_selected() -> None:
    rows = []
    for fam in ["e2_cg_euclid", "e2_sirt", "e2_jacobi", "e2_tik_x0", "e2_tik_dref",
                "e2_mlem", "psf_aware"]:
        cells = []
        for bench in ["real", "synthetic"]:
            s = pd.read_csv(RES / f"analysis_{bench}" / "selected_configs.csv")
            q = s[(s.family == fam) & (s.scheme == "5fold") & (s["mode"] == "A")]
            vc = (q.config.value_counts() / 8).astype(int)
            cells.append("; ".join(describe_config(fam, k).replace("$f=1$, ", "")
                                   + ("" if v == 5 else f" ({v}/5)") for k, v in vc.items()))
        label = E2_LABEL.get(fam, r"\psf-aware (weighted CG, package)")
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    write(SUPP_T / "tab_e2_selected.tex", rf"""\begin{{table}}[t]
\centering
\caption{{Configurations selected by the 5-fold cross-validation for the E2
arms (nominal width). The CG-type arms and SIRT select the largest
iteration count of the extended grid; the selection curves are flat there
(Fig.~\ref{{fig:hyper_curves}}).}}
\label{{tab:e2_selected}}
\setlength{{\tabcolsep}}{{2pt}}
\scriptsize
\begin{{tabular}}{{@{{}}*{{3}}{{>{{\raggedright\arraybackslash}}p{{2.6cm}}}}@{{}}}}
\toprule
Arm & Real & Synthetic \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}""")


def supp_hyper_figure() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.6), sharey="row")
    for row, bench in enumerate(["real", "synthetic"]):
        c = pd.read_csv(RES / f"analysis_{bench}" / "config_curves.csv")
        c1 = c[(c.width_factor.fillna(1.0).round(3) == 1.0)]
        ax = axes[row, 0]
        p = c1[c1.family == "psf_aware"]
        for lam, g in p.groupby("lam"):
            g = g.sort_values("n_iter")
            ax.plot(g.n_iter, g.median_rel, marker="o", ms=2.5, lw=1,
                    label=("0" if lam == 0 else f"{lam:g}"))
        ax.set_xscale("log"); ax.set_title("PSF-aware", fontsize=7)
        ax.legend(title=r"$\lambda$", fontsize=5, title_fontsize=6, frameon=False)
        ax = axes[row, 1]
        p = c1[c1.family == "rl2"]
        for var, g in p.groupby("variant"):
            g = g.sort_values("n_iter")
            ax.plot(g.n_iter, g.median_rel, marker="o", ms=2.5, lw=1, label=var)
        ax.set_xscale("log"); ax.set_title("Richardson--Lucy".replace("--", "–"), fontsize=7)
        ax.legend(fontsize=5, frameon=False)
        ax = axes[row, 2]
        for fam, lab in [("tik_id", "identity"), ("tik_lap", "Laplacian")]:
            g = c1[c1.family == fam].sort_values("lam")
            ax.plot(g.lam, g.median_rel, marker="o", ms=2.5, lw=1, label=lab)
        ax.set_xscale("log"); ax.set_title("Tikhonov", fontsize=7)
        ax.legend(fontsize=5, frameon=False)
        ax = axes[row, 3]
        for fam, lab in [("e2_wcg", "weighted CG"), ("e2_cg_euclid", "unweighted CG"),
                         ("e2_jacobi", "Jacobi CG"), ("e2_tik_x0", r"Tik. from 0"),
                         ("e2_tik_dref", r"Tik. around $By$"), ("e2_mlem", "MLEM")]:
            g = c1[c1.family == fam]
            if "lam" in g and g.lam.notna().any():
                g = g[g.lam == 0]
            g = g.sort_values("n_iter")
            ax.plot(g.n_iter, g.median_rel, marker="o", ms=2.5, lw=1, label=lab)
        g = c1[(c1.family == "e2_sirt") & (c1.omega == 1.9)].sort_values("n_iter")
        ax.plot(g.n_iter, g.median_rel, marker="o", ms=2.5, lw=1, label=r"SIRT $\omega=1.9$")
        ax.set_xscale("log"); ax.set_title(r"E2 arms ($\lambda=0$)", fontsize=7)
        ax.legend(fontsize=4.5, frameon=False, ncol=1)
        for a in axes[row]:
            a.tick_params(labelsize=6)
            a.grid(alpha=0.3, lw=0.5)
        axes[row, 0].set_ylabel(f"{bench}\nmedian RMSE / bicubic", fontsize=7)
    for a in (axes[1, 0], axes[1, 1], axes[1, 3]):
        a.set_xlabel("iterations", fontsize=7)
    axes[1, 2].set_xlabel(r"$\lambda$", fontsize=7)
    for a in axes.ravel():
        lo, hi = a.get_ylim()
        a.set_ylim(max(lo, 0.6), min(hi, 1.3))
    fig.tight_layout(pad=0.4)
    out = SUPP_F / "hyper_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


def supp_e3() -> None:
    rec = pd.read_csv(RES / "analysis_synthetic" / "e3_recovery.csv")
    win = pd.read_csv(RES / "analysis_synthetic" / "e3_within_arm.csv")
    acr = pd.read_csv(RES / "analysis_synthetic" / "e3_across_arms.csv")
    arms = [("S0", "S0: submitted set-up"), ("S1", "S1: point-sampled data"),
            ("S2", "S2: equivalent Gaussian (14.23~m)"), ("S3", "S3: exact kernel")]
    fams = ["psf_aware", "rl2", "tik_id", "wiener_hunt"]
    lines = []
    for arm, label in arms:
        cells = []
        for fam in fams:
            r = pick(rec, arm=arm, family=fam, scene_class="all")
            cells.append(f"{100 * r.mean_recovery:.1f} [{100 * r.ci_low:.1f}, {100 * r.ci_high:.1f}]")
        s = pick(win, arm=arm, competitor="strongest", scene_class="all")
        cells.append(cell(s.mean_rel_margin, s.wins, s.n_regions))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    pc = []
    for cls in CLASSES:
        cells = []
        for arm, _ in arms:
            s = pick(win, arm=arm, competitor="strongest", scene_class=cls)
            cells.append(cell(s.mean_rel_margin, s.wins, s.n_regions))
        pc.append(f"{cls.capitalize()} & " + " & ".join(cells) + r" \\")
    con = []
    for contrast, lab in [("S0-S3", "S0$-$S3: model the aperture exactly"),
                          ("S0-S2", "S0$-$S2: equivalent width"),
                          ("S2-S3", "S2$-$S3: shape beyond 2nd moment"),
                          ("S0-S1", "S0$-$S1: total cost of the aperture")]:
        cells = []
        for fam in fams:
            r = pick(acr, family=fam, contrast=contrast, scene_class="all")
            cells.append(cell(r.mean_rel_margin, r.wins, r.n_regions))
        con.append(f"{lab} & " + " & ".join(cells) + r" \\")
    write(SUPP_T / "tab_e3.tex", rf"""\begin{{table*}}[t]
\centering
\caption{{Response-matched controls (experiment E3, synthetic benchmark, 40
regions). Top: mean recovery fraction (Eq.~\eqref{{eq:recovery_fraction}}),
in percent with 95\% region-bootstrap interval, of each family tuned by
5-fold cross-validation within the arm, and margin of \psf-aware over the
strongest tuned deconvolution (regions won). Middle: that margin by class.
Bottom: paired relative RMSE reduction from the first to the second arm of
each contrast, for each family (positive: the second arm is more accurate;
in brackets, regions where it is).}}
\label{{tab:e3}}
\setlength{{\tabcolsep}}{{3pt}}
\footnotesize
\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\toprule
Arm & \psf-aware & Richardson--Lucy & Tikhonov & Wiener--Hunt & Margin over strongest \\
\midrule
{chr(10).join(lines)}
\midrule
Class & S0 & S1 & S2 & S3 & \\
\midrule
{chr(10).join(l.replace(r" \\", r" & \\") for l in pc)}
\midrule
Contrast & \psf-aware & Richardson--Lucy & Tikhonov & Wiener--Hunt & \\
\midrule
{chr(10).join(l.replace(r" \\", r" & \\") for l in con)}
\bottomrule
\end{{tabular}}
\end{{table*}}""")


def supp_e5() -> None:
    e = pd.read_csv(RES / "analysis_e5" / "e5_comparisons.csv")
    comps = ["rl2", "tik_id", "tik_lap", "wiener_hunt", "geo_nearest", "geo_linear",
             "geo_cubic", "strongest"]
    lines = []
    for comp in comps:
        cells = []
        for lev in (20, 19):
            for ours in ["A/loso", "B/loso", "A/transfer"]:
                r = pick(e, level=lev, ours=ours, competitor=comp, scene_class="all")
                cells.append(cell(r.mean_rel_margin, r.wins, r.n_regions))
        lines.append(f"{COMP_LABEL[comp]} & " + " & ".join(cells) + r" \\")
    write(SUPP_T / "tab_e5.tex", rf"""\begin{{table*}}[t]
\centering
\caption{{Cross-sensor validation (experiment E5, 13 sites, 307 patches).
Mean relative RMSE margin of \psf-aware over each competitor against the
\VENUS{{}} cell-average reference, in percent; in brackets, sites won out of
13. LOSO: leave-one-site-out selection of every method, with the width fixed
(A) or tuned (B). Transfer: configurations selected on the 40 synthetic
regions of E1 at the nominal width.}}
\label{{tab:e5}}
\setlength{{\tabcolsep}}{{3pt}}
\footnotesize
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
& \multicolumn{{3}}{{c}}{{L20 (6.2~m cells)}} & \multicolumn{{3}}{{c}}{{L19 (12.4~m cells)}} \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}
Competitor & LOSO, A & LOSO, B & Transfer & LOSO, A & LOSO, B & Transfer \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table*}}""")
    # site table: patches and spectral budget
    m = pd.read_csv(RES / "e5_manifest.csv")
    b = pd.read_csv(RES / "analysis_e5_spectral" / "e5_spectral_budget.csv")
    rows = []
    for site, g in m.groupby("site"):
        sel = g[g.selected]
        bb = b[(b.site == site) & (b.cells == "L20")]
        ratio = bb.rmse_ratio_optimal.iloc[0] if len(bb) else np.nan
        share = bb.share_out.iloc[0] if len(bb) else np.nan
        rows.append((site, len(g), int(g.usable.sum()), len(sel),
                     np.nanmedian(np.hypot(sel.shift_row_px, sel.shift_col_px)),
                     np.median(sel.fit_a), share, ratio))
    rows.sort(key=lambda r: r[0])
    body = "\n".join(
        f"{s.replace('_', r'\_')} & {n} & {u} & {k} & {sh:.2f} & {a:.2f} & {100 * so:.1f} & {100 * (1 - rr):.1f}"
        + r" \\" for s, n, u, k, sh, a, so, rr in rows)
    write(SUPP_T / "tab_e5_sites.tex", rf"""\begin{{table}}[t]
\centering
\caption{{SEN2\VENUS{{}} sites used in E5. Candidate and usable patches
(after the input-only filters), selected patches, median co-registration
shift (Sentinel-2 pixels), median radiometric gain $a$, share of the
\VENUS{{}} error energy outside the Sentinel-2 band at L20, and largest RMSE
reduction any isotropic linear filter fitted on the same data could achieve
over identity at L20 (exploratory spectral analysis).}}
\label{{tab:e5_sites}}
\setlength{{\tabcolsep}}{{2pt}}
\scriptsize
\begin{{tabular}}{{@{{}}lrrrcccc@{{}}}}
\toprule
Site & Cand. & Usable & Sel. & Shift & $a$ & Out (\%) & Filter (\%) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}""")


def supp_e6() -> None:
    c, x = e6_frames()
    comps = ["rl2", "tik_id", "wiener_hunt", "geo_cubic", "strongest"]
    lines = []
    for arm, label in E6_ARMS:
        a = c[c.arm == arm]
        cells = []
        for scheme, mode in [("transfer", "A"), ("transfer", "B"), ("5fold", "A")]:
            r = pick(a, scheme=scheme, mode=mode, competitor="strongest")
            cells.append(cell(r.mean_rel_margin, r.wins, r.n_regions))
        for comp in comps:
            r = pick(a, scheme="5fold", mode="B", competitor=comp)
            cells.append(cell(r.mean_rel_margin, r.wins, r.n_regions))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    for lev in (19, 20):
        a = x[x.ours == f"psf_aware_l{lev}"]
        cells = []
        for scheme, mode in [("transfer", "A"), ("transfer", "B"), ("5fold", "A")]:
            r = pick(a, scheme=scheme, mode=mode, competitor="strongest")
            cells.append(cell(r.mean_rel_margin, r.wins, r.n_regions))
        for comp in comps:
            r = pick(a, scheme="5fold", mode="B", competitor=comp)
            cells.append(cell(r.mean_rel_margin, r.wins, r.n_regions))
        lines.append(rf"Ratio 0.8, L{lev} then L18$^{{*}}$ & " + " & ".join(cells) + r" \\")
    write(SUPP_T / "tab_e6.tex", rf"""\begin{{table*}}[t]
\centering
\caption{{Generalization arms (experiment E6). Mean relative RMSE margin of
\psf-aware, in percent; in brackets, regions won. The first three columns
compare with the strongest tuned deconvolution of each region under the
transferred configurations (nominal width A, tuned width B) and the internal
5-fold selection at the nominal width; the last five use the internal
selection with the width tuned (mode B), the reproduction criterion of the
protocol. $^{{*}}$Exploratory (E6-X).}}
\label{{tab:e6}}
\setlength{{\tabcolsep}}{{2pt}}
\scriptsize
\begin{{tabular}}{{@{{}}lcccccccc@{{}}}}
\toprule
& \multicolumn{{3}}{{c}}{{vs.\ strongest deconvolution}} & \multicolumn{{5}}{{c}}{{Internal, mode B}} \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-9}}
Arm & Transfer A & Transfer B & Internal A & RL & Tikhonov & Wiener--Hunt & Bicubic & Strongest \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table*}}""")
    p = pd.read_csv(RES / "analysis_e6" / "e6e_patch_size.csv")
    rows = []
    for fam in ["psf_aware", "rl2", "tik_id", "wiener_hunt", "geo_cubic"]:
        g = p[(p.family == fam) & (p.domain == "central")].set_index("size")
        cells = []
        for s in (512, 1024):
            cells.append(f"{100 * g.loc[s, 'median_change_vs_256']:.3f}")
            cells.append(f"{100 * g.loc[s, 'max_abs_change_vs_256']:.3f}")
        rows.append(f"{COMP_LABEL.get(fam, r'\psf-aware')} & " + " & ".join(cells) + r" \\")
    q = p[(p.family == "psf_aware") & (p.domain == "full")].set_index("size")
    rt = " / ".join(f"{q.loc[s, 'median_runtime_s']:.2f}" for s in (256, 512, 1024))
    slope = q.psf_runtime_loglog_slope.iloc[0]
    nreg = int(p[(p.family == "psf_aware") & (p.domain == "central") & (p["size"] == 1024)].n_regions.iloc[0])
    n512 = int(p[(p.family == "psf_aware") & (p.domain == "central") & (p["size"] == 512)].n_regions.iloc[0])
    write(SUPP_T / "tab_e6e.tex", rf"""\begin{{table}}[t]
\centering
\caption{{Patch size (E6e; B04, ratio 1.6, transferred configurations;
{n512} regions at $512^2$ and {nreg} at $1024^2$, one $1024^2$ window being
incomplete). Relative change of the RMSE on the common domain (the interior
of the $256^2$ patch) against the $256^2$ result, in percent: median and
largest absolute value over regions. Median \psf-aware run time
(construction and solve, shared CPU nodes): {rt}~s for
$256^2$/$512^2$/$1024^2$; log--log slope against the number of samples
{slope:.2f}.}}
\label{{tab:e6e}}
\setlength{{\tabcolsep}}{{2.5pt}}
\scriptsize
\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\toprule
& \multicolumn{{2}}{{c}}{{$512^2$}} & \multicolumn{{2}}{{c}}{{$1024^2$}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
Family & median & max & median & max \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}""")


def supp_e7() -> None:
    s = pd.read_csv(RES / "analysis_e7" / "e7_summary.csv").set_index("operator")
    spec = [("psf_aware", r"\psf-aware"), ("psf_aware_cons", r"\psf-aware, constrained"),
            ("overlap1", "Overlap, first order"), ("hard_binning", "Hard binning")]
    rows = []
    for op, lab in spec:
        r = s.loc[op]
        rows.append(
            f"{lab} & {r.rel_median:.3f} [{r.rel_q25:.3f}, {r.rel_q75:.3f}] & "
            f"{100 * r.fill_fraction_median:.0f} & "
            f"{sci(r.abs_cons_err_median)} & {sci(r.abs_cons_err_max)} & "
            f"{sci(r.abs_mean_err_median)} & {r.runtime_s_median:.3f}" + r" \\")
    write(SUPP_T / "tab_e7.tex", rf"""\begin{{table*}}[t]
\centering
\caption{{Operator comparison (experiment E7), 40 real regions. RMSE relative
to bicubic (median and interquartile range) on interior cells (for hard
binning, on the cells it fills), median fill fraction, relative conservation
error on each operator's own support (median and maximum), median relative
error on the regional mean, and median run time.}}
\label{{tab:e7}}
\setlength{{\tabcolsep}}{{4pt}}
\footnotesize
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
Operator & RMSE/bicubic & Fill (\%) & Cons.\ err.\ (median) & (max) & Mean err. & Time (s) \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}""")


def copy_figures() -> None:
    fig = RES / "figures"
    for name in ["e4_width_lambda_maps_it14.pdf", "e4_width_curves.pdf",
                 "e5_width_curves.pdf"]:
        shutil.copy2(fig / name, MAIN_F / name)
    for name in ["e3_arms.pdf", "e4_width_lambda_maps_it30.pdf", "e5_spectral.pdf",
                 "e6_generalization.pdf", "e7_operators.pdf"]:
        shutil.copy2(fig / name, SUPP_F / name)
    print("copied figures")


if __name__ == "__main__":
    main_tuned()
    main_ablation()
    main_generalization()
    main_operators()
    supp_e1_perclass()
    supp_e1_selected()
    supp_e1_variants()
    supp_e2_perclass()
    supp_e2_selected()
    supp_hyper_figure()
    supp_e3()
    supp_e5()
    supp_e6()
    supp_e7()
    copy_figures()
