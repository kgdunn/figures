"""Verify every number the coding-sensitivity section of the chapter quotes.

Run from ``figures/doe``. Prints: R2Y(3) per coding; per-run SPE/T2 diagnostics and the flagged
compound per coding (including sum with A dropped); score-matching inversion reachability under sum
vs cell-means; and the coding-invariant curve-match inversion. Also the minimal OLS-vs-truncated-PLS
demonstration.
"""

# check-scripts: requires pyoptex -- the I-optimal colour design comes from pyoptex
import contextlib
import io

import numpy as np
import pandas as pd
from patsy import dmatrix

from colour_case_study import (
    COMPOUND_LEVELS,
    build_design,
    coded_matrix,
    curve_match_inversion,
    fit_coding,
    goal_projection,
    invert_to_factors,
    simulate_curves,
)
from process_improve.multivariate.methods import PLS


def quiet(fn, *a, **k):
    with contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


def main():
    design = build_design("i_optimal", budget=60)
    curves = simulate_curves(design)
    compound = design.design["compound"].to_numpy()

    print("=" * 70, "\nR2Y(3) and diagnostics per coding (3-component interaction PLS)")
    settings = [("sum (drop F)", "sum", COMPOUND_LEVELS),
                ("sum (drop A)", "sum", list(reversed(COMPOUND_LEVELS))),
                ("treatment", "treatment", COMPOUND_LEVELS),
                ("cell_means", "cell_means", COMPOUND_LEVELS)]
    for label, coding, order in settings:
        pls, _ = quiet(fit_coding, design, curves, coding, order=order)
        t2 = pls.hotellings_t2_.iloc[:, -1].to_numpy()
        spe = pls.spe_.iloc[:, -1].to_numpy()
        r2 = float(np.asarray(pls.r2_cumulative_).ravel()[-1])
        t2lim, spelim = float(pls.hotellings_t2_limit()), float(pls.spe_limit())
        per = {c: round(float(t2[compound == c].max()), 1) for c in COMPOUND_LEVELS}
        overs = int((t2 > t2lim).sum())
        oversp = int((spe > spelim).sum())
        flagged = sorted({c for c in COMPOUND_LEVELS if (t2[compound == c] > t2lim).any()})
        ncols = coded_matrix(design, coding, order=order)[0].shape[1]
        print(f"  {label:14s}: R2Y(3)={r2:.3f}  cols={ncols}  T2lim={t2lim:.1f} SPElim={spelim:.1f}  "
              f"overT2={overs} overSPE={oversp}  flagged={flagged}")
        print(f"                  maxT2 per compound = {per}")

    print("=" * 70, "\nScore-matching inversion (3 comp): reachable set per coding")
    for label, coding in [("sum", "sum"), ("treatment", "treatment"), ("cell_means", "cell_means")]:
        pls, di = quiet(fit_coding, design, curves, coding)
        goal = goal_projection(pls, di)
        tbl = invert_to_factors(pls, di, goal["score"])
        reach = [c for c in ["B", "C", "D", "E", "F"] if bool(tbl.loc[c, "in_range"])]
        print(f"  {label:11s}: goal SPE={goal['spe']:.2f} T2={goal['t2']:.3f}  reachable={reach}")
        for c in ["B", "C", "D", "E", "F"]:
            coded = [float(tbl.loc[c, f'{n}_coded']) for n in
                     ['concentration', 'co_solvent', 'pH', 'temperature']]
            print(f"      {c}: {np.round(coded, 2)}  in={bool(tbl.loc[c, 'in_range'])}")

    print("=" * 70, "\nCurve-match inversion (full-rank OLS): coding-invariant")
    s = curve_match_inversion(design, curves, "sum")
    cm = curve_match_inversion(design, curves, "cell_means")
    for c in ["B", "C", "D", "E", "F"]:
        cs = [float(s.loc[c, f'{n}_coded']) for n in ['concentration', 'co_solvent', 'pH', 'temperature']]
        cc = [float(cm.loc[c, f'{n}_coded']) for n in ['concentration', 'co_solvent', 'pH', 'temperature']]
        diff = float(np.max(np.abs(np.array(cs) - np.array(cc))))
        real = {n: round(float(s.loc[c, n]), 1) for n in ['concentration', 'co_solvent', 'pH', 'temperature']}
        print(f"  {c}: {np.round(cs, 2)}  in_range={bool(s.loc[c, 'in_range'])}  "
              f"sum-vs-cell|diff|={diff:.1e}  real={real}")

    print("=" * 70, "\nMinimal demo: OLS invariant, truncated PLS not (3-level factor + 1 continuous)")
    rng = np.random.default_rng(0)
    n = 30
    g = np.array(["A"] * 11 + ["B"] * 10 + ["C"] * 9)
    rng.shuffle(g)
    x = rng.normal(size=n)
    eff = {"A": 0.0, "B": 1.0, "C": -0.5}
    y = np.array([eff[v] for v in g]) + 0.7 * x + 0.1 * rng.normal(size=n)
    df = pd.DataFrame({"g": g, "x": x})
    Y1 = pd.DataFrame({"y": y})
    Xt = dmatrix("C(g, Treatment) + x", df, return_type="dataframe").drop(columns=["Intercept"])
    Xs = dmatrix("C(g, Sum) + x", df, return_type="dataframe").drop(columns=["Intercept"])
    Z = np.column_stack([np.ones(n), Xt.values])
    Zs = np.column_stack([np.ones(n), Xs.values])
    ols_t = Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    ols_s = Zs @ np.linalg.lstsq(Zs, y, rcond=None)[0]
    print(f"  OLS max|pred diff| = {np.max(np.abs(ols_t - ols_s)):.1e}")
    for a in (1, 2, 3):
        pt = np.asarray(quiet(lambda A=a: PLS(n_components=A, scale=True).fit(Xt, Y1)).predictions_).ravel()
        psv = np.asarray(quiet(lambda A=a: PLS(n_components=A, scale=True).fit(Xs, Y1)).predictions_).ravel()
        print(f"  PLS({a}) max|pred diff| = {np.max(np.abs(pt - psv)):.3f}")


if __name__ == "__main__":
    main()
