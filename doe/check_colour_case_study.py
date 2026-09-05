"""Print every number quoted in the mixed-level split-plot case-study chapter.

Run from this directory:  python check_colour_case_study.py
"""

# check-scripts: requires pyoptex -- the I-optimal colour design comes from pyoptex
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from scipy.optimize import minimize
from sklearn.metrics import r2_score

from colour_case_study import (
    COMPOUND_LEVELS,
    CONT,
    GROUND_TRUTH,
    HARD_TO_CHANGE,
    build_design,
    curve_match_inversion,
    evaluate,
    fit_profile_pls,
    ground_truth_curve,
    model_matrix,
    shape_distance_to_reference,
    shape_floor,
    simulate_curves,
)
from colour_case_study import goal_projection, interaction_matrix, invert_to_factors
from process_improve.experiments import analyze_experiment
from process_improve.multivariate.methods import PLS


def quiet(fn):
    with contextlib.redirect_stderr(io.StringIO()):
        return fn()


def n_changes(s: pd.Series) -> int:
    a = s.to_numpy()
    return int((a[1:] != a[:-1]).sum())


print("=" * 70)
print("STEP 1  design generation (I-optimal, split-plot, quadratic, 60 runs)")
design = build_design("i_optimal", budget=60)
print(f"  runs                : {design.n_runs}")
print(f"  compound balance    : {design.design['compound'].value_counts().sort_index().to_dict()}")
print("  level changes across the run order:")
for n in CONT:
    tag = "  (hard-to-change)" if n in HARD_TO_CHANGE else ""
    print(f"    {n:14s}: {n_changes(design.design[n]):3d}{tag}")

print("=" * 70)
print("STEP 2  design quality, and a 48- vs 60-run / I- vs D-optimal comparison")
for crit in ("i_optimal", "d_optimal"):
    for budget in (48, 60):
        d = build_design(crit, budget=budget)
        m = evaluate(d)
        q = m["fds"]["quantiles"]
        print(f"  {crit:10s} n={budget}: D={m['d_efficiency']:6.2f}  I={m['i_efficiency']:7.2f}  "
              f"G={m['g_efficiency']:6.2f}  FDS median={q['0.5']:.3f}  FDS max={q['1']:.3f}  "
              f"dof_resid={m['degrees_of_freedom']['residual']}")

print("=" * 70)
print("STEP 3  colour-development profile + PLS model")
curves = simulate_curves(design)
pls = fit_profile_pls(design, curves, n_components=5)
yhat = pls.predict(model_matrix(design))
r2y = r2_score(curves, yhat, multioutput="variance_weighted")
print(f"  profile Y           : {curves.shape[0]} runs x {curves.shape[1]} time points")
print(f"  PLS R2Y (var-wtd)   : {r2y:.3f}")
manual = float(np.sqrt(((curves.to_numpy() - yhat.to_numpy()) ** 2).mean(0).mean()))
reported = float(np.asarray(pls.rmse_.iloc[:, -1]).mean())
print(f"  rmse_ (orig scale)  : reported {reported:.4f} vs manual {manual:.4f}")

print("=" * 70)
print("STEP 4  compound x factor interactions (analyze_experiment, C(compound, Sum))")
adf = design.design[["compound"] + list(CONT)].copy()
adf["compound"] = adf["compound"].astype(str)
adf["peak"] = curves.max(axis=1).to_numpy()
formula = "peak ~ C(compound, Sum)*co_solvent + C(compound, Sum)*pH + C(compound, Sum)*temperature + concentration"
res = analyze_experiment(adf, response_column="peak", model=formula, analysis_type=["anova"], coding="coded")
print(f"  model R2            : {res['model_summary']['r_squared']:.3f}")
anova = pd.DataFrame(res["anova_table"])
for _, row in anova[anova["source"].str.contains(":")].iterrows():
    verdict = "significant" if row["p_value"] < 0.05 else "n.s."
    print(f"    {row['source']:34s} F={row['F']:8.1f}  p={row['p_value']:.1e}  {verdict}")

print("=" * 70)
print("STEP 5  objective 4: colour-shape distance to the reference compound A")
dist = shape_distance_to_reference(design, curves)
for cmp_, dd in dist.items():
    tag = " <- reference" if cmp_ == "A" else (" <- closest analog" if cmp_ == dist.index[1] else "")
    print(f"    {cmp_}: {dd:.4f}   |drift|_truth={abs(GROUND_TRUTH[cmp_]['drift']):.2f}{tag}")

print("=" * 70)
print("STEP 6  diagnostics + model inversion (interaction PLS, 3 components)")
X_int, design_info = interaction_matrix(design)
pls_full = PLS(n_components=3, scale=True).fit(X_int, curves)
g = goal_projection(pls_full, design_info)
t2_run = pls_full.hotellings_t2_.iloc[:, -1].to_numpy()
spe_run = pls_full.spe_.iloc[:, -1].to_numpy()
print(f"  limits: T2_95={g['t2_limit']:.1f}  SPE_95={g['spe_limit']:.1f}")
print(f"  goal (A at centre): SPE={g['spe']:.2f}  T2={g['t2']:.3f}  "
      f"(inside: {g['spe'] < g['spe_limit'] and g['t2'] < g['t2_limit']})")
print(f"  runs over T2: {int((t2_run > g['t2_limit']).sum())}  over SPE: {int((spe_run > g['spe_limit']).sum())}  "
      f"max T2={t2_run.max():.1f}")
tab = invert_to_factors(pls_full, design_info, g["score"])
for cmp_ in ["B", "C", "D", "E", "F"]:
    real = {n: f"{tab.loc[cmp_, n]:.1f}" for n in CONT}
    flag = "within ranges" if tab.loc[cmp_, "in_range"] else "OUT of range"
    print(f"    {cmp_}: {real}  resid={tab.loc[cmp_, 'resid']:.1e}  {flag}")

print("=" * 70)
print("STEP 7  peak: 3-component PLS vs least squares, omnibus F, Q2Y per time point")
rhs = ("C(compound, Sum)*co_solvent + C(compound, Sum)*pH "
       "+ C(compound, Sum)*temperature + concentration")
Xp = dmatrix(rhs, adf, return_type="dataframe").drop(columns=["Intercept"])
coefs = analyze_experiment(adf, response_column="peak", model="peak ~ " + rhs,
                           analysis_type=["coefficients"])["coefficients"]
ols = {c["term"]: c["coefficient"] for c in coefs}
se = {c["term"]: c.get("std_error", c.get("standard_error")) for c in coefs}
pls_peak = quiet(lambda: PLS(n_components=3, scale=True).fit(Xp, adf[["peak"]]))
beta = pls_peak.beta_coefficients_.iloc[:, 0]
diff = np.array([abs(beta[t] - ols[t]) for t in Xp.columns])
nse = np.array([abs(beta[t] - ols[t]) / se[t] for t in Xp.columns])
inter_df = sum(":co_solvent" in t for t in Xp.columns)   # contrasts per compound-by-factor term
print(f"  max |PLS - OLS| = {diff.max():.4f}  |  within 1 SE: {(nse <= 1).sum()}/{len(nse)} "
      f"(largest {nse.max():.2f} SE)")
print(f"  concentration: OLS {ols['concentration']:+.3f} PLS {beta['concentration']:+.3f}  |  "
      f"pH: OLS {ols['pH']:+.3f} PLS {beta['pH']:+.3f}")
print(f"  each compound-by-factor interaction is a joint F test over df = {inter_df} contrasts")
r2y_pt = pls_full.r2y_per_variable_.iloc[:, -1].to_numpy()
q2y_pt = np.asarray(quiet(lambda: pls_full.cross_validate(X_int, curves, cv="loo"))["q_squared"]).ravel()
print("  R2Y / Q2Y (leave-one-out) per time point, 3 components:")
for i, (a, b) in enumerate(zip(r2y_pt, q2y_pt)):
    print(f"    t{i}: R2Y={a:.2f}  Q2Y={b:+.2f}")

print("=" * 70)
print("STEP 8  closeness on the developed curve (t1 onward): best match vs validated")
goal_curve = ground_truth_curve("A", [0, 0, 0, 0])
dev = slice(1, None)                                       # drop t0 (near-zero noise)
cmv = curve_match_inversion(design, curves, "treatment")  # coding-invariant
print("  cmp | drift | best(t1-9) | score-match validated | curve-match validated")
for cmp_ in ["B", "F", "C", "D", "E"]:                     # by |drift|, closest first
    best = shape_floor(cmp_)[0]
    sm_cd = [float(tab.loc[cmp_, f"{n}_coded"]) for n in CONT]
    cm_cd = [float(cmv.loc[cmp_, f"{n}_coded"]) for n in CONT]
    sm_r = float(np.sqrt(np.mean((ground_truth_curve(cmp_, sm_cd)[dev] - goal_curve[dev]) ** 2)))
    cm_r = float(np.sqrt(np.mean((ground_truth_curve(cmp_, cm_cd)[dev] - goal_curve[dev]) ** 2)))
    print(f"    {cmp_}: {GROUND_TRUTH[cmp_]['drift']:+.2f}  {best:.3f}   {sm_r:.3f}   {cm_r:.3f}")

print("=" * 70)
print("STEP 9  constrained inversion (Muteki-MacGregor): relaxing T2/SPE does not beat the floor")
full = dmatrix(rhs, adf, return_type="dataframe")          # keep intercept: full rank
beta_full = np.linalg.lstsq(full.to_numpy(), curves.to_numpy(), rcond=None)[0]


def _row(cmp_, cd):
    return pd.DataFrame({"compound": pd.Categorical([cmp_], categories=COMPOUND_LEVELS),
                         "concentration": [cd[0]], "co_solvent": [cd[1]],
                         "pH": [cd[2]], "temperature": [cd[3]]})


def curve_of(cmp_, cd):
    x = build_design_matrices([full.design_info], _row(cmp_, cd), return_type="dataframe")[0]
    return x.to_numpy().ravel() @ beta_full


def diag_row(cmp_, cd):
    return build_design_matrices([design_info], _row(cmp_, cd), return_type="dataframe")[0].drop(columns=["Intercept"])


def t2spe(cmp_, cd):
    d = quiet(lambda: pls_full.diagnose(diag_row(cmp_, cd)))
    return float(d.hotellings_t2.iloc[0]), float(d.spe.iloc[0])


y_goal = curve_of("A", [0, 0, 0, 0])
t2_lim, spe_lim = float(pls_full.hotellings_t2_limit()), float(pls_full.spe_limit())
for cmp_ in ["B", "C", "D", "E", "F"]:
    obj = lambda cd, _c=cmp_: float(np.sum((y_goal - curve_of(_c, cd)) ** 2))  # noqa: E731
    cons = [{"type": "ineq", "fun": lambda cd, _c=cmp_: t2_lim - t2spe(_c, cd)[0]},
            {"type": "ineq", "fun": lambda cd, _c=cmp_: spe_lim - t2spe(_c, cd)[1]}]
    settings = minimize(obj, np.zeros(4), method="SLSQP", constraints=cons).x
    rmse = float(np.sqrt(np.mean((ground_truth_curve(cmp_, settings)[dev] - goal_curve[dev]) ** 2)))
    print(f"    {cmp_}: relaxed RMSE(t1-9) {rmse:.3f}  vs shape floor {shape_floor(cmp_)[0]:.3f}")
