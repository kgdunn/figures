"""Print every number quoted in the mixed-level split-plot case-study chapter.

Run from this directory:  python check_colour_case_study.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from colour_case_study import (
    CONT,
    GROUND_TRUTH,
    HARD_TO_CHANGE,
    build_design,
    evaluate,
    fit_profile_pls,
    model_matrix,
    shape_distance_to_reference,
    simulate_curves,
)
from colour_case_study import goal_projection, interaction_matrix, invert_to_factors
from process_improve.experiments import analyze_experiment
from process_improve.multivariate.methods import PLS


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
