"""Coefficient comparison: least-squares regression vs PLS, same model and response.

Both models are fitted to the peak colour intensity with the same interaction terms (compound
main effects with sum-to-zero coding, their interactions with co-solvent, pH and temperature, and
concentration). The least-squares coefficients and the PLS ``beta_coefficients_`` are shown side by
side; they nearly coincide, so PLS reproduces the regression when enough components are kept.
Regenerates ``colour-coefficient-comparison.png``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrix

from colour_case_study import build_design, simulate_curves
from process_improve.experiments import analyze_experiment
from process_improve.multivariate.methods import PLS

design = build_design("i_optimal", budget=60)
curves = simulate_curves(design)
cont = ["concentration", "co_solvent", "pH", "temperature"]
adf = design.design[["compound"] + cont].copy()
adf["compound"] = adf["compound"].astype(str)
adf["peak"] = curves.max(axis=1).to_numpy()

rhs = ("C(compound, Treatment)*co_solvent + C(compound, Treatment)*pH "
       "+ C(compound, Treatment)*temperature + concentration")
ols = analyze_experiment(adf, response_column="peak", model="peak ~ " + rhs,
                         analysis_type=["coefficients"])["coefficients"]
ols = {c["term"]: c["coefficient"] for c in ols}

X_int = dmatrix(rhs, adf, return_type="dataframe").drop(columns=["Intercept"])
pls = PLS(n_components=3, scale=True).fit(X_int, adf[["peak"]])
beta = np.asarray(pls.beta_coefficients_).ravel()


def short(term):
    return term.replace("C(compound, Treatment)[T.", "cmp").replace("]", "")


coef = pd.DataFrame({"term": [short(c) for c in X_int.columns],
                     "OLS": [ols[c] for c in X_int.columns], "PLS": beta}).sort_values("OLS")
y = np.arange(len(coef))

fig, ax = plt.subplots(figsize=(7.4, 8.2))
ax.hlines(y, coef["OLS"], coef["PLS"], color="0.7", lw=1.0, zorder=1)
ax.scatter(coef["OLS"], y, s=46, color="#1f5fa8", marker="o", label="least squares", zorder=3)
ax.scatter(coef["PLS"], y, s=40, color="#c0392b", marker="X", label="PLS (3 components)", zorder=3)
ax.axvline(0, color="0.5", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(coef["term"], fontsize=8)
ax.set_ylim(-0.7, len(coef) - 0.3)
ax.set_xlabel("Coefficient for the peak colour intensity (coded factors)")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig("colour-coefficient-comparison.png", dpi=300, facecolor="w", edgecolor="w",
            transparent=True)
print("saved colour-coefficient-comparison.png")
