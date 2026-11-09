"""The colour curves modelled one shape at a time, by functional principal components.

Left: the two shapes the sixty runs' curves differ in, the eigenfunctions of their functional PCA,
each with its share of the curves' variance. The first is the height of the curve; the second is
the late drift, which differs by chromogen.

Right: chromogen D at high concentration, the other factors centred. The true curve against the
curve predicted from the scores under the chapter's model, and under that model with the
compound-by-concentration interaction added, each with its 95% band at every time point. In the
ground truth the drift scales with the curve's height, so concentration changes it by a different
amount for each chromogen; without that term the predicted curve ends low.

Uses ``analyze_functional_experiment`` from ``process_improve`` 2.4.0 or later.
Regenerates ``colour-fpca-shapes-and-bands.png``.
"""

import warnings

import matplotlib.pyplot as plt
import pandas as pd
from process_improve.experiments import analyze_functional_experiment

from colour_case_study import RHS, TIME_POINTS, _analysis_frame, build_design, ground_truth_curve, simulate_curves

design = build_design()
curves = simulate_curves(design)
settings = _analysis_frame(design)
settings["compound"] = settings["compound"].astype(str)
profile = curves.set_axis(TIME_POINTS, axis=1)  # the columns are the time points the readings are taken at

models = {
    "the chapter's model": RHS,
    "with compound x concentration": RHS + " + C(compound, Sum):concentration",
}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fits = {name: analyze_functional_experiment(settings, profile, model=rhs) for name, rhs in models.items()}

at = pd.DataFrame([{"compound": "D", "concentration": 1.0, "co_solvent": 0.0, "pH": 0.0, "temperature": 0.0}])
truth = ground_truth_curve("D", [1.0, 0.0, 0.0, 0.0])

fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))

first = next(iter(fits.values()))
shares = first.components["variance_ratio"]
for (name, shape), colour in zip(first.fpca.eigenfunctions_.iloc[:, :2].items(), ["#1f5fa8", "#c0392b"], strict=True):
    left.plot(TIME_POINTS, shape, color=colour, lw=2.2, marker="o", ms=4, label=f"{name} [{shares[name]:.1%}]")
left.axhline(0, color="#999999", lw=0.8)
left.set_xticks(TIME_POINTS)
left.set_xlabel("Time point")
left.set_ylabel("Eigenfunction (shape)")
left.set_title("The two shapes the curves differ in", fontsize=10, loc="left")
left.legend(frameon=False, fontsize=9, loc="lower right")
left.grid(alpha=0.2)

right.plot(TIME_POINTS, truth, color="#111111", lw=2.6, marker="o", ms=4.5, label="true curve", zorder=5)
for (name, fit), colour, dash in zip(fits.items(), ["#d68910", "#1f5fa8"], ["--", "-"], strict=True):
    band = fit.predict(at, n_bootstrap=400, random_state=1)
    right.fill_between(TIME_POINTS, band["pointwise_low"].iloc[0], band["pointwise_high"].iloc[0], color=colour,
                       alpha=0.18, lw=0)
    right.plot(TIME_POINTS, band["curve"].iloc[0], color=colour, lw=2.0, ls=dash, label=f"predicted, {name}")
    print(f"{name}: ends {band['curve'].iloc[0].iloc[-1] - truth[-1]:+.3f} from the truth")
right.set_xticks(TIME_POINTS)
right.set_xlabel("Time point")
right.set_ylabel("Absorbance (colour intensity)")
right.set_title("Chromogen D at high concentration, other factors centred", fontsize=10, loc="left")
right.legend(frameon=False, fontsize=8.5, loc="lower right")
right.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("colour-fpca-shapes-and-bands.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved colour-fpca-shapes-and-bands.png")
