"""Generate the FDS (fraction-of-design-space) figure for the design-quality subchapter.

Compares a 4-factor definitive screening design (9 runs) against a 13-run OMARS
design on the main-effects-plus-quadratic model. The prediction variance is integrated
over the cube [-1, 1]^4 by process_improve's ``evaluate_design`` (v1.44.0+): its
``include_vertices=True`` adds the 2^4 cube vertices (the extreme corner points, where
the worst-case prediction variance can sit and which random interior sampling misses)
to the uniform sample, and ``fds_resolution=200`` returns the dense curve plotted here.
Reproducible; run from this directory: it writes ``fds-plot-dsd-vs-omars.png``
alongside the script.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from process_improve.experiments import Factor, evaluate_design, generate_design

N_EVAL = 80_000  # uniform points for the prediction-variance integration
EVAL_SEED = 1
NAMES = list("ABCD")
MODEL = " + ".join(NAMES + [f"I({c}**2)" for c in NAMES])  # 9 terms: 1 + 4 linear + 4 quadratic


def scaled_fds_curve(design):
    """Scaled prediction-variance FDS curve (fraction, SPV) from ``evaluate_design``.

    ``include_vertices=True`` adds the 2^4 cube vertices to the interior sample, so the
    worst case at a corner anchors the right-hand tail. Here it lifts the DSD's worst
    case from 8.98 to 9.00; the OMARS worst case is interior, so it is unchanged.
    """
    df = pd.DataFrame(np.asarray(design, float), columns=NAMES)
    curve = evaluate_design(df, model=MODEL, metric="fds", n_samples=N_EVAL,
                            random_seed=EVAL_SEED, include_vertices=True,
                            fds_resolution=200)["fds"]["curve"]
    return curve["fraction"], curve["scaled_prediction_variance"]


# 4-factor DSD (9 runs) built with process_improve: a conference-matrix foldover plus a
# centre run. The library has no OMARS generator, so the 13-run OMARS design below is
# built by hand.
_dsd = generate_design([Factor(name=c, low=-1, high=1) for c in "ABCD"], design_type="dsd")
DSD = np.asarray(_dsd.design[NAMES], float)
# 13-run OMARS design (balanced foldover member, 2 estimable interactions)
OM13 = np.array([
    [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, -1, -1], [1, -1, -1, 0], [1, 0, 1, -1],
    [1, 1, 0, 1], [0, 0, 0, -1], [0, 0, -1, 0], [0, -1, 1, 1], [-1, 1, 1, 0],
    [-1, 0, -1, 1], [-1, -1, 0, -1], [0, 0, 0, 0]], float)

fig, ax = plt.subplots(figsize=(7.2, 4.6))
styles = {"DSD [n=9]": dict(color="#1f5fa8", lw=2.0),
          "OMARS [n=13]": dict(color="#c0392b", lw=2.0, ls="--")}
for name, D in [("DSD [n=9]", DSD), ("OMARS [n=13]", OM13)]:
    fraction, spv = scaled_fds_curve(D)
    ax.plot(fraction, spv, label=name, **styles[name])

ax.set_xlabel("Fraction of design space (proportion of region with SPV at or below the curve)")
ax.set_ylabel("Scaled prediction variance, SPV")
ax.set_xlim(0, 1)
ax.set_ylim(0, 13)
ax.axvline(0.5, color="0.6", lw=0.8, ls=":")
ax.text(0.51, 0.6, "median", color="0.4", fontsize=8, rotation=90, va="bottom")
ax.legend(frameon=False, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fds-plot-dsd-vs-omars.png", dpi=300, facecolor="w", edgecolor="w",
            orientation="portrait", format=None, transparent=True)
print("saved figure")
