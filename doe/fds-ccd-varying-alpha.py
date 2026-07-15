"""FDS (fraction-of-design-space) of one central composite design as the axial distance
alpha is varied.

A single central composite design at k = 3 (2^3 factorial cube at +/-1, six axial runs at
+/-alpha, one centre run) is scored on the full second-order model over the cube [-1, 1]^3
for a range of alpha. The axial distance is the one knob the post highlights: alpha = 1 is
the face-centred design, alpha = (2^k)^(1/4) = 1.682 is the rotatable (circumscribed)
design, and larger alpha pushes the star runs further outside the cube.

The FDS curve shows the trade-off. Pulling the stars in to the faces (alpha = 1) keeps the
whole design inside +/-1 and gives the lowest prediction variance across most of the cube,
but the design is no longer rotatable. Pushing the stars out raises the variance in the
cube interior yet extends precise prediction toward the axial extremes; the rotatable choice
is the usual compromise.

Reproducible; run from this directory: writes ``fds-ccd-varying-alpha.png``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from process_improve.experiments import evaluate_design

from ccd_variants_designs import ALPHA_ROTATABLE, EVAL_SEED, FACTOR_NAMES, MODEL_FORMULA, N_EVAL, ccd_matrix

# Axial distances from face-centred (1.0) out past rotatable (1.682). sqrt(2) is the
# spherical-cube value; 2.0 places the stars a full unit beyond the cube face.
ALPHAS = [1.0, float(np.sqrt(2)), ALPHA_ROTATABLE, 2.0]
NAMED = {1.0: "face-centred", round(ALPHA_ROTATABLE, 3): "rotatable"}

norm = Normalize(vmin=min(ALPHAS), vmax=max(ALPHAS))
cmap = cm.viridis

fig, ax = plt.subplots(figsize=(7.2, 5.0))
for alpha in ALPHAS:
    df = pd.DataFrame(ccd_matrix(alpha), columns=FACTOR_NAMES)
    out = evaluate_design(
        df, model=MODEL_FORMULA, metric="fds", n_samples=N_EVAL, random_seed=EVAL_SEED,
        include_vertices=True, region="cuboidal", fds_resolution=200,
    )["fds"]
    curve = out["curve"]
    tag = NAMED.get(round(alpha, 3))
    label = rf"$\alpha$ = {alpha:.3f}" + (f"  ({tag})" if tag else "")
    ax.plot(
        curve["fraction"], curve["scaled_prediction_variance"],
        color=cmap(norm(alpha)), lw=2.2 if tag else 1.8,
        ls="-" if tag else "--", label=label,
    )

ax.set_xlabel("Fraction of design space (SPV at or below the curve)")
ax.set_ylabel("Scaled prediction variance, SPV = N x'(X'X)$^{-1}$x")
ax.set_title("Central composite FDS as the axial distance $\\alpha$ varies (k = 3, cube region)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 20)
ax.grid(alpha=0.25)
ax.legend(frameon=False, loc="upper left", title="axial distance")
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r"axial distance $\alpha$")
fig.tight_layout()
fig.savefig("fds-ccd-varying-alpha.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved fds-ccd-varying-alpha.png")
