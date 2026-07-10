"""PCA of the colour-development profiles: the response lives on a low-dimensional structure.

Each run's ten-point curve is one observation; the first two principal components of the
standardised curves separate the chromogens by their development shape. Regenerates
``colour-pca-scores.png``.
"""

import matplotlib.pyplot as plt
import numpy as np

from colour_case_study import COMPOUND_LEVELS, build_design, simulate_curves
from process_improve.multivariate.methods import PCA, MCUVScaler

design = build_design("i_optimal", budget=60)
curves = simulate_curves(design)
pca = PCA(n_components=3).fit(MCUVScaler().fit_transform(curves))
scores = np.asarray(pca.scores_)
r2cum = np.asarray(pca.r2_cumulative_).ravel()

palette = ["#1f5fa8", "#c0392b", "#2e8b57", "#8e44ad", "#d68910", "#17a2b8"]
compound = design.design["compound"].to_numpy()

fig, ax = plt.subplots(figsize=(6.6, 5.0))
for c, colour in zip(COMPOUND_LEVELS, palette):
    mask = compound == c
    label = "A (reference)" if c == "A" else c
    ax.scatter(scores[mask, 0], scores[mask, 1], s=42, color=colour, edgecolor="w",
               linewidth=0.5, label=label)

ax.axhline(0, color="0.7", lw=0.7)
ax.axvline(0, color="0.7", lw=0.7)
ax.set_xlabel(f"PC1 (R2 cumulative = {r2cum[0]:.2f})")
ax.set_ylabel(f"PC2 (R2 cumulative = {r2cum[1]:.2f})")
ax.grid(alpha=0.2)
ax.legend(frameon=False, fontsize=9, title="chromogen", ncol=2, loc="best")
fig.tight_layout()
fig.savefig("colour-pca-scores.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved colour-pca-scores.png")
