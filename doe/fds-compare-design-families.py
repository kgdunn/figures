"""FDS (fraction-of-design-space) comparison of five response-surface designs at k = 3.

The five families from the central composite design post - circumscribed (rotatable),
face-centred and inscribed CCDs, Box-Behnken and Doehlert - are compared on the full
second-order model over the cube [-1, 1]^3. The y-axis is the scaled prediction variance
SPV = N * x'(X'X)^-1 x; the x-axis is the fraction of the cube whose SPV is at or below the
curve. A low, flat curve is a design that predicts precisely and evenly everywhere.

Two panels share the story:

- left, linear: the central bulk of the cube. The face-centred CCD, Box-Behnken and
  circumscribed CCD sit low and flat.
- right, log: the same curves out to the corners. The inscribed CCD and the Doehlert design
  never leave the unit sphere, so at the cube corners they extrapolate and their SPV runs an
  order of magnitude higher. That is the price of "staying inside +/-1" that the post's
  inscribed and shell designs pay.

Reproducible; run from this directory: writes ``fds-compare-design-families.png``.
"""

import matplotlib.pyplot as plt

from ccd_variants_designs import FAMILY_ORDER, LABELS, STYLES, build_designs, evaluate

designs = build_designs()
results = {name: evaluate(designs[name]) for name in FAMILY_ORDER}

fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(11.0, 4.7), sharex=True)
for name in FAMILY_ORDER:
    curve = results[name]["curve"]
    n = results[name]["N"]
    max_spv = results[name]["max_spv"]
    label = f"{LABELS[name]} [n={n}, max SPV {max_spv:.0f}]"
    for ax in (ax_lin, ax_log):
        ax.plot(curve["fraction"], curve["scaled_prediction_variance"], label=label, **STYLES[name])

ax_lin.set_title("Linear axis: the central bulk of the cube")
ax_lin.set_ylabel("Scaled prediction variance, SPV = N x'(X'X)$^{-1}$x")
ax_lin.set_ylim(0, 25)

ax_log.set_title("Log axis: the same curves out to the corners")
ax_log.set_ylabel("Scaled prediction variance (log scale)")
ax_log.set_yscale("log")

for ax in (ax_lin, ax_log):
    ax.set_xlabel("Fraction of design space (SPV at or below the curve)")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25, which="both")
ax_lin.legend(frameon=False, loc="upper left", fontsize=8.2)

fig.suptitle("Fraction-of-design-space comparison over the cube [-1, 1]$^3$, second-order model (k = 3)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("fds-compare-design-families.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved fds-compare-design-families.png")
