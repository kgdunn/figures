"""Joint FDS (fraction-of-design-space) figure, scaled and unscaled.

Two panels for the four response-surface-capable designs over the cube [-1, 1]^5:

- left: scaled prediction variance (SPV = N * x'(X'X)^-1 x), the per-run geometric view;
- right: unscaled prediction variance (x'(X'X)^-1 x, in sigma^2 units).

The contrast is the lesson. Scaling by the run count flatters the small DSD: on the left
it sits low among the curves, but on the right, in real sigma^2 units, it is clearly the
worst predictor while the larger Box-Behnken is the best. Scaled prediction variance and
G-efficiency are criticised for exactly this reason (Goos, 2009; Piepel, 2009; Goos and
Nunez Ares, 2025): the N-scaling erases the benefit of running more experiments.

Reproducible; run from this directory: writes ``fds-plot-six-designs.png``.
"""

import matplotlib.pyplot as plt

from omnibus_designs import LABELS, RSM_DESIGNS, build_designs, evaluate

designs = build_designs()
results = {name: evaluate(designs[name]) for name in RSM_DESIGNS}

styles = {
    "ccd": dict(color="#1f5fa8", lw=2.0),
    "bbd": dict(color="#2e8b57", lw=2.0, ls="-."),
    "dsd": dict(color="#c0392b", lw=2.0, ls="--"),
    "omars": dict(color="#8e44ad", lw=2.0, ls=":"),
}
fig, (ax_s, ax_u) = plt.subplots(1, 2, figsize=(10.2, 4.6), sharex=True)
for name in RSM_DESIGNS:
    curve = results[name]["curve"]
    n = results[name]["N"]
    label = f"{LABELS[name].split('(')[0].strip()} [{n} runs]"
    ax_s.plot(curve["fraction"], curve["scaled_prediction_variance"], label=label, **styles[name])
    ax_u.plot(curve["fraction"], curve["prediction_variance"], label=label, **styles[name])

ax_s.set_title("Scaled (SPV = N x'(X'X)$^{-1}$x): per-run view")
ax_s.set_ylabel("Scaled prediction variance, SPV")
ax_s.set_ylim(0, 18)
ax_u.set_title("Unscaled (x'(X'X)$^{-1}$x): in $\\sigma^2$ units")
ax_u.set_ylabel("Prediction variance / $\\sigma^2$")
ax_u.set_ylim(0, 1.1)
for ax in (ax_s, ax_u):
    ax.set_xlabel("Fraction of design space (variance at or below curve)")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
ax_s.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig("fds-plot-six-designs.png", dpi=300, facecolor="w", edgecolor="w",
            orientation="portrait", format=None, transparent=True)
print("saved fds-plot-six-designs.png")
