"""Prediction variance of the three-run quadratic design, and the cost of extrapolation.

Illustrates Var(yhat(x))/sigma^2 = 1 - 1.5 x^2 + 1.5 x^4 for the saturated single-factor
quadratic design run at x = -1, 0, +1. The curve equals 1 at the three design points,
dips to 0.625 between them, and climbs steeply once x leaves [-1, 1]. Reproducible; run
from this directory: it writes prediction-variance-extrapolation.png alongside the script.
"""
import numpy as np
import matplotlib.pyplot as plt


def pred_var(x):
    return 1.0 - 1.5 * x ** 2 + 1.5 * x ** 4


x = np.linspace(-1.6, 1.6, 801)
fig, ax = plt.subplots(figsize=(7.2, 4.6))

# Shade the extrapolation region (|x| > 1).
ax.axvspan(1.0, 1.6, color="0.85", alpha=0.6, lw=0)
ax.axvspan(-1.6, -1.0, color="0.85", alpha=0.6, lw=0)
ax.text(1.3, 5.6, "extrapolation", color="0.4", fontsize=9, ha="center")
ax.text(-1.3, 5.6, "extrapolation", color="0.4", fontsize=9, ha="center")

ax.plot(x, pred_var(x), color="#1f5fa8", lw=2.0)

# Design points: prediction variance equals sigma^2 there.
xd = np.array([-1.0, 0.0, 1.0])
ax.plot(xd, pred_var(xd), "o", color="#1f5fa8", ms=8, zorder=5,
        label="design points (x = -1, 0, +1)")
ax.axhline(1.0, color="0.6", lw=0.8, ls=":")

# Interior minima.
xm = np.array([-np.sqrt(0.5), np.sqrt(0.5)])
ax.plot(xm, pred_var(xm), "s", color="#c0392b", ms=7, zorder=5,
        label=r"minima at $x=\pm0.707$ (0.625)")

# Boundary of the design region.
for xb in (-1.0, 1.0):
    ax.axvline(xb, color="0.6", lw=0.8, ls="--")

# Annotate the steep climb just outside the region.
ax.annotate(r"5.2 at $x=1.5$", xy=(1.5, pred_var(1.5)), xytext=(0.55, 5.2),
            fontsize=9, color="0.25",
            arrowprops=dict(arrowstyle="->", color="0.5", lw=1.0))

ax.set_xlabel("Coded factor level, $x$")
ax.set_ylabel(r"Prediction variance, $\mathrm{Var}(\widehat{y})\,/\,\sigma^2$")
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(0, 7)
ax.legend(frameon=False, loc="upper center")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("prediction-variance-extrapolation.png", dpi=300, facecolor="w", edgecolor="w",
            orientation="portrait", format=None, transparent=True)
print("saved figure")
