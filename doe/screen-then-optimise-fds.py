"""Fraction-of-design-space curves of the D-, I- and G-optimal designs of the screen-then-optimise study.

The three 16-run designs are built for the quadratic model in temperature, reaction time and
catalyst loading, inside the heat limit 0.04 T + C <= 9.5, and judged over that same constrained
region. Each curve is the prediction variance, in units of the noise variance, sorted over points
sampled in the region: its height at a fraction f is the value that a fraction f of the region
is predicted at or below. All three designs have 16 runs, so the unscaled variance compares them
directly.

Every number comes from screen_then_optimise_common.py, which reproduces the chapter's study and
checks it against the values the chapter prints.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt

from screen_then_optimise_common import BLUE, GREEN, SPINE, VERMILION, study

S = study()
STYLE = {
    "d_optimal": ("D-optimal", BLUE, "-"),
    "i_optimal": ("I-optimal", VERMILION, "-"),
    "g_optimal": ("G-optimal", GREEN, "--"),
}

fig, ax = plt.subplots(figsize=(7.0, 4.3))
for criterion, (label, colour, ls) in STYLE.items():
    curve = S["curves"][criterion]
    ax.plot(curve["fraction"], curve["prediction_variance"], color=colour, ls=ls, lw=2.0, label=label)

ax.set_xlabel("Fraction of the region inside the heat limit", fontsize=11)
ax.set_ylabel("Prediction variance / noise variance", fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.2)
ax.legend(frameon=False, loc="upper left", fontsize=10)
ax.grid(alpha=0.25)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(SPINE)
ax.tick_params(colors="0.25", labelsize=10)
fig.tight_layout()

for criterion, (label, _, _) in STYLE.items():
    curve = S["curves"][criterion]
    print(f"{label}: median {curve['prediction_variance'][50]:.3f}, maximum {max(curve['prediction_variance']):.3f}")
fig.savefig("screen-then-optimise-fds.png", dpi=300, facecolor="w", edgecolor="w",
            format=None, transparent=True, bbox_inches="tight")
print("saved figure")
