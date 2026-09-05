"""Fraction-of-design-space (FDS) curves for the four design variants.

I-optimal versus D-optimal at 48 and 60 runs. The I-optimal criterion, which minimises
the average prediction variance, gives the flatter, lower FDS curve; more runs lower every
curve. Regenerates ``colour-fds-curves.png``.
"""

# check-scripts: requires pyoptex -- the I-optimal colour design comes from pyoptex
import matplotlib.pyplot as plt
import numpy as np

from colour_case_study import build_design, evaluate

# Colour separates the criterion (I = blue, D = red); line weight and dash pattern
# separate the run count (n = 60 heavy solid, n = 48 light dashed), so all four
# curves are distinguishable from the legend line samples alone.
VARIANTS = [("i_optimal", 60, "#1f5fa8", "-", 3.0), ("i_optimal", 48, "#1f5fa8", (0, (5, 2)), 1.4),
            ("d_optimal", 60, "#c0392b", "-", 3.0), ("d_optimal", 48, "#c0392b", (0, (5, 2)), 1.4)]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for crit, budget, colour, style, width in VARIANTS:
    q = evaluate(build_design(crit, budget=budget))["fds"]["quantiles"]
    frac = np.array([float(k) for k in q])
    pv = np.array([q[k] for k in q])
    label = f"{crit.split('_')[0].upper()}-optimal, n={budget}"
    marker = "o" if budget == 60 else "s"
    ax.plot(frac, pv, color=colour, ls=style, lw=width, marker=marker, ms=4, label=label)

ax.set_xlabel("Fraction of design space")
ax.set_ylabel("Scaled prediction variance")
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=10, loc="upper left", handlelength=3.2)
fig.tight_layout()
fig.savefig("colour-fds-curves.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved colour-fds-curves.png")
