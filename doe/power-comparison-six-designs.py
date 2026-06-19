"""Power comparison for the omnibus design-quality subchapter.

Power to detect a main effect and a pure quadratic effect of one noise standard deviation
(delta = sigma) at alpha = 0.05, for the four response-surface-capable designs. Power
rewards the larger run-richer designs, which the scaled prediction variance hides; this is
the practically meaningful headline metric (Goos and Nunez Ares, 2025, their Figure 3).

Reproducible; run from this directory: writes ``power-comparison-six-designs.png``.
"""

import matplotlib.pyplot as plt
import numpy as np

from omnibus_designs import LABELS, RSM_DESIGNS, build_designs, evaluate

designs = build_designs()
results = {name: evaluate(designs[name]) for name in RSM_DESIGNS}

labels = [f"{LABELS[n].split('(')[0].strip()}\n[{results[n]['N']} runs]" for n in RSM_DESIGNS]
power_main = [results[n]["power_main"] for n in RSM_DESIGNS]
power_quad = [results[n]["power_quad"] for n in RSM_DESIGNS]

x = np.arange(len(RSM_DESIGNS))
width = 0.38
fig, ax = plt.subplots(figsize=(7.2, 4.6))
b1 = ax.bar(x - width / 2, power_main, width, label="main effect", color="#1f5fa8")
b2 = ax.bar(x + width / 2, power_quad, width, label="quadratic effect", color="#c0392b")
for bars in (b1, b2):
    for rect in bars:
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.015,
                f"{rect.get_height():.2f}", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Power to detect the effect (delta = sigma, alpha = 0.05)")
ax.set_ylim(0, 1.08)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.axhline(0.8, color="0.5", lw=0.8, ls=":")
ax.text(len(RSM_DESIGNS) - 0.5, 0.81, "0.80", color="0.4", fontsize=8, va="bottom", ha="right")
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("power-comparison-six-designs.png", dpi=300, facecolor="w", edgecolor="w",
            orientation="portrait", format=None, transparent=True)
print("saved power-comparison-six-designs.png")
