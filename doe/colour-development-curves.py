"""Mean colour-development curve per chromogen.

Absorbance at the complex maximum-absorbance wavelength, averaged over each compound's runs,
at ten time points from mixing to plateau. Compounds differ both in amplitude and in the
late-time drift of the curve shape. Regenerates ``colour-development-curves.png``.
"""

# check-scripts: requires pyoptex -- the I-optimal colour design comes from pyoptex
import matplotlib.pyplot as plt

from colour_case_study import COMPOUND_LEVELS, TIME_POINTS, build_design, mean_curves, simulate_curves

design = build_design("i_optimal", budget=60)
curves = simulate_curves(design)
m = mean_curves(design, curves)

palette = ["#1f5fa8", "#c0392b", "#2e8b57", "#8e44ad", "#d68910", "#17a2b8"]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for compound, colour in zip(COMPOUND_LEVELS, palette):
    label = "A (reference)" if compound == "A" else compound
    lw = 2.4 if compound == "A" else 1.5
    # D, E and F dashed (keeping their colours) so the six curves separate more clearly.
    style = "--" if compound in ("D", "E", "F") else "-"
    ax.plot(TIME_POINTS, m.loc[compound].to_numpy(), color=colour, lw=lw, ls=style, marker="o",
            ms=4, label=label)

ax.set_xlabel("Time point (mixing to plateau)")
ax.set_ylabel("Mean absorbance (colour intensity)")
ax.set_xticks(TIME_POINTS)
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=9, title="chromogen", ncol=2)
fig.tight_layout()
fig.savefig("colour-development-curves.png", dpi=300, facecolor="w", edgecolor="w",
            transparent=True)
print("saved colour-development-curves.png")
