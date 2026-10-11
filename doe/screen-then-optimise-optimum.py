"""Why the screen-then-optimise study recommends the settings it does.

Left: the fitted yield model over temperature and catalyst loading, at the recommended reaction
time, drawn only where the heat limit allows; the grey corner is beyond it. The yield rises
towards that corner, so the best allowed settings, the star, lie on the limit. Right: the individual and overall desirabilities along the
reaction time, at the recommended temperature and loading. A longer time raises the yield but
also the impurity; the overall desirability, their geometric mean, peaks where the two balance.

Every number comes from screen_then_optimise_common.py, which reproduces the chapter's study and
checks it against the values the chapter prints.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from screen_then_optimise_common import (
    BEYOND_C, BEYOND_T, BLUE, HEAT_LIMIT, ORANGE, SPINE, VERMILION, desirability, heat, predict, study,
)

S = study()
fits, setting = S["fits"], S["setting"]

T_axis, C_axis = np.linspace(150, 200, 101), np.linspace(0.5, 2.0, 101)
TT, CC = np.meshgrid(T_axis, C_axis)
plane = pd.DataFrame({"T": TT.ravel(), "t": setting["t"], "C": CC.ravel()})
yield_plane = predict(fits, plane)["yield"].to_numpy().reshape(TT.shape)
# Only the allowed region is drawn: beyond the heat limit the model is not used.
allowed_plane = np.where(heat(plane).to_numpy().reshape(TT.shape) > HEAT_LIMIT, np.nan, yield_plane)
times = pd.DataFrame({"T": setting["T"], "t": np.linspace(10, 40, 121), "C": setting["C"]})
along_t = desirability(predict(fits, times))


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors="0.25", labelsize=10)


fig, (left, right) = plt.subplots(1, 2, figsize=(9.4, 4.1), gridspec_kw={"width_ratios": [1.15, 1]})
levels = np.arange(56, 78.1, 2)
filled = left.contourf(T_axis, C_axis, yield_plane, levels=levels, cmap="Blues")   # covered beyond the limit
lines = left.contour(T_axis, C_axis, allowed_plane, levels=levels, colors="white", linewidths=0.6)
left.clabel(lines, levels=levels[::2], fmt="%d", fontsize=7.5, colors="0.25")
left.fill(BEYOND_T, BEYOND_C, color="#E4E4E4", lw=0, zorder=3)
left.plot(BEYOND_T[:2], BEYOND_C[:2], color="0.45", lw=1.0, zorder=4)
left.text(197.0, 1.87, "beyond\nthe limit", fontsize=7.5, color="0.35", ha="center", va="center", zorder=5)
left.plot(setting["T"], setting["C"], "*", ms=17, color=VERMILION, mec="white", mew=1.2, zorder=6, clip_on=False)
cbar = fig.colorbar(filled, ax=left, pad=0.02)
cbar.set_label("Fitted yield, %", fontsize=10, color="0.25")
cbar.ax.tick_params(colors="0.25", labelsize=9)
cbar.outline.set_edgecolor(SPINE)
left.set_title(f"Reaction time {setting['t']:.1f} min", fontsize=11, color="0.2")
left.set_xlabel("Temperature, °C", fontsize=11)
left.set_ylabel("Catalyst loading, mol%", fontsize=11)
tidy(left)

for column, colour, label, lw in [("yield", BLUE, "yield", 1.6), ("impurity", ORANGE, "impurity", 1.6),
                                  ("overall", "0.15", "overall", 2.6)]:
    right.plot(times["t"], along_t[column], color=colour, lw=lw, label=label)
right.axvline(setting["t"], color="0.45", ls=":", lw=1.0)
right.annotate(f"{setting['t']:.1f} min", (setting["t"], 0.04), xytext=(4, 0), textcoords="offset points",
               fontsize=8.5, color="0.3")
right.set_title(f"T = {setting['T']:.1f} °C, C = {setting['C']:.2f} mol%", fontsize=11, color="0.2")
right.set_xlabel("Reaction time, min", fontsize=11)
right.set_ylabel("Desirability", fontsize=11)
right.set_xlim(10, 40)
right.set_ylim(-0.02, 1.04)
right.legend(frameon=False, loc="center right", fontsize=9.5)
tidy(right)
fig.tight_layout()

peak = times["t"].iloc[int(np.argmax(along_t["overall"]))]
print(f"overall desirability along t peaks at {peak:.2f} min; fitted yield range "
      f"{yield_plane.min():.1f} to {yield_plane.max():.1f} %")
fig.savefig("screen-then-optimise-optimum.png", dpi=300, facecolor="w", edgecolor="w",
            format=None, transparent=True, bbox_inches="tight")
print("saved figure")
