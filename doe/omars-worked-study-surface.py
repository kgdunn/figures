"""The fitted model against the true response, for the worked OMARS study.

Both panels are titer over hold temperature and shift day, with pH at 7.1 and the feed rate
at its high level, 0.070 L/day, where the study's recommendation sits. Left: the four-term
model the staged analysis selected, refitted and back-transformed from log titer. Right: the
simulator with every disturbance switched off, evaluated on a grid. The three marks are the
current recipe, the recipe the study recommends, and the true best in the region. The same
contour levels and colours are used on both panels, so they can be read against each other.

Every number comes from omars_worked_study_common.py, which reproduces the chapter's study and
checks it against the values the chapter prints.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt
import numpy as np

from omars_worked_study_common import (
    FACTORS, GREY, SPINE, VERMILION, decode, encode, model_matrix, study, truth,
)

S = study()
terms, bs, x_rec, best = S["terms"], S["bs"], S["x_rec"], S["best"]

GRID = 31
t_axis = np.linspace(-1, 1, GRID)          # hold temperature, coded
s_axis = np.linspace(-1, 1, GRID)          # shift day, coded
T, Sh = np.meshgrid(t_axis, s_axis)
fitted = np.empty_like(T)
true = np.empty_like(T)
for i in range(GRID):
    for j in range(GRID):
        x = np.array([T[i, j], Sh[i, j], 0.0, 1.0])
        fitted[i, j] = np.exp(float((model_matrix(terms, x) @ bs).ravel()[0]))
        true[i, j] = truth(x)

# Axes in real units.
f_t, f_s = FACTORS[0], FACTORS[1]
t_real = f_t.low + (t_axis + 1) / 2 * (f_t.high - f_t.low)
s_real = f_s.low + (s_axis + 1) / 2 * (f_s.high - f_s.low)
levels = np.arange(5.0, 10.01, 0.5)

marks = {
    "current recipe": (encode({"hold_temp": 30.0, "shift_day": 2.75, "pH": 7.1, "feed_rate": 0.055}), "o"),
    "recommended by the study": (x_rec, "s"),
    "true best in the region": (best.x, "*"),
}

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1), sharey=True)
for ax, Z, title in zip(axes, (fitted, true), ("Fitted four-term model", "True response, no disturbance")):
    cf = ax.contourf(t_real, s_real, Z, levels=levels, cmap="Blues")
    cs = ax.contour(t_real, s_real, Z, levels=levels, colors="white", linewidths=0.7)
    ax.clabel(cs, fmt="%.1f", fontsize=8, colors="0.25")
    for label, (x, marker) in marks.items():
        real = decode(x)
        ax.plot(real["hold_temp"], real["shift_day"], marker, ms=15 if marker == "*" else 9,
                color=VERMILION, mec="white", mew=1.2, zorder=5, clip_on=False, label=label)
    ax.set_title(title, fontsize=11, color="0.2")
    ax.set_xlabel("Hold temperature, °C", fontsize=11)
    ax.set_xlim(f_t.low, f_t.high)
    ax.set_ylim(f_s.low, f_s.high)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors="0.25", labelsize=10)
axes[0].set_ylabel("Shift day", fontsize=11)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.44, -0.13), ncols=3,
           fontsize=9.5, frameon=False, columnspacing=1.8)

cbar = fig.colorbar(cf, ax=axes, shrink=0.9, pad=0.02)
cbar.set_label("Titer, g/L  (pH 7.1, feed 0.070 L/day)", fontsize=10.5, color="0.25")
cbar.ax.tick_params(colors="0.25", labelsize=9.5)
cbar.outline.set_edgecolor(SPINE)

print(f"fitted range {fitted.min():.3f} to {fitted.max():.3f} g/L; true range {true.min():.3f} to {true.max():.3f} g/L")
fig.savefig("omars-worked-study-surface.png", dpi=300, facecolor="w", edgecolor="w",
            format=None, transparent=True, bbox_inches="tight")
print("saved figure")
