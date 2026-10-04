"""The current recipe and its replicate batches, for the worked OMARS study.

Left: the temperature schedule of one batch, warm growth at 36.8 degC, a 1.5-day ramp starting on
the shift day, then the production hold. The current recipe is drawn in full; all four combinations
of the low and high hold temperature and shift day are drawn lightly, to show the region the study
covers. Right: for twenty replicate batches at the current recipe, each with its own disturbance
draw, the titer minus that of the same batch with every disturbance switched off.

Every number comes from omars_worked_study_common.py, which reproduces the chapter's study and
checks it against the values the chapter prints.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt
import numpy as np

from omars_worked_study_common import (
    BLUE, CONFIG, CURRENT, GREY, GROWTH_TEMP, QUIET, RAMP_DAYS, SPINE, VERMILION, simulate,
)

fig, (ax_t, ax_y) = plt.subplots(1, 2, figsize=(8.6, 3.6))

# Left: the setpoint schedule, on a fine day grid so the ramp is a clean line.
days = np.linspace(0, CONFIG.batch_days, 401)


def schedule(hold_temp, shift_day):
    fraction = np.clip((days - shift_day) / RAMP_DAYS, 0.0, 1.0)
    return GROWTH_TEMP - (GROWTH_TEMP - hold_temp) * fraction


for hold, shift in ((28.5, 2.0), (28.5, 3.5), (31.5, 2.0), (31.5, 3.5)):
    ax_t.plot(days, schedule(hold, shift), color="0.78", lw=1.2, zorder=2)
ax_t.plot(days, schedule(CURRENT["hold_temp"], CURRENT["shift_day"]), color=BLUE, lw=2.4, zorder=3,
          label="current recipe")
ax_t.plot([], [], color="0.78", lw=1.2, label="low and high hold\nand shift day")
ax_t.text(0.15, GROWTH_TEMP - 0.35, "growth,\n36.8 °C", fontsize=9.5, color=GREY, ha="left", va="top")
ax_t.text(7.0, CURRENT["hold_temp"] + 0.3, "hold, 30.0 °C", fontsize=9.5, color=GREY, ha="left", va="bottom")
ax_t.annotate("shift starts on day 2.75,\nramp of 1.5 days", xy=(3.55, 33.4), xytext=(5.6, 33.4),
              fontsize=9.5, color=GREY, ha="left", va="center",
              arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.9})
ax_t.set_xlim(0, CONFIG.batch_days)
ax_t.set_ylim(27.5, 38)
ax_t.set_xlabel("Day of the batch", fontsize=11.5)
ax_t.set_ylabel("Temperature setpoint, °C", fontsize=11.5)
ax_t.legend(loc="upper right", fontsize=9.5, frameon=True, facecolor="white", edgecolor="0.85", framealpha=1.0)

# Right: twenty replicate batches at the current recipe, as the chapter runs them, each drawn as
# its departure from the batch with every disturbance switched off.
quiet = simulate(QUIET, **CURRENT, random_state=0).states["titer"]
deviations = []
for s in range(20):
    titer = simulate(CONFIG, **CURRENT, random_state=s).states["titer"]
    deviations.append(titer - quiet)
    ax_y.plot(titer.index, titer - quiet, color=BLUE, lw=1.0, alpha=0.35, zorder=2)
ax_y.axhline(0.0, color=VERMILION, lw=2.2, zorder=4, label="undisturbed")
ax_y.plot([], [], color=BLUE, lw=1.0, alpha=0.6, label="20 replicates")
reach = 1.15 * max(float(abs(d).max()) for d in deviations)
top = 1.45 * reach       # headroom above the curves for the legend
ax_y.axvspan(CURRENT["shift_day"], CURRENT["shift_day"] + RAMP_DAYS, color="0.93", lw=0, zorder=1)
ax_y.text(CURRENT["shift_day"] + RAMP_DAYS / 2, 0.98 * reach, "ramp", ha="center", va="top",
          fontsize=9.5, color=GREY)
ax_y.set_xlim(0, CONFIG.batch_days)
ax_y.set_ylim(-reach, top)
ax_y.set_yticks(np.arange(-0.5, 0.51, 0.25))   # ticks only where the curves are
ax_y.spines["left"].set_bounds(-reach, reach)
ax_y.set_xlabel("Day of the batch", fontsize=11.5)
ax_y.set_ylabel("Titer minus undisturbed titer, g/L", fontsize=11.5)
ax_y.legend(loc="upper right", ncols=2, fontsize=9.5, frameon=True, facecolor="white", edgecolor="0.85", framealpha=1.0)

for ax in (ax_t, ax_y):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors="0.25", labelsize=10)

fig.tight_layout(w_pad=2.5)
fig.savefig("omars-worked-study-recipe.png", dpi=300, facecolor="w", edgecolor="w",
            format=None, transparent=True)
print("saved figure")
