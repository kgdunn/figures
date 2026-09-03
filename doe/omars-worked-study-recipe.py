"""The current recipe and its replicate batches, for the worked OMARS study.

Left: the temperature schedule of one batch, warm growth at 36.8 degC, a 1.5-day ramp starting on
the shift day, then the production hold. The current recipe is drawn in full; the four corners
of the hold-temperature by shift-day range are drawn lightly, to show the region the study
covers. Right: titer against day for twenty replicate batches at the current recipe, each with
its own disturbance draw, over the noise-free curve.

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
ax_t.plot([], [], color="0.78", lw=1.2, label="corners of the study region")
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

# Right: twenty replicate batches at the current recipe, as the chapter runs them.
for s in range(20):
    states = simulate(CONFIG, **CURRENT, random_state=s).states
    ax_y.plot(states.index, states["titer"], color=BLUE, lw=1.0, alpha=0.35, zorder=2)
quiet = simulate(QUIET, **CURRENT, random_state=0).states
ax_y.plot(quiet.index, quiet["titer"], color=VERMILION, lw=2.2, zorder=4, label="no disturbance")
ax_y.plot([], [], color=BLUE, lw=1.0, alpha=0.6, label="20 replicate batches")
ax_y.axvspan(CURRENT["shift_day"], CURRENT["shift_day"] + RAMP_DAYS, color="0.93", lw=0, zorder=1)
ax_y.text(CURRENT["shift_day"] + RAMP_DAYS / 2, 8.9, "ramp", ha="center", va="top", fontsize=9.5,
          color=GREY)
ax_y.set_xlim(0, CONFIG.batch_days)
ax_y.set_ylim(0, 9.2)
ax_y.set_xlabel("Day of the batch", fontsize=11.5)
ax_y.set_ylabel("Titer, g/L", fontsize=11.5)
ax_y.legend(loc="lower right", fontsize=9.5, frameon=True, facecolor="white", edgecolor="0.85", framealpha=1.0)

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
