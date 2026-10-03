"""The thirty-batch campaign plan of the worked OMARS study, in run order.

One column per batch, one row per factor, each cell the level the factor is set to. The
divider is the change of cassette, and with it the change of feed-medium lot. The outlined
columns are the centre runs, two in each cassette and spread through its order rather than
run together. Mirror pairs stay in the same cassette, which is what keeps the main effects
orthogonal to the cassette; it is not visible in this view, and the chapter's code confirms it.

Every number comes from omars_worked_study_common.py, which reproduces the chapter's study and
checks it against the values the chapter prints.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from omars_worked_study_common import BLUE, LABELS, NAMES, SPINE, study

S = study()
plan, C, is_cp = S["plan"], S["C"], S["is_cp"]
runs = plan.index.to_numpy()
n_first = int((plan["cassette"] == 1).sum())

LOW, MID, HIGH = "#F3C46B", "#F4F4F4", BLUE
cmap = ListedColormap([LOW, MID, HIGH])
units = {"hold_temp": "°C", "shift_day": "day", "pH": "", "feed_rate": "L/day"}

fig, ax = plt.subplots(figsize=(8.6, 2.9))
ax.imshow(np.rint(C).T, cmap=cmap, vmin=-1, vmax=1, aspect="auto",
          extent=(0.5, len(runs) + 0.5, len(NAMES) - 0.5, -0.5), interpolation="nearest")

# White grid between cells, then the centre runs outlined and the cassette divider.
for x in np.arange(0.5, len(runs) + 1, 1):
    ax.axvline(x, color="white", lw=1.6)
for y in np.arange(-0.5, len(NAMES), 1):
    ax.axhline(y, color="white", lw=1.6)
for r in runs[is_cp]:
    ax.add_patch(Rectangle((r - 0.5, -0.5), 1, len(NAMES), fill=False, edgecolor="0.2", lw=1.8, zorder=4))
ax.axvline(n_first + 0.5, color="0.2", lw=3.0, zorder=5)
ax.text(n_first / 2 + 0.5, -0.75, f"cassette 1, {n_first} batches", ha="center", va="bottom",
        fontsize=10, color="0.25")
ax.text(n_first + (len(runs) - n_first) / 2 + 0.5, -0.75, f"cassette 2, {len(runs) - n_first} batches",
        ha="center", va="bottom", fontsize=10, color="0.25")

ax.set_xticks(runs)
ax.set_xticklabels([str(r) for r in runs], fontsize=8.5)
ax.set_yticks(range(len(NAMES)))
ax.set_yticklabels([f"{LABELS[n]}" for n in NAMES], fontsize=10.5)
ax.set_xlabel("Run order", fontsize=11.5)
ax.tick_params(length=0, colors="0.25")
for side in ("top", "right", "bottom", "left"):
    ax.spines[side].set_visible(False)

legend = [
    Patch(facecolor=LOW, label="low level"),
    Patch(facecolor=MID, edgecolor="0.8", label="centre"),
    Patch(facecolor=HIGH, label="high level"),
    Patch(facecolor="white", edgecolor="0.2", lw=1.6, label="centre run, all four at centre"),
]
ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=4, fontsize=9.5,
          frameon=False, handlelength=1.3, columnspacing=1.6)

fig.tight_layout()
fig.savefig("omars-worked-study-plan.png", dpi=300, facecolor="w", edgecolor="w",
            format=None, transparent=True)
print("saved figure")
