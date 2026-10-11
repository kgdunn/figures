"""The face-centred CCD and the I-optimal design of the screen-then-optimise study, against the heat limit.

Both panels are the temperature and catalyst-loading plane, with the reaction time shown by the
marker shape. The shaded corner is beyond the pilot reactor's heat limit, 0.04 T + C <= 9.5. The
face-centred central composite design puts two runs in that corner (vermilion); the I-optimal
design, built inside the limit, puts three runs on it instead. Runs at the same temperature and
loading are drawn side by side, the shortest reaction time on the left, and runs repeated at
identical settings are counted above their marker.

Every number comes from screen_then_optimise_common.py, which reproduces the chapter's study and
checks it against the values the chapter prints.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from screen_then_optimise_common import BEYOND_C, BEYOND_T, BLUE, HEAT_LIMIT, SPINE, VERMILION, heat, study

S = study()
SHAPES = {10: "v", 25: "o", 40: "^"}
DODGE = {10: -1.5, 25: 0.0, 40: 1.5}   # degC, so runs sharing a temperature and loading stay visible
SHADE = "#E4E4E4"

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharey=True)
for ax, (title, design) in zip(axes, [("Face-centred CCD, 17 runs", S["ccd"]),
                                      ("I-optimal design, 16 runs", S["stage2"])]):
    runs = design.design_actual
    ax.fill(BEYOND_T, BEYOND_C, color=SHADE, lw=0, zorder=0)
    ax.plot(BEYOND_T[:2], BEYOND_C[:2], color="0.45", lw=1.0, zorder=1)
    for minutes, marker in SHAPES.items():
        at = runs[np.isclose(runs["t"], minutes)]
        edge = np.where(heat(at) > HEAT_LIMIT, VERMILION, BLUE)
        ax.scatter(at["T"] + DODGE[minutes], at["C"], marker=marker, s=70, facecolor="white", edgecolor=edge,
                   lw=1.5, zorder=3, clip_on=False)
    # Count runs repeated at identical settings, above their marker.
    repeats = runs.groupby(["T", "t", "C"]).size()
    for (T, t, C), n in repeats[repeats > 1].items():
        ax.annotate(f"×{n}", (T + DODGE[int(round(t))], C), xytext=(0, 8), textcoords="offset points",
                    fontsize=8.5, color="0.25", ha="center")
    ax.set_title(title, fontsize=11, color="0.2")
    ax.set_xlabel("Temperature, °C", fontsize=11)
    ax.set_xlim(145, 205)
    ax.set_ylim(0.38, 2.12)
    ax.set_xticks([150, 162.5, 175, 187.5, 200])
    ax.set_xticklabels(["150", "162.5", "175", "187.5", "200"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors="0.25", labelsize=10)
axes[0].set_ylabel("Catalyst loading, mol%", fontsize=11)

handles = [Patch(facecolor=SHADE, edgecolor="0.45", lw=0.8, label="beyond the heat limit")]
handles += [Line2D([], [], ls="none", marker=m, ms=8, mfc="white", mec=BLUE, mew=1.4, label=f"t = {t} min")
            for t, m in SHAPES.items()]
handles += [Line2D([], [], ls="none", marker="o", ms=8, mfc="white", mec=VERMILION, mew=1.5,
                   label="run beyond the limit")]
fig.legend(handles=handles, loc="lower center", ncols=5, fontsize=9.5, frameon=False,
           bbox_to_anchor=(0.5, -0.04), columnspacing=1.4, handletextpad=0.4)
fig.tight_layout(rect=(0, 0.06, 1, 1))

on_limit = int(np.isclose(heat(S["stage2"].design_actual), HEAT_LIMIT).sum())
print(f"CCD runs beyond the limit: {int((heat(S['ccd'].design_actual) > HEAT_LIMIT).sum())}; "
      f"I-optimal runs on the limit: {on_limit}")
fig.savefig("screen-then-optimise-designs.png", dpi=300, facecolor="w", edgecolor="w",
            format=None, transparent=True, bbox_inches="tight")
print("saved figure")
