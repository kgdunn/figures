"""Titer of the thirty batches against the feed rate, by cassette, for the worked OMARS study.

The feed rate is the largest effect the study finds, so it is the axis the harvest titers are
read against, one panel per cassette. The short bars are the mean titer of the design runs at
each feed level; the stars are the centre runs, the only batches at identical settings in both
cassettes. The second cassette drew on a new lot of feed medium, and the whole pattern sits
lower there. The chapter estimates that shift from all thirty batches.

Every number comes from omars_worked_study_common.py, which reproduces the chapter's study and
checks it against the values the chapter prints.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt
import numpy as np

from omars_worked_study_common import BLUE, FACTORS, GREY, ORANGE, SPINE, study

S = study()
plan, is_cp, reps = S["plan"], S["is_cp"], S["reps"]
feed = FACTORS[3]
levels = [feed.low, (feed.low + feed.high) / 2, feed.high]
colour = {1: BLUE, 2: ORANGE}
titles = {1: "Cassette 1", 2: "Cassette 2, new feed-medium lot"}
JITTER = 0.0022          # spreads runs at the same level sideways so none hides another
rng = np.random.default_rng(3)

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
for ax, c in zip(axes, (1, 2)):
    ax.axhline(reps.mean(), color="0.82", lw=1.2, zorder=1)
    here = plan["cassette"] == c
    design, centre = here & ~is_cp, here & is_cp
    x = plan.loc[design, "feed_rate"] + rng.uniform(-JITTER, JITTER, design.sum())
    ax.plot(x, plan.loc[design, "titer"], "o", color=colour[c], ms=6.5, zorder=3, alpha=0.9,
            label="design runs")
    ax.plot(plan.loc[centre, "feed_rate"], plan.loc[centre, "titer"], "*", color=colour[c], ms=15,
            mec="0.2", mew=0.8, zorder=4, label="centre runs")
    for lv in levels:
        at = design & np.isclose(plan["feed_rate"], lv)
        if at.sum():
            ax.hlines(plan.loc[at, "titer"].mean(), lv - 0.004, lv + 0.004, color="0.3", lw=2.0, zorder=2,
                      label="mean of the design runs at that level" if lv == levels[0] else None)
    ax.set_title(titles[c], fontsize=11, color="0.2")
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{lv:.3f}" for lv in levels])
    ax.set_xlim(feed.low - 0.008, feed.high + 0.008)
    ax.set_xlabel("Feed rate, L/day per litre", fontsize=11.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors="0.25", labelsize=10)

axes[0].set_ylim(3.5, 9.6)
axes[0].set_ylabel("Titer at harvest, g/L", fontsize=11.5)
axes[1].text(feed.high + 0.007, reps.mean() + 0.08, f"replicates of the current recipe, {reps.mean():.3f} g/L",
             ha="right", va="bottom", fontsize=9, color=GREY)
axes[1].legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="0.85",
               framealpha=1.0, handletextpad=0.5)

fig.tight_layout(w_pad=1.5)
fig.savefig("omars-worked-study-titer.png", dpi=300, facecolor="w", edgecolor="w",
            format=None, transparent=True)
print("saved figure")
