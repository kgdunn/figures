"""Titer of the thirty batches in run order, by cassette, for the worked OMARS study.

The second cassette drew on a new lot of feed medium. The design runs are scattered by their
factor settings, so the shift between cassettes is not obvious from them; the centre runs, two
in each cassette, are the only batches at identical settings, and their means are drawn. The
chapter then estimates the same shift from all thirty batches.

Every number comes from omars_worked_study_common.py, which reproduces the chapter's study and
checks it against the values the chapter prints.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt
import numpy as np

from omars_worked_study_common import BLUE, GREY, ORANGE, SPINE, study

S = study()
plan, is_cp, reps = S["plan"], S["is_cp"], S["reps"]
runs = plan.index.to_numpy()
n_first = int((plan["cassette"] == 1).sum())
colour = {1: BLUE, 2: ORANGE}

fig, ax = plt.subplots(figsize=(8.6, 3.8))

ax.axhline(reps.mean(), color="0.82", lw=1.2, zorder=1)
ax.text(len(runs) + 0.5, reps.mean() - 0.08, f"replicates of the current recipe, {reps.mean():.3f} g/L",
        ha="right", va="top", fontsize=9, color=GREY)
ax.axvline(n_first + 0.5, color="0.6", lw=1.0, ls=(0, (4, 3)), zorder=1)

for c in (1, 2):
    sel = (plan["cassette"] == c) & ~is_cp
    ax.vlines(runs[sel], 0, plan.loc[sel, "titer"], color=colour[c], lw=1.0, alpha=0.35, zorder=2)
    ax.plot(runs[sel], plan.loc[sel, "titer"], "o", color=colour[c], ms=6, zorder=3,
            label=f"cassette {c}, design runs")
    cp = (plan["cassette"] == c) & is_cp
    ax.vlines(runs[cp], 0, plan.loc[cp, "titer"], color=colour[c], lw=1.0, alpha=0.35, zorder=2)
    ax.plot(runs[cp], plan.loc[cp, "titer"], "*", color=colour[c], ms=15, mec="0.2", mew=0.8,
            zorder=4, label=f"cassette {c}, centre runs")
    lo, hi = (0.5, n_first + 0.5) if c == 1 else (n_first + 0.5, len(runs) + 0.5)
    mean = plan.loc[cp, "titer"].mean()
    ax.hlines(mean, lo, hi, color=colour[c], lw=1.6, ls=(0, (5, 3)), zorder=2)
    if c == 1:
        ax.text(hi + 0.4, mean + 0.08, f"centre-run mean {mean:.3f} g/L", ha="left", va="bottom",
                fontsize=9.5, color=colour[c])
    else:
        ax.text(hi + 0.4, mean, f"centre-run mean\n{mean:.3f} g/L", ha="left", va="center",
                fontsize=9.5, color=colour[c])

ax.set_xlim(0.2, len(runs) + 0.8)
ax.set_ylim(3.5, 9.6)
ax.set_xticks(runs[::1])
ax.set_xticklabels([str(r) for r in runs], fontsize=8.5)
ax.set_xlabel("Run order", fontsize=11.5)
ax.set_ylabel("Titer at harvest, g/L", fontsize=11.5)
ax.text(n_first / 2 + 0.5, 9.5, "cassette 1", ha="center", va="top", fontsize=10, color="0.35")
ax.text(n_first + (len(runs) - n_first) / 2 + 0.5, 9.5, "cassette 2, new feed lot", ha="center",
        va="top", fontsize=10, color="0.35")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9, frameon=False, ncols=4,
          columnspacing=1.2, handletextpad=0.5)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(SPINE)
ax.tick_params(colors="0.25", labelsize=10)

fig.tight_layout()
fig.savefig("omars-worked-study-titer.png", dpi=300, facecolor="w", edgecolor="w",
            format=None, transparent=True)
print("saved figure")
