"""Split-plot structure of the mixed-level case-study design.

Across the 60-run order, the two hard-to-change factors (co-solvent fraction and
temperature) hold their level over long stretches (the whole plots), while the two
easy-to-change factors move almost every run. Regenerates
``colour-split-plot-run-order.png``.
"""

import matplotlib.pyplot as plt
import numpy as np

from colour_case_study import CONT, HARD_TO_CHANGE, build_design

design = build_design("i_optimal", budget=60)
run = design.design.sort_values("RunOrder").reset_index(drop=True)
x = np.arange(len(run))

fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.8), sharex=True)
hard = HARD_TO_CHANGE
easy = [n for n in CONT if n not in hard]
palette = {"co_solvent": "#1f5fa8", "temperature": "#c0392b",
           "concentration": "#2e8b57", "pH": "#8e44ad"}


def n_changes(v):
    return int((v[1:] != v[:-1]).sum())


for ax, group, title in [(axes[0], hard, "Hard-to-change factors: held over whole plots"),
                         (axes[1], easy, "Easy-to-change factors: reset almost every run")]:
    for name in group:
        v = run[name].to_numpy(float)
        ax.step(x, v, where="post", color=palette[name], lw=1.6,
                label=f"{name} ({n_changes(v)} changes)")
    ax.set_ylabel("coded level")
    ax.set_ylim(-1.25, 1.25)
    ax.set_yticks([-1, 0, 1])
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)

axes[1].set_xlabel("run order")
fig.tight_layout()
fig.savefig("colour-split-plot-run-order.png", dpi=300, facecolor="w", edgecolor="w",
            transparent=True)
print("saved colour-split-plot-run-order.png")
