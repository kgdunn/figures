"""Which candidate chromogens are reachable depends on how the inversion is posed.

For each candidate, the inversion asks: what settings of the four continuous factors make it develop
colour like the reference goal (chromogen A at the centre point)? Three readings match the goal in
the 3-component *score* space, one per categorical coding; each can return a different reachable set,
because a truncated PLS score is coding-dependent. The fourth reading matches the predicted *ten-point
curve* at full rank, which is coding-invariant (identical to ~1e-13 under any coding), and is
over-determined, so it returns the least-squares closest curve. A cell is filled when the candidate
can be brought onto the goal within the studied factor ranges (all coded settings in [-1, 1]).
Regenerates ``colour-coding-inversion.png``.
"""

import contextlib
import io

import matplotlib.pyplot as plt

from colour_case_study import (
    build_design,
    curve_match_inversion,
    fit_coding,
    goal_projection,
    invert_to_factors,
    simulate_curves,
)

design = build_design("i_optimal", budget=60)
curves = simulate_curves(design)
candidates = ["B", "C", "D", "E", "F"]


def score_reach(coding):
    with contextlib.redirect_stderr(io.StringIO()):
        pls, di = fit_coding(design, curves, coding)
    tbl = invert_to_factors(pls, di, goal_projection(pls, di)["score"])
    return {c: bool(tbl.loc[c, "in_range"]) for c in candidates}


curve_tbl = curve_match_inversion(design, curves, "sum")
readings = [
    ("Score match\nsum coding", score_reach("sum")),
    ("Score match\ntreatment coding", score_reach("treatment")),
    ("Score match\ncell-means coding", score_reach("cell_means")),
    ("Curve match\n(coding-invariant)", {c: bool(curve_tbl.loc[c, "in_range"]) for c in candidates}),
]

reach_fill, reach_edge = "#2e8b57", "#1e5e3a"
miss_fill, miss_edge = "#f0ede8", "#b8b0a4"

fig, ax = plt.subplots(figsize=(8.4, 4.8))
for j, (_, reach) in enumerate(readings):
    for i, c in enumerate(candidates):
        y = len(candidates) - 1 - i
        ok = reach[c]
        ax.add_patch(plt.Rectangle((j - 0.44, y - 0.44), 0.88, 0.88,
                                   facecolor=reach_fill if ok else miss_fill,
                                   edgecolor=reach_edge if ok else miss_edge, linewidth=1.1))
        ax.text(j, y, "reachable" if ok else "out of\nrange", ha="center", va="center",
                fontsize=8.5, color="w" if ok else "0.45",
                fontweight="bold" if ok else "normal")

ax.set_xticks(range(len(readings)))
ax.set_xticklabels([r[0] for r in readings], fontsize=9)
ax.set_yticks([len(candidates) - 1 - i for i in range(len(candidates))])
ax.set_yticklabels([f"chromogen {c}" for c in candidates], fontsize=9)
ax.set_xlim(-0.6, len(readings) - 0.4)
ax.set_ylim(-0.6, len(candidates) - 0.4)
ax.set_title("Reachable candidates depend on how the inversion is posed", fontsize=11, loc="left")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
fig.savefig("colour-coding-inversion.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved colour-coding-inversion.png")
for label, reach in readings:
    print(label.replace(chr(10), " "), "->", [c for c in candidates if reach[c]])
