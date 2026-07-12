"""Validate the inversion against the withheld ground truth: how close can each candidate get to A?

Left: the reference colour-development curve (chromogen A at the centre point) with each candidate's
closest attainable emulation, the best amplitude for that compound's fixed shape. Because the
continuous factors move only amplitude and each compound carries a fixed late-time drift, the
candidates separate at the tail: B and F track the reference, C is close, D and E lift away. Right:
the emulation error (RMSE to the reference) per candidate against the measurement-noise scale; the
open marker is each compound's best attainable match (best at any amplitude), the filled marker is what the
coding-invariant curve-match inversion actually reaches when its real-unit settings are pushed back
through the simulator. Ordered by late-time drift. Regenerates ``colour-inversion-validation.png``.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from colour_case_study import (
    CONT, GROUND_TRUTH, REF_SHAPE, TAIL_BASIS, TIME_POINTS,
    build_design, coded_to_real, curve_match_inversion, ground_truth_curve, shape_floor, simulate_curves,
)

NOISE = 0.03
design = build_design("i_optimal", 60)
curves = simulate_curves(design)
goal = ground_truth_curve("A", [0, 0, 0, 0])          # reference: A at centre, noiseless

order = ["B", "F", "C", "D", "E"]                       # by |drift|, closest first
palette = {"A": "#111111", "B": "#c0392b", "C": "#2e8b57", "D": "#8e44ad", "E": "#d68910", "F": "#17a2b8"}
cm = curve_match_inversion(design, curves, "treatment")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.7), gridspec_kw={"width_ratios": [1.35, 1]})

# ---- Left: reference + best-emulation curves ----
axL.plot(TIME_POINTS, goal, color=palette["A"], lw=3.0, label="A (reference)", zorder=5)
for c in order:
    floor_rmse, amp = shape_floor(c)
    emul = amp * np.clip(REF_SHAPE + GROUND_TRUTH[c]["drift"] * TAIL_BASIS, 0.0, None)
    axL.plot(TIME_POINTS, emul, color=palette[c], lw=1.6, marker="o", ms=3.5,
             label=f"{c}  (RMSE {floor_rmse:.3f})")
axL.set_xlabel("time point")
axL.set_ylabel("absorbance (colour intensity)")
axL.set_title("Closest emulation of the reference by each candidate", fontsize=10, loc="left")
axL.legend(frameon=False, fontsize=8, loc="lower right")
axL.grid(alpha=0.2)

# ---- Right: emulation error vs the noise scale ----
y = {c: len(order) - 1 - i for i, c in enumerate(order)}
axR.axvspan(0, NOISE, color="#e3efe7", zorder=0)
axR.axvline(NOISE, color="#2e8b57", lw=1.0, ls="--", zorder=1)
axR.axvline(2 * NOISE, color="0.6", lw=1.0, ls=":", zorder=1)
axR.text(NOISE, len(order) - 0.45, " 1x noise", color="#2e8b57", fontsize=8, va="top", ha="left")
axR.text(2 * NOISE, len(order) - 0.45, " 2x noise", color="0.5", fontsize=8, va="top", ha="left")
for c in order:
    floor_rmse, _ = shape_floor(c)                     # developed curve (t1 onward) by default
    cd = [float(cm.loc[c, f"{n}_coded"]) for n in CONT]
    got = float(np.sqrt(np.mean((ground_truth_curve(c, cd)[1:] - goal[1:]) ** 2)))   # t1 onward
    axR.plot([floor_rmse], [y[c]], marker="o", ms=9, mfc="w", mec=palette[c], mew=1.6, zorder=4)
    axR.plot([got], [y[c]], marker="o", ms=8, color=palette[c], zorder=4)
    axR.plot([floor_rmse, got], [y[c], y[c]], color=palette[c], lw=1.0, alpha=0.5, zorder=3)
axR.set_yticks(list(y.values()))
axR.set_yticklabels([f"{c} (drift {GROUND_TRUTH[c]['drift']:+.2f})" for c in order])
axR.set_ylim(-0.5, len(order) - 0.5)
axR.set_xlim(0, 0.16)
axR.set_xlabel("RMSE of emulated curve vs the reference")
axR.set_title("Emulation error against the noise scale", fontsize=10, loc="left")
axR.grid(axis="x", alpha=0.2)
legend = [Line2D([], [], marker="o", ls="", mfc="w", mec="0.35", mew=1.6, label="best attainable match (any amplitude)"),
          Line2D([], [], marker="o", ls="", color="0.35", label="curve-match inversion (validated)")]
axR.legend(handles=legend, frameon=False, fontsize=7.5, loc="lower right")

fig.suptitle("F is not unique: B best, then F, then C; ordered by fixed curve shape, not by amplitude",
             fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("colour-inversion-validation.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved colour-inversion-validation.png")
