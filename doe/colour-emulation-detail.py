"""Zoomed detail: each candidate's best inverted solution against the reference, t1 onward.

A companion to ``colour-inversion-validation.png`` that magnifies the developed part of the curve.
The first time point (t0), where every curve is near zero and carries little information, is dropped,
and the vertical axis is tightened to the range of the plotted points (it does not include zero), so
the accuracy of the emulation is visible: each candidate's curve is the best amplitude for that
compound's fixed shape (its best attainable inverted solution). B and F sit on the reference, C is
close, and D and E lift away at the tail by their late-time drift, the shape no setting can change.
Regenerates ``colour-emulation-detail.png``.
"""

# check-scripts: requires pyoptex -- the I-optimal colour design comes from pyoptex
import numpy as np
import matplotlib.pyplot as plt

from colour_case_study import GROUND_TRUTH, REF_SHAPE, TAIL_BASIS, TIME_POINTS, ground_truth_curve, shape_floor

goal = ground_truth_curve("A", [0, 0, 0, 0])          # reference: A at centre, noiseless
order = ["B", "F", "C", "D", "E"]                      # by |drift|, closest first
palette = {"A": "#111111", "B": "#c0392b", "C": "#2e8b57", "D": "#8e44ad", "E": "#d68910", "F": "#17a2b8"}

t = TIME_POINTS[1:]                                     # drop t0
emul = {c: shape_floor(c)[1] * np.clip(REF_SHAPE + GROUND_TRUTH[c]["drift"] * TAIL_BASIS, 0.0, None)
        for c in order}                                # best-amplitude (best inverted) curve per candidate

stack = np.concatenate([goal[1:]] + [emul[c][1:] for c in order])
ymin, ymax = float(stack.min()), float(stack.max())
pad = 0.03 * (ymax - ymin)

fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.plot(t, goal[1:], color=palette["A"], lw=3.0, marker="o", ms=5, label="A (reference)", zorder=5)
for c in order:
    rmse, _ = shape_floor(c)
    ax.plot(t, emul[c][1:], color=palette[c], lw=1.7, marker="o", ms=4.5,
            label=f"{c}  (RMSE {rmse:.3f})")

ax.set_ylim(ymin - pad, ymax + pad)                    # tight; excludes zero
ax.set_xticks(t)
ax.set_xlabel("time point (t0 omitted)")
ax.set_ylabel("absorbance (colour intensity)")
ax.set_title("Emulation accuracy, developed part of the curve (t1 onward, zoomed)",
             fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("colour-emulation-detail.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved colour-emulation-detail.png")
print(f"y-range plotted: {ymin:.3f} to {ymax:.3f}")
