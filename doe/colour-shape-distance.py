"""Objective 4: which chromogen's colour-development shape is closest to the reference.

Euclidean distance between each compound's unit-peak-normalised mean curve and compound A's.
Amplitude is divided out first, so this compares shape (development rate and late drift), not
depth of colour. Regenerates ``colour-shape-distance.png``.
"""

import matplotlib.pyplot as plt

from colour_case_study import build_design, shape_distance_to_reference, simulate_curves

design = build_design("i_optimal", budget=60)
curves = simulate_curves(design)
dist = shape_distance_to_reference(design, curves)
alternatives = dist.drop("A")  # A is the reference, distance 0 by construction

colours = ["#2e8b57" if c == alternatives.index[0] else "#1f5fa8" for c in alternatives.index]

fig, ax = plt.subplots(figsize=(7.0, 4.0))
bars = ax.barh(list(alternatives.index)[::-1], list(alternatives.to_numpy())[::-1],
               color=colours[::-1])
for rect, val in zip(bars, list(alternatives.to_numpy())[::-1]):
    ax.text(rect.get_width() + 0.005, rect.get_y() + rect.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8)

ax.set_xlabel("Shape distance to reference A  (smaller = more reference-like)")
ax.set_ylabel("Candidate chromogen")
ax.set_xlim(0, max(alternatives) * 1.18)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig("colour-shape-distance.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved colour-shape-distance.png")
