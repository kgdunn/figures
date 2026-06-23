"""The screening-to-response-surface design spectrum.

Illustrates section 8 of the "Optimal designs and OMARS designs" chapter. A single axis
places the three design families in order. Moving from left to right spends more runs and
reduces the aliasing among the second-order effects. Reproducible; run from this directory
to write the PNG alongside it.
"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9.2, 3.4))

# The main axis.
ax.annotate("", xy=(9.6, 0), xytext=(0.4, 0),
            arrowprops=dict(arrowstyle="-", color="0.4", lw=1.4))

families = [
    (1.5, "Definitive screening\ndesign", "#1f5fa8"),
    (5.0, "Mid-sized OMARS\ndesigns", "#6a51a3"),
    (8.5, "Central composite /\nBox-Behnken", "#c0392b"),
]
for x, label, colour in families:
    ax.plot(x, 0, marker="o", ms=12, color=colour, zorder=5)
    ax.text(x, 0.32, label, ha="center", va="bottom", fontsize=13, color=colour)

# Two labelled arrows: number of runs increases to the right, aliasing increases to the left.
ax.annotate("", xy=(9.2, -0.7), xytext=(0.8, -0.7),
            arrowprops=dict(arrowstyle="->", color="0.3", lw=1.6))
ax.text(5.0, -0.55, "number of runs increases", ha="center", va="bottom", fontsize=12)

ax.annotate("", xy=(0.8, -1.25), xytext=(9.2, -1.25),
            arrowprops=dict(arrowstyle="->", color="0.3", lw=1.6))
ax.text(5.0, -1.10, "aliasing among second-order effects increases",
        ha="center", va="bottom", fontsize=12)

ax.set_xlim(0, 10)
ax.set_ylim(-1.6, 1.1)
ax.axis("off")
fig.tight_layout()
fig.savefig("design-spectrum.png", dpi=300, facecolor="w", edgecolor="w",
            orientation="portrait", format=None, transparent=True)
print("saved figure")
