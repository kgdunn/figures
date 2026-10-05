"""Variance components in a blocked experiment: block-to-block vs within-block spread.

Illustrates the "Fixed and random block effects" material in the pid-book Design and Analysis
of Experiments chapter. Each run's response is a grand mean, plus a block effect (the block's
offset from the grand mean), plus a residual (the run's scatter within its block). The spread
of the block means measures sigma_gamma (the random block effect); the scatter within a block
measures sigma_eps (the residual). The within-block correlation is
rho = sigma_gamma^2 / (sigma_gamma^2 + sigma_eps^2). Reproducible; run from this directory to
write the PNG alongside it. Values are hard-coded so the figure is deterministic.
"""
import numpy as np
import matplotlib.pyplot as plt

grand = 50.0
# Five blocks (e.g. days), each with a mean offset from the grand mean, and four runs whose
# responses scatter about that block mean.
block_means = [47.2, 52.4, 48.6, 53.6, 49.2]
runs = [
    [46.4, 47.9, 46.8, 47.7],
    [51.6, 53.1, 52.0, 52.9],
    [47.9, 49.3, 48.1, 49.1],
    [52.9, 54.4, 53.1, 54.0],
    [48.5, 49.9, 48.6, 49.8],
]

fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.axhline(grand, color="0.55", lw=1.1, ls="--", zorder=1)
ax.text(5.62, grand, "grand\nmean", color="0.4", fontsize=8, va="center")

for i, (mean, ys) in enumerate(zip(block_means, runs), start=1):
    xs = i + np.linspace(-0.13, 0.13, len(ys))
    ax.scatter(xs, ys, s=42, color="#1f5fa8", zorder=5, edgecolor="white", lw=0.6)
    ax.plot([i - 0.22, i + 0.22], [mean, mean], color="#c0392b", lw=2.2, zorder=4)

ax.text(1 - 0.22, block_means[0] - 0.05, "block mean", color="#c0392b", fontsize=8,
        ha="left", va="top")

# Bracket 1: within-block scatter (residual, sigma_eps) inside block 3.
xb = 3.32
ax.annotate("", xy=(xb, min(runs[2])), xytext=(xb, max(runs[2])),
            arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2))
ax.text(xb + 0.06, block_means[2], r"within-block" "\n" r"scatter, $\sigma_\varepsilon$",
        fontsize=8, va="center", color="0.25")

# Bracket 2: block-to-block scatter (random block effect, sigma_gamma) across the block means.
xg = 5.34
ax.annotate("", xy=(xg, min(block_means)), xytext=(xg, max(block_means)),
            arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.2))
ax.text(xg + 0.06, np.mean(block_means) + 2.7, r"block-to-block" "\n" r"scatter, $\sigma_\gamma$",
        fontsize=8, va="center", color="#c0392b")

ax.set_xlabel("Block (e.g. day)")
ax.set_ylabel("Response")
ax.set_xticks(range(1, 6))
ax.set_xlim(0.5, 6.35)
ax.set_ylim(44.5, 56)
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig("blocking-variance-components.png", dpi=300, facecolor="w", edgecolor="w",
            transparent=True)
print("saved blocking-variance-components.png")
