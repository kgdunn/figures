"""Why the OMARS trade-off table reports estimability and not a quality score.

Illustrates the "OMARS designs" section of the Design and Analysis of Experiments
chapter. It takes a single column of the trade-off table, three factors, and plots six
candidate cell measures down that column: the five alphabetic optimality criteria and the
largest absolute correlation between two second-order terms.

Panels are arranged so that the related pairs sit in columns:

    A  average variance of an estimated coefficient   I  average prediction variance
    E  smallest eigenvalue of X'X                     G  worst prediction variance
    D  generalised variance                           max |r|  worst entanglement

The left column averages, the middle column takes a worst case, the right column does
neither. The top row is a summary of the eigenvalues of X'X; the bottom row is prediction
variance over the design region.

Every point is the best value attainable at that run count, found by enumerating every
OMARS foldover of that size, so the curves are frontiers rather than the results of a
search. The enumeration reduces the design to a count per sign class of {-1, 0, 1}^k,
which is what makes it exhaustive rather than heuristic: for second-order terms a half-row
and its negation are interchangeable. It takes hours, so its output is carried below as
literal data rather than recomputed here. Twelve of the cells were re-derived from naive
numpy model matrices and agree to 5e-7.

The model scored is the same everywhere, main effects and quadratics, p = 2k + 1 terms, so
that a value means the same thing in every cell. I is exact against the cuboidal region
moments; G is maximised over a seven-level grid per factor.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt
import numpy as np

# Best attainable value at each run count, three factors, keyed by centre-point count.
# Columns: A/p, D, E, I, G. N and the centre count share parity, since N = 2h + c.
BEST = {
    1: {7: (1.000000, 1.811447, 0.227998, 1.066667, 7.000000),
        9: (0.396825, 3.970330, 0.811221, 0.577778, 0.777778),
        11: (0.333333, 4.636775, 0.964242, 0.474444, 0.766667),
        13: (0.273810, 5.498742, 1.173238, 0.383333, 0.705263),
        15: (0.217262, 6.321042, 2.000000, 0.301389, 0.600000),
        17: (0.198649, 7.401203, 2.000000, 0.272823, 0.553488),
        19: (0.174286, 8.322191, 2.437608, 0.244932, 0.386667),
        21: (0.152015, 9.077880, 3.422761, 0.210256, 0.367965),
        23: (0.139604, 10.010817, 3.504283, 0.194148, 0.335526),
        25: (0.126732, 10.868684, 3.631717, 0.177871, 0.331731),
        27: (0.114911, 11.910990, 4.204939, 0.161419, 0.259259),
        29: (0.108179, 12.696601, 4.325162, 0.152030, 0.257310),
        31: (0.101456, 13.578326, 4.492595, 0.142591, 0.247657)},
    2: {8: (0.714286, 2.000000, 0.417424, 0.933333, 5.000000),
        10: (0.321429, 4.310458, 1.416995, 0.450000, 0.770833),
        12: (0.279762, 4.958794, 1.502798, 0.391667, 0.708333),
        14: (0.238095, 5.943977, 1.669697, 0.333333, 0.606250),
        16: (0.196429, 6.713077, 2.143594, 0.275000, 0.596552),
        18: (0.180952, 7.940660, 2.375332, 0.251852, 0.388889),
        20: (0.160714, 8.689800, 2.972244, 0.225000, 0.385057),
        22: (0.145963, 9.567842, 4.000000, 0.200000, 0.364583),
        24: (0.133503, 10.412568, 4.000000, 0.185501, 0.332042),
        26: (0.119898, 11.411133, 4.000000, 0.168651, 0.312500),
        28: (0.111111, 12.309772, 4.730678, 0.155908, 0.258170),
        30: (0.104702, 13.104206, 4.825413, 0.147201, 0.249206)},
    3: {9: (0.619048, 2.119268, 0.575571, 0.888889, 4.333333),
        11: (0.291925, 4.539822, 2.000000, 0.400000, 0.768116),
        13: (0.255411, 5.189597, 2.000000, 0.354040, 0.696970),
        15: (0.217262, 6.298440, 2.000000, 0.301389, 0.600000),
        17: (0.183929, 7.002495, 2.634575, 0.259167, 0.553488),
        19: (0.169855, 8.322191, 2.707761, 0.238701, 0.386667),
        21: (0.152015, 9.048783, 3.493057, 0.210256, 0.367965),
        23: (0.139604, 10.010817, 4.000000, 0.192453, 0.335526),
        25: (0.126732, 10.810243, 4.000000, 0.177871, 0.331731),
        27: (0.114911, 11.910990, 4.241623, 0.161419, 0.259259),
        29: (0.108120, 12.696601, 5.246134, 0.151570, 0.257310),
        31: (0.101456, 13.578326, 5.310658, 0.142591, 0.247657)},
}

# Largest absolute correlation between any two second-order terms, same designs.
# The saturated sizes are omitted: there the correlation is not defined for every pair.
MAX_R = {
    1: {9: 0.707107, 11: 0.677003, 13: 0.300000, 15: 0.377964, 17: 0.367315, 19: 0.266667,
        21: 0.050000, 23: 0.178571, 25: 0.161165, 27: 0.000000, 29: 0.033333, 31: 0.091747},
    2: {10: 0.645497, 12: 0.612372, 14: 0.166667, 16: 0.258199, 18: 0.316228, 20: 0.200000,
        22: 0.083333, 24: 0.250000, 26: 0.133333, 28: 0.066667, 30: 0.040291},
    3: {11: 0.605530, 13: 0.570088, 15: 0.071429, 17: 0.169031, 19: 0.232621, 21: 0.145455,
        23: 0.162650, 25: 0.200082, 27: 0.200000, 29: 0.121212, 31: 0.091747},
}

# Absolute correlation matrices of five of the plotted designs, used as insets so the reader
# can see what a value of max |r| looks like. Keyed by (centre-point count, run count), and
# ordered as the three quadratics then the three two-factor interactions.
#
# The first three are the smallest design at each centre-point count. All three have the same
# four half-rows, so what separates them is the centre runs alone: as those are added, the
# quadratic-to-interaction correlation falls from 0.707 to 0.645 to 0.606 while the
# quadratic-to-quadratic correlation rises from 0 to 0.167 to 0.267.
INSETS = {
    (1, 9): [[1.000, 0.000, 0.000, 0.000, 0.000, 0.707],
             [0.000, 1.000, 0.000, 0.000, 0.707, 0.000],
             [0.000, 0.000, 1.000, 0.707, 0.000, 0.000],
             [0.000, 0.000, 0.707, 1.000, 0.500, 0.500],
             [0.000, 0.707, 0.000, 0.500, 1.000, 0.500],
             [0.707, 0.000, 0.000, 0.500, 0.500, 1.000]],
    (2, 10): [[1.000, 0.167, 0.167, 0.000, 0.000, 0.645],
              [0.167, 1.000, 0.167, 0.000, 0.645, 0.000],
              [0.167, 0.167, 1.000, 0.645, 0.000, 0.000],
              [0.000, 0.000, 0.645, 1.000, 0.500, 0.500],
              [0.000, 0.645, 0.000, 0.500, 1.000, 0.500],
              [0.645, 0.000, 0.000, 0.500, 0.500, 1.000]],
    (3, 11): [[1.000, 0.267, 0.267, 0.000, 0.000, 0.606],
              [0.267, 1.000, 0.267, 0.000, 0.606, 0.000],
              [0.267, 0.267, 1.000, 0.606, 0.000, 0.000],
              [0.000, 0.000, 0.606, 1.000, 0.500, 0.500],
              [0.000, 0.606, 0.000, 0.500, 1.000, 0.500],
              [0.606, 0.000, 0.000, 0.500, 0.500, 1.000]],
    (2, 24): [[1.000, 0.250, 0.120, 0.0, 0.0, 0.0],
              [0.250, 1.000, 0.239, 0.0, 0.0, 0.0],
              [0.120, 0.239, 1.000, 0.0, 0.0, 0.0],
              [0.000, 0.000, 0.000, 1.0, 0.0, 0.0],
              [0.000, 0.000, 0.000, 0.0, 1.0, 0.0],
              [0.000, 0.000, 0.000, 0.0, 0.0, 1.0]],
    (1, 31): [[1.000, 0.092, 0.092, 0.0, 0.0, 0.0],
              [0.092, 1.000, 0.033, 0.0, 0.0, 0.0],
              [0.092, 0.033, 1.000, 0.0, 0.0, 0.0],
              [0.000, 0.000, 0.000, 1.0, 0.0, 0.0],
              [0.000, 0.000, 0.000, 0.0, 1.0, 0.0],
              [0.000, 0.000, 0.000, 0.0, 0.0, 1.0]],
}

# Okabe-Ito, matching the other figures in this chapter.
BLUE, GREEN, VERM = "#0072B2", "#009E73", "#D55E00"
SERIES = [(1, BLUE, "o", "1 centre point"),
          (2, GREEN, "s", "2 centre points"),
          (3, VERM, "^", "3 centre points")]

# (index into BEST, title, direction, log scale, monotone in N)
PANELS = [
    (0, "$A/p$   average coefficient variance", "lower better", True, True),
    (2, "$E$   smallest eigenvalue of $\\mathbf{X}^T\\mathbf{X}$", "higher better", False, True),
    (1, "$D$   $|\\mathbf{X}^T\\mathbf{X}|^{1/p}$", "higher better", False, True),
    (3, "$I$   average prediction variance", "lower better", True, True),
    (4, "$G$   worst prediction variance", "lower better", True, True),
    (None, "max $|r|$   worst second-order correlation", "lower better", False, False),
]
COLUMN_LABELS = ["AVERAGED OVER THE WHOLE", "WORST CASE ONLY", "NEITHER"]

fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.0))

for ax, (idx, title, direction, log_y, monotone) in zip(axes.ravel(), PANELS):
    source = BEST if idx is not None else MAX_R
    for centre, colour, marker, label in SERIES:
        runs = sorted(source[centre])
        values = [source[centre][n][idx] if idx is not None else source[centre][n]
                  for n in runs]
        ax.plot(runs, values, color=colour, marker=marker, markersize=4.2, linewidth=1.6,
                label=label, zorder=3)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Number of runs, $N$", fontsize=9.5)
    # Keep both notes off the curve: a rising series leaves the lower right free and the
    # upper left free, a falling one the reverse. The max |r| panel carries insets along
    # its top, so its note sits lower than the rest.
    rising = direction.startswith("higher")
    ax.text(0.98, 0.06 if rising else (0.95 if monotone else 0.60), direction,
            transform=ax.transAxes, ha="right", va="bottom" if rising else "top",
            fontsize=8.5, color="#5A6570")
    ax.text(0.02, 0.94 if rising else 0.03, "monotone" if monotone else "reverses",
            transform=ax.transAxes, va="top" if rising else "bottom",
            fontsize=9.5, fontweight="bold", color="#00785A" if monotone else "#C1541F")
    ax.grid(axis="y", color="#D8DEE3", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C1541F" if not monotone else "#98A2AB")
        ax.spines[side].set_linewidth(1.6 if not monotone else 1.0)

# Insets on the max |r| panel: the correlation maps behind three of its points, on a
# common 0-to-1 scale so the shading can be compared across them.
ax_r = axes[1, 2]
ax_r.set_ylim(-0.03, 1.02)
SIZE = 0.145
PLACEMENT = {(1, 9): 0.02, (2, 10): 0.175, (3, 11): 0.33, (2, 24): 0.575, (1, 31): 0.80}
SERIES_COLOUR = {centre: colour for centre, colour, _, _ in SERIES}
for (centre, n_runs), x0 in PLACEMENT.items():
    colour, y0 = SERIES_COLOUR[centre], 0.79
    inset = ax_r.inset_axes([x0, y0, SIZE, SIZE])
    inset.imshow(np.array(INSETS[(centre, n_runs)]), cmap="Blues", vmin=0, vmax=1,
                 interpolation="nearest")
    inset.axhline(2.5, color="#7A848D", linewidth=0.7)   # quadratics | interactions
    inset.axvline(2.5, color="#7A848D", linewidth=0.7)
    inset.set_xticks([])
    inset.set_yticks([])
    # Border and leader line take the series colour, so each inset is tied to its own
    # centre-point count without needing a label.
    for spine in inset.spines.values():
        spine.set_color(colour)
        spine.set_linewidth(1.3)
    ax_r.annotate("", xy=(n_runs, MAX_R[centre][n_runs]), xycoords="data",
                  xytext=(x0 + SIZE / 2, y0), textcoords=ax_r.transAxes,
                  arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.9, alpha=0.75,
                                  shrinkA=1, shrinkB=4))
    ax_r.plot([n_runs], [MAX_R[centre][n_runs]], marker="o", markersize=9,
              markerfacecolor="none", markeredgecolor=colour, markeredgewidth=1.3,
              zorder=5)

for column, label in enumerate(COLUMN_LABELS):
    axes[0, column].text(0.5, 1.17, label, transform=axes[0, column].transAxes,
                         ha="center", fontsize=8.5, color="#8A949D")
axes[0, 0].legend(frameon=False, loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 0.90))

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("omars-metric-choice.png", dpi=250, facecolor="w", edgecolor="w",
            bbox_inches="tight")
print("saved figure")
