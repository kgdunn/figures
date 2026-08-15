"""Why the OMARS trade-off table reports estimability and not a quality score.

Illustrates the "OMARS designs" section of the Design and Analysis of Experiments
chapter. It takes a single column of the trade-off table, three factors, and plots nine
candidate cell measures down that column.

The first two rows are the five alphabetic optimality criteria and the largest absolute
correlation between two second-order terms, arranged so that the related pairs sit in
columns:

    A  average variance of an estimated coefficient   I  average prediction variance
    E  smallest eigenvalue of X'X                     G  worst prediction variance
    D  generalised variance                           max |r|  worst entanglement

The left column averages, the middle column takes a worst case, the right column does
neither. The top row is a summary of the eigenvalues of X'X; the second row is prediction
variance over the design region. Those six all score the same model, main effects and
quadratics, p = 2k + 1 terms, so a value means the same thing in every cell. I is exact
against the cuboidal region moments; G is maximised over a seven-level grid per factor.

The third row is power, the measure a practitioner is most likely to reach for. It scores
the full second-order model instead, p = 1 + 2k + k(k-1)/2, because the two-factor
interactions have to be in the model for the middle panel to exist at all. The row
therefore starts at the estimability frontier, N = k^2 + k + 1 = 13, rather than at 7 or 9
like the rows above. Power needs two things from the reader that the other six do not, an
effect size and a significance level, and it needs one panel per term type; both are
stated in the row's own header.

Every point is the best value attainable at that run count, found by enumerating every
OMARS foldover of that size, so the curves are frontiers rather than the results of a
search. The enumeration reduces the design to a count per sign class of {-1, 0, 1}^k,
which is what makes it exhaustive rather than heuristic: for second-order terms a half-row
and its negation are interchangeable. It takes hours, so its output is carried below as
literal data rather than recomputed here. Twelve of the cells were re-derived from naive
numpy model matrices and agree to 5e-7.

Two standard designs are marked in every panel where they are defined, so a reader can see
where a design they already know sits against the frontier: the definitive screening design
at 9 runs, and the Box-Behnken design at 15. Both were built by
process_improve.experiments and then scored by the same code that checked the panels. The
DSD is absent from the third row because 9 runs cannot fit a ten-parameter model. The
Box-Behnken sits on the frontier for A, D, I, max |r| and quadratic power, and clearly off
it for E, G, main-effect power and interaction power: it spends its runs on curvature.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

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

# Smallest attainable coefficient variance under the FULL second-order model, the diagonal
# of (X'X)^-1, as (main effect, two-factor interaction, pure quadratic), keyed by
# centre-point count then run count. Power is derived from these below rather than stored,
# so the effect size and significance level stay visible and adjustable.
#
# Main effects are orthogonal to everything in an OMARS design, so their entry is
# 1 / (2 n_j) with n_j the number of half-rows in which factor j is off zero. The other two
# come from the even block over the intercept, the quadratics and the interactions. The
# three are minimised independently, so each is a frontier in its own right and one design
# need not attain all three.
#
# The row starts where the full model becomes estimable. That needs h >= k(k+1)/2 half-rows
# whatever the run count, which is why three centre runs at 13 is absent: it clears
# N >= p but leaves h = 5, one short.
POWER_C = {
    1: {13: (0.083333, 0.125000, 0.437500),
        15: (0.071429, 0.109375, 0.388889),
        17: (0.062500, 0.093750, 0.333333),
        19: (0.055556, 0.078125, 0.261905),
        21: (0.050000, 0.062500, 0.211538),
        23: (0.045455, 0.057292, 0.193478),
        25: (0.041667, 0.052083, 0.174779),
        27: (0.038462, 0.046875, 0.151652),
        29: (0.035714, 0.041667, 0.142424),
        31: (0.033333, 0.039062, 0.133028)},
    2: {14: (0.083333, 0.125000, 0.312500),
        16: (0.071429, 0.109375, 0.296875),
        18: (0.062500, 0.093750, 0.261905),
        20: (0.055556, 0.078125, 0.222222),
        22: (0.050000, 0.062500, 0.206522),
        24: (0.045455, 0.057292, 0.190299),
        26: (0.041667, 0.052083, 0.171256),
        28: (0.038462, 0.046875, 0.148148),
        30: (0.035714, 0.041667, 0.139785)},
    3: {15: (0.083333, 0.125000, 0.270833),
        17: (0.071429, 0.109375, 0.255208),
        19: (0.062500, 0.093750, 0.228070),
        21: (0.055556, 0.078125, 0.200855),
        23: (0.050000, 0.062500, 0.193038),
        25: (0.045455, 0.057292, 0.179688),
        27: (0.041667, 0.052083, 0.165201),
        29: (0.038462, 0.046875, 0.145390),
        31: (0.035714, 0.041667, 0.137681)},
}

# The two standard designs marked in every panel. Built by dispatch_dsd and
# dispatch_box_behnken, then scored by the code that checked the six enumerated panels.
# "c" is the same triple as POWER_C, or None when the full second-order model does not fit:
# the three-factor DSD has 9 runs against ten parameters.
#
# The DSD lands at 9 runs rather than 2k + 1 = 7 because the construction folds a
# conference matrix of order k, which exists only for an even k. An odd k uses one of order
# k + 1 and drops a column, so the design arrives with two runs to spare.
ANCHORS = {
    "bbd": {"n_runs": 15, "A": 0.217262, "E": 1.634575, "D": 6.298440,
            "I": 0.301389, "G": 0.645833, "maxr": 0.071429,
            "c": (0.125000, 0.250000, 0.270833)},
    "dsd": {"n_runs": 9, "A": 0.396825, "E": 0.811221, "D": 3.970330,
            "I": 0.577778, "G": 0.777778, "maxr": 0.707107,
            "c": None},
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

K = 3
P_FULL = 1 + 2 * K + K * (K - 1) // 2      # terms in the full second-order model
DELTA = 1.0                                # effect size, |beta| / sigma
ALPHA = 0.05

# Five marks have to stay apart from each other: three centre-run series plus the two anchor
# designs. Green and orange are spoken for, since they are what the Box-Behnken and the
# definitive screening design carry in the trade-off table, so the series take blue, the
# reddish purple of the Okabe-Ito palette, and a dark gold-brown. Simulated under
# deuteranopia and protanopia the brown stays at least 33 units in CIE76 Lab from its
# nearest neighbour, where a deep violet collapses onto the blue at 9.
BLUE, PURPLE, BROWN = "#0072B2", "#CC79A7", "#946000"
BBD_GREEN, DSD_ORANGE = "#009E73", "#E69F00"
SPINE = "#98A2AB"

# One x axis for all nine panels, so a run count sits at the same horizontal position in
# every panel and a column can be read straight down.
XLIM = (6, 32)
XTICKS = [10, 15, 20, 25, 30]

# The corner notes sit wherever the curve is not, which on a log axis can still land them on
# a gridline, and in the max |r| panel on an inset leader. A white patch behind the text
# keeps them readable without moving them off the free corner.
NOTE_BOX = {"facecolor": "white", "edgecolor": "none", "pad": 0.25}

SERIES = [(1, BLUE, "o", "1 centre point"),
          (2, PURPLE, "s", "2 centre points"),
          (3, BROWN, "^", "3 centre points")]

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

# Key into ANCHORS for each panel of the first two rows, in the order PANELS lists them.
ANCHOR_KEYS = ["A", "E", "D", "I", "G", "maxr"]

# Third row, left to right. The index is into the POWER_C and ANCHORS["c"] triples, which
# run (main effect, two-factor interaction, pure quadratic).
POWER_PANELS = [(0, "a main effect"), (1, "a two-factor interaction"),
                (2, "a pure quadratic")]


def power(c, n_runs):
    """Power of the two-sided test on one coefficient, from the non-central F."""
    df = n_runs - P_FULL
    if df <= 0:
        return None
    return float(1 - stats.ncf.cdf(stats.f.ppf(1 - ALPHA, 1, df), 1, df, DELTA**2 / c))


def mark_anchors(ax, value_of):
    """Put the Box-Behnken star and the definitive screening circle on one panel."""
    for name, colour, marker, size in (("bbd", BBD_GREEN, "*", 20),
                                       ("dsd", DSD_ORANGE, "o", 10)):
        value = value_of(ANCHORS[name])
        if value is None:
            continue
        ax.plot([ANCHORS[name]["n_runs"]], [value], marker=marker, markersize=size,
                markerfacecolor=colour, markeredgecolor="white", markeredgewidth=1.1,
                linestyle="none", zorder=6)


def style(ax, direction, monotone):
    """The shared furniture: notes, grid, ticks and frame."""
    ax.set_xlabel("Number of runs, $N$", fontsize=9.5)
    ax.set_xlim(*XLIM)
    ax.set_xticks(XTICKS)
    ax.grid(axis="y", color="#D8DEE3", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # One frame colour and weight for all nine panels; the monotone / reverses note already
    # says in words which panel is the odd one.
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(1.0)


fig, axes = plt.subplots(3, 3, figsize=(13.4, 12.0))

for ax, panel, anchor_key in zip(axes.ravel()[:6], PANELS, ANCHOR_KEYS):
    idx, title, direction, log_y, monotone = panel
    source = BEST if idx is not None else MAX_R
    for centre, colour, marker, label in SERIES:
        runs = sorted(source[centre])
        values = [source[centre][n][idx] if idx is not None else source[centre][n]
                  for n in runs]
        ax.plot(runs, values, color=colour, marker=marker, markersize=4.2, linewidth=1.6,
                label=label, zorder=3)
    mark_anchors(ax, lambda entry, key=anchor_key: entry[key])
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=11, pad=8)
    # Keep both notes off the curve: a rising series leaves the lower right free and the
    # upper left free, a falling one the reverse. The max |r| panel carries insets along
    # its top, so its note sits lower than the rest.
    rising = direction.startswith("higher")
    ax.text(0.98, 0.06 if rising else (0.95 if monotone else 0.60), direction,
            transform=ax.transAxes, ha="right", va="bottom" if rising else "top",
            fontsize=8.5, color="#5A6570", bbox=NOTE_BOX)
    ax.text(0.02, 0.94 if rising else 0.03, "monotone" if monotone else "reverses",
            transform=ax.transAxes, va="top" if rising else "bottom",
            fontsize=9.5, fontweight="bold", color="#00785A" if monotone else "#C1541F",
            bbox=NOTE_BOX)
    style(ax, direction, monotone)

# Insets on the max |r| panel: the correlation maps behind five of its points, on a
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

# Third row: power for each term type, all under the full second-order model.
for ax, (slot, what) in zip(axes[2], POWER_PANELS):
    for centre, colour, marker, label in SERIES:
        runs = sorted(POWER_C[centre])
        xs = [n for n in runs if power(POWER_C[centre][n][slot], n) is not None]
        ys = [power(POWER_C[centre][n][slot], n) for n in xs]
        ax.plot(xs, ys, color=colour, marker=marker, markersize=4.2, linewidth=1.6,
                label=label, zorder=3)
    mark_anchors(ax, lambda entry, slot=slot: None if entry["c"] is None
                 else power(entry["c"][slot], entry["n_runs"]))
    ax.axhline(0.8, color="#8A949D", linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.set_title(f"Power, {what}", fontsize=11, pad=8)
    ax.set_ylim(0, 1.02)
    ax.text(0.98, 0.06, "higher better", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="#5A6570", bbox=NOTE_BOX)
    ax.text(0.02, 0.94, "monotone", transform=ax.transAxes, va="top", fontsize=9.5,
            fontweight="bold", color="#00785A", bbox=NOTE_BOX)
    style(ax, "higher better", monotone=True)

axes[2, 0].set_ylabel("Power at $\\alpha = 0.05$", fontsize=9.5)

for column, label in enumerate(COLUMN_LABELS):
    axes[0, column].text(0.5, 1.17, label, transform=axes[0, column].transAxes,
                         ha="center", fontsize=8.5, color="#8A949D")
axes[2, 1].text(0.5, 1.20, "FULL SECOND-ORDER MODEL, $|\\beta|/\\sigma = 1$, "
                           "$\\alpha = 0.05$, dashed line at 0.8",
                transform=axes[2, 1].transAxes, ha="center", fontsize=8.5, color="#8A949D")

handles, labels = axes[0, 0].get_legend_handles_labels()
handles += [Line2D([], [], marker="*", markersize=13, markerfacecolor=BBD_GREEN,
                   markeredgecolor="white", markeredgewidth=1.0, linestyle="none"),
            Line2D([], [], marker="o", markersize=8, markerfacecolor=DSD_ORANGE,
                   markeredgecolor="white", markeredgewidth=1.0, linestyle="none")]
labels += ["Box-Behnken design, 15 runs", "Definitive screening design, 9 runs"]
axes[0, 0].legend(handles, labels, frameon=False, loc="upper right", fontsize=9,
                  bbox_to_anchor=(1.02, 0.98))

fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig("omars-metric-choice.png", dpi=250, facecolor="w", edgecolor="w",
            bbox_inches="tight")
print("saved figure")
