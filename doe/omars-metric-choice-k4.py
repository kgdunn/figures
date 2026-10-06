"""The nine candidate cell measures of omars-metric-choice.py, for four factors.

Companion to omars-metric-choice.py, which plots the three-factor column of the OMARS
trade-off table. This is the four-factor column, drawn the same way: the five alphabetic
criteria and the largest second-order correlation for the main-effects-and-quadratics model
(p = 2k + 1 = 9 terms) in the first two rows, and power under the full second-order model
(p = 15 terms) in the third, which therefore starts at the estimability frontier,
N = k^2 + k + 1 = 21.

Where the values come from
--------------------------
The three-factor figure is exact everywhere. At four factors the design family is too
large for that past ten half-rows, so the data are of two kinds:

* **Exact (filled markers).** Every OMARS foldover with up to ten half-rows was scored:
  every run count up to 21 with one centre point, 22 with two and 23 with three.
* **Best found (open markers).** Past that, an iterated local search over the same family.
  Those points are the best design the search found, a bound on the frontier rather than
  the frontier. Run on the exact range as if it were out of reach, the same search
  reached the exact best in every cell.

omars_metric_choice_k4_data.py produced every literal below and documents both methods.

The definitive screening design (9 runs) and the Box-Behnken design (27 runs, three centre
points) are marked where they are defined, built by process_improve.experiments and scored
by the same code as the panels.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

# Best value at each run count, four factors, keyed by centre-point count. Columns:
# A/p, D, E, I, G for the main-effects-and-quadratics model. N = 2h + c.
BEST = {
    1: {9: (0.407407, 3.851794, 0.683994, 0.733333, 1.000000),
        11: (0.335648, 4.434810, 0.918320, 0.581944, 1.000000),
        13: (0.279630, 5.310664, 1.172649, 0.476667, 0.885714),
        15: (0.227273, 6.090681, 1.475825, 0.384848, 0.801562),
        17: (0.187831, 7.132557, 2.429841, 0.325397, 0.761798),
        19: (0.174411, 8.058584, 2.582404, 0.302020, 0.500000),
        21: (0.155556, 8.741520, 3.355418, 0.262222, 0.479853),
        23: (0.144264, 9.680818, 3.486513, 0.243431, 0.438953),
        25: (0.129470, 10.446051, 4.000000, 0.224020, 0.422101),
        27: (0.117145, 11.555382, 4.000000, 0.201941, 0.333333),
        29: (0.110670, 12.279738, 5.140698, 0.190416, 0.333333),
        31: (0.104260, 13.160193, 5.264721, 0.178131, 0.314685)},
    2: {10: (0.327160, 4.160168, 1.350889, 0.559259, 1.000000),
        12: (0.288889, 4.693803, 1.485696, 0.490000, 0.933333),
        14: (0.251684, 5.672403, 1.639320, 0.426768, 0.805263),
        16: (0.201389, 6.349604, 1.875485, 0.345833, 0.795238),
        18: (0.179487, 7.703588, 2.968780, 0.312821, 0.500000),
        20: (0.163580, 8.382636, 3.079486, 0.279630, 0.500000),
        22: (0.150206, 9.265551, 4.000000, 0.250617, 0.462500),
        24: (0.138036, 10.067194, 4.000000, 0.234251, 0.428571),
        26: (0.122160, 11.046345, 4.000000, 0.210924, 0.419207),
        28: (0.113492, 11.930713, 4.557075, 0.195397, 0.333333),
        30: (0.107529, 12.661108, 5.715729, 0.185395, 0.323611)},
    3: {11: (0.300412, 4.351876, 2.000000, 0.501235, 1.000000),
        13: (0.268849, 4.872606, 2.000000, 0.450595, 0.931548),
        15: (0.231481, 5.910295, 2.000000, 0.394444, 0.781818),
        17: (0.187831, 6.544384, 2.429841, 0.325397, 0.761404),
        19: (0.173835, 8.058584, 3.492189, 0.302020, 0.500000),
        21: (0.155556, 8.687211, 3.553982, 0.262222, 0.479853),
        23: (0.144264, 9.643688, 4.000000, 0.242328, 0.438953),
        25: (0.129470, 10.362217, 4.000000, 0.224020, 0.422101),
        27: (0.117145, 11.555382, 4.000000, 0.201941, 0.333333),
        29: (0.110670, 12.279738, 5.140698, 0.190416, 0.333333),
        31: (0.104260, 13.160193, 6.000000, 0.178131, 0.314685)},
}

# Largest absolute correlation between any two second-order terms.
MAX_R = {
    1: {9: 0.707107, 11: 0.677003, 13: 0.500000, 15: 0.500000, 17: 0.367315, 19: 0.457143, 21: 0.311805, 23: 0.320435, 25: 0.257172, 27: 0.253546, 29: 0.216974, 31: 0.197842},
    2: {10: 0.645497, 12: 0.612372, 14: 0.500000, 16: 0.500000, 18: 0.357143, 20: 0.500000, 22: 0.385758, 24: 0.306186, 26: 0.238095, 28: 0.241523, 30: 0.206431},
    3: {11: 0.605530, 13: 0.570088, 15: 0.500000, 17: 0.500000, 19: 0.457143, 21: 0.500000, 23: 0.393939, 25: 0.305556, 27: 0.200000, 29: 0.231741, 31: 0.197842},
}

# Smallest coefficient variance under the full second-order model, as (main effect,
# two-factor interaction, pure quadratic); each minimised on its own, over designs
# that fit the full model. The row starts at h = k(k+1)/2 = 10 half-rows.
POWER_C = {
    1: {21: (0.062500, 0.138889, 0.416667),
        23: (0.055556, 0.062500, 0.221939),
        25: (0.045455, 0.062500, 0.202778),
        27: (0.045455, 0.062500, 0.166667),
        29: (0.041667, 0.057065, 0.162698),
        31: (0.038462, 0.046875, 0.144097)},
    2: {22: (0.062500, 0.138889, 0.290123),
        24: (0.055556, 0.062500, 0.213710),
        26: (0.045455, 0.062500, 0.195767),
        28: (0.045455, 0.062500, 0.157407),
        30: (0.038462, 0.054688, 0.150000)},
    3: {23: (0.062500, 0.138889, 0.245370),
        25: (0.055556, 0.062500, 0.208333),
        27: (0.045455, 0.062500, 0.187500),
        29: (0.045455, 0.062500, 0.151852),
        31: (0.038462, 0.051044, 0.142612)},
}

# The largest run count, per centre-point count, at which the values are exact: every
# foldover with up to ten half-rows was scored. Larger run counts are best found.
EXACT = { 1: 21, 2: 22, 3: 23}

# The two standard designs, built by process_improve and scored by the same code.
ANCHORS = {
    "bbd": {"n_runs": 27, "A": 0.157407, "E": 1.453171, "D": 9.705913,
            "I": 0.233333, "G": 0.833333, "maxr": 0.200000,
            "c": (0.083333, 0.250000, 0.187500)},
    "dsd": {"n_runs": 9, "A": 0.407407, "E": 0.683994, "D": 3.851794,
            "I": 0.733333, "G": 1.000000, "maxr": 0.707107,
            "c": None},
}

# Absolute correlation matrices behind five of the max |r| points, quadratics then
# interactions, for the insets.
INSETS = {
    (1, 9): [[1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.707, 0.707, 0.707],
             [0.000, 1.000, 0.000, 0.000, 0.000, 0.707, 0.707, 0.000, 0.000, 0.707],
             [0.000, 0.000, 1.000, 0.000, 0.707, 0.000, 0.707, 0.000, 0.707, 0.000],
             [0.000, 0.000, 0.000, 1.000, 0.707, 0.707, 0.000, 0.707, 0.000, 0.000],
             [0.000, 0.000, 0.707, 0.707, 1.000, 0.500, 0.500, 0.500, 0.500, 0.000],
             [0.000, 0.707, 0.000, 0.707, 0.500, 1.000, 0.500, 0.500, 0.000, 0.500],
             [0.000, 0.707, 0.707, 0.000, 0.500, 0.500, 1.000, 0.000, 0.500, 0.500],
             [0.707, 0.000, 0.000, 0.707, 0.500, 0.500, 0.000, 1.000, 0.500, 0.500],
             [0.707, 0.000, 0.707, 0.000, 0.500, 0.000, 0.500, 0.500, 1.000, 0.500],
             [0.707, 0.707, 0.000, 0.000, 0.000, 0.500, 0.500, 0.500, 0.500, 1.000]],
    (2, 10): [[1.000, 0.167, 0.167, 0.167, 0.000, 0.000, 0.000, 0.645, 0.645, 0.645],
             [0.167, 1.000, 0.167, 0.167, 0.000, 0.645, 0.645, 0.000, 0.000, 0.645],
             [0.167, 0.167, 1.000, 0.167, 0.645, 0.000, 0.645, 0.000, 0.645, 0.000],
             [0.167, 0.167, 0.167, 1.000, 0.645, 0.645, 0.000, 0.645, 0.000, 0.000],
             [0.000, 0.000, 0.645, 0.645, 1.000, 0.500, 0.500, 0.500, 0.500, 0.000],
             [0.000, 0.645, 0.000, 0.645, 0.500, 1.000, 0.500, 0.500, 0.000, 0.500],
             [0.000, 0.645, 0.645, 0.000, 0.500, 0.500, 1.000, 0.000, 0.500, 0.500],
             [0.645, 0.000, 0.000, 0.645, 0.500, 0.500, 0.000, 1.000, 0.500, 0.500],
             [0.645, 0.000, 0.645, 0.000, 0.500, 0.000, 0.500, 0.500, 1.000, 0.500],
             [0.645, 0.645, 0.000, 0.000, 0.000, 0.500, 0.500, 0.500, 0.500, 1.000]],
    (3, 11): [[1.000, 0.267, 0.267, 0.267, 0.000, 0.000, 0.000, 0.606, 0.606, 0.606],
             [0.267, 1.000, 0.267, 0.267, 0.000, 0.606, 0.606, 0.000, 0.000, 0.606],
             [0.267, 0.267, 1.000, 0.267, 0.606, 0.000, 0.606, 0.000, 0.606, 0.000],
             [0.267, 0.267, 0.267, 1.000, 0.606, 0.606, 0.000, 0.606, 0.000, 0.000],
             [0.000, 0.000, 0.606, 0.606, 1.000, 0.500, 0.500, 0.500, 0.500, 0.000],
             [0.000, 0.606, 0.000, 0.606, 0.500, 1.000, 0.500, 0.500, 0.000, 0.500],
             [0.000, 0.606, 0.606, 0.000, 0.500, 0.500, 1.000, 0.000, 0.500, 0.500],
             [0.606, 0.000, 0.000, 0.606, 0.500, 0.500, 0.000, 1.000, 0.500, 0.500],
             [0.606, 0.000, 0.606, 0.000, 0.500, 0.000, 0.500, 0.500, 1.000, 0.500],
             [0.606, 0.606, 0.000, 0.000, 0.000, 0.500, 0.500, 0.500, 0.500, 1.000]],
    (3, 27): [[1.000, 0.200, 0.200, 0.200, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
             [0.200, 1.000, 0.200, 0.200, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
             [0.200, 0.200, 1.000, 0.200, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
             [0.200, 0.200, 0.200, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
             [0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000],
             [0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000],
             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000],
             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000],
             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000],
             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000]],
    (1, 31): [[1.000, 0.111, 0.111, 0.111, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
             [0.111, 1.000, 0.061, 0.061, 0.000, 0.000, 0.000, 0.000, 0.000, 0.198],
             [0.111, 0.061, 1.000, 0.061, 0.000, 0.000, 0.000, 0.000, 0.198, 0.000],
             [0.111, 0.061, 0.061, 1.000, 0.000, 0.000, 0.000, 0.198, 0.000, 0.000],
             [0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000],
             [0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000],
             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000],
             [0.000, 0.000, 0.000, 0.198, 0.000, 0.000, 0.000, 1.000, 0.125, 0.125],
             [0.000, 0.000, 0.198, 0.000, 0.000, 0.000, 0.000, 0.125, 1.000, 0.125],
             [0.000, 0.198, 0.000, 0.000, 0.000, 0.000, 0.000, 0.125, 0.125, 1.000]],
}

K = 4
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
XLIM = (7, 32)
XTICKS = [10, 15, 20, 25, 30]

# The corner notes sit wherever the curve is not, which on a log axis can still land them on
# a gridline, and in the max |r| panel on an inset leader. A white patch behind the text
# keeps them readable without moving them off the free corner.
NOTE_BOX = {"facecolor": "white", "edgecolor": "none", "pad": 0.25}

SERIES = [(1, BLUE, "o", "1 centre point"),
          (2, PURPLE, "s", "2 centre points"),
          (3, BROWN, "^", "3 centre points")]

# (index into BEST, title, direction, log scale). Whether a panel is monotone in N is read
# off the data below, not asserted: at four factors it need not match three.
PANELS = [
    (0, "$A/p$   average coefficient variance", "lower better", True),
    (2, "$E$   smallest eigenvalue of $\\mathbf{X}^T\\mathbf{X}$", "higher better", False),
    (1, "$D$   $|\\mathbf{X}^T\\mathbf{X}|^{1/p}$", "higher better", False),
    (3, "$I$   average prediction variance", "lower better", True),
    (4, "$G$   worst prediction variance", "lower better", True),
    (None, "max $|r|$   worst second-order correlation", "lower better", False),
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


def is_monotone(table, value_of, higher_better):
    """True if no centre-point series ever gets worse as the run count grows."""
    for centre in table:
        values = [value_of(table[centre][n]) for n in sorted(table[centre])]
        values = [v for v in values if v is not None]
        steps = np.diff(values) * (1 if higher_better else -1)
        if (steps < -1e-9).any():
            return False
    return True


def plot_series(ax, runs, values, centre, colour, marker, label):
    """One centre-point series: a line through every point, filled markers where the value
    is exact and open markers where it is the best a search found."""
    ax.plot(runs, values, color=colour, linewidth=1.6, label=None, zorder=3)
    exact = [(n, v) for n, v in zip(runs, values) if n <= EXACT[centre]]
    found = [(n, v) for n, v in zip(runs, values) if n > EXACT[centre]]
    ax.plot(*zip(*exact), color=colour, marker=marker, markersize=4.2, linestyle="none",
            label=label, zorder=4)
    if found:
        ax.plot(*zip(*found), marker=marker, markersize=4.6, linestyle="none",
                markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.2,
                zorder=4)


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
    idx, title, direction, log_y = panel
    source = BEST if idx is not None else MAX_R
    monotone = is_monotone(source, (lambda v, i=idx: v[i]) if idx is not None else (lambda v: v),
                           direction.startswith("higher"))
    for centre, colour, marker, label in SERIES:
        runs = sorted(source[centre])
        values = [source[centre][n][idx] if idx is not None else source[centre][n]
                  for n in runs]
        plot_series(ax, runs, values, centre, colour, marker, label)
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
PLACEMENT = {(1, 9): 0.02, (2, 10): 0.175, (3, 11): 0.33, (3, 27): 0.60, (1, 31): 0.80}
SERIES_COLOUR = {centre: colour for centre, colour, _, _ in SERIES}
for (centre, n_runs), x0 in PLACEMENT.items():
    colour, y0 = SERIES_COLOUR[centre], 0.79
    inset = ax_r.inset_axes([x0, y0, SIZE, SIZE])
    inset.imshow(np.array(INSETS[(centre, n_runs)]), cmap="Blues", vmin=0, vmax=1,
                 interpolation="nearest")
    inset.axhline(K - 0.5, color="#7A848D", linewidth=0.7)   # quadratics | interactions
    inset.axvline(K - 0.5, color="#7A848D", linewidth=0.7)
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
        plot_series(ax, xs, ys, centre, colour, marker, label)
    mark_anchors(ax, lambda entry, slot=slot: None if entry["c"] is None
                 else power(entry["c"][slot], entry["n_runs"]))
    ax.axhline(0.8, color="#8A949D", linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.set_title(f"Power, {what}", fontsize=11, pad=8)
    ax.set_ylim(0, 1.02)
    ax.text(0.98, 0.06, "higher better", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="#5A6570", bbox=NOTE_BOX)
    monotone = is_monotone({c: {n: power(v[slot], n) for n, v in POWER_C[c].items()}
                            for c in POWER_C}, lambda v: v, True)
    ax.text(0.02, 0.94, "monotone" if monotone else "reverses", transform=ax.transAxes,
            va="top", fontsize=9.5, fontweight="bold",
            color="#00785A" if monotone else "#C1541F", bbox=NOTE_BOX)
    style(ax, "higher better", monotone=monotone)

axes[2, 0].set_ylabel("Power at $\\alpha = 0.05$", fontsize=9.5)

for column, label in enumerate(COLUMN_LABELS):
    axes[0, column].text(0.5, 1.17, label, transform=axes[0, column].transAxes,
                         ha="center", fontsize=8.5, color="#8A949D")
axes[2, 1].text(0.5, 1.20, "FULL SECOND-ORDER MODEL, $|\\beta|/\\sigma = 1$, "
                           "$\\alpha = 0.05$, dashed line at 0.8",
                transform=axes[2, 1].transAxes, ha="center", fontsize=8.5, color="#8A949D")

handles = [Line2D([], [], color=colour, marker=marker, markersize=4.2, linewidth=1.6)
           for _, colour, marker, _ in SERIES]
labels = [label for _, _, _, label in SERIES]
handles += [Line2D([], [], marker="*", markersize=13, markerfacecolor=BBD_GREEN,
                   markeredgecolor="white", markeredgewidth=1.0, linestyle="none"),
            Line2D([], [], marker="o", markersize=8, markerfacecolor=DSD_ORANGE,
                   markeredgecolor="white", markeredgewidth=1.0, linestyle="none"),
            Line2D([], [], marker="o", markersize=5, markerfacecolor="white",
                   markeredgecolor="#5A6570", markeredgewidth=1.2, linestyle="none")]
labels += [f"Box-Behnken design, {ANCHORS['bbd']['n_runs']} runs",
           f"Definitive screening design, {ANCHORS['dsd']['n_runs']} runs",
           "open marker: best found by search"]
fig.legend(handles, labels, frameon=False, loc="upper center", ncol=6, fontsize=9.5,
           bbox_to_anchor=(0.5, 0.952), handletextpad=0.4, columnspacing=1.6)

fig.suptitle(f"Four factors ($k$ = {K}): the best value at each run count, "
             "exact where filled, best found by search where open",
             fontsize=15, fontweight="bold", x=0.5, y=0.972)
fig.tight_layout(rect=[0, 0, 1, 0.935])
fig.savefig("omars-metric-choice-k4.png", dpi=250, facecolor="w", edgecolor="w",
            bbox_inches="tight")
print("saved figure")
