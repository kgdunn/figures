"""What each design size buys on the fed-batch bioreactor, in grams per litre.

Illustrates the section "A worked OMARS study" in the Design and Analysis of Experiments
chapter. The same four-factor study is run at five OMARS sizes and with the two classical
27-run designs, two hundred times each with fresh disturbance draws. Each campaign is scored
the same way: fit the staged analysis, follow the recipe the fitted model recommends, and
read the true titer there from the simulator with every disturbance switched off. The gain is
that titer minus the titer of the recipe the team started with, 7.436 g/L; the most any
recipe in the region can reach is 2.006 g/L more.

The upper panel is the gain in titer; the lower panel is how often each of the three real
effects that the recommendation depends on was declared active, over the same campaigns.

Every number is carried as a literal because two hundred campaigns per design take minutes;
omars_worked_study_data.py regenerates them.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt

# Gain in titer over the current recipe [g/L] at the recipe each campaign recommended, over
# two hundred campaigns per design, and how often each of four real effects was found;
# the figure plots three of them.
GAINS = {
    "OMARS 13": {"runs": 13, "mean": 0.1690, "p10": -0.02071, "p50": 0.000, "p90": 0.8573, "worst": -3.246,
                 "found": {"feed_rate": 0.28, "hold_temp^2": 0.02, "hold_temp:shift_day": 0.41, "shift_day^2": 0.10}},
    "OMARS 17": {"runs": 17, "mean": 0.4468, "p10": -0.8849, "p50": 0.8321, "p90": 1.142, "worst": -3.246,
                 "found": {"feed_rate": 0.98, "hold_temp^2": 0.20, "hold_temp:shift_day": 0.41, "shift_day^2": 0.38}},
    "OMARS 21": {"runs": 21, "mean": 0.8082, "p10": 0.000, "p50": 0.8573, "p90": 1.199, "worst": -1.093,
                 "found": {"feed_rate": 0.89, "hold_temp^2": 0.50, "hold_temp:shift_day": 0.81, "shift_day^2": 0.01}},
    "OMARS 27": {"runs": 27, "mean": 0.7612, "p10": -0.02071, "p50": 0.9244, "p90": 1.094, "worst": -1.118,
                 "found": {"feed_rate": 0.91, "hold_temp^2": 0.85, "hold_temp:shift_day": 1.00, "shift_day^2": 0.04}},
    "OMARS 31": {"runs": 31, "mean": 0.8506, "p10": 0.5546, "p50": 0.9236, "p90": 1.158, "worst": -0.3901,
                 "found": {"feed_rate": 1.00, "hold_temp^2": 0.85, "hold_temp:shift_day": 1.00, "shift_day^2": 0.23}},
    "Box-Behnken 27": {"runs": 27, "mean": 0.8848, "p10": -0.2215, "p50": 1.199, "p90": 1.677, "worst": -3.246,
                       "found": {"feed_rate": 0.99, "hold_temp^2": 0.77, "hold_temp:shift_day": 0.41, "shift_day^2": 0.06}},
    "CCD 27": {"runs": 27, "mean": 0.7597, "p10": -0.03423, "p50": 0.9403, "p90": 1.133, "worst": -1.149,
               "found": {"feed_rate": 0.91, "hold_temp^2": 0.88, "hold_temp:shift_day": 1.00, "shift_day^2": 0.09}},
}
PRIZE = 2.0064

# Okabe-Ito, matching the other figures in this chapter.
BLUE, GREEN, ORANGE, GREY, VERM = "#0072B2", "#009E73", "#E69F00", "#666666", "#D55E00"
PURPLE, SPINE = "#CC79A7", "#98A2AB"

omars = [v for k, v in GAINS.items() if k.startswith("OMARS")]
runs = [v["runs"] for v in omars]

fig, (ax, ax_f) = plt.subplots(2, 1, figsize=(8.6, 8.6), sharex=True,
                               gridspec_kw={"height_ratios": [1.5, 1.0]})

ax.axhline(PRIZE, color=GREY, lw=1.2, ls=(0, (5, 4)), zorder=1)
ax.text(31.4, PRIZE + 0.04, f"best possible, {PRIZE:.3f} g/L", ha="right", va="bottom",
        fontsize=10.5, color=GREY)
ax.axhline(0, color="#98A2AB", lw=1.0, zorder=1)
# In the gap below the zero line between the 21- and 27-run marks, clear of every band and bar.
ax.text(21.5, -0.08, "no better than\ntoday's recipe", ha="left", va="top", fontsize=10.5,
        color="#5A6570")

ax.fill_between(runs, [v["p10"] for v in omars], [v["p90"] for v in omars],
                color=BLUE, alpha=0.14, lw=0, zorder=2, label="OMARS, 10th to 90th percentile")
ax.plot(runs, [v["p50"] for v in omars], color=BLUE, marker="o", ms=7, lw=2.2, zorder=4,
        label="OMARS, median campaign")
ax.plot(runs, [v["worst"] for v in omars], color=BLUE, marker="v", ms=7, lw=0, zorder=4,
        markerfacecolor="white", markeredgewidth=1.6, label="OMARS, worst of 200 campaigns")

# The two classical 27-run designs, offset so they do not sit on the OMARS marks.
for name, colour, dx, marker in (("Box-Behnken 27", GREEN, -0.55, "*"), ("CCD 27", ORANGE, 0.55, "D")):
    v = GAINS[name]
    x = v["runs"] + dx
    ax.plot([x, x], [v["p10"], v["p90"]], color=colour, lw=2.0, alpha=0.55, zorder=3)
    ax.plot([x], [v["p50"]], color=colour, marker=marker, ms=13 if marker == "*" else 8, lw=0,
            zorder=5, label=f"{name.replace(' 27', '')}, 27 runs: median and 10th to 90th")
    ax.plot([x], [v["worst"]], color=colour, marker="v", ms=7, lw=0, markerfacecolor="white",
            markeredgewidth=1.6, zorder=5)

ax.set_ylim(-3.5, 2.4)
ax.set_ylabel("Titer gained over the current recipe, g/L", fontsize=12)
# Legend above the axes, so no marker can land in it.
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, fontsize=10, frameon=False,
          columnspacing=1.6, handlelength=2.0)

# Lower panel: how often each real effect was found, over the same campaigns.
EFFECTS = (("feed_rate", "feed rate, main effect", BLUE, "o"),
           ("hold_temp:shift_day", "hold temperature × shift day", VERM, "s"),
           ("hold_temp^2", "hold temperature, quadratic", PURPLE, "D"))
for key, label, colour, marker in EFFECTS:
    ax_f.plot(runs, [100 * v["found"][key] for v in omars], color=colour, marker=marker, ms=6.5,
              lw=2.0, zorder=4, label=label)
    for name, dx, mk in (("Box-Behnken 27", -0.55, "*"), ("CCD 27", 0.55, "D")):
        ax_f.plot([GAINS[name]["runs"] + dx], [100 * GAINS[name]["found"][key]], mk,
                  ms=12 if mk == "*" else 7, color=colour, markerfacecolor="white", markeredgewidth=1.6,
                  zorder=5)
ax_f.set_ylim(-4, 104)
ax_f.set_yticks([0, 25, 50, 75, 100])
ax_f.set_ylabel("Campaigns finding the effect, %", fontsize=12)
ax_f.set_xticks(runs)
ax_f.set_xlim(11.5, 32.5)
ax_f.set_xlabel("Number of runs, $N$", fontsize=13)
ax_f.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=10, frameon=True, facecolor="white",
            edgecolor="0.85", framealpha=1.0)
# Top left, above the 13-run points and left of the feed-rate line's climb to 17 runs.
ax_f.text(11.8, 101, "hollow star: Box-Behnken\nhollow diamond: CCD\nboth at 27 runs",
          ha="left", va="top", fontsize=9, color=GREY)

for axis in (ax, ax_f):
    axis.grid(axis="y", color="0.9", lw=0.9)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(SPINE)
    axis.tick_params(colors="0.25", labelsize=10.5)

fig.tight_layout()
fig.savefig("omars-worked-study-tradeoff.png", dpi=300, facecolor="w", edgecolor="w",
            format=None, transparent=True)
print("saved figure")
