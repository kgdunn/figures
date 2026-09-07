"""The unfolded batch row, and what is known about it at a decision point.

One committed PNG for the batch monitoring and control chapter:

- ``mcc-decision-point-row.png``: the batchwise-unfolded row of the
  chapter's bioreactor model (11 pre-batch values, then 20 samples of
  each of 5 recorded tags: 111 columns, regressed onto the final titer),
  drawn twice. The upper row is a training batch, complete. The lower row
  is a running batch at the day-4 decision point (8 of 20 samples
  recorded): the pre-batch values and the first 8 samples of every tag
  are known; the remaining samples of the three responding tags are
  missing and are estimated by the model; the remaining samples of the
  two manipulated tags (pH and temperature) are not missing at all: they
  are the schedule under consideration, nominal for the no-change
  prediction and free for the correction.

Schematic only: no simulation is run, so it regenerates in a second.

Usage
-----
    uv run --with matplotlib python mcc-decision-point-row.py [output_dir]
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
SKY = "#56B4E9"
GREY = "#666666"
LIGHT = "#E8E8E8"

DPI = 300

mpl.rcParams.update({"font.size": 17, "axes.titlesize": 20})

N_Z = 11
N_SAMPLES = 20
TAGS = [("pH", "mv"), ("temperature", "mv"), ("dissolved oxygen", "response"),
        ("offgas CO2", "response"), ("volume", "response")]
K_DECISION = 8   # samples recorded at the decision point (day 4 of 10, two samples a day)
CELL = 1.0
GAP = 2.2        # cells of space between blocks


def save(fig, outdir: pathlib.Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote {outdir / name}")


def draw_row(ax, y: float, *, at_decision_point: bool) -> None:
    """Draw one unfolded row at height ``y``; colour by what is known if at a decision point."""
    x = 0.0
    height = 1.0

    def cell(x0, colour, hatch=None, edge="white"):
        ax.add_patch(Rectangle((x0, y), CELL, height, facecolor=colour, edgecolor=edge,
                               linewidth=0.6, hatch=hatch))

    # Z block
    for j in range(N_Z):
        cell(x, BLUE if at_decision_point else LIGHT)
        x += CELL
    z_end = x
    ax.text((z_end - N_Z * CELL) / 2 + z_end / 2 - z_end / 2 + (N_Z * CELL) / 2 - N_Z * CELL / 2 + 0.0 + N_Z * CELL / 2,
            y + height + 0.25, "Z: 11 values\nbefore the batch", ha="center", va="bottom", fontsize=15)
    x += GAP
    # tag blocks
    for name, kind in TAGS:
        start = x
        for s in range(N_SAMPLES):
            if not at_decision_point:
                colour, hatch = LIGHT, None
            elif s < K_DECISION:
                colour, hatch = BLUE, None
            elif kind == "mv":
                colour, hatch = ORANGE, None
            else:
                colour, hatch = "white", "////"
            cell(x, colour, hatch=hatch, edge="white" if colour != "white" else GREY)
            x += CELL
        label = f"{name}\n20 samples" if not at_decision_point else name
        ax.text(start + N_SAMPLES * CELL / 2, y + height + 0.25, label, ha="center", va="bottom", fontsize=15)
        if at_decision_point:
            xd = start + K_DECISION * CELL
            ax.plot([xd, xd], [y - 0.25, y + height + 0.15], color=VERMILLION, lw=2.0)
        x += GAP
    # the quality
    x += GAP * 0.4
    cell(x, LIGHT if not at_decision_point else "white", edge=GREY)
    ax.text(x + CELL / 2, y + height + 0.25, "titer" if at_decision_point else "titer\n(measured\nat the end)",
            ha="center", va="bottom", fontsize=15)
    if at_decision_point:
        ax.text(x + CELL / 2, y - 0.35, "predicted", ha="center", va="top", fontsize=14, color=GREY)


def main(outdir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 6.4))
    total = N_Z + GAP + len(TAGS) * (N_SAMPLES + GAP) + GAP * 0.4 + CELL

    draw_row(ax, 5.2, at_decision_point=False)
    ax.text(-1.0, 5.2 + 0.5, "A training batch:\none complete row", ha="right", va="center", fontsize=16)

    draw_row(ax, 1.2, at_decision_point=True)
    ax.text(-1.0, 1.2 + 0.5, "A running batch at the\nday-4 decision point", ha="right", va="center", fontsize=16)
    ax.text(N_Z + GAP + K_DECISION * CELL, 0.55, "decision point", ha="center", va="top",
            fontsize=14, color=VERMILLION)

    handles = [
        Patch(facecolor=BLUE, edgecolor="white", label="known: pre-batch values and the samples recorded so far"),
        Patch(facecolor="white", edgecolor=GREY, hatch="////", label="missing: future samples of the responding tags, estimated by the model"),
        Patch(facecolor=ORANGE, edgecolor="white", label="the remaining schedule: nominal for the no-change prediction, free for the correction"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False,
              fontsize=14.5, ncol=1, handlelength=1.6)

    ax.set_xlim(-14, total + 0.5)
    ax.set_ylim(-1.9, 8.0)
    ax.axis("off")
    ax.set_title("One batch is one row of 111 columns; at a decision point, three kinds of column", pad=8)
    save(fig, outdir, "mcc-decision-point-row.png")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent)
