"""Generate the boards (2x6 lumber) monitoring figures for the PID book exercise.

The exercise builds a Shewhart chart on the ``six-point-board-thickness``
dataset. Each board carries six thickness measurements (``Pos1`` ... ``Pos6``)
across its width; the single monitored value per board is the **median** of
those six positions. Subgroups are seven consecutive boards; phase 1 is boards
1 to 500, phase 2 is boards 501 to 2000.

This script replaces ``boards-monitoring-assignment4-2010.R``, whose data
source (``board-thickness.csv``, a single-column file) is no longer available
on openmv.net.

Usage
-----
    python monitoring/boards_monitoring_figures.py [output_dir]

Writes five PNGs into ``output_dir`` (default: this script's own directory):
``boards-monitoring-raw-data.png``,
``boards-monitoring-find-outliers-phase1.png``,
``boards-monitoring-Shewhart-phase1.png``,
``boards-monitoring-Shewhart-phase2.png`` and
``boards-monitoring-subgroup-standard-deviation.png``.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gamma
from statsmodels.nonparametric.smoothers_lowess import lowess

URL = "https://openmv.net/file/six-point-board-thickness.csv"
POSITIONS = ["Pos1", "Pos2", "Pos3", "Pos4", "Pos5", "Pos6"]
N_SUB = 7
N_PHASE1 = 500
N_PHASE2_END = 2000
DPI = 300


def a_n(n: int) -> float:
    """Unbiasing constant c4 for a subgroup of size ``n``."""
    return float(np.sqrt(2) * gamma(n / 2) / (np.sqrt(n - 1) * gamma((n - 1) / 2)))


def board_thickness() -> np.ndarray:
    """Return one thickness per board: the median of the six positions."""
    boards = pd.read_csv(URL)
    return boards[POSITIONS].median(axis=1).to_numpy()


def subgroup_stats(thickness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-subgroup means and standard deviations (subgroups of N_SUB)."""
    n_groups = len(thickness) // N_SUB
    grouped = thickness[: n_groups * N_SUB].reshape(-1, N_SUB).T
    return grouped.mean(axis=0), grouped.std(axis=0, ddof=1)


def limits(xbar: np.ndarray, sd: np.ndarray) -> tuple[float, float, float]:
    """Return (LCL, target, UCL) three-sigma Shewhart limits."""
    target = xbar.mean()
    half = 3 * sd.mean() / (a_n(N_SUB) * np.sqrt(N_SUB))
    return target - half, target, target + half


def _save(fig: plt.Figure, out: pathlib.Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=DPI)
    plt.close(fig)


def main() -> None:
    """Generate all five boards figures into the output directory."""
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)

    thickness = board_thickness()
    xbar, sd = subgroup_stats(thickness)
    phase1_end = N_PHASE1 // N_SUB       # 71 subgroups
    phase2_end = N_PHASE2_END // N_SUB   # 285 subgroups

    # Round 2 limits: drop any phase-1 subgroup outside the round-1 limits.
    lcl1, _, ucl1 = limits(xbar[:phase1_end], sd[:phase1_end])
    keep = (xbar[:phase1_end] >= lcl1) & (xbar[:phase1_end] <= ucl1)
    lcl, target, ucl = limits(xbar[:phase1_end][keep], sd[:phase1_end][keep])

    # 1. All raw data.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, len(thickness) + 1), thickness, ".", ms=2, color="black")
    ax.set_title("Thickness: all data")
    ax.set_xlabel("Board number")
    ax.set_ylabel("Board thickness (median of 6 positions)")
    _save(fig, out, "boards-monitoring-raw-data.png")

    # 2. Phase 1 raw data (boards 1 to 500).
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, N_PHASE1 + 1), thickness[:N_PHASE1], ".", ms=3, color="black")
    ax.set_title("Phase 1 raw data (boards 1 to 500)")
    ax.set_xlabel("Board number")
    ax.set_ylabel("Board thickness")
    _save(fig, out, "boards-monitoring-find-outliers-phase1.png")

    # 3. Phase 1 Shewhart chart (subgroup means, with the final limits).
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, phase1_end + 1), xbar[:phase1_end], "-o", ms=3, color="black")
    for y, c in [(ucl, "red"), (lcl, "red"), (target, "green")]:
        ax.axhline(y, color=c)
    ax.set_title("Shewhart chart for phase I: training data")
    ax.set_xlabel("Subgroup number")
    ax.set_ylabel("Subgroup mean")
    _save(fig, out, "boards-monitoring-Shewhart-phase1.png")

    # 4. Phase 2 Shewhart chart (subgroups 72 to 285, boards ~501 to 2000).
    x2 = xbar[phase1_end:phase2_end]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(phase1_end + 1, phase2_end + 1), x2, "-o", ms=3, color="black")
    for y, c in [(ucl, "red"), (lcl, "red"), (target, "green")]:
        ax.axhline(y, color=c)
    ax.set_title("Shewhart chart for phase II: testing data")
    ax.set_xlabel("Subgroup number")
    ax.set_ylabel("Subgroup mean")
    _save(fig, out, "boards-monitoring-Shewhart-phase2.png")

    # 5. Subgroup standard deviation over time, with a lowess trend.
    idx = np.arange(1, phase2_end + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, sd[:phase2_end], ".", ms=4, color="black")
    smooth = lowess(sd[:phase2_end], idx, frac=0.3, return_sorted=True)
    ax.plot(smooth[:, 0], smooth[:, 1], color="red", lw=2)
    ax.set_title("Slow increase in subgroup variability over the day")
    ax.set_xlabel("Subgroup number")
    ax.set_ylabel("Subgroup standard deviation")
    _save(fig, out, "boards-monitoring-subgroup-standard-deviation.png")


if __name__ == "__main__":
    main()
