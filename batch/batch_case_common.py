"""Shared style and plot helpers for the three batch case-study figure scripts.

Used by ``batch-case-dupont-figures.py``, ``batch-case-sbr-figures.py`` and
``batch-case-fmc-figures.py`` in this directory, which write the PNGs for the
batch case-study pages of the pid-book Applications chapter. This module writes
no figures itself.

The colours are the book's house colours (the same ones the adaptive
soft-sensor figures use), checked as a categorical set with the data-viz
palette validator: dark blue for the main series, orange for the batch under
discussion, aqua for a second highlighted batch, grey for everything else.
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

DARK_BLUE = "#1f3d7a"
ORANGE = "#c55a11"
AQUA = "#1baf7a"
GREY = "0.55"
PALE_GREY = "0.82"
BAND = "#e9edf4"  # alternate shading of the tag blocks in a contribution vector
FIGSIZE_WIDE = (9.0, 3.6)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10.5,
        "axes.titleweight": "normal",
        "legend.frameon": False,
        "legend.fontsize": 9,
    }
)


def save(fig: Figure, out_dir: pathlib.Path, name: str) -> None:
    """Write ``name.png`` into ``out_dir`` and close the figure."""
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def overlay_panels(
    batches: dict,
    tags: list[str],
    highlight: dict[int, str],
    *,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    xlabel: str = "Sample [aligned time]",
) -> Figure:
    """One panel per tag: every batch in grey, the highlighted batches in colour."""
    nrows = int(np.ceil(len(tags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (4.5 * ncols, 2.6 * nrows), squeeze=False)
    for ax, tag in zip(axes.ravel(), tags, strict=False):
        for batch_id, batch in batches.items():
            if batch_id not in highlight:
                ax.plot(batch[tag].to_numpy(), color=PALE_GREY, lw=0.7, zorder=1)
        for batch_id, colour in highlight.items():
            ax.plot(batches[batch_id][tag].to_numpy(), color=colour, lw=1.8, label=f"batch {batch_id}", zorder=3)
        ax.set_title(tag)
        ax.set_xlabel(xlabel)
    for ax in axes.ravel()[len(tags) :]:
        ax.set_visible(False)
    axes.ravel()[0].legend(loc="best")
    fig.tight_layout()
    return fig


def score_plot(
    model,
    *,
    pc_horiz: int = 1,
    pc_vert: int = 2,
    highlight: dict[int, str] | None = None,
    labels: list[int] | None = None,
    conf_level: float = 0.95,
    title: str = "",
    ax=None,
) -> Figure:
    """Scores on two components with the Hotelling's T2 ellipse; selected batches coloured and labelled."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
    else:
        fig = ax.figure
    scores = model.scores_
    x, y = scores.iloc[:, pc_horiz - 1], scores.iloc[:, pc_vert - 1]
    ex, ey = model.ellipse_coordinates(score_horiz=pc_horiz, score_vert=pc_vert, conf_level=conf_level)
    ax.plot(ex, ey, color=GREY, lw=1, ls="--", label=f"{conf_level:.0%} confidence ellipse")
    highlight = highlight or {}
    others = [b for b in scores.index if b not in highlight]
    ax.scatter(x.loc[others], y.loc[others], s=28, color=DARK_BLUE, edgecolor="white", linewidth=1, zorder=3)
    for batch_id, colour in highlight.items():
        ax.scatter(x.loc[batch_id], y.loc[batch_id], s=46, color=colour, edgecolor="white", linewidth=1, zorder=4)
    for batch_id in labels or []:
        ax.annotate(str(batch_id), (x.loc[batch_id], y.loc[batch_id]), xytext=(4, 4), textcoords="offset points", fontsize=8.5)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel(f"$t_{pc_horiz}$")
    ax.set_ylabel(f"$t_{pc_vert}$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right")
    return fig


def influence_plot(
    model,
    *,
    highlight: dict[int, str] | None = None,
    labels: list[int] | None = None,
    conf_level: float = 0.95,
    title: str = "",
    ax=None,
) -> Figure:
    """Hotelling's T2 against SPE for every batch, with both limits drawn.

    Each batch is one dot: how far it sits *along* the model's components
    (Hotelling's T2, horizontal) against how far it sits *away* from them (SPE,
    vertical). The two limits split the plot into quadrants, which separates a
    batch that is extreme in a direction the model knows from one the model
    cannot describe. Plotting either statistic against the batch number instead
    would put the order the batches happen to appear in on an axis, which
    carries no information about the batch.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
    else:
        fig = ax.figure
    t2 = model.hotellings_t2_.iloc[:, -1]
    spe = model.spe_.iloc[:, -1]
    t2_limit = float(model.hotellings_t2_limit(conf_level=conf_level))
    spe_limit = float(model.spe_limit(conf_level=conf_level))
    highlight = highlight or {}

    x_max = max(float(t2.max()), t2_limit) * 1.14
    y_min, y_max = float(spe.min()) * 0.94, max(float(spe.max()), spe_limit) * 1.06
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)

    ax.axvline(t2_limit, color=GREY, ls="--", lw=1)
    ax.axhline(spe_limit, color=GREY, ls="--", lw=1)
    ax.text(t2_limit, 0.99, f" {conf_level:.0%} limit", transform=ax.get_xaxis_transform(), va="top", ha="left", fontsize=8.5, color=GREY)
    ax.text(0.995, spe_limit, f"{conf_level:.0%} limit", transform=ax.get_yaxis_transform(), va="bottom", ha="right", fontsize=8.5, color=GREY)

    others = [batch_id for batch_id in t2.index if batch_id not in highlight]
    ax.scatter(t2.loc[others], spe.loc[others], s=30, color=DARK_BLUE, edgecolor="white", linewidth=1, zorder=3)
    for batch_id, colour in highlight.items():
        ax.scatter(t2.loc[batch_id], spe.loc[batch_id], s=52, color=colour, edgecolor="white", linewidth=1, zorder=4)
    for batch_id in labels or []:
        to_the_left = float(t2.loc[batch_id]) > 0.82 * x_max
        ax.annotate(
            str(batch_id),
            (t2.loc[batch_id], spe.loc[batch_id]),
            xytext=(-6, 5) if to_the_left else (6, 5),
            textcoords="offset points",
            ha="right" if to_the_left else "left",
            fontsize=8.5,
        )

    ax.set_xlabel("Hotelling's $T^2$")
    ax.set_ylabel("SPE")
    ax.set_title(title)
    return fig


def _tag_blocks(row: pd.Series) -> list[tuple[str, int, int]]:
    """Return (tag, start, stop) for each contiguous tag block of an unfolded row (positions are 0-based)."""
    tags = row.index.get_level_values("tag").to_numpy()
    blocks: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(tags) + 1):
        if i == len(tags) or tags[i] != tags[start]:
            blocks.append((str(tags[start]), start, i))
            start = i
    return blocks


def contribution_vector(row: pd.Series, *, ax, ylabel: str, colour: str = DARK_BLUE) -> None:
    """Draw one batch's contribution vector over the unfolded (tag, time) axis with shaded tag blocks."""
    values = row.to_numpy(dtype=float)
    positions = np.arange(len(values))
    for k, (tag, start, stop) in enumerate(_tag_blocks(row)):
        if k % 2 == 1:
            ax.axvspan(start - 0.5, stop - 0.5, color=BAND, zorder=0, lw=0)
        ax.text((start + stop - 1) / 2, 1.02, tag, ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
    ax.bar(positions, values, width=1.0, color=colour, lw=0, zorder=2)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlim(-0.5, len(values) - 0.5)
    ax.set_xticks([])
    ax.set_xlabel("Unfolded (tag, time) cells, one block per tag")
    ax.set_ylabel(ylabel)
    ax.grid(False, axis="x")


def contribution_triptych(row: pd.Series, *, what: str, title: str) -> Figure:
    """The full contribution vector, its sum per tag, and its sum per time sample, in three panels."""
    by_tag = row.groupby(level="tag", sort=False).sum()
    by_time = row.groupby(level="sequence").sum()
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), gridspec_kw={"height_ratios": [1.3, 1, 1]})
    contribution_vector(row, ax=axes[0], ylabel=what)
    axes[0].set_title(title, pad=18)
    axes[1].bar(range(len(by_tag)), by_tag.to_numpy(), color=DARK_BLUE, width=0.6, zorder=2)
    axes[1].set_xticks(range(len(by_tag)), [str(tag) for tag in by_tag.index], rotation=20, ha="right")
    axes[1].axhline(0, color=GREY, lw=0.8)
    axes[1].set_ylabel(f"{what}, summed per tag")
    axes[2].bar(by_time.index.to_numpy(), by_time.to_numpy(), width=1.0, color=DARK_BLUE, lw=0, zorder=2)
    axes[2].axhline(0, color=GREY, lw=0.8)
    axes[2].set_xlabel("Sample [aligned time]")
    axes[2].set_ylabel(f"{what}, summed per sample")
    fig.tight_layout()
    return fig


def tag_panels(grid: pd.DataFrame, *, ylabel: str, ncols: int = 5, colour: str = DARK_BLUE, second: pd.DataFrame | None = None, second_label: str = "", first_label: str = "") -> Figure:
    """Small multiples: one panel per tag (row of ``grid``), the values over time; an optional second grid overlaid."""
    tags = list(grid.index)
    nrows = int(np.ceil(len(tags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.0 * nrows), sharex=True, sharey=True, squeeze=False)
    for ax, tag in zip(axes.ravel(), tags, strict=False):
        ax.plot(grid.columns.to_numpy(), grid.loc[tag].to_numpy(), color=colour, lw=1.4, label=first_label)
        if second is not None:
            ax.plot(second.columns.to_numpy(), second.loc[tag].to_numpy(), color=ORANGE, lw=1.4, label=second_label)
        ax.axhline(0, color=GREY, lw=0.7)
        ax.set_title(str(tag))
    for ax in axes.ravel()[len(tags) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Sample")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    if second is not None:
        axes.ravel()[0].legend(loc="best")
    fig.tight_layout()
    return fig


def parity_plot(observed: pd.Series, predicted: pd.Series, *, highlight: dict[int, str], ax, title: str) -> None:
    """Observed against fitted values with the y = x line; selected batches coloured and labelled."""
    others = [b for b in observed.index if b not in highlight]
    ax.scatter(observed.loc[others], predicted.loc[others], s=26, color=DARK_BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    for batch_id, colour in highlight.items():
        ax.scatter(observed.loc[batch_id], predicted.loc[batch_id], s=46, color=colour, edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(str(batch_id), (observed.loc[batch_id], predicted.loc[batch_id]), xytext=(4, 4), textcoords="offset points", fontsize=8.5)
    lo, hi = float(min(observed.min(), predicted.min())), float(max(observed.max(), predicted.max()))
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=1, ls="--", label="y = x")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Fitted")
    ax.set_title(title)
    ax.legend(loc="upper left")
