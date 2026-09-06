"""Generate the committed PNGs for the FMC multiblock batch PLS case study.

Mirrors the analysis in the pid-book chapter
``product-development-product-improvement/batch-case-study-fmc.rst``: the
ladder of two-component models on the FMC batch dryer data (PCA on quality,
PLS from the initial conditions, multiblock PLS on both initial-condition
blocks, batch PCA and batch PLS on the trajectories, and the batch multiblock
PLS that joins all three blocks, with ``ClockTime`` as the eleventh
trajectory), and the four batches classed good whose trajectories sit with
the abnormal batches. The chapter shows the equivalent Plotly code; the
committed figures are these matplotlib renderings.

Requires the ``process_improve`` package (``pip install 'process-improve[batch]'``,
version 1.81 or later) for ``load_fmc``, ``dict_to_wide`` and ``MBPLS``.

Usage::

    python batch/batch-case-fmc-figures.py [output_dir]

``output_dir`` defaults to this script's own directory (``batch/``). The data
are downloaded from https://openmv.net/file/batch-dryer.xlsx.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from batch_case_common import (
    AQUA,
    DARK_BLUE,
    GREY,
    GOLD,
    ORANGE,
    PALE_GREY,
    PURPLE,
    compact_legend,
    contribution_triptych,
    group_scatter,
    influence_plot,
    label_bars,
    overlay_panels,
    parity_plot,
    phase_lines,
    save,
    score_plot,
    shade_alternate_tags,
    tag_panels,
)
from matplotlib.lines import Line2D

from process_improve.batch import dict_to_wide, load_fmc
from process_improve.multivariate import PCA, PLS, MCUVScaler
from process_improve.multivariate.methods import MBPLS

OPERATING_OUTLIER = 20
QUALITY_GROUP = [61, 14]
TRAJECTORY_BATCHES = [13, 5, 7]
DISPOSITION = {"good": 33, "abnormal": 61, "high solvent": 71}  # the plant's classes: the last batch number of each
DISPOSITION_STYLES = {"good": (DARK_BLUE, "o"), "abnormal": (PURPLE, "^"), "high solvent": (GOLD, "s")}  # colour, marker
PHASE_ENDS = (175, 249)  # the first sample of the high-speed phase and the sample of the peak temperature
RAW_TAGS = ["CTankLvl", "ClockTime", "D-Temp", "D-Temp-SP"]  # the raw trajectories shown beside the Zop contributions


def grouped_bars(ax, table, *, colours: list[str], ylabel: str, title: str) -> None:
    """One group of bars per row of ``table``, one bar per column, on alternately shaded positions."""
    n_rows, n_cols = table.shape
    width = 0.8 / n_cols
    x = np.arange(n_rows)
    for j, column in enumerate(table.columns):
        ax.bar(x + (j - (n_cols - 1) / 2) * width, table[column].to_numpy(dtype=float), width=width, color=colours[j], label=str(column), zorder=2)
    ax.set_xticks(x, [str(name) for name in table.index], rotation=30, ha="right")
    ax.axhline(0, color=GREY, lw=0.8)
    shade_alternate_tags(ax, n_rows)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")


def weight_plot(ax, weights: pd.DataFrame, *, title: str, label_left: tuple[str, ...] = ()) -> None:
    """The weights of the first two components of one block, one labelled point per variable.

    ``label_left`` names the variables whose label goes to the lower left of the point instead of the upper
    right, for a label that would otherwise sit on a neighbour's.
    """
    w1, w2 = weights.iloc[:, 0].to_numpy(dtype=float), weights.iloc[:, 1].to_numpy(dtype=float)
    ax.scatter(w1, w2, s=40, color=DARK_BLUE, edgecolor="white", linewidth=1, zorder=3)
    for name, x, y in zip(weights.index, w1, w2, strict=True):
        left = str(name) in label_left
        ax.annotate(str(name), (x, y), xytext=(-4, -4) if left else (4, 3), textcoords="offset points",
                    ha="right" if left else "left", va="top" if left else "bottom", fontsize=8.5)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("block weight $w_1$")
    ax.set_ylabel("block weight $w_2$")
    ax.set_title(title)


def main(out_dir: pathlib.Path) -> None:
    fmc = load_fmc()
    keep = [b for b in fmc.batch_ids if b not in fmc.missing_chemistry]
    X = {b: fmc.X[b] for b in keep}
    Y, Zop, Zchem = fmc.Y.loc[keep], fmc.Zop.loc[keep], fmc.Zchem.loc[keep]
    groups = pd.Series(pd.cut(keep, bins=[0, *DISPOSITION.values()], labels=list(DISPOSITION)).astype(str), index=keep)
    coded = {"groups": groups, "group_styles": DISPOSITION_STYLES}  # colour and marker by the plant's disposition

    fig = overlay_panels(X, ["D-Temp", "J-Temp", "CTankLvl", "ClockTime"], {OPERATING_OUTLIER: ORANGE}, vlines=PHASE_ENDS)
    save(fig, out_dir, "batch-case-fmc-raw-trajectories")

    y_scaled = MCUVScaler().fit_transform(Y)
    pca_y = PCA(n_components=2).fit(y_scaled)
    contributions = pca_y.score_contributions(y_scaled, component=1).loc[QUALITY_GROUP].T
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    score_plot(pca_y, highlight={61: ORANGE, 14: AQUA}, labels=QUALITY_GROUP, title="PCA on the quality block: scores", legend_loc="upper left", ax=axes[0], **coded)
    grouped_bars(axes[1], contributions.rename(columns=lambda b: f"batch {b}"), colours=[ORANGE, AQUA], ylabel="Contribution to $t_1$", title="Contributions of one batch from each group")
    fig.tight_layout()
    save(fig, out_dir, "batch-case-fmc-quality-pca")

    mb_z = MBPLS(n_components=2).fit({"Zchem": Zchem, "Zop": Zop}, Y)
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.6))
    ss = mb_z.super_scores_
    group_scatter(axes[0, 0], ss.iloc[:, 0], ss.iloc[:, 1], {OPERATING_OUTLIER: ORANGE}, highlight_size=200, **coded)
    axes[0, 0].annotate("20", (ss.loc[OPERATING_OUTLIER].iloc[0], ss.loc[OPERATING_OUTLIER].iloc[1]), xytext=(6, 4), textcoords="offset points", fontsize=8.5)
    r2y = mb_z.r2_y_per_component_.to_numpy()
    axes[0, 0].set_xlabel(f"super score $t_1$ [$R^2_Y$ {r2y[0]:.1%}]")
    axes[0, 0].set_ylabel(f"super score $t_2$ [$R^2_Y$ {r2y[1]:.1%}]")
    axes[0, 0].set_title("Super scores: batch 20 (orange) at the lower left")
    compact_legend(axes[0, 0], "upper left")
    weights = mb_z.super_weights_.copy()
    weights.columns = [f"component {c}" for c in weights.columns]
    grouped_bars(axes[1, 0], weights, colours=[DARK_BLUE, ORANGE], ylabel="super weight", title="Super weights: how much each block pulls")
    titles = {"Zchem": ("Chemistry block scores: batch 20 inside the cloud", "Chemistry block weights"), "Zop": ("Operating-condition block scores: batch 20 far outside", "Operating-condition block weights")}
    for col, (name, block_scores) in enumerate(mb_z.block_scores_.items(), start=1):
        ax = axes[0, col]
        group_scatter(ax, block_scores.iloc[:, 0], block_scores.iloc[:, 1], {OPERATING_OUTLIER: ORANGE}, highlight_size=200, **coded)
        ax.annotate("20", (block_scores.loc[OPERATING_OUTLIER].iloc[0], block_scores.loc[OPERATING_OUTLIER].iloc[1]), xytext=(6, 4), textcoords="offset points", fontsize=8.5)
        r2 = np.diff([0.0, *mb_z.r2_x_per_block_cumulative_.loc[name].to_numpy(dtype=float)])
        ax.set_xlabel(f"block score $t_1$ [{r2[0]:.1%}]")
        ax.set_ylabel(f"block score $t_2$ [{r2[1]:.1%}]")
        ax.set_title(titles[name][0])
        weight_plot(axes[1, col], mb_z.block_weights_[name], title=titles[name][1], label_left=("Time1",))
    for ax in axes[0]:
        ax.axhline(0, color=GREY, lw=0.8)
        ax.axvline(0, color=GREY, lw=0.8)
    fig.tight_layout()
    save(fig, out_dir, "batch-case-fmc-mbpls-z")

    wide = dict_to_wide(X)  # ten process tags plus ClockTime, the eleventh trajectory
    x_scaled = MCUVScaler().fit_transform(wide)
    x_scaled.columns = wide.columns
    pca_x = PCA(n_components=2).fit(x_scaled)
    squared = pca_x.spe_contributions(x_scaled) ** 2  # NaN only at the missing cells; the rest from the observed cells
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    score_plot(pca_x, highlight={OPERATING_OUTLIER: ORANGE}, labels=[OPERATING_OUTLIER], title="Batch PCA on the trajectories: scores", legend_loc="upper left", ax=axes[0], **coded)
    influence_plot(pca_x, highlight={OPERATING_OUTLIER: ORANGE}, labels=[OPERATING_OUTLIER, 41, 51], title="Batch PCA: Hotelling's $T^2$ against SPE", legend_loc="center right", ax=axes[1], **coded)
    fig.tight_layout()
    save(fig, out_dir, "batch-case-fmc-batch-pca")

    tags = list(X[keep[0]].columns)
    p1 = pca_x.loadings_.iloc[:, 0].unstack(level="sequence").reindex(index=tags)
    r2_cell = pd.Series(pca_x.r2_per_variable_.iloc[:, -1].to_numpy(dtype=float), index=x_scaled.columns)
    r2_grid = r2_cell.unstack(level="sequence").reindex(index=tags)
    fig = tag_panels(p1, ylabel="$p_1$", ncols=4, secondary=r2_grid, secondary_label="$R^2$ per cell, two components", vlines=PHASE_ENDS)
    fig.suptitle("Batch PCA: loading $p_1$ (blue) and $R^2$ per cell (orange) over the batch; the phases end at samples 175 and 249", y=1.02)
    save(fig, out_dir, "batch-case-fmc-loadings-p1")

    spe_share = squared.div(squared.sum(axis=1), axis=0) * 100
    share_20 = spe_share.loc[OPERATING_OUTLIER].fillna(0.0)  # a missing cell has no residual: drawn as an empty position
    fig = contribution_triptych(share_20, what="Share of SPE [%]", title=f"Batch {OPERATING_OUTLIER}: share of the SPE carried by each (tag, time) cell", vlines=PHASE_ENDS)
    save(fig, out_dir, f"batch-case-fmc-batch-{OPERATING_OUTLIER}-spe-contributions")

    pls_x = PLS(n_components=2, scale=False).fit(x_scaled, y_scaled)
    t1 = pls_x.score_contributions(x_scaled, component=1)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    score_plot(pls_x, highlight={13: ORANGE, 5: AQUA, 7: AQUA}, labels=TRAJECTORY_BATCHES, title="Batch PLS to quality: scores", legend_loc="upper left", ax=axes[0], **coded)
    by_tag = t1.loc[13].groupby(level="tag", sort=False).sum()
    axes[1].bar(range(len(by_tag)), by_tag.to_numpy(), color=DARK_BLUE, width=0.6)
    axes[1].set_xticks(range(len(by_tag)), [str(t) for t in by_tag.index], rotation=30, ha="right")
    axes[1].axhline(0, color=GREY, lw=0.8)
    axes[1].set_ylabel("Contribution to $t_1$, summed per tag")
    axes[1].set_title("Batch 13: contributions to $t_1$")
    fig.tight_layout()
    save(fig, out_dir, "batch-case-fmc-batch-pls")
    fig = overlay_panels(X, ["D-Temp", "CTankLvl", "ClockTime", "J-Temp-SP"], {13: ORANGE, 5: AQUA, 7: DARK_BLUE}, vlines=PHASE_ENDS)
    save(fig, out_dir, "batch-case-fmc-raw-batches-13-5-7")

    blocks = {"Zchem": Zchem, "Zop": Zop, "X": wide}
    mb = MBPLS(n_components=2).fit(blocks, Y)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), gridspec_kw={"width_ratios": [1, 1, 1]})
    ss = mb.super_scores_
    group_scatter(axes[0], ss.iloc[:, 0], ss.iloc[:, 1], {13: ORANGE, 5: AQUA, 7: AQUA}, **coded)
    for b in TRAJECTORY_BATCHES:
        axes[0].annotate(str(b), (ss.loc[b].iloc[0], ss.loc[b].iloc[1]), xytext=(4, 4), textcoords="offset points", fontsize=8.5)
    compact_legend(axes[0], "upper left")
    axes[0].axhline(0, color=GREY, lw=0.8)
    axes[0].axvline(0, color=GREY, lw=0.8)
    r2y = mb.r2_y_per_component_.to_numpy()
    axes[0].set_xlabel(f"super score $t_1$ [$R^2_Y$ {r2y[0]:.1%}]")
    axes[0].set_ylabel(f"super score $t_2$ [$R^2_Y$ {r2y[1]:.1%}]")
    axes[0].set_title("Batch multiblock PLS: super scores")
    summary = mb.r2_x_per_block_cumulative_.iloc[:, -1].to_frame("$R^2_X$ after two components")
    summary["super VIP"] = mb.super_vip_
    grouped_bars(axes[1], summary, colours=[DARK_BLUE, ORANGE], ylabel="", title="Per block: $R^2_X$ and super VIP")
    observed = Y["SolventConc"].dropna()
    parity_plot(observed, mb.predictions_["SolventConc"].loc[observed.index], highlight={13: ORANGE}, ax=axes[2], title="SolventConc: observed and fitted")
    fig.tight_layout()
    save(fig, out_dir, "batch-case-fmc-batch-mbpls")

    # Off-spec trajectories, on-spec product: place every batch, block by block, with the nearer group average
    def nearer_group(scores: pd.DataFrame) -> pd.Series:
        centres = {name: scores.loc[groups == name].mean() for name in ("good", "abnormal")}
        return pd.DataFrame({name: ((scores - centre) ** 2).sum(axis=1) for name, centre in centres.items()}).idxmin(axis=1)

    placed = pd.DataFrame({name: nearer_group(scores) for name, scores in mb.block_scores_.items()})
    anomalous = [b for b in keep if groups[b] == "good" and placed.loc[b, "X"] == "abnormal" and (placed.loc[b, ["Zchem", "Zop"]] == "good").all()]
    x_scores = mb.block_scores_["X"]
    abnormal_scores = x_scores.loc[groups == "abnormal"]
    neighbours = sorted({int(b) for a in anomalous for b in ((abnormal_scores - x_scores.loc[a]) ** 2).sum(axis=1).nsmallest(2).index})
    print(f"anomalous {anomalous}; nearest abnormal neighbours {neighbours}")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3))
    label_offsets = {2: (5, 4), 3: (-5, 4), 6: (5, -4), 7: (5, 4)}  # keep the four labels off each other in the X block
    for ax, (name, scores) in zip(axes, mb.block_scores_.items(), strict=True):
        group_scatter(ax, scores.iloc[:, 0], scores.iloc[:, 1], dict.fromkeys(anomalous, ORANGE), highlight_size=170, **coded)
        ax.scatter([], [], s=170, color=ORANGE, marker="o", edgecolor="white", linewidth=1, label="the four batches (classed good)")
        for b in anomalous:
            dx, dy = label_offsets.get(b, (5, 4))
            ax.annotate(str(b), (scores.loc[b].iloc[0], scores.loc[b].iloc[1]), xytext=(dx, dy), textcoords="offset points", ha="right" if dx < 0 else "left", va="top" if dy < 0 else "bottom", fontsize=8.5, zorder=6)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.axvline(0, color=GREY, lw=0.8)
        r2 = np.diff([0.0, *mb.r2_x_per_block_cumulative_.loc[name].to_numpy(dtype=float)])
        ax.set_xlabel(f"block score $t_1$ [{r2[0]:.1%}]")
        ax.set_ylabel(f"block score $t_2$ [{r2[1]:.1%}]")
        ax.set_title(f"{name} block")
    handles, labels = axes[0].get_legend_handles_labels()  # one legend for the three panels, in a row below them
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out_dir, "batch-case-fmc-block-scores")

    contributions = mb.score_contributions(blocks, component=1)["Zop"]
    move = contributions.loc[anomalous].mean() - contributions.loc[neighbours].mean()
    fig = plt.figure(figsize=(13.0, 5.8), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1])
    ax = fig.add_subplot(grid[:, 0])
    ax.bar(range(len(move)), move.to_numpy(dtype=float), color=DARK_BLUE, width=0.6, zorder=2)
    ax.set_xticks(range(len(move)), [str(name) for name in move.index], rotation=30, ha="right")
    ax.axhline(0, color=GREY, lw=0.8)
    shade_alternate_tags(ax, len(move))
    label_bars(ax, move.to_numpy(dtype=float), fmt="{:.2f}", floor=0.005)
    ax.set_ylabel("Contribution to the Zop block score $t_1$")
    ax.set_title("From the neighbours' average to the four batches' average")
    highlight = {**dict.fromkeys(neighbours, AQUA), **dict.fromkeys(anomalous, ORANGE)}
    for k, tag in enumerate(RAW_TAGS):
        ax = fig.add_subplot(grid[k // 2, 1 + k % 2])
        phase_lines(ax, PHASE_ENDS)
        for b, batch in X.items():
            if b not in highlight:
                ax.plot(batch[tag].to_numpy(), color=PALE_GREY, lw=0.7, zorder=1)
        for b, colour in highlight.items():
            ax.plot(X[b][tag].to_numpy(), color=colour, lw=1.4, zorder=3 if colour == ORANGE else 2)
        ax.set_title(tag)
        if k >= 2:
            ax.set_xlabel("Sample [aligned time]")
    handles = [Line2D([], [], color=ORANGE, lw=1.8, label="the four batches (classed good)"), Line2D([], [], color=AQUA, lw=1.8, label="their nearest abnormal batches")]
    fig.axes[1].legend(handles=handles, loc="upper left", fontsize=8)
    save(fig, out_dir, "batch-case-fmc-anomalous")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent)
