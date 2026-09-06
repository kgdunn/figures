"""Generate the committed PNGs for the DuPont batch PCA case study.

Mirrors the analysis in the pid-book chapter
``product-development-product-improvement/batch-case-study-dupont.rst``: a
batchwise-unfolded PCA on the 55 batches of the DuPont polymerization reactor,
the SPE and score outliers, the contribution plots that name the variables and
the time of an event, the two rebuilt models, and the poor-quality batches the
trajectories cannot reveal. The chapter shows the equivalent Plotly code; the
committed figures are these matplotlib renderings.

Requires the ``process_improve`` package (``pip install 'process-improve[batch]'``,
version 1.79 or later) for ``BatchPCA`` and ``load_dupont``.

Usage::

    python batch/batch-case-dupont-figures.py [output_dir]

``output_dir`` defaults to this script's own directory (``batch/``). The data are
downloaded from https://openmv.net/file/polymerization.csv.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from batch_case_common import AQUA, DARK_BLUE, GREY, ORANGE, PALE_GREY, PURPLE, contribution_triptych, influence_plot, overlay_panels, save, score_plot, tag_panels

from process_improve.batch import BatchPCA, load_dupont

SPE_OUTLIER = 49
LAST_SIX = [50, 51, 52, 53, 54, 55]
ABOVE_SPE_LIMIT = [49, 51]  # a residual the two components cannot describe
ABOVE_T2_LIMIT = [50, 52, 53, 54, 55]  # extreme along the components themselves
FLAGGED = {**{b: AQUA for b in ABOVE_T2_LIMIT}, **{b: ORANGE for b in ABOVE_SPE_LIMIT}}
SECOND_CLUSTER = [37, 39, 43, 44, 45, 46, 47, 48]
ARROW = "0.3"  # the contribution direction drawn on the model B score plot
RAW_TAGS = ["TempC-1", "Press-3", "Press-2", "Flow-2"]  # the three largest |t2| + |t3| contributions of the cluster, and Flow-2
RAW_WINDOW = 30  # the raw panels stop here: samples 0 to 25 carry 66% of the cluster's t2 and 90% of its t3 contribution
MEMBER_DOT = "0.25"  # edge colour of the member markers: white face and dark edge read on the bars and on the background
POOR_QUALITY_NOT_VISIBLE = [38, 40, 41, 42]


def main(out_dir: pathlib.Path) -> None:
    batches = load_dupont()

    fig = overlay_panels(batches, ["TempC-1", "Press-1", "Flow-1", "TempR-1"], {SPE_OUTLIER: ORANGE, 54: AQUA})
    save(fig, out_dir, "batch-case-dupont-raw-trajectories")

    model_a = BatchPCA(n_components=2).fit(batches)
    # Batch 49 sits inside the central cloud, so its label is pulled clear of the points on a leader line.
    fig = score_plot(model_a, highlight=FLAGGED, labels=LAST_SIX + [SPE_OUTLIER], label_leader={SPE_OUTLIER: (-16, -12)},
                     title="Model A: 55 batches, two components")
    save(fig, out_dir, "batch-case-dupont-model-a-scores")
    fig = influence_plot(model_a, highlight=FLAGGED, labels=ABOVE_SPE_LIMIT + ABOVE_T2_LIMIT, title="Model A: Hotelling's $T^2$ against SPE")
    save(fig, out_dir, "batch-case-dupont-model-a-influence")

    scaled = model_a.unfold_and_scale(batches)
    squared = model_a.spe_contributions(scaled) ** 2
    spe_share = squared.div(squared.sum(axis=1), axis=0) * 100  # each cell's share of the batch's SPE, in percent
    fig = contribution_triptych(spe_share.loc[SPE_OUTLIER], what="Share of SPE [%]", title="Batch 49: share of the SPE carried by each (tag, time) cell")
    save(fig, out_dir, "batch-case-dupont-batch-49-spe-contributions")

    p1 = model_a.loadings_.iloc[:, 0].unstack(level="sequence").reindex(index=model_a.tag_names_)
    fig = tag_panels(p1, ylabel="$p_1$")
    fig.suptitle("Model A: loading $p_1$ over the batch, one panel per tag", y=1.02)
    save(fig, out_dir, "batch-case-dupont-loadings-p1")

    t1 = model_a.score_contributions(scaled, component=1)
    fig = contribution_triptych(t1.loc[54], what="Contribution to $t_1$", title="Batch 54: score contributions to $t_1$")
    save(fig, out_dir, "batch-case-dupont-batch-54-t1-contributions")

    kept_b = {b: batch for b, batch in batches.items() if b < SPE_OUTLIER}
    model_b = BatchPCA(n_components=3).fit(kept_b)
    fig = score_plot(model_b, pc_horiz=2, pc_vert=3, highlight={b: ORANGE for b in SECOND_CLUSTER}, labels=SECOND_CLUSTER,
                     label_left=(39, 47), legend_loc="lower left", title="Model B: batches 1 to 48, components 2 and 3")
    # The contributions below compare the group with the model centre: draw that direction, from the group's average
    # point to the origin, with its label riding along the arrow (the axes have equal scales, so the data angle holds).
    group_t2, group_t3 = model_b.scores_.loc[SECOND_CLUSTER].iloc[:, 1:3].mean()
    ax = fig.axes[0]
    ax.scatter(group_t2, group_t3, marker="s", s=42, color=ARROW, edgecolor="white", linewidth=0.8, zorder=6)
    ax.annotate("", xy=(0, 0), xytext=(group_t2, group_t3), zorder=5,
                arrowprops={"arrowstyle": "-|>", "lw": 2.5, "color": ARROW, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 18})
    ax.annotate("contribution direction", (group_t2 / 2, group_t3 / 2), xytext=(-3, 4), textcoords="offset points",
                rotation=np.degrees(np.arctan2(group_t3, group_t2)), rotation_mode="anchor", ha="center", va="bottom",
                fontsize=8, color=ARROW, zorder=5)
    save(fig, out_dir, "batch-case-dupont-model-b-scores")

    # The columns are centred, so the model centre is the origin and a group's mean row is
    # its displacement from the centre. Contributions are linear in the row, so the mean of
    # the members' contributions is the contribution of the group mean, and it sums to the
    # group's mean score. That is what makes a whole cluster, rather than one representative
    # batch, the thing to plot here.
    scaled_b = model_b.unfold_and_scale(kept_b)
    per_component = {a: model_b.score_contributions(scaled_b, component=a) for a in (2, 3)}
    group = {a: c.loc[SECOND_CLUSTER].mean(axis=0) for a, c in per_component.items()}

    # Left: the group's contribution per tag, each member as a dot. Right: the raw trajectories of the three
    # tags with the largest contributions and of Flow-2, over the window the contributions point at (over the
    # whole batch the difference is under 2% of the panel height for two of these tags).
    others = [b for b in kept_b if b not in SECOND_CLUSTER]
    fig = plt.figure(figsize=(13.0, 5.8), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1])
    ax_bars = fig.add_subplot(grid[:, 0])
    raw_axes = [fig.add_subplot(grid[row, col]) for row in (0, 1) for col in (1, 2)]
    per_tag = pd.DataFrame({a: group[a].groupby(level="tag", sort=False).sum() for a in (2, 3)})
    positions, width = np.arange(len(per_tag)), 0.38
    for offset, (component, colour) in zip((-width / 2, width / 2), ((2, DARK_BLUE), (3, ORANGE)), strict=True):
        ax_bars.bar(positions + offset, per_tag[component].to_numpy(), width=width, color=colour, label=f"$t_{component}$", zorder=2)
        members = [per_component[component].loc[b].groupby(level="tag", sort=False).sum().to_numpy() for b in SECOND_CLUSTER]
        for row in members:
            ax_bars.scatter(positions + offset, row, s=16, facecolor="white", edgecolor=MEMBER_DOT, linewidth=0.8, zorder=3)
    ax_bars.axhline(0, color=GREY, lw=0.8)
    ax_bars.set_xticks(positions, [str(tag) for tag in per_tag.index], rotation=30, ha="right")
    ax_bars.set_ylabel("Contribution to the score, summed per tag")
    ax_bars.set_title("The cluster against the model centre, per tag")
    ax_bars.legend(loc="upper left")
    ax_bars.text(0.985, 0.965, "dots: the eight members", transform=ax_bars.transAxes, ha="right", va="top", fontsize=8.5, color=MEMBER_DOT)
    for ax, tag in zip(raw_axes, RAW_TAGS, strict=True):
        for b in others:
            ax.plot(kept_b[b][tag].to_numpy(), color=PALE_GREY, lw=0.8, zorder=1)
        for b in SECOND_CLUSTER:
            ax.plot(kept_b[b][tag].to_numpy(), color=ORANGE, lw=1.0, alpha=0.9, zorder=3, label="the eight-batch group" if b == SECOND_CLUSTER[0] else None)
        ax.plot([], [], color=PALE_GREY, lw=1.4, label=f"the other {len(others)} batches")
        ax.set_xlim(0, RAW_WINDOW)
        values = np.concatenate([kept_b[b][tag].to_numpy()[: RAW_WINDOW + 1] for b in kept_b])
        pad = 0.06 * np.ptp(values)
        ax.set_ylim(values.min() - pad, values.max() + pad)
        ax.set_title(f"{tag}, samples 0 to {RAW_WINDOW}")
    for ax in raw_axes[2:]:
        ax.set_xlabel("Sample [aligned time]")
    raw_axes[0].legend(loc="best")
    save(fig, out_dir, "batch-case-dupont-group-contribution")

    excluded = set(range(SPE_OUTLIER, 56)) | set(SECOND_CLUSTER)
    kept_c = {b: batch for b, batch in batches.items() if b not in excluded}
    model_c = BatchPCA(n_components=3).fit(kept_c)
    # The 15 batches left out of model C, projected onto it: the on-line projection at the last
    # sample of a complete batch gives its scores, T2 and SPE against model C's centre and scale.
    left_out = {
        "batch 49": ([SPE_OUTLIER], ORANGE),
        "batches 50 to 55": (list(range(50, 56)), AQUA),
        "the second group": (SECOND_CLUSTER, PURPLE),
    }
    projected = {b: model_c.predict_online(batches[b], upto_k=model_c.n_timesteps_) for ids, _ in left_out.values() for b in ids}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    score_plot(model_c, highlight={b: ORANGE for b in POOR_QUALITY_NOT_VISIBLE}, labels=POOR_QUALITY_NOT_VISIBLE, title="Model C: 40 batches, scores", ax=axes[0])
    influence_plot(model_c, title="Model C and the 15 batches left out of it", ax=axes[1])
    for label, (ids, colour) in left_out.items():
        axes[1].scatter([float(projected[b].hotellings_t2) for b in ids], [float(projected[b].spe) for b in ids],
                        s=40, color=colour, edgecolor="white", linewidth=1, zorder=5, label=label)
    for b in (SPE_OUTLIER, 37):
        axes[1].annotate(str(b), (float(projected[b].hotellings_t2), float(projected[b].spe)), xytext=(6, 4), textcoords="offset points", fontsize=8.5)
    # Logarithmic axes: the projected batches sit up to two decades beyond the training cloud and its limits.
    all_t2 = [*model_c.hotellings_t2_.iloc[:, -1].tolist(), *(float(r.hotellings_t2) for r in projected.values())]
    all_spe = [*model_c.spe_.iloc[:, -1].tolist(), *(float(r.spe) for r in projected.values())]
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(min(all_t2) * 0.6, max(all_t2) * 2.0)
    axes[1].set_ylim(min(all_spe) * 0.8, max(all_spe) * 1.6)
    axes[1].legend(loc="upper left", fontsize=8, title="projected onto model C", title_fontsize=8)  # lower right holds the SPE-limit label
    fig.tight_layout()
    save(fig, out_dir, "batch-case-dupont-model-c")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent)
