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
from batch_case_common import AQUA, BAND, DARK_BLUE, GREY, ORANGE, PALE_GREY, PURPLE, contribution_triptych, influence_plot, overlay_panels, save, score_plot, tag_panels

from process_improve.batch import BatchPCA, load_dupont

SPE_OUTLIER = 49
LAST_SIX = [50, 51, 52, 53, 54, 55]
ABOVE_SPE_LIMIT = [49, 51]  # a residual the two components cannot describe
ABOVE_T2_LIMIT = [50, 52, 53, 54, 55]  # extreme along the components themselves
FLAGGED = {**{b: AQUA for b in ABOVE_T2_LIMIT}, **{b: ORANGE for b in ABOVE_SPE_LIMIT}}
SECOND_CLUSTER = [37, 39, 43, 44, 45, 46, 47, 48]
CLUSTER_TAGS = ["TempC-1", "Press-3", "Press-2"]  # the three largest |t2| + |t3| contributions of the cluster
EARLY_WINDOW = 25  # samples 0 to 25 carry 66% of the cluster's t2 and 90% of its t3 contribution
RAW_WINDOW = 30
MEMBER_DOT = "0.35"  # darker than the axis grey, so the eight members read against the bars
POOR_QUALITY_NOT_VISIBLE = [38, 40, 41, 42]


def main(out_dir: pathlib.Path) -> None:
    batches = load_dupont()

    fig = overlay_panels(batches, ["TempC-1", "Press-1", "Flow-1", "TempR-1"], {SPE_OUTLIER: ORANGE, 54: AQUA})
    save(fig, out_dir, "batch-case-dupont-raw-trajectories")

    model_a = BatchPCA(n_components=2).fit(batches)
    fig = score_plot(model_a, highlight=FLAGGED, labels=LAST_SIX + [SPE_OUTLIER], title="Model A: 55 batches, two components")
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
    fig = score_plot(model_b, pc_horiz=2, pc_vert=3, highlight={b: ORANGE for b in SECOND_CLUSTER}, labels=SECOND_CLUSTER, title="Model B: batches 1 to 48, components 2 and 3")
    save(fig, out_dir, "batch-case-dupont-model-b-scores")

    # The columns are centred, so the model centre is the origin and a group's mean row is
    # its displacement from the centre. Contributions are linear in the row, so the mean of
    # the members' contributions is the contribution of the group mean, and it sums to the
    # group's mean score. That is what makes a whole cluster, rather than one representative
    # batch, the thing to plot here.
    scaled_b = model_b.unfold_and_scale(kept_b)
    per_component = {a: model_b.score_contributions(scaled_b, component=a) for a in (2, 3)}
    group = {a: c.loc[SECOND_CLUSTER].mean(axis=0) for a, c in per_component.items()}

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    per_tag = pd.DataFrame({a: group[a].groupby(level="tag", sort=False).sum() for a in (2, 3)})
    positions, width = np.arange(len(per_tag)), 0.38
    for offset, (component, colour) in zip((-width / 2, width / 2), ((2, DARK_BLUE), (3, ORANGE)), strict=True):
        axes[0].bar(positions + offset, per_tag[component].to_numpy(), width=width, color=colour, label=f"$t_{component}$", zorder=2)
        members = [per_component[component].loc[b].groupby(level="tag", sort=False).sum().to_numpy() for b in SECOND_CLUSTER]
        for row in members:
            axes[0].scatter(positions + offset, row, s=8, color=MEMBER_DOT, alpha=0.9, zorder=3, linewidths=0)
    axes[0].axhline(0, color=GREY, lw=0.8)
    axes[0].set_xticks(positions, [str(tag) for tag in per_tag.index], rotation=30, ha="right")
    axes[0].set_ylabel("Contribution to the score, summed per tag")
    axes[0].set_title("The cluster against the model centre, per tag")
    axes[0].legend(loc="upper left")
    axes[0].text(0.985, 0.965, "dots: the eight members", transform=axes[0].transAxes, ha="right", va="top", fontsize=8.5, color=MEMBER_DOT)
    for component, colour in ((2, DARK_BLUE), (3, ORANGE)):
        by_time = group[component].groupby(level="sequence").sum()
        axes[1].plot(by_time.index.to_numpy(), by_time.to_numpy(), color=colour, lw=1.5, label=f"$t_{component}$")
    axes[1].axhline(0, color=GREY, lw=0.8)
    axes[1].axvspan(0, EARLY_WINDOW, color=BAND, zorder=0, lw=0)
    axes[1].text(EARLY_WINDOW, 0.97, f" samples 0 to {EARLY_WINDOW}", transform=axes[1].get_xaxis_transform(), ha="left", va="top", fontsize=8.5, color=GREY)
    axes[1].set_xlabel("Sample [aligned time]")
    axes[1].set_ylabel("Contribution to the score, summed per sample")
    axes[1].set_title("... and when in the batch it happens")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, out_dir, "batch-case-dupont-group-contribution")

    # The same difference in the raw data. Over the whole batch it is under 2% of the panel
    # height for two of these three tags, so the panels are limited to the window the
    # contributions point at.
    others = [b for b in kept_b if b not in SECOND_CLUSTER]
    fig, axes = plt.subplots(1, len(CLUSTER_TAGS), figsize=(12.0, 3.7))
    for ax, tag in zip(axes, CLUSTER_TAGS, strict=True):
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
        ax.set_xlabel("Sample [aligned time]")
    axes[0].legend(loc="best")
    fig.tight_layout()
    save(fig, out_dir, "batch-case-dupont-group-raw")

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
