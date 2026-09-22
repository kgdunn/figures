"""Generate the committed PNGs for the DuPont batch PCA case study.

Mirrors the analysis in the pid-book chapter
``product-development-product-improvement/batch-case-study-dupont.rst``: a
batchwise-unfolded PCA on the 55 batches of the DuPont polymerization reactor,
the SPE and score outliers, the contribution plots that name the variables and
the time of an event, the two rebuilt models, the poor-quality batches the
trajectories cannot reveal, and batch 49 sample by sample under the
observation-wise, lagged and batchwise layouts of the model C reference set
(``batch-case-dupont-lagged-layout.png``). The chapter shows the equivalent
Plotly code; the committed figures are these matplotlib renderings.

Requires the ``process_improve`` package (``pip install 'process-improve[batch]'``,
version 1.82.0 or later) for ``BatchPCA`` and ``load_dupont``.

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
from batch_case_common import AQUA, DARK_BLUE, GOLD, GREY, MAGENTA, ORANGE, PALE_GREY, PURPLE, contribution_triptych, influence_plot, overlay_panels, save, score_plot, tag_panels, shade_alternate_tags

from process_improve.batch import BatchPCA, load_dupont

SPE_OUTLIER = 49
LAST_SIX = [50, 51, 52, 53, 54, 55]
ABOVE_SPE_LIMIT = [49, 51]  # a residual the two components cannot describe
ABOVE_T2_LIMIT = [50, 52, 53, 54, 55]  # extreme along the components themselves
FLAGGED = {**{b: AQUA for b in ABOVE_T2_LIMIT}, **{b: ORANGE for b in ABOVE_SPE_LIMIT}}
SECOND_CLUSTER = [37, 39, 43, 44, 45, 46, 47, 48]
# One colour and one marker for the cluster in every figure it appears in, so that it is recognised
# across the page; orange already stands for batch 49 and aqua for batches 50 to 55.
CLUSTER_COLOUR, CLUSTER_MARKER = PURPLE, "^"
ARROW = "0.3"  # the contribution direction drawn on the model B score plot
RAW_TAGS = ["TempC-1", "Press-3", "Press-2", "Flow-2"]  # the three largest |t2| + |t3| contributions of the cluster, and Flow-2
RAW_WINDOW = 30  # the raw panels stop here: samples 0 to 25 carry 66% of the cluster's t2 and 90% of its t3 contribution
RAW_49_TAGS = ["TempC-1", "TempH-1", "Press-2", "Flow-2"]  # the four largest shares of batch 49's SPE: 19, 14, 15 and 18%
RAW_49_EVENT = (56, 65)  # samples holding 80% of that SPE, marked on the panels
RAW_49_WINDOW = (40, 80)  # and the range the panels cover, enough either side of the event to read it
MEMBER_DOT = "0.25"  # edge colour of the member markers: white face and dark edge read on the bars and on the background
POOR_QUALITY_NOT_VISIBLE = [38, 40, 41, 42]
# Their own colour and shape as well: the two panels of the model C figure sit side by side, and
# orange there already means batch 49. Magenta clears the colour-vision and contrast checks against
# the four colours it shares the figure with.
POOR_QUALITY_COLOUR, POOR_QUALITY_MARKER = MAGENTA, "D"


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

    # The same four tags in the raw record. Zoomed: the deviation is large between batches but small
    # against the full trajectory range, so at full scale it cannot be seen.
    fig = overlay_panels(batches, RAW_49_TAGS, {SPE_OUTLIER: ORANGE, 54: AQUA},
                         vlines=RAW_49_EVENT, xlim=RAW_49_WINDOW)
    save(fig, out_dir, "batch-case-dupont-batch-49-raw")

    p1 = model_a.loadings_.iloc[:, 0].unstack(level="sequence").reindex(index=model_a.tag_names_)
    fig = tag_panels(p1, ylabel="$p_1$")
    fig.suptitle("Model A: loading $p_1$ over the batch, one panel per tag", y=1.02)
    save(fig, out_dir, "batch-case-dupont-loadings-p1")

    t1 = model_a.score_contributions(scaled, component=1)
    fig = contribution_triptych(t1.loc[54], what="Contribution to $t_1$", title="Batch 54: score contributions to $t_1$")
    save(fig, out_dir, "batch-case-dupont-batch-54-t1-contributions")

    kept_b = {b: batch for b, batch in batches.items() if b < SPE_OUTLIER}
    model_b = BatchPCA(n_components=3).fit(kept_b)
    fig = score_plot(model_b, pc_horiz=2, pc_vert=3, highlight={b: CLUSTER_COLOUR for b in SECOND_CLUSTER},
                     highlight_marker=CLUSTER_MARKER, labels=SECOND_CLUSTER,
                     legend_loc="lower left", title="Model B: batches 1 to 48, components 2 and 3")
    # The contributions below are the group's displacement from the model centre, so the arrow runs from the
    # origin out to the group's average point, with its label riding along it (the axes have equal scales, so
    # the data angle holds). Drawn the other way it would point against the quantity it labels.
    group_t2, group_t3 = model_b.scores_.loc[SECOND_CLUSTER].iloc[:, 1:3].mean()
    ax = fig.axes[0]
    ax.scatter(group_t2, group_t3, marker="s", s=42, color=ARROW, edgecolor="white", linewidth=0.8, zorder=6)
    ax.annotate("", xy=(group_t2, group_t3), xytext=(0, 0), zorder=5,
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
    shade_alternate_tags(ax_bars, len(per_tag))
    ax_bars.set_xticks(positions, [str(tag) for tag in per_tag.index], rotation=30, ha="right")
    ax_bars.set_ylabel("Contribution to the score, summed per tag")
    ax_bars.set_title("The cluster against the model centre, per tag")
    ax_bars.legend(loc="upper left")
    ax_bars.text(0.985, 0.965, "dots: the eight members", transform=ax_bars.transAxes, ha="right", va="top", fontsize=8.5, color=MEMBER_DOT)
    for ax, tag in zip(raw_axes, RAW_TAGS, strict=True):
        for b in others:
            ax.plot(kept_b[b][tag].to_numpy(), color=PALE_GREY, lw=0.8, zorder=1)
        for b in SECOND_CLUSTER:
            ax.plot(kept_b[b][tag].to_numpy(), color=CLUSTER_COLOUR, lw=1.0, alpha=0.9, zorder=3, label="the eight-batch group" if b == SECOND_CLUSTER[0] else None)
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
        "batch 49": ([SPE_OUTLIER], ORANGE, "o"),
        "batches 50 to 55": (list(range(50, 56)), AQUA, "o"),
        "the second group": (SECOND_CLUSTER, CLUSTER_COLOUR, CLUSTER_MARKER),
    }
    projected = {b: model_c.predict_online(batches[b], upto_k=model_c.n_timesteps_) for ids, _, _ in left_out.values() for b in ids}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    # 38 sits below batch 21, and 40, 41 and 42 sit together near the centre: the four labels are
    # placed away from the neighbour each would otherwise land on.
    score_plot(model_c, highlight={b: POOR_QUALITY_COLOUR for b in POOR_QUALITY_NOT_VISIBLE},
               highlight_marker=POOR_QUALITY_MARKER, labels=POOR_QUALITY_NOT_VISIBLE,
               title="Model C: 40 batches, scores", ax=axes[0])
    # The same four, with the same colour and shape, in the influence plot: the text reads their SPE
    # off this panel, and among 36 plain dots they are otherwise not findable.
    # Batch 38 sits in the thickest part of the training cloud, where no side of its marker is clear:
    # its label goes up into the empty band on a leader, the other three place themselves.
    influence_plot(model_c, highlight={b: POOR_QUALITY_COLOUR for b in POOR_QUALITY_NOT_VISIBLE},
                   highlight_marker=POOR_QUALITY_MARKER, labels=POOR_QUALITY_NOT_VISIBLE,
                   label_leader={38: (4, 30)},
                   title="Model C and the 15 batches left out of it", ax=axes[1])
    for label, (ids, colour, marker) in left_out.items():
        axes[1].scatter([float(projected[b].hotellings_t2) for b in ids], [float(projected[b].spe) for b in ids],
                        s=40 if marker == "o" else 52, color=colour, marker=marker,
                        edgecolor="white", linewidth=1, zorder=5, label=label)
    for b in (SPE_OUTLIER, 37):
        axes[1].annotate(str(b), (float(projected[b].hotellings_t2), float(projected[b].spe)), xytext=(6, 4), textcoords="offset points", fontsize=8.5)
    # Logarithmic axes: the projected batches sit up to two decades beyond the training cloud and its limits.
    all_t2 = [*model_c.hotellings_t2_.iloc[:, -1].tolist(), *(float(r.hotellings_t2) for r in projected.values())]
    all_spe = [*model_c.spe_.iloc[:, -1].tolist(), *(float(r.spe) for r in projected.values())]
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(min(all_t2) * 0.6, max(all_t2) * 2.0)
    # Extra room below the training cloud: it sits on the floor of the panel, and the four labels
    # there would otherwise crowd the axis.
    axes[1].set_ylim(min(all_spe) * 0.6, max(all_spe) * 1.6)
    axes[1].legend(loc="upper left", fontsize=8, title="projected onto model C", title_fontsize=8)  # lower right holds the SPE-limit label
    fig.tight_layout()
    save(fig, out_dir, "batch-case-dupont-model-c")

    lagged_layout_figure(batches, kept_c, model_c, out_dir)


LAGS = 2  # the lagged layout appends this many preceding samples to each sample's row
LAYOUT_WINDOW = (40, 80)  # the samples shown, the same window as the batch 49 raw panels
ROW_SHOWN = 65  # the sample whose lagged row is bracketed: it still holds sample 63, the last one displaced
LAYOUT_COLOURS = {"observation-wise": (GOLD, "s"), "lagged, 2 lags": ("0.4", "o"), "batchwise, so far": (DARK_BLUE, "^")}


def lagged_rows(table: np.ndarray, lags: int) -> np.ndarray:
    """One row per sample: its tags, then the same tags at each of the preceding ``lags`` samples."""
    n = len(table)
    return np.concatenate([table[lags - lag : n - lag] for lag in range(lags + 1)], axis=1)


def lagged_layout_figure(batches: dict, kept_c: dict, model_c: BatchPCA, out_dir: pathlib.Path) -> None:
    """Batch 49 sample by sample under the observation-wise, lagged and batchwise layouts of the model C reference set.

    The 40 reference batches are scaled as model C scales them, then arranged three ways: one row
    per sample (observation-wise), one row per sample with its two preceding samples appended
    (lagged), and one row per batch (batchwise, model C itself). A three-component PCA is fitted
    to each sample-wise layout and batch 49, which is in none of the reference sets, is run
    through all three; ``BatchMonitor`` gives model C a limit at every sample. The lower panel
    shows batch 49's SPE at every sample as a multiple of each layout's 95% limit.
    """
    from process_improve.batch import BatchMonitor
    from process_improve.multivariate import PCA

    def cells(group: dict) -> dict:
        wide = model_c.unfold_and_scale(group)
        return {b: wide.loc[b].unstack(level="tag").to_numpy() for b in wide.index}

    reference = cells(kept_c)
    sample_rows = {
        "observation-wise": (np.concatenate(list(reference.values())), 0),
        f"lagged, {LAGS} lags": (np.concatenate([lagged_rows(t, LAGS) for t in reference.values()]), LAGS),
    }
    table_49 = cells({SPE_OUTLIER: batches[SPE_OUTLIER]})[SPE_OUTLIER]
    ratio: dict[str, pd.Series] = {}
    for name, (rows, lags) in sample_rows.items():
        model = PCA(n_components=model_c.n_components).fit(pd.DataFrame(rows))
        spe = np.asarray(model.diagnose(pd.DataFrame(lagged_rows(table_49, lags))).spe, dtype=float)
        ratio[name] = pd.Series(spe / model.spe_limit(conf_level=0.95), index=np.arange(lags, len(table_49)))
    trace = BatchMonitor(model_c, conf_level=0.95).fit(kept_c).monitor(batches[SPE_OUTLIER])
    ratio["batchwise, so far"] = pd.Series(trace.spe / trace.spe_limit, index=trace.time - 1)  # 0-based samples, as the page counts them
    lo, hi = LAYOUT_WINDOW
    for name, r in ratio.items():
        inside = r.loc[lo:hi]
        print(f"{name}: above the limit at samples {inside.index[inside > 1].tolist()}")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True, gridspec_kw={"height_ratios": [1, 1.15]})
    tag = "TempC-1"
    for b, batch in kept_c.items():
        top.plot(batch[tag].to_numpy(), color=PALE_GREY, lw=0.7, zorder=1)
    top.plot([], [], color=PALE_GREY, lw=1.2, label="40 reference batches")
    top.plot(batches[SPE_OUTLIER][tag].to_numpy(), color=ORANGE, lw=1.8, zorder=3, label=f"batch {SPE_OUTLIER}")
    window = np.array([batch[tag].to_numpy()[lo : hi + 1] for batch in (*kept_c.values(), batches[SPE_OUTLIER])])
    pad = 0.05 * (window.max() - window.min())
    top.set_ylim(window.min() - pad, window.max() + 5 * pad)
    # The lagged row at sample 65 holds samples 63, 64 and 65: sample 63 is the last one batch 49's
    # early transition displaced, so the row is still flagged although the batch has rejoined the others.
    y_bracket = window.max() + 2.2 * pad
    held = (ROW_SHOWN - LAGS, ROW_SHOWN)
    top.plot([held[0] - 0.3, held[0] - 0.3, held[1] + 0.3, held[1] + 0.3],
             [y_bracket - 0.6 * pad, y_bracket, y_bracket, y_bracket - 0.6 * pad], color=GREY, lw=1.1, zorder=2)
    top.annotate(f"the lagged row at sample {ROW_SHOWN}:\nsamples {held[0]}, {held[0] + 1} and {held[1]}, all ten tags",
                 xy=(held[1] + 0.6, y_bracket), xytext=(held[1] + 1.2, y_bracket - 0.3 * pad), fontsize=8, color=GREY,
                 ha="left", va="center")
    top.set_title(tag)
    top.legend(loc="lower left", fontsize=8)
    for name, r in ratio.items():
        colour, marker = LAYOUT_COLOURS[name]
        inside = r.loc[lo:hi]
        bottom.plot(inside.index, inside.to_numpy(), color=colour, lw=1.4, marker=marker, ms=4, label=name)
    bottom.axhline(1.0, color=GREY, ls="--", lw=1.0)
    bottom.text(lo + 0.5, 1.25, "95% limit", color=GREY, fontsize=8, va="bottom")
    bottom.set_xlim(lo, hi)
    bottom.set_ylim(0, 11.5)
    bottom.set_xlabel("Sample [aligned time]")
    bottom.set_ylabel("SPE / limit")
    bottom.set_title("SPE of batch 49 as a multiple of its 95% limit")
    bottom.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, out_dir, "batch-case-dupont-lagged-layout")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent)
