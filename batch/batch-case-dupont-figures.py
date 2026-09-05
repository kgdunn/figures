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
from batch_case_common import AQUA, DARK_BLUE, GREY, ORANGE, contribution_triptych, influence_plot, overlay_panels, save, score_plot, tag_panels

from process_improve.batch import BatchPCA, load_dupont

SPE_OUTLIER = 49
LAST_SIX = [50, 51, 52, 53, 54, 55]
ABOVE_SPE_LIMIT = [49, 51]  # a residual the two components cannot describe
ABOVE_T2_LIMIT = [50, 52, 53, 54, 55]  # extreme along the components themselves
FLAGGED = {**{b: AQUA for b in ABOVE_T2_LIMIT}, **{b: ORANGE for b in ABOVE_SPE_LIMIT}}
SECOND_CLUSTER = [37, 39, 43, 44, 45, 46, 47, 48]
POOR_QUALITY_NOT_VISIBLE = [38, 40, 41, 42]


def main(out_dir: pathlib.Path) -> None:
    batches = load_dupont()

    fig = overlay_panels(batches, ["TempC-1", "Press-1", "Flow-1", "TempR-1"], {SPE_OUTLIER: ORANGE, 54: DARK_BLUE})
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

    t3 = model_b.score_contributions(model_b.unfold_and_scale(kept_b), component=3)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
    by_tag = t3.loc[39].groupby(level="tag", sort=False).sum()
    axes[0].bar(range(len(by_tag)), by_tag.to_numpy(), color=DARK_BLUE, width=0.6)
    axes[0].set_xticks(range(len(by_tag)), [str(t) for t in by_tag.index], rotation=30, ha="right")
    axes[0].axhline(0, color="0.55", lw=0.8)
    axes[0].set_ylabel("Contribution to $t_3$, summed per tag")
    axes[0].set_title("Batch 39: contributions to $t_3$")
    for b, batch in batches.items():
        if b not in SECOND_CLUSTER:
            axes[1].plot(batch["Press-3"].to_numpy(), color="0.82", lw=0.7)
    for b in SECOND_CLUSTER:
        axes[1].plot(batches[b]["Press-3"].to_numpy(), color=ORANGE, lw=1.0, label="batches 37, 39, 43 to 48" if b == 37 else None)
    axes[1].plot(batches[39]["Press-3"].to_numpy(), color=DARK_BLUE, lw=1.8, label="batch 39")
    axes[1].set_xlabel("Sample [aligned time]")
    axes[1].set_title("Press-3: the second cluster runs a different pressure profile")
    axes[1].legend(loc="best")
    fig.tight_layout()
    save(fig, out_dir, "batch-case-dupont-batch-39")

    excluded = set(range(SPE_OUTLIER, 56)) | set(SECOND_CLUSTER)
    kept_c = {b: batch for b, batch in batches.items() if b not in excluded}
    model_c = BatchPCA(n_components=3).fit(kept_c)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    score_plot(model_c, highlight={b: ORANGE for b in POOR_QUALITY_NOT_VISIBLE}, labels=POOR_QUALITY_NOT_VISIBLE, title="Model C: 40 batches, scores", ax=axes[0])
    influence_plot(model_c, highlight={b: ORANGE for b in POOR_QUALITY_NOT_VISIBLE}, labels=POOR_QUALITY_NOT_VISIBLE, title="Model C: Hotelling's $T^2$ against SPE", ax=axes[1])
    fig.tight_layout()
    save(fig, out_dir, "batch-case-dupont-model-c")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent)
