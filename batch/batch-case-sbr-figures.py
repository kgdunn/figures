"""Generate the committed PNGs for the SBR batch PLS case study.

Mirrors the analysis in the pid-book chapter
``product-development-product-improvement/batch-case-study-sbr.rst``: a
batchwise-unfolded PLS from six trajectories of the simulated
styrene-butadiene rubber reactor to five quality attributes, the score and SPE
plots that flag batches 34 and 37, the weights, the contribution plots for
each faulty batch, and the observed-versus-fitted quality. The chapter shows
the equivalent Plotly code; the committed figures are these matplotlib
renderings.

Requires the ``process_improve`` package (``pip install 'process-improve[batch]'``,
version 1.79 or later) for ``BatchPLS`` and ``load_sbr``.

Usage::

    python batch/batch-case-sbr-figures.py [output_dir] [--data-url URL]

``output_dir`` defaults to this script's own directory (``batch/``). The data
are downloaded from https://openmv.net/file/sbr-batch-reactor.xlsx; pass
``--data-url file:///path/to/sbr-batch-reactor.xlsx`` to use a local copy.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from batch_case_common import AQUA, BAND, DARK_BLUE, GREY, ORANGE, contribution_triptych, influence_plot, overlay_panels, parity_plot, save, score_plot, tag_panels

from process_improve.batch import BatchPLS, load_sbr

FAULT_FROM_START = 37
FAULT_PARTWAY = 34
HIGHLIGHT = {FAULT_PARTWAY: ORANGE, FAULT_FROM_START: AQUA}
SPE_OUTLIERS = [8, 15, 16]  # flagged by the SPE, and not the batches carrying the injected fault


def main(out_dir: pathlib.Path, data_url: str | None) -> None:
    sbr = load_sbr(url=data_url)
    trajectories = {batch_id: batch[sbr.trajectory_tags] for batch_id, batch in sbr.X.items()}
    quality = sbr.Y

    fig = overlay_panels(trajectories, sbr.trajectory_tags, HIGHLIGHT, ncols=3)
    save(fig, out_dir, "batch-case-sbr-raw-trajectories")

    model = BatchPLS(n_components=2).fit(trajectories, quality)
    save(score_plot(model, highlight=HIGHLIGHT, labels=list(HIGHLIGHT), title="Batch PLS: scores of the 53 batches"), out_dir, "batch-case-sbr-scores")
    save(influence_plot(model, highlight=HIGHLIGHT, labels=[*HIGHLIGHT, *SPE_OUTLIERS], title="Batch PLS: Hotelling's $T^2$ against SPE"), out_dir, "batch-case-sbr-influence")

    r2_grid = model.r2_per_variable_.iloc[:, -1].unstack(level="sequence").reindex(index=model.tag_names_)
    fig = tag_panels(r2_grid, ylabel="$R^2$", ncols=3)
    fig.suptitle("$R^2$ of every (tag, time) cell after two components", y=1.02)
    save(fig, out_dir, "batch-case-sbr-r2-over-time")

    w1 = model.x_weights_.iloc[:, 0].unstack(level="sequence").reindex(index=model.tag_names_)
    w2 = model.x_weights_.iloc[:, 1].unstack(level="sequence").reindex(index=model.tag_names_)
    fig = tag_panels(w1, ylabel="weight", ncols=3, second=w2, first_label="$w_1$", second_label="$w_2$")
    fig.suptitle("Time-varying weights of the two components", y=1.02)
    save(fig, out_dir, "batch-case-sbr-weights")

    scaled = model.unfold_and_scale(trajectories)
    t1 = model.score_contributions(scaled, component=1)
    t2 = model.score_contributions(scaled, component=2)
    save(contribution_triptych(t1.loc[FAULT_FROM_START], what="Contribution to $t_1$", title="Batch 37: contributions to $t_1$"), out_dir, "batch-case-sbr-batch-37-contributions")
    save(contribution_triptych(t2.loc[FAULT_PARTWAY], what="Contribution to $t_2$", title="Batch 34: contributions to $t_2$"), out_dir, "batch-case-sbr-batch-34-contributions")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    for ax, variable in zip(axes, ["Composition", "ParticleSize"], strict=True):
        parity_plot(quality[variable], model.predictions_[variable], highlight=HIGHLIGHT, ax=ax, title=variable)
    fig.tight_layout()
    save(fig, out_dir, "batch-case-sbr-observed-vs-fitted")

    # When does each trajectory of the two faulty batches leave the band of the other batches?
    # One panel per tag and batch: z, the signed deviation from the other batches' mean in their
    # standard deviations. Signed rather than absolute, so the direction of the departure shows.
    others = np.stack([batch.to_numpy() for key, batch in trajectories.items() if key not in HIGHLIGHT])
    mean, sd = others.mean(axis=0), others.std(axis=0, ddof=1)
    fig, axes = plt.subplots(2, len(sbr.trajectory_tags), figsize=(13.0, 4.4), sharex=True, sharey=True)
    for row, batch_id in enumerate([FAULT_FROM_START, FAULT_PARTWAY]):
        z = (trajectories[batch_id].to_numpy() - mean) / sd
        for j, tag in enumerate(sbr.trajectory_tags):
            ax = axes[row, j]
            ax.axhspan(-2, 2, color=BAND, zorder=0, lw=0)
            ax.plot(z[:, j], lw=1.2, color=HIGHLIGHT[batch_id], zorder=2)
            ax.axhline(0, color=GREY, lw=0.8)
            for level in (-2, 2):
                ax.axhline(level, color="0.55", lw=0.8, ls="--")
            if row == 0:
                ax.set_title(tag)
            if j == 0:
                ax.set_ylabel(f"batch {batch_id}\nz [sd of the others]")
            if row == 1:
                ax.set_xlabel("Sample")
    for level, label in ((2, "+2 sd"), (-2, "-2 sd")):
        axes[0, -1].text(0.99, level, label, transform=axes[0, -1].get_yaxis_transform(), ha="right",
                         va="bottom" if level > 0 else "top", fontsize=8.5)
    fig.suptitle("Distance of batches 37 (top) and 34 (bottom) from the other batches, tag by tag", y=1.01)
    fig.tight_layout()
    save(fig, out_dir, "batch-case-sbr-departure")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dir", nargs="?", type=pathlib.Path, default=pathlib.Path(__file__).parent)
    parser.add_argument("--data-url", default=None)
    args = parser.parse_args()
    main(args.output_dir, args.data_url)
