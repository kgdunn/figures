"""Generate the committed PNGs for the SBR batch PLS case study.

Mirrors the analysis in the pid-book chapter
``product-development-product-improvement/batch-case-study-sbr.rst``: a
batchwise-unfolded PLS from six trajectories of the simulated
styrene-butadiene rubber reactor to five quality attributes, the score and SPE
plots that flag batches 34 and 37, the weights, the contribution plots for
each faulty batch, and the observed-versus-fitted quality. The last four
figures ask what the model would have shown while a batch was still running:
how the error of the evolving quality prediction falls as more of the batch
is observed, that prediction for the near-average batch 4, per-sample T2 and
SPE charts of the two faulty batches against a reference model of the normal
batches, and the model's forecast of the rest of each faulty batch. The
chapter shows the equivalent Plotly code; the committed figures are these
matplotlib renderings.

Requires the ``process_improve`` package (``pip install 'process-improve[batch]'``,
version 1.80 or later) for ``BatchPLS``, ``BatchMonitor`` and ``load_sbr``. The
leave-one-batch-out sweep behind the prediction-error figure refits the model
53 times and takes about a minute; everything else runs in seconds.

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
import pandas as pd
from batch_case_common import (
    AQUA,
    BAND,
    DARK_BLUE,
    GREY,
    MAGENTA,
    ORANGE,
    PALE_GREY,
    PURPLE,
    contribution_triptych,
    influence_plot,
    label_bars,
    online_chart,
    overlay_panels,
    parity_plot,
    save,
    score_plot,
    shade_alternate_tags,
    tag_panels,
)
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from process_improve.batch import BatchMonitor, BatchPLS, load_sbr
from process_improve.univariate import median_absolute_deviation

FAULT_FROM_START = 37
FAULT_PARTWAY = 34
HIGHLIGHT = {FAULT_PARTWAY: ORANGE, FAULT_FROM_START: AQUA}
SPE_OUTLIERS = [8, 15, 16]  # flagged by the SPE, and not the batches carrying the injected fault
AVERAGE_BATCH = 4  # the batch whose trajectories lie closest to the average
ATTRIBUTE_COLOURS = {
    "Composition": DARK_BLUE,
    "ParticleSize": ORANGE,
    "Branching": AQUA,
    "CrossLinking": PURPLE,
    "Polydispersity": MAGENTA,
}
REPORT_SAMPLES = [10, 50, 100, 150, 200]
FIRST_SAMPLE_SHOWN = 5  # the score estimate from fewer samples than this is too poor to be worth drawing
MONITOR_CONF_LEVEL = 0.99
ALARM_RUN = 3  # consecutive samples above the limit before an alarm counts
FAULT_SAMPLE = 100  # the impurity enters batch 34 at about this sample
EWMA_LAMBDA = 0.3  # smoothing of the robust departure chart, the value the book's EWMA chapter uses
TRANSIENT_END = 15  # the cooling-water temperature settles after this many samples; the forecast panel starts here
FORECASTS = {FAULT_FROM_START: ("Conversion", [30, 60]), FAULT_PARTWAY: ("CoolingTemp", [60, 115])}


def first_sustained_alarm(alarm: np.ndarray, run: int = ALARM_RUN) -> int | None:
    """Return the 1-based sample after which a statistic first stays above its limit for ``run`` samples in a row."""
    runs = np.convolve(np.asarray(alarm, dtype=int), np.ones(run, dtype=int), mode="valid") == run
    return int(np.argmax(runs)) + 1 if runs.any() else None


def leave_one_batch_out_rmse(trajectories: dict, quality: pd.DataFrame, samples: list[int]) -> pd.DataFrame:
    """RMSEP of the evolving prediction: refit the model without each batch and predict that batch as it runs.

    ``online_rmse`` on the training batches gives the estimation error (RMSEE);
    this pools the same per-sample squared error over the 53 held-out
    predictions instead. Only the rows in ``samples`` are kept, since the
    53 refits are what take the time, not the rows.
    """
    squared = pd.DataFrame(0.0, index=pd.Index(samples, name="upto_k"), columns=quality.columns)
    for batch_id, batch in trajectories.items():
        others = {other: trajectory for other, trajectory in trajectories.items() if other != batch_id}
        model_without = BatchPLS(n_components=2).fit(others, quality.loc[list(others)])
        squared += model_without.online_rmse({batch_id: batch}, quality.loc[[batch_id]]).loc[samples] ** 2
    return np.sqrt(squared / len(trajectories))


def rule_label(ax, value: float, text: str, *, x: float, above: bool, colour: str) -> None:
    """Write ``text`` just above or below the horizontal rule at ``value``, at axis fraction ``x``."""
    ax.annotate(
        text,
        (x, value),
        xytext=(0, 3 if above else -3),
        textcoords="offset points",
        xycoords=("axes fraction", "data"),
        ha="left",
        va="bottom" if above else "top",
        fontsize=8.5,
        color=colour,
        zorder=5,
    )


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
    # One panel per tag and batch, two signed distances from the other batches (signed rather than
    # absolute, so the direction of the departure shows): dashed, the deviation from the others'
    # mean in their standard deviations; solid, a robust version with the others' median as centre
    # and 1.4826 times their median absolute deviation as scale (the factor makes the MAD equal the
    # standard deviation of normal values, and a batch among the others that is itself unusual at a
    # sample does not widen the band), smoothed with an EWMA (lambda = 0.3, the value the book's
    # EWMA chapter uses; pandas starts the average at the first sample's value).
    others = np.stack([batch.to_numpy() for key, batch in trajectories.items() if key not in HIGHLIGHT])
    mean, sd = others.mean(axis=0), others.std(axis=0, ddof=1)
    centre, spread = np.median(others, axis=0), median_absolute_deviation(others, axis=0, scale="normal")
    fig, axes = plt.subplots(2, len(sbr.trajectory_tags), figsize=(13.0, 4.4), sharex=True, sharey=True)
    for row, batch_id in enumerate([FAULT_FROM_START, FAULT_PARTWAY]):
        z = (trajectories[batch_id].to_numpy() - mean) / sd
        z_robust = pd.DataFrame((trajectories[batch_id].to_numpy() - centre) / spread).ewm(alpha=EWMA_LAMBDA, adjust=False).mean().to_numpy()
        for j, tag in enumerate(sbr.trajectory_tags):
            ax = axes[row, j]
            ax.axhspan(-2, 2, color=BAND, zorder=0, lw=0)
            ax.plot(z[:, j], lw=1.0, ls="--", color=HIGHLIGHT[batch_id], alpha=0.8, zorder=2, label="mean and sd")
            ax.plot(z_robust[:, j], lw=1.4, color=HIGHLIGHT[batch_id], zorder=3, label="robust, smoothed")
            ax.axhline(0, color=GREY, lw=0.8)
            for level in (-2, 2):
                ax.axhline(level, color="0.55", lw=0.8, ls=":")
            if row == 0:
                ax.set_title(tag)
            if j == 0:
                ax.set_ylabel(f"batch {batch_id}")  # the suptitle names the quantity; a longer label collides between the rows
            if row == 1:
                ax.set_xlabel("Sample")
    axes[0, 0].legend(loc="lower right", fontsize=8)  # lower left is where this trace starts, at -6
    for level, label in ((2, "+2"), (-2, "-2")):
        # white backing so the label reads over the traces, which end near -2 in this panel
        axes[0, -1].text(0.99, level, label, transform=axes[0, -1].get_yaxis_transform(), ha="right",
                         va="bottom" if level > 0 else "top", fontsize=8.5, zorder=5,
                         bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.85})
    fig.suptitle(
        "Distance of batches 37 (top) and 34 (bottom) from the other batches, tag by tag: "
        "robust (median, MAD, EWMA; solid) and mean-and-sd (dashed)",
        y=1.01,
    )
    fig.tight_layout()
    save(fig, out_dir, "batch-case-sbr-departure")

    # -- On-line prediction: how the error of the evolving quality prediction falls as the batch is observed.
    # RMSEE from the 53-batch model on its own training batches (solid), RMSEP with each batch held out of the
    # fit in turn (dashed, every fifth sample to keep the 53 refits near a minute), both relative to the
    # attribute's standard deviation over the 53 batches: a ratio of 1 is the error of predicting the average.
    sd = quality.std(ddof=1)
    rmsee = model.online_rmse(trajectories, quality)
    loo_samples = list(range(10, model.n_timesteps_ + 1, 5))
    rmsep = leave_one_batch_out_rmse(trajectories, quality, loo_samples)
    for attribute in ("ParticleSize", "Composition"):
        print(
            f"{attribute}: RMSEE / sd after "
            + ", ".join(f"{k} samples {rmsee.loc[k, attribute] / sd[attribute]:.2f}" for k in REPORT_SAMPLES)
        )
        print(
            f"{attribute}: leave-one-batch-out RMSEP / sd after "
            + ", ".join(f"{k} samples {rmsep.loc[k, attribute] / sd[attribute]:.2f}" for k in REPORT_SAMPLES)
        )
    rmsee_ratio, rmsep_ratio = (rmsee / sd).loc[10:], rmsep / sd
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.axhline(1.0, color=GREY, lw=1, zorder=1)
    ax.text(0.99, 1.0, "as good as the average batch", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8.5, color=GREY)
    for attribute, colour in ATTRIBUTE_COLOURS.items():
        # Branching and CrossLinking coincide; the wider line underneath lets both colours show.
        lw = 2.8 if attribute == "Branching" else 1.5
        ax.plot(rmsee_ratio.index, rmsee_ratio[attribute].to_numpy(), color=colour, lw=lw, zorder=3)
        ax.plot(rmsep_ratio.index, rmsep_ratio[attribute].to_numpy(), color=colour, lw=lw, ls="--", zorder=3)
    swatch = {attribute: Line2D([], [], color=colour, lw=2) for attribute, colour in ATTRIBUTE_COLOURS.items()}
    handles = [
        swatch["Composition"],
        swatch["ParticleSize"],
        (swatch["Branching"], swatch["CrossLinking"]),
        swatch["Polydispersity"],
        Line2D([], [], color=GREY, lw=1.5),
        Line2D([], [], color=GREY, lw=1.5, ls="--"),
    ]
    labels = [
        "Composition",
        "ParticleSize",
        "Branching, CrossLinking (coincide)",
        "Polydispersity",
        "RMSEE: model fitted on all 53 batches",
        "RMSEP: each batch held out of the fit in turn",
    ]
    ax.legend(handles, labels, loc="upper right", handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)},
              handlelength=3.2)
    ax.set_xlim(0, model.n_timesteps_ + 2)
    ax.set_ylim(0, None)
    ax.set_xlabel("Samples observed")
    ax.set_ylabel("RMSE / standard deviation of the attribute")
    ax.set_title("How the mid-batch prediction error falls as the batch is observed")
    fig.tight_layout()
    save(fig, out_dir, "batch-case-sbr-online-rmse")

    # -- The evolving prediction for the near-average batch 4: ParticleSize, and the attribute whose measured value
    # and final prediction differ the most (in standard deviations), so that the two rules are visibly apart.
    trace = model.predict_online_trace(trajectories[AVERAGE_BATCH])
    final, measured = model.predictions_.loc[AVERAGE_BATCH], quality.loc[AVERAGE_BATCH]
    gap = ((final - measured).abs() / sd).drop("ParticleSize")
    second = str(gap.idxmax())
    print(f"batch {AVERAGE_BATCH}: |final prediction - measured| / sd = " + ", ".join(f"{a} {g:.3f}" for a, g in gap.items()))
    print(
        f"batch {AVERAGE_BATCH}: ParticleSize measured {measured['ParticleSize']:.1f}, final prediction "
        f"{final['ParticleSize']:.1f}; second panel {second}: measured {measured[second]:.4g}, final prediction "
        f"{final[second]:.4g}, sd {sd[second]:.3g}"
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    for ax, attribute in zip(axes, ["ParticleSize", second], strict=True):
        y_hat = trace.y_hat[attribute].loc[FIRST_SAMPLE_SHOWN:]
        half_width = 2 * rmsee[attribute].loc[FIRST_SAMPLE_SHOWN:]
        ax.fill_between(y_hat.index, (y_hat - half_width).to_numpy(), (y_hat + half_width).to_numpy(), color=BAND,
                        lw=0, zorder=1, label="$\\pm 2$ RMSEE after this many samples")
        ax.plot(y_hat.index, y_hat.to_numpy(), color=DARK_BLUE, lw=1.6, zorder=3, label="predicted from the samples so far")
        ax.axhline(final[attribute], color=GREY, lw=1, ls="--", zorder=2)
        ax.axhline(measured[attribute], color="black", lw=1, zorder=2)
        decimals = max(0, int(np.ceil(-np.log10(sd[attribute]))) + 1)
        upper_is_final = final[attribute] > measured[attribute]
        # The labels start where the trace is clear of the rules: from the first sample for ParticleSize (the
        # early predictions sit well below), a little later for the second attribute, whose trace descends
        # through the rules over the first samples.
        label_x = 0.03 if attribute == "ParticleSize" else 0.12
        rule_label(ax, final[attribute], f"final prediction {final[attribute]:.{decimals}f}", x=label_x, above=upper_is_final, colour=GREY)
        rule_label(ax, measured[attribute], f"measured {measured[attribute]:.{decimals}f}", x=label_x, above=not upper_is_final, colour="black")
        low, high = float(y_hat.min()), float(y_hat.max())
        pad = 0.3 * (high - low)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlim(0, model.n_timesteps_ + 2)
        ax.set_xlabel("Samples observed")
        ax.set_ylabel(attribute)
        ax.set_title(attribute)
        ax.legend(loc="lower right")
    fig.suptitle(f"Batch {AVERAGE_BATCH}: the final quality predicted while the batch runs", y=1.02)
    fig.tight_layout()
    save(fig, out_dir, f"batch-case-sbr-online-prediction-batch-{AVERAGE_BATCH}")

    # -- On-line monitoring against a reference model of the normal batches only (34 and 37 left out): per-sample
    # T2 and instantaneous SPE limits from BatchMonitor. An alarm counts once the statistic stays above its limit
    # for three consecutive samples; the third panel names the tags behind batch 34's SPE at its alarm sample.
    normal = {batch_id: batch for batch_id, batch in trajectories.items() if batch_id not in HIGHLIGHT}
    reference = BatchPLS(n_components=2).fit(normal, quality.loc[list(normal)])
    monitor = BatchMonitor(reference, conf_level=MONITOR_CONF_LEVEL, spe_statistic="instantaneous").fit(normal)
    results = {batch_id: monitor.monitor(trajectories[batch_id]) for batch_id in HIGHLIGHT}
    t2_alarm = first_sustained_alarm(results[FAULT_FROM_START].t2_alarm)
    spe_alarm = first_sustained_alarm(results[FAULT_PARTWAY].spe_alarm)
    if t2_alarm is None or spe_alarm is None:
        raise RuntimeError("expected a sustained T2 alarm for batch 37 and a sustained SPE alarm for batch 34")
    print(f"reference model on {len(normal)} batches; T2 limit {monitor.t2_limit_over_time_[0]:.2f} at every sample")
    print(
        f"batch {FAULT_FROM_START}: first {ALARM_RUN} consecutive T2 samples above the limit after {t2_alarm} samples; "
        f"batch {FAULT_PARTWAY}: first {ALARM_RUN} consecutive SPE samples above the limit after {spe_alarm} samples"
    )
    squared = reference.predict_online(trajectories[FAULT_PARTWAY], upto_k=spe_alarm).residuals.xs(spe_alarm - 1, level="sequence") ** 2
    shares = (100 * squared / squared.sum()).reindex(reference.tag_names_)
    print(f"batch {FAULT_PARTWAY} after {spe_alarm} samples, share of the squared residual: " + ", ".join(f"{t} {v:.1f}%" for t, v in shares.items()))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    online_chart(axes[0], results[FAULT_FROM_START], "t2", colour=AQUA, mean_trace=monitor.t2_mean_over_time_,
                 conf_level=MONITOR_CONF_LEVEL)
    axes[0].set_title(f"Batch {FAULT_FROM_START}: Hotelling's $T^2$,\nfirst sustained alarm after {t2_alarm} samples")
    online_chart(axes[1], results[FAULT_PARTWAY], "spe", colour=ORANGE, mean_trace=monitor.spe_mean_over_time_,
                 conf_level=MONITOR_CONF_LEVEL, fault_at=FAULT_SAMPLE, fault_label="impurity enters", legend_loc="upper left")
    axes[1].set_title(f"Batch {FAULT_PARTWAY}: SPE of the newest sample,\nfirst sustained alarm after {spe_alarm} samples")
    ax = axes[2]
    ax.bar(range(len(shares)), shares.to_numpy(), color=DARK_BLUE, width=0.6, zorder=2)
    ax.set_xticks(range(len(shares)), [str(tag) for tag in shares.index], rotation=20, ha="right")
    shade_alternate_tags(ax, len(shares))
    label_bars(ax, shares.to_numpy(dtype=float))
    ax.set_ylabel("Share of the squared residual [%]")
    ax.set_title(f"Batch {FAULT_PARTWAY} after {spe_alarm} samples:\nshare of the residual per tag")
    fig.tight_layout()
    save(fig, out_dir, "batch-case-sbr-online-monitoring")

    # -- What the reference model expected the rest of each faulty batch to look like (Wold et al. 2009, Eq. 4):
    # the scores estimated from the samples so far, mapped back through the loadings onto the unobserved cells.
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    time = np.arange(1, model.n_timesteps_ + 1)
    for ax, (batch_id, (tag, sample_points)) in zip(axes, FORECASTS.items(), strict=True):
        colour = HIGHLIGHT[batch_id]
        for j, batch in enumerate(normal.values()):
            ax.plot(time, batch[tag].to_numpy(), color=PALE_GREY, lw=0.6, zorder=1, label="normal batches" if j == 0 else None)
        actual = trajectories[batch_id][tag].to_numpy()
        first = sample_points[0]
        ax.plot(time, actual, color=colour, lw=1.4, alpha=0.35, zorder=2, label=f"batch {batch_id}, what happened")
        ax.plot(time[:first], actual[:first], color=colour, lw=1.8, zorder=3, label=f"batch {batch_id}, first {first} samples")
        for k, style in zip(sample_points, ["--", (0, (1.0, 1.6))], strict=True):
            forecast = reference.predict_online(trajectories[batch_id], upto_k=k).forecast[tag]
            print(
                f"batch {batch_id} {tag}, mean over the samples after {k}: forecast {forecast.iloc[k:].mean():.4f}, "
                f"actual {trajectories[batch_id][tag].iloc[k:].mean():.4f}, "
                f"normal batches {np.mean([batch[tag].iloc[k:].mean() for batch in normal.values()]):.4f}"
            )
            ax.plot(time[k:], forecast.to_numpy()[k:], color=colour, lw=1.7, ls=style, zorder=4, label=f"forecast from {k} samples")
        if batch_id == FAULT_PARTWAY:
            ax.axvline(FAULT_SAMPLE, color=GREY, lw=1, ls=":", zorder=2)
            # Written along the bottom, to the right of the line: the legend fills the upper right of this panel.
            ax.text(FAULT_SAMPLE + 2, 0.03, "impurity enters", transform=ax.get_xaxis_transform(), va="bottom", ha="left", fontsize=8.5, color=GREY)
        ax.set_xlim(0, model.n_timesteps_ + 2)
        ax.set_xlabel("Sample [aligned time]")
        ax.set_ylabel(tag)
    axes[0].set_title(f"Batch {FAULT_FROM_START}: conversion,\nforecast of the remainder")
    axes[0].legend(loc="lower right")
    # The cooling-water temperature swings through 5 degrees in the first 15 samples of every batch; the panel
    # starts after that transient so the 0.4 degree departure of batch 34 is not squashed by it. The vertical
    # axis is set from the samples shown (autoscaling would still span the transient), with room for the legend.
    settled = [batch["CoolingTemp"].iloc[TRANSIENT_END:] for batch in [*normal.values(), trajectories[FAULT_PARTWAY]]]
    low, high = min(float(s.min()) for s in settled), max(float(s.max()) for s in settled)
    axes[1].set_xlim(TRANSIENT_END, model.n_timesteps_ + 2)
    axes[1].set_ylim(low - 0.25 * (high - low), high + 0.65 * (high - low))
    axes[1].set_xlabel(f"Sample [aligned time], after the start-up transient (samples 1 to {TRANSIENT_END})")
    axes[1].set_title(f"Batch {FAULT_PARTWAY}: cooling-water temperature,\nforecast of the remainder")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, out_dir, "batch-case-sbr-forecast")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dir", nargs="?", type=pathlib.Path, default=pathlib.Path(__file__).parent)
    parser.add_argument("--data-url", default=None)
    args = parser.parse_args()
    main(args.output_dir, args.data_url)
