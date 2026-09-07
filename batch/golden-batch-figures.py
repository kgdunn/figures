"""Golden-batch baseline figures for the batch monitoring and control chapter.

Four committed PNGs establishing, with real data and with the package's
bioreactor simulator, why replaying a golden batch's schedule does not
reproduce its outcome:

- ``golden-batch-nylon-spread.png``: all 57 aligned batches of the
  industrial nylon dataset for two recorded variables. Tag06, a
  tightly-regulated variable, repeats to a 0.5% average relative spread;
  Tag10 spreads 7.4%, fourteen times wider, although every batch ran the
  same recipe. Real data: the recipe repeats, the batch does not.
- ``golden-batch-replay-spread.png``: a 200-batch replay campaign of the
  simulator (every batch requests the identical nominal schedule,
  seed 0; the same campaign the chapter regresses on Z). Left: the
  recorded reactor temperature lies in a band a fraction of a degree
  wide. Right: the final titers spread around the disturbance-free
  reference of 8.01 g/L; the script prints the range and the CV.
- ``golden-batch-z-scores.png``: score plot of a two-component PCA on the
  standardised 11-variable upstream (Z) block of the chapter's 200-batch
  historical training campaign (seed chain of
  ``evaluate_control_policies(random_state=0)``). The three feed classes
  A, B and C occupy overlapping ranges along the dominant direction t1
  (the feed-quality axis) rather than separate clusters; assigning a batch
  to the nearest class centroid in standardised Z recovers the true label
  about 85% of the time, with confusions only between adjacent classes.
  The per-class models in the chapter are local models along this axis.
- ``golden-batch-variance-decomposition.png``: the titer-variance split
  of a 200-batch replay campaign (seed 0) into measured initial
  conditions, the unmeasured within-batch disturbance, control and
  measurement noise, and the interaction residual of the nonlinear
  process, reported as shares of the total variance. The first share is
  addressable before the batch starts; the second only while it runs;
  the noise floor by neither.

The nylon numbers come from the package's bundled dataset; every
simulator number is reproducible from the stated seeds.

Usage
-----
    uv run --with "process-improve[batch]>=1.79" --with matplotlib \
        python golden-batch-figures.py [output_dir]

Writes the four PNGs into ``output_dir`` (default: this script's own
directory), refreshing the committed images in place. Runs in about two
minutes; the variance decomposition simulates four 200-batch campaigns.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colourblind-safe palette entries used in these figures.
BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
SKY = "#56B4E9"
GREEN = "#009E73"
GREY = "#666666"
GRID = "#DDDDDD"

DPI = 300

CLASS_COLOURS = {"A": BLUE, "B": ORANGE, "C": VERMILLION}

mpl.rcParams.update(
    {
        "font.size": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 21,
        "axes.labelsize": 21,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 17,
        "axes.axisbelow": True,
    }
)


def save(fig, outdir: pathlib.Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote {outdir / name}")


def nylon_spread(outdir: pathlib.Path) -> None:
    """All nylon batches for one tight and one wide variable, side by side."""
    from process_improve.batch import load_nylon, resample_to_reference

    batches = load_nylon()
    tags = list(next(iter(batches.values())).columns)
    aligned = resample_to_reference(batches, columns_to_align=tags, reference_batch=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for ax, tag, colour in ((axes[0], "Tag06", BLUE), (axes[1], "Tag10", VERMILLION)):
        matrix = np.array([b[tag].to_numpy() for b in aligned.values()])
        for row in matrix:
            ax.plot(row, color=colour, alpha=0.25, lw=0.9)
        mean_trajectory = matrix.mean(axis=0)
        relative = 100 * np.mean(matrix.std(axis=0, ddof=1)) / np.mean(np.abs(mean_trajectory))
        ax.plot(mean_trajectory, color="black", lw=2.0, label="average trajectory")
        ax.set_xlabel("Time [sample]")
        ax.set_title(f"{tag}: average spread {relative:.1f}%")
        ax.grid(color=GRID, lw=0.7)
    axes[0].set_ylabel("Recorded value")
    axes[0].legend(frameon=False, loc="lower right")
    save(fig, outdir, "golden-batch-nylon-spread.png")


def replay_spread(outdir: pathlib.Path) -> None:
    """Identical requested schedule; recorded temperatures and final titers."""
    import dataclasses

    from process_improve.simulation import BioreactorConfig, BioreactorSimulator

    simulator = BioreactorSimulator()
    campaign = simulator.simulate_campaign(200, policy="replay", random_state=0)
    reference = BioreactorSimulator(
        dataclasses.replace(BioreactorConfig(), ic_scale=0.0, within_batch_scale=0.0, noise_scale=0.0)
    ).simulate_batch(None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), width_ratios=[2.2, 1.0])
    ax = axes[0]
    for batch in campaign.batches.values():
        ax.plot(batch.index, batch["temperature"], color=BLUE, alpha=0.12, lw=0.8)
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("Recorded temperature [°C]")
    ax.set_title("Every batch requests the same schedule")
    ax.grid(color=GRID, lw=0.7)

    ax = axes[1]
    titer = campaign.quality["titer"]
    ax.hist(titer, bins=12, color=SKY, edgecolor="white")
    ax.axvline(reference.titer, color=VERMILLION, lw=2.2, ls="--")
    ax.annotate(
        f"disturbance-free\nreference {reference.titer:.2f} g/L",
        xy=(reference.titer, ax.get_ylim()[1] * 0.94),
        xytext=(titer.min() + 0.1, ax.get_ylim()[1] * 0.78),
        fontsize=15,
        color=VERMILLION,
        arrowprops={"arrowstyle": "->", "color": VERMILLION},
    )
    cv = 100 * titer.std(ddof=1) / titer.mean()
    print(
        f"replay spread: {len(titer)} batches, mean {titer.mean():.2f}, sd {titer.std(ddof=1):.2f}, "
        f"range {titer.min():.2f} to {titer.max():.2f} g/L, CV {cv:.1f}%; reference {reference.titer:.2f}"
    )
    ax.set_xlabel("Final titer [g/L]")
    ax.set_ylabel("Batches")
    ax.set_title(f"The outcomes: {cv:.1f}% CV")
    ax.grid(color=GRID, lw=0.7)
    save(fig, outdir, "golden-batch-replay-spread.png")


def z_scores(outdir: pathlib.Path) -> None:
    """PCA score plot of the training campaign's Z block, coloured by feed class.

    The campaign is the chapter's own 200-batch historical campaign (the
    first seed drawn from the master seed 0, as in
    ``evaluate_control_policies(random_state=0)``), so the plotted batches
    are the ones the per-class models are fitted on.
    """
    from process_improve.multivariate import MCUVScaler, PCA
    from process_improve.simulation import BioreactorSimulator

    train_seed = int(np.random.default_rng(0).integers(2**31))
    train = BioreactorSimulator().simulate_campaign(200, policy="historical", mv_variation=2.5, random_state=train_seed)
    z_train = train.initial_conditions
    labels = np.asarray(list(train.classes))
    scaled = MCUVScaler().fit_transform(z_train)
    model = PCA(n_components=2).fit(scaled)
    scores = model.scores_

    explained = 100 * model.r2_per_component_
    z_mean, z_sd = z_train.mean(), z_train.std(ddof=1)
    standardised = (z_train - z_mean) / z_sd
    centroids = {g: standardised[labels == g].mean() for g in ("A", "B", "C")}
    assigned = np.array(
        [min(centroids, key=lambda g: float(((standardised.iloc[i] - centroids[g]) ** 2).sum())) for i in range(len(z_train))]
    )
    print(
        f"z-scores: classes {dict(zip(*np.unique(labels, return_counts=True)))}; "
        f"nearest-centroid agreement {np.mean(assigned == labels):.3f}; "
        f"t1, t2 explain {explained.iloc[0]:.1f}%, {explained.iloc[1]:.1f}% of Z"
    )

    fig, ax = plt.subplots(figsize=(9.5, 7))
    for label in ("A", "B", "C"):
        mask = labels == label
        ax.scatter(
            scores.iloc[mask, 0],
            scores.iloc[mask, 1],
            s=42,
            color=CLASS_COLOURS[label],
            label=f"class {label}",
            alpha=0.85,
            edgecolor="white",
            lw=0.5,
        )
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel(f"$t_1$ [{explained.iloc[0]:.0f}% of Z]")
    ax.set_ylabel(f"$t_2$ [{explained.iloc[1]:.0f}% of Z]")
    ax.set_title("Feed classes: overlapping ranges along $t_1$")
    ax.legend(frameon=False)
    ax.grid(color=GRID, lw=0.7)
    save(fig, outdir, "golden-batch-z-scores.png")


def variance_split(outdir: pathlib.Path) -> None:
    """Horizontal bars of the replay-campaign titer-variance decomposition."""
    from process_improve.simulation import BioreactorSimulator, variance_decomposition

    decomposition = variance_decomposition(BioreactorSimulator(), n_batches=200, random_state=0)
    shares = decomposition.drop(index="total")["pct_of_total"]
    labels = {
        "measured initial conditions": "Measured initial conditions\n(addressable before the batch)",
        "within-batch disturbance": "Within-batch disturbance\n(visible only while it runs)",
        "control and measurement noise": "Control and measurement noise",
        "interaction residual": "Interaction of the sources\n(nonlinear process)",
    }
    colours = [BLUE, ORANGE, GREY, SKY]

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    positions = np.arange(len(shares))[::-1]
    ax.barh(positions, shares.to_numpy(), color=colours, edgecolor="white")
    for pos, value in zip(positions, shares.to_numpy()):
        ax.text(value + 1.0, pos, f"{value:.0f}%", va="center", fontsize=17)
    ax.set_yticks(positions)
    ax.set_yticklabels([labels[i] for i in shares.index], fontsize=16)
    ax.set_xlabel("Share of replay-campaign titer variance [%]")
    ax.set_xlim(0, max(shares) * 1.18)
    ax.grid(axis="x", color=GRID, lw=0.7)
    save(fig, outdir, "golden-batch-variance-decomposition.png")


if __name__ == "__main__":
    outdir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent
    nylon_spread(outdir)
    replay_spread(outdir)
    z_scores(outdir)
    variance_split(outdir)
