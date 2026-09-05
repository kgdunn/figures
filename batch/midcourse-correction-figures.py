"""Mid-course correction figures for the batch monitoring and control chapter.

Five committed PNGs, every quantitative claim executed on the bioreactor
simulator (corrected schedules re-simulated with the identical disturbance
seed, so gains are same-batch counterfactuals, never model predictions).
The seeds replicate ``evaluate_control_policies(y_target=8.0,
random_state=0)`` from ``process_improve``, so the numbers match that
function's documented headline exactly.

- ``mcc-monitoring-funnel.png``: the predicted final titer of one
  batch from the poorest feed class (test batch 28; replay outcome
  3.66 g/L) at every decision point, with the 95% prediction-interval
  band built at each decision point (``MidCourseCorrector.predict``).
  From the first day the projection sits near 3.2 g/L, far below the
  8 g/L target; the interval narrows from +/-4.0 g/L at day 0.5 to
  +/-1.75 at day 4 and +/-0.9 at day 9.5, and its upper end drops below
  the target from day 2.5 on.
- ``mcc-correction-at-k.png``: the same batch's temperature and pH
  schedules before and after the day-4 correction. Instead of
  completing the drop to the 29 degC production hold on schedule, the
  corrected schedule holds near 34.7 degC at day 4 and ramps down to
  reach the hold only around day 7.5, with a small transient pH dip;
  executed, the batch finishes at 5.79 g/L instead of 3.66 g/L (+58%).
- ``mcc-policy-comparison.png``: forty fresh batches under four executed
  policies: replay 7.51 +/- 1.20 g/L, mid-course correction
  7.75 +/- 0.78 (five corrected, none harmed; dead band 1.0, the whole
  interval must fall short of the target), the
  oracle-from-the-decision-point ceiling 7.87 +/- 0.63, and the
  perfect-feedforward (adapted) ceiling 7.82 +/- 1.01. Left: each
  corrected batch's jump. Right: the distributions; correction removes
  the low tail, which feedforward adaptation cannot reach.
- ``mcc-decision-point-window.png``: the executed gain of the corrected
  batches as the single decision point moves through the batch. The
  window is mid-batch (days 3 to 5, largest gain at day 4): earlier, the
  interval at the decision point is wide, so the dead band admits only
  the clearest shortfalls and the gain is about half of the peak; later,
  the remaining schedule has no leverage and corrections turn harmful.
- ``mcc-exploration-dial.png``: predicted versus executed titer of the
  five corrected batches as the T2 (stay-where-the-model-has-data)
  penalty is relaxed, hard caps off. On this process the executed
  outcome improves monotonically and sits above the prediction at every
  setting; the late-decision-point harm in the window figure shows the
  same freedom working against you once the model's leverage is gone.

Usage
-----
    uv run --with "process-improve[control]>=1.80" --with matplotlib \
        python midcourse-correction-figures.py [output_dir]

Writes the five PNGs into ``output_dir`` (default: this script's own
directory). Full regeneration re-runs the executed policy comparison and
the two sweeps: roughly 20 minutes on a laptop.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
SKY = "#56B4E9"
GREEN = "#009E73"
GREY = "#666666"
GRID = "#DDDDDD"

DPI = 300
TARGET = 8.0

mpl.rcParams.update(
    {
        "font.size": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 21,
        "axes.labelsize": 21,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "axes.axisbelow": True,
    }
)


def save(fig, outdir: pathlib.Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote {outdir / name}")


def _headline_setup():
    """Replicate the seed chain of evaluate_control_policies(random_state=0).

    Returns the simulator, the per-class correctors fitted on the same
    200-batch historical campaign, the class-assignment helper, the test
    campaign's Z block, and the per-batch execution seeds.
    """
    from process_improve.batch import BatchPLS
    from process_improve.batch.control import MidCourseCorrector
    from process_improve.simulation import BioreactorSimulator

    simulator = BioreactorSimulator()
    rng = np.random.default_rng(0)
    train_seed = int(rng.integers(2**31))
    test_seed = int(rng.integers(2**31))
    batch_seeds = rng.integers(2**31, size=40)

    train = simulator.simulate_campaign(200, policy="historical", mv_variation=2.5, random_state=train_seed)
    z_train = train.initial_conditions
    labels = np.asarray(list(train.classes))
    z_mean, z_sd = z_train.mean(), z_train.std(ddof=1)
    z_standardised = (z_train - z_mean) / z_sd

    nominal = simulator.nominal_trajectory().reset_index(drop=True)
    config = simulator.config
    bounds = {
        "temperature": (config.temp_bounds[0] + 0.3, config.temp_bounds[1] - 0.3),
        "pH": (config.ph_bounds[0] + 0.04, config.ph_bounds[1] - 0.04),
    }
    correctors, centroids = {}, {}
    for group in sorted(set(labels)):
        ids = [i for i, c in zip(train.batches, labels) if c == group]
        model = BatchPLS(n_components=4).fit(
            {i: train.batches[i] for i in ids}, train.quality.loc[ids], initial_conditions=z_train.loc[ids]
        )
        centroids[group] = z_standardised.loc[ids].mean()
        correctors[group] = MidCourseCorrector(
            model,
            nominal,
            mv_tags=["pH", "temperature"],
            mode="target",
            y_target=TARGET,
            target_side="below",
            dead_band=1.0,
            weights={"target": 1.0, "movement": 0.1},
            bounds=bounds,
            rate_limits={"temperature": 3.0, "pH": 0.5},
            spe_cap="limit",
            t2_cap="limit",
            n_knots=4,
        )

    def assign(z_row):
        z_std = (z_row - z_mean) / z_sd
        return min(centroids, key=lambda g: float(((z_std - centroids[g]) ** 2).sum()))

    test = simulator.simulate_campaign(40, policy="replay", random_state=test_seed)
    return simulator, correctors, assign, test.initial_conditions, batch_seeds


def monitoring_funnel_and_correction(outdir: pathlib.Path) -> None:
    """Per-decision-point prediction of one poor batch, and its day-4 correction."""
    simulator, correctors, assign, z_test, batch_seeds = _headline_setup()
    batch_id = 28  # the headline's deepest corrected batch (replay 3.66 g/L)
    position = list(z_test.index).index(batch_id)
    seed = int(batch_seeds[position])
    z_row = z_test.loc[batch_id]
    base = simulator.simulate_batch(z_row, random_state=seed)
    corrector = correctors[assign(z_row)]
    model = corrector.model

    # Prediction and interval at every decision point, no intervention: the
    # corrector's monitoring question, with the interval at that decision
    # point (wide early, narrowing as the batch reveals itself).
    ks, y_hat, half = [], [], []
    for k in range(1, model.n_timesteps_):
        prediction = corrector.predict(base.tags.iloc[:k].reset_index(drop=True), initial_conditions=z_row, k=k)
        ks.append(k)
        y_hat.append(float(prediction.y_hat.iloc[0]))
        half.append(float(prediction.half_width.iloc[0]))

    days = np.array(ks) * 0.5
    y_hat = np.array(y_hat)
    half = np.array(half)
    print("funnel (batch 28): day, predicted titer, half-width")
    for d, y, h in zip(days, y_hat, half):
        print(f"  {d:4.1f} {y:6.2f} +/- {h:5.2f}")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.fill_between(days, y_hat - half, y_hat + half, color=SKY, alpha=0.35, label="95% prediction interval")
    ax.plot(days, y_hat, color=BLUE, lw=2.2, marker="o", ms=5, label="predicted final titer")
    ax.axhline(TARGET, color=GREEN, lw=2.0, ls="--", label=f"target {TARGET:.0f} g/L")
    ax.axhline(base.titer, color=VERMILLION, lw=2.0, ls=":", label=f"replay outcome {base.titer:.2f} g/L")
    ax.axvline(4.0, color=GREY, lw=1.4)
    ax.annotate("decision point (day 4)", xy=(4.1, 7.4), fontsize=15, color=GREY, ha="left")
    ax.set_xlabel("Decision point [day]")
    ax.set_ylabel("Predicted final titer [g/L]")
    ax.set_title("Watching one batch mid-flight (test batch 28)")
    ax.legend(frameon=False, loc="lower left", fontsize=15)
    ax.grid(color=GRID, lw=0.7)
    save(fig, outdir, "mcc-monitoring-funnel.png")

    # --- The correction at day 4, and its executed outcome ------------------
    outcome = corrector.correct(base.tags.iloc[:8].reset_index(drop=True), initial_conditions=z_row, k=8)
    trajectory = outcome.schedule.copy()
    trajectory.index = simulator.nominal_trajectory().index
    redo = simulator.simulate_batch(z_row, trajectory, random_state=seed)
    print(f"correction at day 4: replay {base.titer:.3f} -> executed {redo.titer:.3f} g/L "
          f"(predicted {float(outcome.y_hat.iloc[0]):.3f}); reason {outcome.reason}")

    nominal = corrector.nominal_schedule
    day_axis = np.asarray(simulator.nominal_trajectory().index, dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, height_ratios=[2.0, 1.0])
    ax = axes[0]
    ax.step(day_axis, nominal["temperature"], where="post", color=GREY, lw=2.0, label="nominal schedule")
    ax.step(
        day_axis,
        outcome.schedule["temperature"],
        where="post",
        color=VERMILLION,
        lw=2.4,
        label="corrected at day 4",
    )
    ax.axvspan(day_axis[0], 4.0, color=GRID, alpha=0.5)
    ax.annotate("already run", xy=(1.4, 30.2), fontsize=15, color=GREY)
    ax.set_ylabel("Temperature setpoint [°C]")
    ax.set_title(
        f"The correction: executed titer {base.titer:.2f} " f"→ {redo.titer:.2f} g/L"
    )
    ax.legend(frameon=False, loc="upper right", fontsize=15)
    ax.grid(color=GRID, lw=0.7)

    ax = axes[1]
    ax.step(day_axis, nominal["pH"], where="post", color=GREY, lw=2.0)
    ax.step(day_axis, outcome.schedule["pH"], where="post", color=VERMILLION, lw=2.4)
    ax.axvspan(day_axis[0], 4.0, color=GRID, alpha=0.5)
    ax.set_ylabel("pH setpoint")
    ax.set_xlabel("Time [day]")
    ax.grid(color=GRID, lw=0.7)
    save(fig, outdir, "mcc-correction-at-k.png")


def policy_comparison(outdir: pathlib.Path) -> None:
    """Executed four-policy comparison; recomputes the documented headline."""
    from process_improve.batch.control import evaluate_control_policies
    from process_improve.simulation import BioreactorSimulator

    result = evaluate_control_policies(BioreactorSimulator(), y_target=TARGET, random_state=0)
    batches = result.batches
    corrected = batches[batches["corrected"]]
    print("policy comparison (executed titers, g/L):")
    print(result.summary.round(3).to_string())
    print(f"{result.n_corrected} corrected, {result.n_harmed} harmed; reasons: {batches['reason'].value_counts().to_dict()}")
    print(corrected[["class_assigned", "replay", "midcourse", "y_hat_no_change", "half_width", "y_hat_predicted"]].round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), width_ratios=[1.0, 1.6])

    ax = axes[0]
    # Stagger the gain labels vertically when two corrected outcomes nearly
    # coincide, so neither annotation overprints the other.
    label_positions: list[float] = []
    for _batch_id, row in corrected.sort_values("midcourse").iterrows():
        ax.plot([0, 1], [row["replay"], row["midcourse"]], color=GREY, lw=1.4, zorder=1)
        ax.scatter([0], [row["replay"]], color=BLUE, s=70, zorder=2)
        ax.scatter([1], [row["midcourse"]], color=VERMILLION, s=70, zorder=2)
        label_y = float(row["midcourse"])
        while any(abs(label_y - other) < 0.22 for other in label_positions):
            label_y += 0.22
        label_positions.append(label_y)
        ax.annotate(
            f"+{row['midcourse'] - row['replay']:.2f}",
            xy=(1.04, label_y),
            fontsize=14,
            va="center",
        )
    ax.axhline(TARGET, color=GREEN, lw=1.6, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["replay", "corrected"])
    ax.set_xlim(-0.35, 1.55)
    ax.set_ylabel("Final titer [g/L]")
    ax.set_title("The four corrected batches\n(executed gains)")
    ax.grid(axis="y", color=GRID, lw=0.7)

    ax = axes[1]
    order = [
        ("replay", batches["replay"], BLUE),
        ("mid-course", batches["midcourse"], VERMILLION),
        ("oracle-from-k", batches["oracle_from_k"].fillna(batches["midcourse"]), ORANGE),
        ("adapted\n(feedforward)", batches["adapted"], GREEN),
    ]
    positions = np.arange(len(order))
    rng = np.random.default_rng(1)
    for position, (label, values, colour) in zip(positions, order):
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(position + jitter, values, s=26, color=colour, alpha=0.65, edgecolor="none")
        ax.hlines(values.mean(), position - 0.24, position + 0.24, color="black", lw=2.4, zorder=3)
        ax.annotate(
            f"{values.mean():.2f}\n±{values.std(ddof=1):.2f}",
            xy=(position, batches[["replay"]].to_numpy().min() - 0.15),
            ha="center",
            va="top",
            fontsize=14,
        )
    ax.axhline(TARGET, color=GREEN, lw=1.6, ls="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([label for label, _, _ in order], fontsize=16)
    ax.set_ylim(batches[["replay"]].to_numpy().min() - 1.15, None)
    ax.set_title("All 40 batches, four executed policies")
    ax.grid(axis="y", color=GRID, lw=0.7)
    save(fig, outdir, "mcc-policy-comparison.png")


def decision_point_window(outdir: pathlib.Path) -> None:
    """Executed gain of corrected batches versus decision-point placement."""
    from process_improve.batch.control import evaluate_control_policies
    from process_improve.simulation import BioreactorSimulator

    simulator = BioreactorSimulator()
    rows = []
    for k in (4, 6, 8, 10, 12, 14):
        result = evaluate_control_policies(
            simulator,
            y_target=TARGET,
            decision_points=(k,),
            include_adapted=False,
            oracle="none",
            random_state=0,
        )
        corrected = result.batches[result.batches["corrected"]]
        gain = corrected["midcourse"] - corrected["replay"]
        rows.append(
            {
                "day": 0.5 * k,
                "mean_gain": float(gain.mean()) if len(gain) else 0.0,
                "n_corrected": int(result.n_corrected),
                "n_harmed": int(result.n_harmed),
            }
        )
    sweep = pd.DataFrame(rows)
    print("decision-point sweep:")
    print(sweep.round(3).to_string())

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.axhline(0, color=GREY, lw=1.0)
    ax.plot(sweep["day"], sweep["mean_gain"], color=BLUE, lw=2.4, marker="o", ms=9)
    for _, row in sweep.iterrows():
        note = f"{row['n_corrected']:.0f} corrected"
        if row["n_harmed"]:
            note += f"\n{row['n_harmed']:.0f} harmed"
        below = row["mean_gain"] > 1.5  # keep the peak's label clear of the title
        ax.annotate(
            note,
            xy=(row["day"], row["mean_gain"]),
            xytext=(14, -18) if below else (0, 14),
            textcoords="offset points",
            ha="left" if below else "center",
            fontsize=13.5,
            color=GREY,
        )
    ax.axvspan(2.9, 5.1, color=GREEN, alpha=0.12)
    ax.annotate("the window", xy=(4.0, ax.get_ylim()[0] + 0.12), ha="center", fontsize=16, color=GREEN)
    ax.set_xlabel("Decision point [day]")
    ax.set_ylabel("Mean executed gain [g/L]")
    ax.set_ylim(top=sweep["mean_gain"].max() + 0.45)
    ax.set_title("Executed gain of the corrected batches\nversus decision-point placement")
    ax.grid(color=GRID, lw=0.7)
    save(fig, outdir, "mcc-decision-point-window.png")


def exploration_dial(outdir: pathlib.Path) -> None:
    """Predicted versus executed titer of the corrected batches as T2 relaxes."""
    from process_improve.batch.control import evaluate_control_policies
    from process_improve.simulation import BioreactorSimulator

    simulator = BioreactorSimulator()
    rows = []
    for w4 in (3.0, 1.0, 0.3, 0.1, 0.03, 0.0):
        result = evaluate_control_policies(
            simulator,
            y_target=TARGET,
            decision_points=(8,),
            weights={"target": 1.0, "movement": 0.1, "t2": w4},
            spe_cap=None,
            t2_cap=None,
            include_adapted=False,
            oracle="none",
            random_state=0,
        )
        corrected = result.batches[result.batches["corrected"]]
        rows.append(
            {
                "w4": w4,
                "predicted": float(corrected["y_hat_predicted"].mean()),
                "executed": float(corrected["midcourse"].mean()),
                "replay": float(corrected["replay"].mean()),
            }
        )
    dial = pd.DataFrame(rows)
    print("exploration dial:")
    print(dial.round(3).to_string())

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(dial))
    ax.plot(x, dial["executed"], color=VERMILLION, lw=2.4, marker="o", ms=9, label="executed titer")
    ax.plot(x, dial["predicted"], color=BLUE, lw=2.4, marker="s", ms=9, label="model's own prediction")
    ax.axhline(dial["replay"].iloc[0], color=GREY, lw=1.8, ls=":", label="replay (no correction)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w:g}" for w in dial["w4"]])
    ax.set_xlabel("T2 penalty weight (left: conservative; right: free to explore)")
    ax.set_ylabel("Mean titer of the corrected batches [g/L]")
    ax.set_title("The exploration dial, measured against the true process")
    ax.legend(frameon=False, loc="lower right", fontsize=15)
    ax.grid(color=GRID, lw=0.7)
    save(fig, outdir, "mcc-exploration-dial.png")


if __name__ == "__main__":
    outdir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent
    monitoring_funnel_and_correction(outdir)
    policy_comparison(outdir)
    decision_point_window(outdir)
    exploration_dial(outdir)
