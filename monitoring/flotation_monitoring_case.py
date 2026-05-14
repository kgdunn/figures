"""
Multivariate process monitoring case study for pid-book §7.3.

Fits a 2-component PCA on the phase-1 stretch (first 479 observations,
15 December 2004) of the openmv flotation-cell dataset, aggregated into
2-minute subgroups (size n=4) so the multivariate analysis is on the
same statistical object as the univariate Shewhart chart. Then monitors
the phase-2 stretch (16 December onwards, 2443 observations / 610
subgroups) with Hotelling's T^2 and SPE. Also produces a univariate
Shewhart trace on the Feed rate column for comparison, and a dual-axis
side-by-side of the Feed-rate Shewhart and the multivariate T^2.

Generates six PNG figures referenced by
`product-development-product-improvement/multivariate-process-monitoring.rst`,
writing them next to this script in the figures repository.

Run:

    python monitoring/flotation_monitoring_case.py

Requires:
    process_improve, matplotlib, numpy, pandas
    network access to openmv.net for the input CSV
"""

from __future__ import annotations

import pathlib
from math import gamma, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_improve.multivariate import PCA, MCUVScaler

FIGURES_DIR = pathlib.Path(__file__).resolve().parent
DATA_URL = "https://openmv.net/file/flotation-cell.csv"

N_PHASE1 = 479
N_SUB = 4
CONF_LEVEL = 0.95


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_URL)


def split_phases(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num = df.drop(columns=["Date and time"])
    return num.iloc[:N_PHASE1].copy(), num.iloc[N_PHASE1:].copy()


def subgroup(x, n_sub: int) -> np.ndarray:
    n_groups = len(x) // n_sub
    return np.asarray(x[: n_groups * n_sub]).reshape((n_groups, n_sub))


def subgroup_means(df: pd.DataFrame, n_sub: int) -> pd.DataFrame:
    n_groups = len(df) // n_sub
    arr = df.values[: n_groups * n_sub].reshape((n_groups, n_sub, df.shape[1]))
    return pd.DataFrame(arr.mean(axis=1), columns=df.columns)


def shewhart_limits(xbar_p1: np.ndarray, s_p1: np.ndarray, n_sub: int):
    target = float(xbar_p1.mean())
    sbar = float(s_p1.mean())
    a_n = sqrt(2) * gamma(n_sub / 2) / (sqrt(n_sub - 1) * gamma((n_sub - 1) / 2))
    sigma_hat = sbar / a_n
    lcl = target - 3 * sigma_hat / sqrt(n_sub)
    ucl = target + 3 * sigma_hat / sqrt(n_sub)
    return target, lcl, ucl


def figure_shewhart(phase1, phase2, out: pathlib.Path) -> None:
    sub_p1 = subgroup(phase1["Feed rate"].values, N_SUB)
    sub_p2 = subgroup(phase2["Feed rate"].values, N_SUB)
    xbar_p1 = sub_p1.mean(axis=1)
    s_p1 = sub_p1.std(axis=1, ddof=1)
    xbar_p2 = sub_p2.mean(axis=1)
    target, lcl, ucl = shewhart_limits(xbar_p1, s_p1, N_SUB)
    first_alarm_p2 = int(np.where((xbar_p2 < lcl) | (xbar_p2 > ucl))[0][0])

    fig, ax = plt.subplots(figsize=(11, 4.4))
    idx_p1 = np.arange(len(xbar_p1))
    idx_p2 = np.arange(len(xbar_p1), len(xbar_p1) + len(xbar_p2))
    ax.plot(idx_p1, xbar_p1, "k.-", linewidth=1.0, markersize=4, label="Phase 1 (15 Dec)")
    ax.plot(idx_p2, xbar_p2, "C0.-", linewidth=1.0, markersize=3, label="Phase 2 (16 Dec onwards)")
    ax.axhline(target, color="grey", linestyle=":", linewidth=1, label="target")
    ax.axhline(ucl, color="red", linestyle="--", linewidth=1, label="UCL / LCL (3 sigma)")
    ax.axhline(lcl, color="red", linestyle="--", linewidth=1)
    ax.axvline(len(xbar_p1) - 0.5, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Subgroup index (2 min each)")
    ax.set_ylabel("Feed rate (subgroup mean, t/h)")
    ax.set_title(
        f"Shewhart chart on Feed rate (subgroup size {N_SUB}); "
        f"first phase-2 alarm at subgroup {first_alarm_p2}"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_score_phase1(model, n_phase1_sub, out: pathlib.Path) -> None:
    scores = model.scores_
    ci_x, ci_y = model.ellipse_coordinates(score_horiz=1, score_vert=2, conf_level=CONF_LEVEL)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(scores.iloc[:, 0], scores.iloc[:, 1], "k.", markersize=5, alpha=0.8, label="Phase 1 subgroup scores")
    ax.plot(ci_x, ci_y, color="palevioletred", linewidth=1.8, label="95% T^2 ellipse")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("$t_1$")
    ax.set_ylabel("$t_2$")
    ax.set_title(f"Phase-1 score plot ({n_phase1_sub} subgroups, 15 Dec)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_loadings(model, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for k, ax in enumerate(axes):
        p = model.loadings_.iloc[:, k]
        ax.bar(p.index, p.values, color="#4c72b0")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel(f"$p_{k + 1}$ loading")
        ax.tick_params(axis="x", rotation=25)
        ax.set_title(f"Loading {k + 1}")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_t2_spe(t2, spe, t2_lim, spe_lim, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(t2.index, t2.values, color="#1f77b4", linewidth=1.0, marker=".", markersize=3)
    axes[0].axhline(t2_lim, color="red", linestyle="--", linewidth=1, label="95% limit")
    axes[0].set_ylabel("Hotelling's $T^2$")
    axes[0].set_title("Phase-2 multivariate monitoring (2-min subgroups)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(spe.index, spe.values, color="#d62728", linewidth=1.0, marker=".", markersize=3)
    axes[1].axhline(spe_lim, color="red", linestyle="--", linewidth=1, label="95% limit")
    axes[1].set_ylabel("SPE")
    axes[1].set_xlabel("Phase-2 subgroup index (2 min each)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_contribution(contribs: pd.Series, first_alarm_idx: int, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(contribs.index, contribs.values, color="#d62728")
    ax.set_ylabel("$(x_k - \\hat{x}_k)^2$ in scaled units")
    ax.set_title(
        f"Per-variable SPE contributions at phase-2 subgroup {first_alarm_idx} "
        f"(16 Dec, first SPE alarm)"
    )
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_comparison(
    p2_feed_sub: np.ndarray,
    feed_target: float,
    feed_lcl: float,
    feed_ucl: float,
    t2: pd.Series,
    t2_lim: float,
    feed_first_alarm: int,
    t2_first_alarm: int,
    out: pathlib.Path,
) -> None:
    """Twin-axis overlay of phase-2 Feed-rate subgroups and the multivariate T^2,
    so a reader can read off the timing dividend directly."""
    fig, ax1 = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(p2_feed_sub))

    # Left axis: Feed-rate subgroup mean + Shewhart limits.
    ax1.plot(x, p2_feed_sub, color="#1f77b4", linewidth=1.0, marker=".", markersize=3,
             label="Feed rate (subgroup mean, left)")
    ax1.axhline(feed_target, color="#1f77b4", linestyle=":", linewidth=0.9, alpha=0.6)
    ax1.axhline(feed_ucl, color="#1f77b4", linestyle="--", linewidth=0.9, alpha=0.6)
    ax1.axhline(feed_lcl, color="#1f77b4", linestyle="--", linewidth=0.9, alpha=0.6)
    ax1.axvline(feed_first_alarm, color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.8)
    ax1.set_xlabel("Phase-2 subgroup index (2 min each)")
    ax1.set_ylabel("Feed rate (t/h)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    # Right axis: Hotelling's T^2 + 95 % limit.
    ax2 = ax1.twinx()
    ax2.plot(x, t2.values, color="#d62728", linewidth=1.0, marker=".", markersize=3,
             label="Hotelling's $T^2$ (right)")
    ax2.axhline(t2_lim, color="#d62728", linestyle="--", linewidth=0.9, alpha=0.6)
    ax2.axvline(t2_first_alarm, color="#d62728", linestyle=":", linewidth=1.2, alpha=0.8)
    ax2.set_ylabel("Hotelling's $T^2$", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # Title + a shared legend below.
    ax1.set_title(
        f"Univariate vs multivariate alarms on phase 2: feed rate alarms at subgroup "
        f"{feed_first_alarm}, T^2 alarms at subgroup {t2_first_alarm}"
    )
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    phase1, phase2 = split_phases(df)
    print(f"phase1 shape: {phase1.shape}  phase2 shape: {phase2.shape}")

    # 1. Univariate Shewhart on Feed rate (subgroups)
    figure_shewhart(phase1, phase2, FIGURES_DIR / "Flotation-MSPC-shewhart.png")
    print("[1/6] shewhart fig done")

    # 2-5. Multivariate PCA on phase-1 subgroup means.
    p1_sub = subgroup_means(phase1, N_SUB)
    p2_sub = subgroup_means(phase2, N_SUB)
    print(f"subgrouped phase1: {p1_sub.shape}   subgrouped phase2: {p2_sub.shape}")

    scaler = MCUVScaler().fit(p1_sub)
    model = PCA(n_components=2).fit(scaler.transform(p1_sub))
    print(f"R^2 cumulative (subgrouped): {model.r2_cumulative_.values}")

    figure_score_phase1(model, len(p1_sub), FIGURES_DIR / "Flotation-MSPC-score-phase1.png")
    print("[2/6] score plot done")

    figure_loadings(model, FIGURES_DIR / "Flotation-MSPC-loadings.png")
    print("[3/6] loadings done")

    result = model.predict(scaler.transform(p2_sub))
    t2 = result.hotellings_t2.iloc[:, -1]
    spe = result.spe
    t2_lim = float(model.hotellings_t2_limit(conf_level=CONF_LEVEL))
    spe_lim = float(model.spe_limit(conf_level=CONF_LEVEL))
    print(f"95% T^2 limit: {t2_lim:.2f}   95% SPE limit: {spe_lim:.2f}")

    figure_t2_spe(t2, spe, t2_lim, spe_lim, FIGURES_DIR / "Flotation-MSPC-t2-spe.png")
    print("[4/6] T^2/SPE done")

    flagged_t2 = t2[t2 > t2_lim]
    flagged_spe = spe[spe > spe_lim]
    first_t2_alarm = int(flagged_t2.index[0])
    first_spe_alarm = int(flagged_spe.index[0])
    print(f"first T^2 alarm at phase-2 subgroup {first_t2_alarm}")
    print(f"first SPE alarm at phase-2 subgroup {first_spe_alarm}")

    row_scaled = scaler.transform(p2_sub).loc[first_spe_alarm].values
    row_hat = row_scaled @ model.loadings_.values @ model.loadings_.values.T
    contribs = pd.Series(
        (row_scaled - row_hat) ** 2,
        index=p1_sub.columns,
        name="SPE contribution",
    )
    print(contribs.round(3).to_dict())
    figure_contribution(contribs, first_spe_alarm,
                        FIGURES_DIR / "Flotation-MSPC-contributions.png")
    print("[5/6] contributions done")

    # 6. Twin-axis comparison of phase-2 Feed-rate subgroups vs T^2 subgroups.
    feed_sub_p1 = subgroup(phase1["Feed rate"].values, N_SUB).mean(axis=1)
    feed_sub_p2 = subgroup(phase2["Feed rate"].values, N_SUB).mean(axis=1)
    feed_target, feed_lcl, feed_ucl = shewhart_limits(
        feed_sub_p1,
        subgroup(phase1["Feed rate"].values, N_SUB).std(axis=1, ddof=1),
        N_SUB,
    )
    feed_first_alarm = int(np.where((feed_sub_p2 < feed_lcl) | (feed_sub_p2 > feed_ucl))[0][0])

    figure_comparison(
        p2_feed_sub=feed_sub_p2,
        feed_target=feed_target,
        feed_lcl=feed_lcl,
        feed_ucl=feed_ucl,
        t2=t2,
        t2_lim=t2_lim,
        feed_first_alarm=feed_first_alarm,
        t2_first_alarm=first_t2_alarm,
        out=FIGURES_DIR / "Flotation-MSPC-comparison.png",
    )
    print("[6/6] comparison fig done")

    print()
    print(f"Univariate Shewhart on Feed rate first alarm: subgroup {feed_first_alarm}")
    print(f"Multivariate T^2 first alarm:                  subgroup {first_t2_alarm}")
    print(f"Dividend in subgroups: {feed_first_alarm - first_t2_alarm} "
          f"(~ {(feed_first_alarm - first_t2_alarm) * 2} minutes earlier)")
    print()
    print("Wrote 6 PNG figures to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
