"""
Multivariate process monitoring case study for pid-book §7.3.

Discards the first 479 observations of the openmv flotation-cell dataset
(15 December 2004; the process was unsettled here and not useful as an
in-control reference), then takes the next 1000 observations as phase 1
and the remaining 1443 observations as phase 2. Both are aggregated into
4-minute subgroups (size 8 at 30-second sampling); the size is chosen
from the autocorrelation of the Pulp-level tag on phase 1, which falls
within the 95% noise band around lag 9. A 2-component PCA is fit on the
subgrouped phase 1 and used to monitor the subgrouped phase 2 with
Hotelling's T^2, SPE, and a contribution decomposition at the first
T^2 alarm. A univariate Shewhart chart on the Pulp level column is
shown side by side for comparison.

Generates seven PNG figures referenced by
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

# Drop the first 479 raw observations entirely (the 15 December stretch is
# not in-control and was already known to be a poor training set). The next
# 1000 raw observations become phase 1, and everything after that is phase 2.
# Subgroup size is set from the Pulp-level autocorrelation: at 30-second
# sampling the lag-to-lag autocorrelation falls within the 95% noise band
# around lag 9, so an 8-sample subgroup (~4 minutes) averages out
# essentially all of the short-range autocorrelation without smearing the
# longer-period process structure.
N_DROP = 479
N_PHASE1_RAW = 1000
N_SUB = 8
UNIVARIATE_TAG = "Pulp level"
CONF_LEVEL = 0.95


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_URL)


def split_phases(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num = df.drop(columns=["Date and time"])
    phase1 = num.iloc[N_DROP : N_DROP + N_PHASE1_RAW].reset_index(drop=True)
    phase2 = num.iloc[N_DROP + N_PHASE1_RAW :].reset_index(drop=True)
    return phase1, phase2


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


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Return the sample autocorrelation function for lags 0..nlags."""
    xc = np.asarray(x, dtype=float) - float(np.mean(x))
    var = float((xc ** 2).sum())
    return np.array(
        [(xc[: len(xc) - k] * xc[k:]).sum() / var for k in range(nlags + 1)]
    )


def figure_acf(phase1, out: pathlib.Path) -> None:
    """Autocorrelation of UNIVARIATE_TAG on phase 1; motivates the subgroup size."""
    x = phase1[UNIVARIATE_TAG].values
    nlags = 40
    rho = acf(x, nlags)
    ci = 1.96 / sqrt(len(x))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(rho)), rho, color="#4c72b0")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(ci, color="red", linestyle="--", linewidth=0.8, label=f"95% noise band ($\\pm{ci:.3f}$)")
    ax.axhline(-ci, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Lag (30-second samples)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"Autocorrelation of {UNIVARIATE_TAG} on phase 1 ({len(x)} obs)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_shewhart(phase1, phase2, out: pathlib.Path) -> None:
    sub_p1 = subgroup(phase1[UNIVARIATE_TAG].values, N_SUB)
    sub_p2 = subgroup(phase2[UNIVARIATE_TAG].values, N_SUB)
    xbar_p1 = sub_p1.mean(axis=1)
    s_p1 = sub_p1.std(axis=1, ddof=1)
    xbar_p2 = sub_p2.mean(axis=1)
    target, lcl, ucl = shewhart_limits(xbar_p1, s_p1, N_SUB)
    first_alarm_p2 = int(np.where((xbar_p2 < lcl) | (xbar_p2 > ucl))[0][0])

    fig, ax = plt.subplots(figsize=(11, 4.4))
    idx_p1 = np.arange(len(xbar_p1))
    idx_p2 = np.arange(len(xbar_p1), len(xbar_p1) + len(xbar_p2))
    minutes_per_sub = N_SUB * 0.5  # 30-second sampling
    ax.plot(idx_p1, xbar_p1, "k.-", linewidth=1.0, markersize=4,
            label=f"Phase 1 (training, {len(xbar_p1)} subgroups)")
    ax.plot(idx_p2, xbar_p2, "C0.-", linewidth=1.0, markersize=3,
            label=f"Phase 2 (monitoring, {len(xbar_p2)} subgroups)")
    ax.axhline(target, color="grey", linestyle=":", linewidth=1, label="target")
    ax.axhline(ucl, color="red", linestyle="--", linewidth=1, label="UCL / LCL (3 sigma)")
    ax.axhline(lcl, color="red", linestyle="--", linewidth=1)
    ax.axvline(len(xbar_p1) - 0.5, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel(f"Subgroup index ({minutes_per_sub:.0f} min each)")
    ax.set_ylabel(f"{UNIVARIATE_TAG} (subgroup mean)")
    ax.set_title(
        f"Shewhart chart on {UNIVARIATE_TAG} (subgroup size {N_SUB}); "
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
    ax.set_title(f"Phase-1 score plot ({n_phase1_sub} subgroups)")
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
    axes[0].set_title(f"Phase-2 multivariate monitoring ({N_SUB * 0.5:.0f}-min subgroups)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(spe.index, spe.values, color="#d62728", linewidth=1.0, marker=".", markersize=3)
    axes[1].axhline(spe_lim, color="red", linestyle="--", linewidth=1, label="95% limit")
    axes[1].set_ylabel("SPE")
    axes[1].set_xlabel(f"Phase-2 subgroup index ({N_SUB * 0.5:.0f} min each)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_contribution(contribs: pd.Series, first_alarm_idx: int, out: pathlib.Path) -> None:
    """Bar chart of T^2-style contributions at the first T^2 alarm."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#4c72b0" if v >= 0 else "#4c72b0" for v in contribs.values]  # neutral
    ax.bar(contribs.index, contribs.values, color=colors)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Contribution to $T^2$ score movement (scaled units)")
    ax.set_title(
        f"Per-variable T^2 contributions at phase-2 subgroup {first_alarm_idx} "
        f"(first T^2 alarm)"
    )
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_comparison(
    p2_univ_sub: np.ndarray,
    univ_target: float,
    univ_lcl: float,
    univ_ucl: float,
    t2: pd.Series,
    t2_lim: float,
    univ_first_alarm: int,
    t2_first_alarm: int,
    out: pathlib.Path,
) -> None:
    """Twin-axis overlay of phase-2 univariate-tag subgroups and the multivariate T^2."""
    fig, ax1 = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(p2_univ_sub))
    minutes_per_sub = N_SUB * 0.5

    ax1.plot(x, p2_univ_sub, color="#1f77b4", linewidth=1.0, marker=".", markersize=3,
             label=f"{UNIVARIATE_TAG} (subgroup mean, left)")
    ax1.axhline(univ_target, color="#1f77b4", linestyle=":", linewidth=0.9, alpha=0.6)
    ax1.axhline(univ_ucl, color="#1f77b4", linestyle="--", linewidth=0.9, alpha=0.6)
    ax1.axhline(univ_lcl, color="#1f77b4", linestyle="--", linewidth=0.9, alpha=0.6)
    ax1.axvline(univ_first_alarm, color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.8)
    ax1.set_xlabel(f"Phase-2 subgroup index ({minutes_per_sub:.0f} min each)")
    ax1.set_ylabel(UNIVARIATE_TAG, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, t2.values, color="#d62728", linewidth=1.0, marker=".", markersize=3,
             label="Hotelling's $T^2$ (right)")
    ax2.axhline(t2_lim, color="#d62728", linestyle="--", linewidth=0.9, alpha=0.6)
    ax2.axvline(t2_first_alarm, color="#d62728", linestyle=":", linewidth=1.2, alpha=0.8)
    ax2.set_ylabel("Hotelling's $T^2$", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_title(
        f"Univariate vs multivariate alarms on phase 2: "
        f"{UNIVARIATE_TAG} alarms at subgroup {univ_first_alarm}, "
        f"T^2 alarms at subgroup {t2_first_alarm}"
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

    # 1. ACF of the univariate tag on phase 1: motivates the subgroup size.
    figure_acf(phase1, FIGURES_DIR / "Flotation-MSPC-acf.png")
    print("[1/7] ACF fig done")

    # 2. Univariate Shewhart on Pulp level (subgroups of N_SUB).
    figure_shewhart(phase1, phase2, FIGURES_DIR / "Flotation-MSPC-shewhart.png")
    print("[2/7] shewhart fig done")

    # 3-6. Multivariate PCA on phase-1 subgroup means.
    p1_sub = subgroup_means(phase1, N_SUB)
    p2_sub = subgroup_means(phase2, N_SUB)
    print(f"subgrouped phase1: {p1_sub.shape}   subgrouped phase2: {p2_sub.shape}")

    scaler = MCUVScaler().fit(p1_sub)
    model = PCA(n_components=2).fit(scaler.transform(p1_sub))
    print(f"R^2 cumulative (subgrouped): {model.r2_cumulative_.values}")

    figure_score_phase1(model, len(p1_sub), FIGURES_DIR / "Flotation-MSPC-score-phase1.png")
    print("[3/7] score plot done")

    figure_loadings(model, FIGURES_DIR / "Flotation-MSPC-loadings.png")
    print("[4/7] loadings done")

    result = model.predict(scaler.transform(p2_sub))
    t2 = result.hotellings_t2.iloc[:, -1]
    spe = result.spe
    t2_lim = float(model.hotellings_t2_limit(conf_level=CONF_LEVEL))
    spe_lim = float(model.spe_limit(conf_level=CONF_LEVEL))
    print(f"95% T^2 limit: {t2_lim:.2f}   95% SPE limit: {spe_lim:.2f}")

    figure_t2_spe(t2, spe, t2_lim, spe_lim, FIGURES_DIR / "Flotation-MSPC-t2-spe.png")
    print("[5/7] T^2/SPE done")

    flagged_t2 = t2[t2 > t2_lim]
    flagged_spe = spe[spe > spe_lim]
    first_t2_alarm = int(flagged_t2.index[0])
    first_spe_alarm = int(flagged_spe.index[0])
    print(f"first T^2 alarm at phase-2 subgroup {first_t2_alarm}")
    print(f"first SPE alarm at phase-2 subgroup {first_spe_alarm}")

    # T^2-style contribution from the library: how each X-variable contributes
    # to the (t_alarm - 0) movement in score space, weighted by 1/sqrt(explained
    # variance) per component. Library API, no manual residual arithmetic.
    t_at_alarm = result.scores.loc[first_t2_alarm].values
    contribs = model.score_contributions(t_at_alarm, weighted=True)
    print("T^2 contributions at the first T^2 alarm:")
    print(contribs.round(3).to_dict())
    figure_contribution(contribs, first_t2_alarm,
                        FIGURES_DIR / "Flotation-MSPC-contributions.png")
    print("[6/7] contributions done")

    # 7. Twin-axis comparison of phase-2 univariate-tag subgroups vs T^2.
    univ_sub_p1 = subgroup(phase1[UNIVARIATE_TAG].values, N_SUB).mean(axis=1)
    univ_sub_p2 = subgroup(phase2[UNIVARIATE_TAG].values, N_SUB).mean(axis=1)
    univ_target, univ_lcl, univ_ucl = shewhart_limits(
        univ_sub_p1,
        subgroup(phase1[UNIVARIATE_TAG].values, N_SUB).std(axis=1, ddof=1),
        N_SUB,
    )
    univ_first_alarm = int(np.where((univ_sub_p2 < univ_lcl) | (univ_sub_p2 > univ_ucl))[0][0])

    figure_comparison(
        p2_univ_sub=univ_sub_p2,
        univ_target=univ_target,
        univ_lcl=univ_lcl,
        univ_ucl=univ_ucl,
        t2=t2,
        t2_lim=t2_lim,
        univ_first_alarm=univ_first_alarm,
        t2_first_alarm=first_t2_alarm,
        out=FIGURES_DIR / "Flotation-MSPC-comparison.png",
    )
    print("[7/7] comparison fig done")

    print()
    print(f"Univariate Shewhart on {UNIVARIATE_TAG} first alarm: subgroup {univ_first_alarm}")
    print(f"Multivariate T^2 first alarm:                          subgroup {first_t2_alarm}")
    minutes_per_sub = N_SUB * 0.5
    print(f"Dividend in subgroups: {univ_first_alarm - first_t2_alarm} "
          f"(~ {(univ_first_alarm - first_t2_alarm) * minutes_per_sub:.0f} minutes earlier)")
    print()
    print("Wrote 7 PNG figures to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
