"""Bayesian-complement figures for the univariate-review chapter of the PID book.

Four committed PNGs, supporting the sections that add Bayesian estimates
as complementary tools alongside the chapter's confidence intervals:

- ``bayes-viscosity-prior-likelihood-posterior.png``: two panels for the
  nine viscosity values (23, 19, ..., 18) with the standard deviation
  taken as known (sigma = 3.5). Left: a prior so wide it expresses no
  preference, so the posterior lies on top of the likelihood. Right: an
  informative prior N(18, 2^2) from long-run plant records; the
  posterior N(19.49, 1.01^2) sits between the prior and the sample
  mean, with its central 95% region shaded.
- ``bayes-viscosity-sequential-updating.png``: the posterior for the
  viscosity mean after 0, 1, 3, 5 and 9 of the observations have been
  processed one at a time, starting from the N(18, 2^2) prior. The
  final curve is the same posterior as the right panel above.
- ``bayes-yield-difference-posterior.png``: the posterior for the
  difference in long-run yields, mu_B - mu_A, of the two batch reactor
  control systems: a t-distribution with 18 degrees of freedom centred
  at 3.04 with scale 3.02. The area to the right of zero (84%) is
  shaded; the left tail (16%) matches the pooled-variance calculation
  earlier in the same section.
- ``bayes-proportion-beta-posterior.png``: the Beta(195, 7) posterior
  for the proportion of acceptable tablets after inspecting 200 and
  finding 194 acceptable, starting from a uniform Beta(1, 1) prior.
  The central 95% region [0.936, 0.986] is shaded and the asymmetry
  around the sample proportion 0.97 is visible.

All curves are closed-form densities (no random numbers), so the
committed images regenerate exactly.

Usage
-----
    uv run --with numpy --with scipy --with matplotlib python bayesian_univariate_figures.py [output_dir]

Writes the four PNGs into ``output_dir`` (default: this script's own
directory), refreshing the committed images in place.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Okabe-Ito colourblind-safe palette entries used in these figures.
BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GREY = "#666666"
GRID = "#DDDDDD"

DPI = 300

# The nine viscosity values quoted in the chapter (mean 20.0).
VISCOSITY = np.array([23, 19, 17, 18, 24, 26, 21, 14, 18])
SIGMA = 3.5  # the chapter's known-standard-deviation case

# Informative prior from long-run plant records of the same polymer grade.
PRIOR_MEAN, PRIOR_SD = 18.0, 2.0

# Font sizes match the other univariate figure modules.
mpl.rcParams.update(
    {
        "font.size": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": 18,
        "axes.axisbelow": True,
    }
)


def save(fig, outdir: pathlib.Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote {outdir / name}")


def conjugate_update(m0: float, s0: float, xbar: float, n: int, sigma: float):
    """Posterior (mean, sd) for a N(m0, s0^2) prior and n observations."""
    precision = 1 / s0**2 + n / sigma**2
    mean = (m0 / s0**2 + n * xbar / sigma**2) / precision
    return mean, 1 / np.sqrt(precision)


def prior_likelihood_posterior(outdir: pathlib.Path) -> None:
    n = len(VISCOSITY)
    xbar = VISCOSITY.mean()
    mu = np.linspace(12, 26, 701)
    # The likelihood of the nine values, as a function of mu, scaled to
    # unit area: a normal curve centred at the sample mean.
    likelihood = stats.norm.pdf(mu, loc=xbar, scale=SIGMA / np.sqrt(n))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True, sharex=True)

    # Left: a prior so wide it expresses no preference among plausible
    # values, so the posterior falls on top of the likelihood.
    ax = axes[0]
    ax.grid(color=GRID, linewidth=0.8)
    wide = stats.norm.pdf(mu, loc=xbar, scale=50)
    ax.plot(mu, wide, color=GREY, linestyle="--", linewidth=2, label="Prior (very wide)")
    ax.plot(mu, likelihood, color=ORANGE, linewidth=5, alpha=0.55, label="Likelihood of the data")
    ax.plot(mu, likelihood, color=BLUE, linewidth=2, label="Posterior")
    ax.set_title("Weak prior", fontsize=20)
    ax.set_xlabel(r"Viscosity mean, $\mu$")
    ax.set_ylabel("Probability density")
    ax.legend(loc="upper left", frameon=False, fontsize=15)

    # Right: the informative prior pulls the posterior toward 18, and
    # the extra information narrows it.
    ax = axes[1]
    ax.grid(color=GRID, linewidth=0.8)
    prior = stats.norm.pdf(mu, loc=PRIOR_MEAN, scale=PRIOR_SD)
    m_post, s_post = conjugate_update(PRIOR_MEAN, PRIOR_SD, xbar, n, SIGMA)
    posterior = stats.norm.pdf(mu, loc=m_post, scale=s_post)
    lo, hi = stats.norm.ppf([0.025, 0.975], loc=m_post, scale=s_post)
    region = (mu >= lo) & (mu <= hi)
    ax.fill_between(mu[region], posterior[region], color=BLUE, alpha=0.15)
    ax.plot(mu, prior, color=GREY, linestyle="--", linewidth=2, label="Prior N(18, $2^2$)")
    ax.plot(mu, likelihood, color=ORANGE, linewidth=2.5, label="Likelihood of the data")
    ax.plot(mu, posterior, color=BLUE, linewidth=2.5, label="Posterior")
    ax.axvline(PRIOR_MEAN, color=GREY, linewidth=1, linestyle=":")
    ax.axvline(xbar, color=ORANGE, linewidth=1, linestyle=":")
    ax.annotate(
        "posterior mean = 19.5,\nbetween 18 and 20",
        xy=(m_post - s_post, stats.norm.pdf(s_post, 0, s_post)),
        xytext=(12.3, 0.22), fontsize=15,
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.text(m_post, 0.05, "95%", ha="center", fontsize=15, color=BLUE)
    ax.set_title("Informative prior", fontsize=20)
    ax.set_xlabel(r"Viscosity mean, $\mu$")
    ax.legend(loc="upper left", frameon=False, fontsize=15)
    save(fig, outdir, "bayes-viscosity-prior-likelihood-posterior.png")


def sequential_updating(outdir: pathlib.Path) -> None:
    mu = np.linspace(12, 26, 701)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.grid(color=GRID, linewidth=0.8)

    stops = {0: None, 1: None, 3: None, 5: None, 9: None}
    m, v = PRIOR_MEAN, PRIOR_SD**2
    stops[0] = (m, np.sqrt(v))
    for i, x in enumerate(VISCOSITY, 1):
        v_new = 1 / (1 / v + 1 / SIGMA**2)
        m = (m / v + x / SIGMA**2) * v_new
        v = v_new
        if i in stops:
            stops[i] = (m, np.sqrt(v))

    # Colour ramp from the grey prior toward the blue final posterior.
    shades = ["#999999", "#8CA6C4", "#5D8DB9", "#2E7AB5", BLUE]
    widths = [2, 2, 2, 2, 3.5]
    styles = ["--", "-", "-", "-", "-"]
    for (k, (mean, sd)), colour, lw, ls in zip(stops.items(), shades, widths, styles):
        label = "Prior (0 observations)" if k == 0 else f"After {k} observation{'s' if k > 1 else ''}"
        ax.plot(mu, stats.norm.pdf(mu, mean, sd), color=colour, linewidth=lw,
                linestyle=ls, label=label)

    final_mean, final_sd = stops[9]
    ax.annotate(
        f"after all 9: mean = {final_mean:.1f}\nsame as processing\nthem all at once",
        xy=(final_mean + final_sd, stats.norm.pdf(final_sd, 0, final_sd)),
        xytext=(22.3, 0.30), fontsize=15,
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.set_xlabel(r"Viscosity mean, $\mu$")
    ax.set_ylabel("Probability density")
    ax.set_title("The posterior narrows as observations arrive", fontsize=20)
    ax.legend(loc="upper left", frameon=False, fontsize=15)
    save(fig, outdir, "bayes-viscosity-sequential-updating.png")


def yield_difference_posterior(outdir: pathlib.Path) -> None:
    centre, scale, df = 3.04, 3.02, 18
    d = np.linspace(-8, 14, 881)
    density = stats.t.pdf((d - centre) / scale, df) / scale

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.grid(color=GRID, linewidth=0.8)
    right = d >= 0
    ax.fill_between(d[right], density[right], color=BLUE, alpha=0.25)
    ax.plot(d, density, color=BLUE, linewidth=3)
    ax.axvline(0, color=VERMILLION, linestyle="--", linewidth=2)
    ax.annotate(
        r"$p(\mu_B - \mu_A > 0 \,|\, \mathrm{data}) = 84\%$",
        xy=(3.5, 0.055), xytext=(6.1, 0.105), fontsize=17,
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.annotate(
        "16%: the pooled-variance\ncalculation, seen from\nthe other side",
        xy=(-1.8, 0.02), xytext=(-8.0, 0.075), fontsize=15,
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.set_xlabel(r"Difference in long-run yields, $\mu_B - \mu_A$ [%]")
    ax.set_ylabel("Posterior probability density")
    ax.set_title("Posterior for the difference between systems B and A", fontsize=20)
    save(fig, outdir, "bayes-yield-difference-posterior.png")


def proportion_beta_posterior(outdir: pathlib.Path) -> None:
    a, b = 195, 7
    p = np.linspace(0.85, 1.0, 601)
    posterior = stats.beta.pdf(p, a, b)
    lo, hi = stats.beta.ppf([0.025, 0.975], a, b)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.grid(color=GRID, linewidth=0.8)
    region = (p >= lo) & (p <= hi)
    ax.fill_between(p[region], posterior[region], color=BLUE, alpha=0.15)
    ax.plot(p, posterior, color=BLUE, linewidth=3, label="Posterior Beta(195, 7)")
    ax.plot(p, np.ones_like(p), color=GREY, linestyle="--", linewidth=2,
            label="Prior Beta(1, 1): uniform")
    ax.axvline(0.97, color=VERMILLION, linestyle=":", linewidth=2)
    ax.text(0.9705, 26, r"$\hat{p} = 0.97$", color=VERMILLION, fontsize=16)
    ax.annotate(
        "the density is not symmetric:\nthe left tail is longer, and the\ncurve is zero beyond $p = 1$",
        xy=(0.935, 5.5), xytext=(0.856, 16), fontsize=15,
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.text((lo + hi) / 2, 2.5, "95%", ha="center", fontsize=16, color=BLUE)
    ax.set_xlabel("Proportion of acceptable tablets, $p$")
    ax.set_ylabel("Probability density")
    ax.set_title("Posterior for the proportion after 194 of 200 pass", fontsize=20)
    ax.legend(loc="upper left", frameon=False, fontsize=15)
    save(fig, outdir, "bayes-proportion-beta-posterior.png")


if __name__ == "__main__":
    outdir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent
    prior_likelihood_posterior(outdir)
    sequential_updating(outdir)
    yield_difference_posterior(outdir)
    proportion_beta_posterior(outdir)
