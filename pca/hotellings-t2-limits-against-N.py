"""The three 95% Hotelling's T-squared limits against the number of observations.

One committed PNG, ``hotellings-t2-limits-against-N.png``, for the section of
the PCA chapter that explains why a new observation needs a wider T-squared
limit than an observation used to build the model (``LVM-Hotellings-T2``).

For a model of ``N`` observations and ``A = 2`` components it draws:

- the beta limit, for the observations used to build the model:
  ``(N - 1)^2 / N * B(A/2, (N - A - 1)/2)``;
- the limit ``A (N - 1) / (N - A) * F(A, N - A)``, often drawn for those same
  observations;
- the F limit for a new observation, ``A (N^2 - 1) / (N (N - A)) * F(A, N - A)``;
- the bound ``(N - 1)^2 / N``, the largest T-squared an observation used to build
  the model can have.

The middle limit lies above that bound for small ``N`` (up to ``N = 11`` here),
where it cannot flag any of the observations it is drawn for. The book's code
block computes the same four curves with ``scipy`` alone; this script checks
the beta and F curves against ``process_improve``'s ``hotellings_t2_limit``
(``judged_rows="fitted"`` and ``"new"``), so a change to the library's formulas
fails here instead of leaving the figure on the old ones.

Usage
-----
    uv run --with 'process-improve[all]' python hotellings-t2-limits-against-N.py [output_dir]
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from process_improve.multivariate.methods import hotellings_t2_limit
from scipy import stats

BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GREY = "#666666"
GRID = "#DDDDDD"

DPI = 300
HERE = pathlib.Path(__file__).parent
CONF_LEVEL = 0.95
N_COMPONENTS = 2
N_OBS = np.arange(N_COMPONENTS + 2, 31)  # the beta limit needs N > A + 1
Y_MAX = 25.0

mpl.rcParams.update(
    {
        "font.size": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12.5,
        "axes.axisbelow": True,
    }
)


def t2_limits(n_obs: int, n_comp: int, conf_level: float = CONF_LEVEL) -> tuple[float, float, float, float]:
    """The beta, middle and F limits, and the largest T2 of an observation in the model (as the book)."""
    largest = (n_obs - 1) ** 2 / n_obs
    fitted = largest * stats.beta.ppf(conf_level, n_comp / 2, (n_obs - n_comp - 1) / 2)
    middle = n_comp * (n_obs - 1) / (n_obs - n_comp) * stats.f.ppf(conf_level, n_comp, n_obs - n_comp)
    new = middle * (n_obs + 1) / n_obs
    for judged_rows, value in (("fitted", fitted), ("new", new)):
        library = hotellings_t2_limit(conf_level, n_components=n_comp, n_rows=n_obs, judged_rows=judged_rows)
        if not np.isclose(library, value, rtol=1e-9):
            raise ValueError(
                f"process_improve's {judged_rows!r} limit for N = {n_obs} is {library:.6f}, not {value:.6f}: "
                "the library changed its formula, so update t2_limits here and in the book."
            )
    return fitted, middle, new, largest


def main(outdir: pathlib.Path) -> None:
    curves = np.array([t2_limits(int(n), N_COMPONENTS) for n in N_OBS])
    above = N_OBS[curves[:, 1] > curves[:, 3]]
    print(f"A = {N_COMPONENTS}: the A(N-1)/(N-A) F limit lies above the bound up to N = {above.max()}")
    for n in (10, 20):
        row = curves[N_OBS == n][0]
        print(f"N = {n}: beta {row[0]:.2f}, middle {row[1]:.2f}, new {row[2]:.2f}, bound {row[3]:.2f}")

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.grid(color=GRID, linewidth=0.8)
    # Where the middle limit lies above the bound, and so can flag none of the observations it is drawn for.
    crossing = int(above.max())
    ax.axvspan(N_OBS[0] - 0.5, crossing + 0.5, color=ORANGE, alpha=0.10, linewidth=0)
    # The note sits right of the band, in the corner no curve reaches, with an arrow into the band.
    ax.annotate(f"$A(N-1)/(N-A)\\,F$ above the bound\nup to $N$ = {crossing}", (crossing - 0.6, 22.0),
                xytext=(crossing + 1.6, 22.0), ha="left", va="center", fontsize=11.5, color=GREY,
                arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.0})
    ax.plot(N_OBS, curves[:, 3], ":", color=GREY, linewidth=2.2,
            label="Bound, $(N-1)^2/N$")
    ax.plot(N_OBS, curves[:, 2], "-", color=VERMILLION, linewidth=2.4, label="New observation ($F$)")
    ax.plot(N_OBS, curves[:, 1], "--", color=ORANGE, linewidth=2.4, label="$A(N-1)/(N-A)\\,F$")
    ax.plot(N_OBS, curves[:, 0], "-", color=BLUE, linewidth=2.4,
            label="Used to build the model (beta)")
    ax.set_xlim(N_OBS[0] - 0.5, N_OBS[-1] + 0.5)
    ax.set_ylim(0, Y_MAX)
    ax.set_xticks(np.arange(5, N_OBS[-1] + 1, 5))
    ax.set_xlabel("Number of observations used to build the model, $N$")
    ax.set_ylabel(f"{CONF_LEVEL:.0%} limit for $T^2$, $A$ = {N_COMPONENTS}")
    # In the gap between the F curves and the bound, where no curve runs.
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.37), frameon=False)
    out = outdir / "hotellings-t2-limits-against-N.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
