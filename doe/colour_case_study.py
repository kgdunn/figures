"""Shared construction, response simulation, and modelling for the mixed-level
split-plot case study (``design-analysis-experiments/mixed-level-profile-case-study.rst``).

The scenario: an analytical laboratory develops a coloured metal-chelate complex with an
arylazo *chromogen* and follows the colour as it develops, logging absorbance at the
complex maximum-absorbance wavelength at ten time points from mixing to plateau. Six
candidate chromogens (a reference and five single-substituent analogs) are screened against
four continuous process factors, two of which are hard to change, so the design is a
split-plot mixed-level optimal design.

Everything here is built with Kevin Dunn's ``process_improve`` library
(``pip install 'process-improve[expt]'`` plus a separate ``pip install pyoptex`` for the
coordinate-exchange optimiser): ``generate_design`` builds the mixed-level split-plot
optimal design directly (v1.52.0+ routes a categorical factor and a per-factor quadratic
model through the public entry point), ``evaluate_design`` scores D/I/G-efficiency and the
prediction variance integrated over the whole region (the FDS curve), and the multivariate
``PLS`` models the ten-point colour-development profile with ``scale=True`` (v1.51.3+ scales
both blocks; v1.51.4+ reports ``rmse_`` on the original response scale).

The response is simulated from a fixed ground truth so the recovered effects can be checked
against what was injected. Deterministic: the coordinate exchange is seeded through NumPy's
global generator and the simulation through an explicit ``default_rng``.

Run ``check_colour_case_study.py`` to print the numbers quoted in the chapter; run the
``colour-*.py`` scripts to regenerate the figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from process_improve.experiments import Factor, evaluate_design, generate_design
from process_improve.multivariate.methods import PLS

# ---------------------------------------------------------------------------
# Factor space: one six-level categorical chromogen + four continuous factors,
# two of them (co-solvent fraction and temperature) hard to change.
# ---------------------------------------------------------------------------
COMPOUND_LEVELS = ["A", "B", "C", "D", "E", "F"]

#: Plausible arylazo (pyridylazo-resorcinol) chromogen family; A is the incumbent.
COMPOUND_NAMES = {
    "A": "4-(2-pyridylazo)benzene-1,3-diol",
    "B": "4-(2-pyridylazo)-6-methylbenzene-1,3-diol",
    "C": "4-(2-thiazolylazo)benzene-1,3-diol",
    "D": "4-(2-pyridylazo)-6-chlorobenzene-1,3-diol",
    "E": "4-(5-methyl-2-pyridylazo)benzene-1,3-diol",
    "F": "4-(2-quinolylazo)benzene-1,3-diol",
}

#: Continuous factors: name -> (low, high, units).
CONT = {
    "concentration": (2.0, 8.0, "umol/L"),
    "co_solvent": (5.0, 25.0, "% v/v"),
    "pH": (4.0, 7.0, "-"),
    "temperature": (15.0, 35.0, "degC"),
}
HARD_TO_CHANGE = ["co_solvent", "temperature"]

FACTORS = [Factor(name="compound", type="categorical", levels=COMPOUND_LEVELS)] + [
    Factor(name=n, type="continuous", low=lo, high=hi, units=u) for n, (lo, hi, u) in CONT.items()
]

# ---------------------------------------------------------------------------
# Colour-development ground truth: absorbance at the complex lambda-max, logged
# at ten equally spaced time points from mixing (t0) to plateau (t9).
# ---------------------------------------------------------------------------
TIME_POINTS = np.arange(10)

#: Reference rise-to-plateau shape (fast chelation, levelling off), unit peak.
_REF = 1.0 - np.exp(-TIME_POINTS / 2.0)
REF_SHAPE = _REF / _REF.max()

#: Late-drift basis: extra absorbance at long times (a slow secondary reaction).
_TAIL = np.clip((TIME_POINTS - 4) / 5.0, 0.0, None)
TAIL_BASIS = _TAIL / _TAIL.max()

#: Per-compound truth. ``drift`` is the late-time shape departure from the
#: reference (objective 4); the three slopes are how strongly co-solvent, pH and
#: temperature move the colour amplitude for that compound (objectives 1-3).
GROUND_TRUTH = {
    "A": {"drift": 0.00, "co_solvent": -0.05, "pH": -0.06, "temperature": +0.02},
    "B": {"drift": 0.05, "co_solvent": -0.08, "pH": -0.10, "temperature": +0.03},
    "C": {"drift": 0.20, "co_solvent": -0.20, "pH": -0.28, "temperature": +0.12},
    "D": {"drift": 0.30, "co_solvent": -0.25, "pH": -0.30, "temperature": +0.14},
    "E": {"drift": 0.35, "co_solvent": -0.10, "pH": -0.08, "temperature": +0.05},
    "F": {"drift": -0.10, "co_solvent": -0.15, "pH": -0.22, "temperature": -0.10},
}
_BASE_AMP = 1.0        # baseline colour amplitude at the centre point
_CONC_SLOPE = 0.35     # concentration raises amplitude for every compound
_NOISE_SD = 0.03       # absorbance measurement noise


def build_design(criterion: str = "i_optimal", budget: int = 60, *, seed: int = 42):
    """Build the mixed-level split-plot optimal design through the public API.

    ``criterion`` is ``"i_optimal"`` or ``"d_optimal"``; ``budget`` is the run count.
    The coordinate exchange draws its restarts from NumPy's global generator, so we
    seed it here for reproducibility.
    """
    np.random.seed(seed)  # noqa: NPY002  (pyoptex reads the legacy global RNG)
    return generate_design(
        FACTORS,
        design_type=criterion,
        budget=budget,
        hard_to_change=HARD_TO_CHANGE,
        model_type="quadratic",
    )


def simulate_curves(design, *, seed: int = 20260710) -> pd.DataFrame:
    """Return an (n_runs x 10) colour-development profile from the ground truth.

    The design's continuous columns are coded to [-1, 1]; amplitude is a
    compound-specific linear function of them, and the curve shape is the
    reference rise-to-plateau plus a compound-specific late drift.
    """
    rng = np.random.default_rng(seed)
    coded = design.design
    rows = []
    for _, r in coded.iterrows():
        g = GROUND_TRUTH[r["compound"]]
        amp = (_BASE_AMP + _CONC_SLOPE * r["concentration"] + g["co_solvent"] * r["co_solvent"]
               + g["pH"] * r["pH"] + g["temperature"] * r["temperature"])
        amp = max(amp, 0.05)
        shape = np.clip(REF_SHAPE + g["drift"] * TAIL_BASIS, 0.0, None)
        rows.append(amp * shape + rng.normal(0.0, _NOISE_SD, TIME_POINTS.size))
    cols = [f"t{t}" for t in TIME_POINTS]
    return pd.DataFrame(np.vstack(rows), columns=cols, index=coded.index)


def model_matrix(design) -> pd.DataFrame:
    """X for PLS: coded continuous factors + one-hot compound (reference A dropped)."""
    coded = design.design
    X = coded[list(CONT)].astype(float).copy()
    dummies = pd.get_dummies(coded["compound"], prefix="cmp").astype(float).drop(columns=["cmp_A"])
    return pd.concat([X, dummies], axis=1)


def fit_profile_pls(design, curves, *, n_components: int = 5) -> PLS:
    """PLS on raw X and raw curves; scale=True standardises both blocks internally."""
    return PLS(n_components=n_components, scale=True).fit(model_matrix(design), curves)


def evaluate(design, *, n_samples: int = 60_000, seed: int = 1) -> dict:
    """D/I/G-efficiency, degrees of freedom, and the FDS quantiles for a design."""
    return evaluate_design(
        design,
        model="quadratic",
        metric=["d_efficiency", "i_efficiency", "g_efficiency", "degrees_of_freedom", "fds"],
        n_samples=n_samples,
        random_seed=seed,
    )


def mean_curves(design, curves) -> pd.DataFrame:
    """Mean colour-development curve per compound (rows = compounds, cols = t0..t9)."""
    joined = pd.concat([design.design["compound"], curves], axis=1)
    return joined.groupby("compound")[list(curves.columns)].mean()


def shape_distance_to_reference(design, curves) -> pd.Series:
    """Objective 4: distance from each compound's unit-peak-normalised mean curve to A's."""
    m = mean_curves(design, curves)
    shapes = m.div(m.max(axis=1), axis=0)
    ref = shapes.loc["A"]
    return (((shapes - ref) ** 2).sum(axis=1) ** 0.5).sort_values()
