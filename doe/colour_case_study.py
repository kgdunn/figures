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
from patsy import build_design_matrices, dmatrix
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
    "G": "4-(2-pyridylazo)-6-bromobenzene-1,3-diol",  # hypothetical seventh chromogen (see GROUND_TRUTH)
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
    # G is a hypothetical seventh chromogen, kept for the "adding a new chromogen" discussion. It is
    # NOT one of the six levels in COMPOUND_LEVELS and takes no part in the 60-run design; its ground
    # truth is defined here so a small block of G runs can be simulated when the augmentation is done.
    "G": {"drift": 0.12, "co_solvent": -0.16, "pH": -0.24, "temperature": +0.08},
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


# ---------------------------------------------------------------------------
# Interaction model, goal projection, and model inversion (objective 4 revisited).
# The compound-by-factor interaction PLS is the model used for the scores/loadings,
# the SPE/T2 diagnostics, and the inversion that finds the continuous-factor settings
# which make each candidate reproduce the reference chromogen's goal profile.
# ---------------------------------------------------------------------------

#: Right-hand side shared by the interaction analysis, its PLS twin, and the inversion.
RHS = ("C(compound, Sum)*co_solvent + C(compound, Sum)*pH "
       "+ C(compound, Sum)*temperature + concentration")


def _analysis_frame(design) -> pd.DataFrame:
    """design.design reduced to compound (fixed-level categorical) + the four continuous factors."""
    adf = design.design[["compound"] + list(CONT)].copy()
    adf["compound"] = pd.Categorical(adf["compound"].astype(str), categories=COMPOUND_LEVELS)
    return adf


def interaction_matrix(design):
    """Return (X_int, design_info): the 24-term interaction model matrix (Intercept dropped) and
    the patsy design_info, so new rows are encoded with the same contrasts and column order."""
    full = dmatrix(RHS, _analysis_frame(design), return_type="dataframe")
    return full.drop(columns=["Intercept"]), full.design_info


def encode_row(design_info, compound: str, coded) -> pd.DataFrame:
    """One X_int row for ``compound`` at coded continuous settings [concentration, co_solvent,
    pH, temperature], using the fitted design_info so the coding matches the training matrix.

    The ``Intercept`` column (present for sum/treatment coding, absent for cell-means) is dropped,
    to match the fitted matrices, which drop it because ``PLS`` centres the columns."""
    row = pd.DataFrame({
        "compound": pd.Categorical([compound], categories=COMPOUND_LEVELS),
        "concentration": [coded[0]], "co_solvent": [coded[1]], "pH": [coded[2]], "temperature": [coded[3]],
    })
    full = build_design_matrices([design_info], row, return_type="dataframe")[0]
    return full.drop(columns=["Intercept"], errors="ignore")


# ---------------------------------------------------------------------------
# Coding the categorical factor (the coding-sensitivity subsection).
# The six-level chromogen can be written with three contrast codings that span the
# same model space. Ordinary least squares (a full-rank fit) is invariant to the choice;
# a truncated, scaled PLS is not, which shows up in the SPE/T2 diagnostics and in the
# score-matching model inversion. Only a full-rank curve match is coding-invariant.
# ---------------------------------------------------------------------------

#: The three codings, as interaction right-hand sides. ``sum`` writes each compound as a
#: departure from the average (the last level is dropped and carried as the negative sum of
#: the rest); ``treatment`` measures each compound against a reference level (all-zero row);
#: ``cell_means`` gives every compound its own indicator (no intercept, no dropped level).
CODINGS = {
    "sum": ("C(compound, Sum)*co_solvent + C(compound, Sum)*pH "
            "+ C(compound, Sum)*temperature + concentration"),
    "treatment": ("C(compound, Treatment)*co_solvent + C(compound, Treatment)*pH "
                  "+ C(compound, Treatment)*temperature + concentration"),
    "cell_means": ("0 + C(compound)*co_solvent + C(compound)*pH "
                   "+ C(compound)*temperature + concentration"),
}


def coded_matrix(design, coding: str = "sum", *, order=None):
    """Return (X, design_info) for a chosen categorical ``coding`` ("sum", "treatment",
    "cell_means"). ``order`` overrides the level order (e.g. reversed, to drop A instead of F
    under sum coding); the ``Intercept`` column, when present, is dropped as in ``interaction_matrix``."""
    adf = _analysis_frame(design)
    if order is not None:
        adf["compound"] = pd.Categorical(adf["compound"].astype(str), categories=list(order))
    full = dmatrix(CODINGS[coding], adf, return_type="dataframe")
    return full.drop(columns=["Intercept"], errors="ignore"), full.design_info


def fit_coding(design, curves, coding: str = "sum", *, order=None, n_components: int = 3):
    """Fit the 3-component interaction PLS under one coding; return (pls, design_info)."""
    X, design_info = coded_matrix(design, coding, order=order)
    return PLS(n_components=n_components, scale=True).fit(X, curves), design_info


def ground_truth_curve(compound: str, coded, *, noise_sd: float = 0.0, rng=None) -> np.ndarray:
    """The withheld ground-truth colour curve for ``compound`` at coded continuous settings
    [concentration, co_solvent, pH, temperature].

    The amplitude model is linear by construction, so this extrapolates smoothly for ``|coded| > 1``:
    that is the point of validating an inversion that steps outside the studied window (a designed
    experiment establishes the factor effects, so mild extrapolation is a testable next-run
    hypothesis, as in RSM steepest-ascent exploration). ``noise_sd = 0`` gives the clean curve for a
    like-for-like comparison against the reference.
    """
    g = GROUND_TRUTH[compound]
    amp = (_BASE_AMP + _CONC_SLOPE * coded[0] + g["co_solvent"] * coded[1]
           + g["pH"] * coded[2] + g["temperature"] * coded[3])
    amp = max(amp, 0.05)
    curve = amp * np.clip(REF_SHAPE + g["drift"] * TAIL_BASIS, 0.0, None)
    if noise_sd and rng is not None:
        curve = curve + rng.normal(0.0, noise_sd, curve.size)
    return curve


def shape_floor(compound: str, reference: str = "A") -> tuple[float, float]:
    """Smallest RMSE to the ``reference`` (at its centre point) that ``compound`` can reach at ANY
    amplitude: the limit set by its fixed late-time shape (drift), which no continuous-factor setting
    can move. Returns (rmse, best_amplitude). This is the floor an inversion can approach but not beat.
    """
    goal = np.clip(REF_SHAPE + GROUND_TRUTH[reference]["drift"] * TAIL_BASIS, 0.0, None)
    shp = np.clip(REF_SHAPE + GROUND_TRUTH[compound]["drift"] * TAIL_BASIS, 0.0, None)
    a = float(shp @ goal / (shp @ shp))
    return float(np.sqrt(np.mean((a * shp - goal) ** 2))), a


def curve_match_inversion(design, curves, coding: str = "sum",
                          compounds=("B", "C", "D", "E", "F")) -> pd.DataFrame:
    """Coding-invariant inversion: for each compound, the continuous settings whose *predicted
    ten-point curve* is closest (least squares) to the reference goal curve (chromogen A at the
    centre point).

    Uses the full-rank ordinary-least-squares fit of the interaction model, so the predicted curve,
    and therefore this inversion, does not depend on the categorical coding (verify by passing a
    different ``coding``: the coded settings agree to ~1e-13). The curve match is over-determined
    (ten time points, four factors), so it returns the least-squares closest curve rather than an
    exact score match. Columns match :func:`invert_to_factors`.
    """
    adf = _analysis_frame(design)
    full = dmatrix(CODINGS[coding], adf, return_type="dataframe")   # keep the intercept: full rank
    design_info = full.design_info
    beta = np.linalg.lstsq(full.to_numpy(), curves.to_numpy(), rcond=None)[0]

    def curve_of(compound, coded):
        row = pd.DataFrame({
            "compound": pd.Categorical([compound], categories=COMPOUND_LEVELS),
            "concentration": [coded[0]], "co_solvent": [coded[1]], "pH": [coded[2]], "temperature": [coded[3]],
        })
        x = build_design_matrices([design_info], row, return_type="dataframe")[0].to_numpy().ravel()
        return x @ beta

    y_goal = curve_of("A", [0.0, 0.0, 0.0, 0.0])
    recs = {}
    for cmp in compounds:
        y0 = curve_of(cmp, [0.0, 0.0, 0.0, 0.0])
        jac = np.column_stack([curve_of(cmp, [float(k == j) for k in range(4)]) - y0 for j in range(4)])
        c_sol, *_ = np.linalg.lstsq(jac, y_goal - y0, rcond=None)
        real = coded_to_real(c_sol)
        rec = {f"{n}_coded": float(c_sol[i]) for i, n in enumerate(CONT)}
        rec.update({n: float(real[n]) for n in CONT})
        rec["resid"] = float(np.linalg.norm(jac @ c_sol - (y_goal - y0)))
        rec["in_range"] = bool(np.all(np.abs(c_sol) <= 1.0 + 1e-9))
        recs[cmp] = rec
    return pd.DataFrame(recs).T


def coded_to_real(coded) -> dict:
    """Map coded [-1, 1] continuous settings to real units using the CONT low/high ranges."""
    out = {}
    for name, val in zip(CONT, coded):
        lo, hi, _ = CONT[name]
        out[name] = (lo + hi) / 2 + (hi - lo) / 2 * val
    return out


def project_point(pls, x_row: pd.DataFrame) -> dict:
    """Score, SPE, and Hotelling's T2 for a single new observation ``x_row`` (an X_int row).

    SPE and T2 are computed directly from the fitted loadings and score scaling rather than through
    ``pls.diagnose``: as of the version used here ``diagnose().spe`` returns 0 for every row, so it
    cannot be trusted for the goal check (verified against the fitted ``spe_``). The score and T2
    match ``transform``/``diagnose``; only SPE is recomputed. SPE uses the same reconstruction as the
    fitted ``spe_`` (``sqrt`` of the summed squared residual of the standardized X).
    """
    cols = list(pls.direct_weights_.index)
    center = pls._x_scaler.center_.reindex(cols).to_numpy()
    scale = pls._x_scaler.scale_.reindex(cols).to_numpy()
    weights = pls.direct_weights_.to_numpy()
    loadings = pls.x_loadings_.to_numpy()
    scaling = np.asarray(pls.scaling_factor_for_scores_).ravel()
    xs = (x_row.reindex(columns=cols, fill_value=0.0).to_numpy().ravel() - center) / scale
    score = xs @ weights
    resid = xs - score @ loadings.T
    return {
        "score": score,
        "spe": float(np.sqrt((resid ** 2).sum())),
        "t2": float(((score / scaling) ** 2).sum()),
    }


def goal_projection(pls, design_info) -> dict:
    """Project chromogen A at the centre point (all continuous at coded 0) onto the fitted PLS.

    Returns the goal score, its SPE and Hotelling's T2, and the 95% limits, so the caller can
    confirm the goal sits inside the model before using the prediction as an inversion target.
    """
    goal_x = encode_row(design_info, "A", [0.0, 0.0, 0.0, 0.0])
    p = project_point(pls, goal_x)
    return {
        "score": p["score"],
        "spe": p["spe"],
        "t2": p["t2"],
        "spe_limit": float(pls.spe_limit(0.95)),
        "t2_limit": float(pls.hotellings_t2_limit(0.95)),
    }


def invert_to_factors(pls, design_info, target_score, compounds=("B", "C", "D", "E", "F")) -> pd.DataFrame:
    """Minimum-adjustment inversion: for each compound, the coded continuous settings whose score
    matches ``target_score``.

    With sum coding and the compound fixed, the score map is affine in the four continuous factors,
    so ``score(c) = t0 + M c`` (M is n_components x 4). Matching a 3-component target leaves one
    free direction (the operating window); ``lstsq`` returns the minimum-norm (least-adjustment)
    solution. Columns: the four coded settings, the four real-unit settings, the score residual,
    and ``in_range`` (all coded settings within [-1, 1]).
    """
    cols = list(pls.direct_weights_.index)
    center = pls._x_scaler.center_.reindex(cols).to_numpy()
    scale = pls._x_scaler.scale_.reindex(cols).to_numpy()
    weights = pls.direct_weights_.to_numpy()
    target = np.asarray(target_score)

    def score_of(compound, coded):
        x = encode_row(design_info, compound, coded).reindex(columns=cols, fill_value=0.0).to_numpy().ravel()
        return ((x - center) / scale) @ weights

    recs = {}
    for cmp in compounds:
        t0 = score_of(cmp, [0.0, 0.0, 0.0, 0.0])
        jac = np.column_stack([score_of(cmp, [float(k == j) for k in range(4)]) - t0 for j in range(4)])
        c_sol, *_ = np.linalg.lstsq(jac, target - t0, rcond=None)
        real = coded_to_real(c_sol)
        rec = {f"{n}_coded": float(c_sol[i]) for i, n in enumerate(CONT)}
        rec.update({n: float(real[n]) for n in CONT})
        rec["resid"] = float(np.linalg.norm(jac @ c_sol - (target - t0)))
        rec["in_range"] = bool(np.all(np.abs(c_sol) <= 1.0 + 1e-9))
        recs[cmp] = rec
    return pd.DataFrame(recs).T
