"""The worked screen-then-optimise study, reproduced for its figures.

The book section "A worked screen-then-optimise study"
(design-analysis-experiments/screen-then-optimise-study.rst) screens six factors of a simulated
reactor with a definitive screening design, builds a second design inside a heat-release limit,
and chooses settings by desirability. The three figure scripts beside this module each draw one
view of that study, so the study lives here once, written as the chapter's code blocks write it:
the same simulator, the same seeds and the same calls, in the same order. ``study()`` then checks
the values the chapter prints, so a figure cannot drift away from the text it illustrates.

Import from a figure script run in this directory::

    from screen_then_optimise_common import study
    S = study()          # a dict of everything the chapter computes, checked
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from patsy import dmatrix

from process_improve.experiments import (
    Constraint, DesignRegion, Factor, analyze_experiment, analyze_omars, evaluate_design,
    generate_design, optimize_responses,
)

# Okabe-Ito, matching the other figures in this chapter.
BLUE, ORANGE, GREEN, VERMILION = "#0072B2", "#E69F00", "#009E73", "#D55E00"
SKY, PURPLE, GREY, SPINE = "#56B4E9", "#CC79A7", "#666666", "#98A2AB"

FACTORS = [
    Factor(name="T", low=150, high=200, units="degC"),
    Factor(name="t", low=10, high=40, units="min"),
    Factor(name="C", low=0.5, high=2.0, units="mol%"),
    Factor(name="P", low=2, high=10, units="bar"),
    Factor(name="S", low=300, high=900, units="rpm"),
    Factor(name="R", low=1.0, high=1.5),
]
NAMES = [f.name for f in FACTORS]
ACTIVE = FACTORS[:3]
QUADRATIC = "T + t + C + T:t + T:C + t:C + I(T ** 2) + I(t ** 2) + I(C ** 2)"
TRUE_MODELS = {
    "yield": {"Intercept": 70, "T": 6, "t": 4, "C": 5, "T:C": 2.5,
              "I(T ** 2)": -3, "I(t ** 2)": -2, "I(C ** 2)": -1.5},
    "impurity": {"Intercept": 1.6, "T": 0.2, "t": 0.5, "C": 0.1, "I(t ** 2)": 0.3},
}
NOISE_SD = {"yield": 0.5, "impurity": 0.05}
GOALS = [
    {"response": "yield", "goal": "maximize", "low": 70, "high": 80},
    {"response": "impurity", "goal": "minimize", "low": 1.6, "high": 2.0},
]
RANGES = {f.name: {"low": f.low, "high": f.high} for f in ACTIVE}
HEAT_LIMIT = 9.5
# The corner of the temperature and catalyst plane beyond the heat limit, as a closed polygon.
BEYOND_T, BEYOND_C = [187.5, 200.0, 200.0], [2.0, 1.5, 2.0]


def quadratic_terms(settings):
    """The quadratic model's terms at settings in actual units, coded -1 (low) to +1 (high)."""
    coded = pd.DataFrame({f.name: (settings[f.name] - (f.low + f.high) / 2) / ((f.high - f.low) / 2)
                          for f in ACTIVE})
    return dmatrix(QUADRATIC, coded, return_type="dataframe")


def true_response(settings):
    """Yield and impurity, in %, without noise."""
    X = quadratic_terms(settings)
    return pd.DataFrame({r: X[list(model)] @ pd.Series(model) for r, model in TRUE_MODELS.items()})


def heat(settings):
    """Heat released, in the units of the limit."""
    return 0.04 * settings["T"] + settings["C"]


def predict(fits, settings):
    """The fitted models' yield and impurity, in %, at settings in actual units."""
    X = quadratic_terms(settings)
    return pd.DataFrame({fit["response_name"]: X @ pd.Series({c["term"]: c["coefficient"] for c in fit["coefficients"]})
                         for fit in fits})


def desirability(responses):
    """The two ramps of the goals, and their geometric mean."""
    d = pd.DataFrame({"yield": ((responses["yield"] - 70) / (80 - 70)).clip(0, 1),
                      "impurity": ((2.0 - responses["impurity"]) / (2.0 - 1.6)).clip(0, 1)})
    d["overall"] = np.sqrt(d["yield"] * d["impurity"])
    return d


def _check(name, got, expected):
    """The chapter prints these values in exactly this form."""
    if got != expected:
        msg = f"{name}: figure module gives {got!r}, the chapter prints {expected!r}"
        raise AssertionError(msg)


def study():
    """Run the study as the chapter does and return everything the figures need."""
    rng = np.random.default_rng(2026)

    def run_reactor(settings):
        noise = {r: rng.normal(0, sd, len(settings)) for r, sd in NOISE_SD.items()}
        return true_response(settings) + pd.DataFrame(noise, index=settings.index)

    screen = generate_design(FACTORS, "dsd", budget=17, random_state=1)
    screen_obs = run_reactor(screen.design_actual)
    selected = {}
    for response in ["yield", "impurity"]:
        staged = analyze_omars(screen.design[NAMES], screen_obs[response],
                               quadratic_heredity="strong", interaction_heredity="strong")
        selected[response] = staged.active_main_effects + staged.active_quadratics + staged.active_interactions

    heat_limit = Constraint(expression="0.04*T + C <= 9.5")
    ccd = generate_design(ACTIVE, "ccd", alpha="face_centered", n_center_points=3, random_state=1)

    designs, curves, quality = {}, {}, {}
    for criterion in ["d_optimal", "i_optimal", "g_optimal"]:
        design = generate_design(ACTIVE, criterion, budget=16, model_type="quadratic",
                                 constraints=[heat_limit], random_state=1)
        q = evaluate_design(design, model="quadratic", fds_resolution=101,
                            metric=["d_efficiency", "average_prediction_variance", "fds"])
        designs[criterion], curves[criterion], quality[criterion] = design, q["fds"]["curve"], q

    stage2 = designs["i_optimal"]
    stage2_obs = run_reactor(stage2.design_actual)
    fits = []
    for response in ["yield", "impurity"]:
        data = stage2.design[["T", "t", "C"]].assign(**{response: stage2_obs[response].to_numpy()})
        fit = analyze_experiment(data, response_column=response, model="quadratic", analysis_type="coefficients")
        fits.append({"response_name": response, "factor_names": ["T", "t", "C"], "coefficients": fit["coefficients"]})

    region = DesignRegion.from_dict(stage2.metadata["region"])
    best = optimize_responses(fits, GOALS, factor_ranges=RANGES, region=region)["desirability"]
    setting = pd.Series(best["optimal_actual"])

    true_fits = [{"response_name": r, "factor_names": ["T", "t", "C"],
                  "coefficients": [{"term": term, "coefficient": b} for term, b in model.items()]}
                 for r, model in TRUE_MODELS.items()]
    truth = optimize_responses(true_fits, GOALS, factor_ranges=RANGES, region=region)["desirability"]

    # The values the chapter prints. Any drift stops every figure script here.
    _check("screen", (screen.n_runs, screen.metadata["fake_factors"]), (17, 2))
    _check("yield terms", selected["yield"], ["T", "t", "C", "T^2", "t^2", "C^2", "T:C"])
    _check("impurity terms", selected["impurity"], ["T", "t", "C", "t^2"])
    _check("CCD beyond", int((heat(ccd.design_actual) > HEAT_LIMIT).sum()), 2)
    for criterion, (d_eff, apv, worst) in {"d_optimal": ("43.2", "0.523", "0.965"),
                                          "i_optimal": ("39.0", "0.347", "1.132"),
                                          "g_optimal": ("40.7", "0.506", "0.775")}.items():
        q = quality[criterion]
        _check(f"{criterion} D-efficiency", f"{q['d_efficiency']:.1f}", d_eff)
        _check(f"{criterion} average", f"{q['average_prediction_variance']:.3f}", apv)
        _check(f"{criterion} maximum", f"{q['fds']['max_prediction_variance']:.3f}", worst)
    _check("optimum", f"{setting['T']:.1f} {setting['t']:.1f} {setting['C']:.2f}", "190.1 20.4 1.90")
    _check("overall desirability", f"{best['composite_desirability']:.3f}", "0.699")
    _check("true optimum", f"{truth['optimal_actual']['T']:.1f} {truth['optimal_actual']['C']:.2f}", "187.6 2.00")

    return {"screen": screen, "ccd": ccd, "designs": designs, "curves": curves, "quality": quality,
            "stage2": stage2, "fits": fits, "best": best, "setting": setting, "truth": truth}
