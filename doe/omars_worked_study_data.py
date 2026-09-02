"""Regenerate the literal data carried in omars-worked-study-tradeoff.py.

What a design of each size buys on the fed-batch bioreactor, in the currency that matters:
grams per litre of product at the recipe the fitted model recommends, against the recipe the
team started with. Every design is simulated many times with fresh disturbance draws, so the
figure shows the distribution of outcomes a team could expect, not one lucky campaign.

The study region, factors, disturbance level and analysis are exactly those of the book
section "A worked OMARS study" (design-analysis-experiments/omars-worked-study.rst); this
script exists so the section's figure can be re-derived rather than trusted.

    python omars_worked_study_data.py            # prints the literals, takes minutes
    python omars_worked_study_data.py --quick    # 20 seeds, for a smoke test

Factors absent from the fitted model are held at the current recipe: a team does not move a
setting it has no evidence about. The true titer at the recommended recipe is read from the
simulator with every disturbance channel off.
"""
from __future__ import annotations

import dataclasses
import itertools
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import optimize

from process_improve.experiments import Factor, analyze_omars, generate_omars
from process_improve.experiments.designs_response_surface import dispatch_box_behnken, dispatch_ccd
from process_improve.simulation import BioreactorConfig, BioreactorSimulator

warnings.filterwarnings("ignore")

FACTORS = [
    Factor(name="hold_temp", low=28.5, high=31.5),
    Factor(name="shift_day", low=2.0, high=3.5),
    Factor(name="pH", low=6.9, high=7.3),
    Factor(name="feed_rate", low=0.040, high=0.070),
]
NAMES = [f.name for f in FACTORS]
GROWTH_TEMP, RAMP_DAYS = 36.8, 1.5
CONFIG = dataclasses.replace(BioreactorConfig(), within_batch_scale=0.7)
QUIET = dataclasses.replace(CONFIG, ic_scale=0.0, within_batch_scale=0.0, noise_scale=0.0)
N_SEEDS = 20 if "--quick" in sys.argv[1:] else 200


def recipe(cfg, hold_temp, shift_day, pH):
    days = cfg.interval_start_days
    fraction = np.clip((days - shift_day) / RAMP_DAYS, 0.0, 1.0)
    return pd.DataFrame({"pH": np.full_like(days, pH),
                         "temperature": GROWTH_TEMP - (GROWTH_TEMP - hold_temp) * fraction},
                        index=pd.Index(days, name="day"))


def run_batch(cfg, hold_temp, shift_day, pH, feed_rate, random_state):
    sim = BioreactorSimulator(dataclasses.replace(cfg, feed_rate=float(feed_rate)))
    return sim.simulate_batch(trajectory=recipe(cfg, hold_temp, shift_day, pH), random_state=random_state).titer


def decode(x):
    return {n: f.low + (x[i] + 1) / 2 * (f.high - f.low) for i, (n, f) in enumerate(zip(NAMES, FACTORS))}


def truth(x):
    return run_batch(QUIET, **decode(np.clip(np.asarray(x, float), -1, 1)), random_state=0)


def model_matrix(terms, C):
    C = np.atleast_2d(np.asarray(C, float))
    cols = [np.ones(len(C))]
    for kind, j in terms:
        cols.append(C[:, j] if kind == "m" else C[:, j] ** 2 if kind == "q" else C[:, j[0]] * C[:, j[1]])
    return np.column_stack(cols)


def campaign_gain(C, seed):
    """Titer gained over the current recipe by following what this design recommends."""
    rng = np.random.default_rng(seed)
    real = pd.DataFrame([decode(r) for r in C])
    y = pd.Series(np.log([run_batch(CONFIG, **row, random_state=int(rng.integers(1 << 30)))
                          for row in real.to_dict("records")]), index=real.index)
    res = analyze_omars(real, y, quadratic_heredity="none", interaction_heredity="none")
    terms = [("m", NAMES.index(m)) for m in res.active_main_effects]
    terms += [("q", NAMES.index(q[:-2])) for q in res.active_quadratics]
    terms += [("i", tuple(NAMES.index(v) for v in it.split(":"))) for it in res.active_interactions]
    found = set(res.active_main_effects) | set(res.active_quadratics) | set(res.active_interactions)
    moved = sorted({j for k, j in terms if k != "i"} | {v for k, j in terms if k == "i" for v in j})
    if not moved:
        return 0.0, found
    b = np.linalg.lstsq(model_matrix(terms, C), y.to_numpy(), rcond=None)[0]

    def predicted(z):
        x = np.zeros(4); x[moved] = z
        return -float((model_matrix(terms, x) @ b).ravel()[0])

    starts = [np.zeros(len(moved)), -np.ones(len(moved)), np.ones(len(moved))]
    opt = min((optimize.minimize(predicted, s, method="Powell", bounds=[(-1, 1)] * len(moved)) for s in starts),
              key=lambda r: r.fun)
    x = np.zeros(4); x[moved] = opt.x
    return truth(x) - T_NOW, found


T_NOW = truth(np.zeros(4))
best = max((optimize.minimize(lambda x: -truth(x), s, method="Powell", bounds=[(-1, 1)] * 4)
            for s in [np.zeros(4), np.array([-1.0, -1, 0, 1]), np.array([-0.5, 0.5, 0, 1])]), key=lambda r: -r.fun)
PRIZE = -best.fun - T_NOW

designs = {}
for n in (13, 17, 21, 27, 31):
    designs[f"OMARS {n}"] = generate_omars(FACTORS, n_runs=n, model="main_quadratic", random_seed=42).design[NAMES].to_numpy(float)
designs["Box-Behnken 27"] = np.asarray(dispatch_box_behnken(FACTORS)[0], float)
designs["CCD 27"] = np.asarray(dispatch_ccd(FACTORS, alpha="face_centered")[0], float)

KEY_TERMS = ["feed_rate", "hold_temp^2", "hold_temp:shift_day", "shift_day^2"]
print(f"# current recipe {T_NOW:.4f} g/L, best in region {-best.fun:.4f} g/L, prize {PRIZE:.4f} g/L, "
      f"{N_SEEDS} seeds per design")
print("GAINS = {")
for name, C in designs.items():
    gains, hits = [], {t: 0 for t in KEY_TERMS}
    for seed in range(N_SEEDS):
        g, found = campaign_gain(C, 500 + seed)
        gains.append(g)
        for t in KEY_TERMS:
            hits[t] += t in found
    q = np.percentile(gains, [10, 50, 90])
    print(f'    "{name}": {{"runs": {len(C)}, "mean": {np.mean(gains):.4f}, "p10": {q[0]:.4f}, '
          f'"p50": {q[1]:.4f}, "p90": {q[2]:.4f}, "worst": {np.min(gains):.4f}, '
          f'"found": {{' + ", ".join(f'"{t}": {hits[t] / N_SEEDS:.2f}' for t in KEY_TERMS) + "}},", flush=True)
print("}")
print(f"PRIZE = {PRIZE:.4f}")
