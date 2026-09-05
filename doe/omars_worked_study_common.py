"""The worked OMARS study, reproduced for its figures.

The book section "A worked OMARS study" (design-analysis-experiments/omars-worked-study.rst)
runs a four-factor study on the fed-batch bioreactor simulator in process_improve. The six
figure scripts beside this module each draw one view of that study, so the study itself lives
here once, written exactly as the chapter's code blocks write it: the same factors, the same
disturbance level, the same design seed, the same pair split, the same run order and the same
disturbance seeds. ``study()`` then asserts the values the chapter prints, so a figure cannot
quietly drift away from the text it illustrates.

Import from a figure script run in this directory::

    from omars_worked_study_common import study
    S = study()          # a dict of everything the chapter computes, checked

The two hundred-campaign trade-off figure has its own data script,
omars_worked_study_data.py, because its numbers take minutes to regenerate.
"""
from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pandas as pd
from scipy import optimize, stats

from process_improve.experiments import Factor, analyze_omars, generate_omars
from process_improve.simulation import BioreactorConfig, BioreactorSimulator

# Okabe-Ito, matching the other figures in this chapter.
BLUE, ORANGE, GREEN, VERMILION = "#0072B2", "#E69F00", "#009E73", "#D55E00"
SKY, PURPLE, GREY, SPINE = "#56B4E9", "#CC79A7", "#666666", "#98A2AB"

FACTORS = [
    Factor(name="hold_temp", low=28.5, high=31.5),
    Factor(name="shift_day", low=2.0, high=3.5),
    Factor(name="pH", low=6.9, high=7.3),
    Factor(name="feed_rate", low=0.040, high=0.070),
]
NAMES = [f.name for f in FACTORS]
LABELS = {"hold_temp": "Hold temperature", "shift_day": "Shift day", "pH": "pH",
          "feed_rate": "Feed rate"}
GROWTH_TEMP, RAMP_DAYS = 36.8, 1.5
CONFIG = dataclasses.replace(BioreactorConfig(), within_batch_scale=0.7)
QUIET = dataclasses.replace(CONFIG, ic_scale=0.0, within_batch_scale=0.0, noise_scale=0.0)
CURRENT = {"hold_temp": 30.0, "shift_day": 2.75, "pH": 7.1, "feed_rate": 0.055}


def recipe(cfg, hold_temp, shift_day, pH):
    """The setpoint schedule for one batch: warm growth, a ramp, then the cold hold."""
    days = cfg.interval_start_days
    fraction = np.clip((days - shift_day) / RAMP_DAYS, 0.0, 1.0)
    temperature = GROWTH_TEMP - (GROWTH_TEMP - hold_temp) * fraction
    return pd.DataFrame({"pH": np.full_like(days, pH), "temperature": temperature},
                        index=pd.Index(days, name="day"))


def simulate(cfg, hold_temp, shift_day, pH, feed_rate, random_state):
    """The whole simulator result for one batch, states and all."""
    sim = BioreactorSimulator(dataclasses.replace(cfg, feed_rate=feed_rate))
    return sim.simulate_batch(trajectory=recipe(cfg, hold_temp, shift_day, pH),
                              random_state=random_state)


def run_batch(cfg, hold_temp, shift_day, pH, feed_rate, random_state):
    """Final titer, in g/L, of one batch at these settings."""
    return simulate(cfg, hold_temp, shift_day, pH, feed_rate, random_state).titer


def decode(x):
    """Coded (-1..+1) settings to real units, in factor order."""
    return {n: f.low + (x[i] + 1) / 2 * (f.high - f.low) for i, (n, f) in enumerate(zip(NAMES, FACTORS))}


def encode(real):
    """Real units to coded (-1..+1), in factor order."""
    return np.array([(real[n] - f.low) / (f.high - f.low) * 2 - 1 for n, f in zip(NAMES, FACTORS)])


def truth(x):
    """True titer, g/L, at coded settings x, with every disturbance switched off."""
    return run_batch(QUIET, **decode(np.clip(np.asarray(x, float), -1, 1)), random_state=0)


def model_matrix(terms, C):
    C = np.atleast_2d(np.asarray(C, float))
    cols = [np.ones(len(C))]
    for kind, j in terms:
        cols.append(C[:, j] if kind == "m" else C[:, j] ** 2 if kind == "q" else C[:, j[0]] * C[:, j[1]])
    return np.column_stack(cols)


def second_order_columns(C):
    cols = [C[:, i] ** 2 for i in range(C.shape[1])]
    cols += [C[:, i] * C[:, j] for i, j in itertools.combinations(range(C.shape[1]), 2)]
    return np.column_stack(cols)


def _check(name, value, expected):
    """Four significant figures, the way the chapter prints them."""
    got = float(f"{value:#.4g}")
    if not math.isclose(got, expected, rel_tol=0, abs_tol=1e-12):
        msg = f"{name}: figure module gives {got}, the chapter states {expected}"
        raise AssertionError(msg)


def study():
    """Run the study as the chapter does and return everything the figures need."""
    reps = np.array([run_batch(CONFIG, **CURRENT, random_state=s) for s in range(20)])

    design = generate_omars(FACTORS, n_runs=27, model="main_quadratic", random_seed=42)
    coded = design.design[NAMES].to_numpy(float)
    is_centre = np.all(coded == 0, axis=1)
    rows = [i for i in range(len(coded)) if not is_centre[i]]
    pairs, seen = [], set()
    for i in rows:
        if i in seen:
            continue
        j = next(k for k in rows if k not in seen and k != i and np.allclose(coded[k], -coded[i]))
        pairs.append((i, j))
        seen.update((i, j))

    so = second_order_columns(coded)
    keep = ~is_centre
    best_split, best_r = None, np.inf
    for chosen in itertools.combinations(range(len(pairs)), 7):
        block = np.full(len(coded), -1.0)
        for p in chosen:
            block[list(pairs[p])] = 1.0
        r = max(abs(np.corrcoef(block[keep], so[keep, c])[0, 1])
                for c in range(so.shape[1]) if so[keep, c].std() > 0)
        if r < best_r:
            best_split, best_r = chosen, r
    block = np.full(len(coded), -1.0)
    for p in best_split:
        block[list(pairs[p])] = 1.0

    rng = np.random.default_rng(7)
    plan = pd.DataFrame(coded, columns=NAMES)
    plan["cassette"] = np.where(block > 0, 1, 2)
    plan.loc[is_centre, "cassette"] = 1
    extra = pd.DataFrame(np.zeros((3, 4)), columns=NAMES)
    extra["cassette"] = [1, 2, 2]
    plan = pd.concat([plan, extra], ignore_index=True)
    order = []
    for c in (1, 2):
        idx = plan.index[plan["cassette"] == c].to_numpy()
        centres = idx[np.all(plan.loc[idx, NAMES] == 0, axis=1)]
        seq = list(rng.permutation(idx[~np.isin(idx, centres)]))
        for k, cpt in enumerate(centres):
            seq.insert(int(round((k + 1) * len(idx) / (len(centres) + 1))), cpt)
        order.extend(seq)
    plan = plan.loc[order].reset_index(drop=True)
    plan.index = pd.RangeIndex(1, len(plan) + 1, name="run")
    for n, f in zip(NAMES, FACTORS):
        plan[n] = f.low + (plan[n] + 1) / 2 * (f.high - f.low)

    lot = {1: CONFIG, 2: dataclasses.replace(CONFIG, feed_substrate=0.88 * CONFIG.feed_substrate)}
    seeds = np.random.default_rng(2026).integers(1 << 30, size=len(plan))
    plan["titer"] = [run_batch(lot[int(r.cassette)], r.hold_temp, r.shift_day, r.pH, r.feed_rate, int(s))
                     for r, s in zip(plan.itertuples(), seeds)]
    plan["log_titer"] = np.log(plan["titer"])
    is_cp = np.all(np.isclose(plan[NAMES], list(CURRENT.values())), axis=1)

    C = np.column_stack([(plan[n] - f.low) / (f.high - f.low) * 2 - 1 for n, f in zip(NAMES, FACTORS)])
    cassette = np.where(plan["cassette"] == 2, 1.0, 0.0)
    X = np.column_stack([np.ones(len(plan)), cassette, C])
    b = np.linalg.lstsq(X, plan["log_titer"], rcond=None)[0]
    plan["log_titer_adj"] = plan["log_titer"] - b[1] * cassette

    result = analyze_omars(plan[NAMES], plan["log_titer_adj"],
                           quadratic_heredity="none", interaction_heredity="none")
    terms = [("m", NAMES.index(m)) for m in result.active_main_effects]
    terms += [("q", NAMES.index(q[:-2])) for q in result.active_quadratics]
    terms += [("i", tuple(NAMES.index(v) for v in it.split(":"))) for it in result.active_interactions]
    bs = np.linalg.lstsq(model_matrix(terms, C), plan["log_titer_adj"], rcond=None)[0]
    moved = sorted({j for k, j in terms if k != "i"} | {v for k, j in terms if k == "i" for v in j})

    def predicted(z):
        x = np.zeros(4)
        x[moved] = z
        return -float((model_matrix(terms, x) @ bs).ravel()[0])

    starts = [np.zeros(len(moved)), -np.ones(len(moved)), np.ones(len(moved))]
    opt = min((optimize.minimize(predicted, s, method="Powell", bounds=[(-1, 1)] * len(moved))
               for s in starts), key=lambda r: r.fun)
    x_rec = np.zeros(4)
    x_rec[moved] = opt.x

    best = max((optimize.minimize(lambda x: -truth(x), s, method="Powell", bounds=[(-1, 1)] * 4)
                for s in [np.zeros(4), np.array([-1.0, -1, 0, 1]), np.array([-0.5, 0.5, 0, 1])]),
               key=lambda r: -r.fun)

    # The values the chapter prints. Any drift stops every figure script here.
    _check("replicate mean", reps.mean(), 7.477)
    _check("titer min", plan["titer"].min(), 4.290)
    _check("titer max", plan["titer"].max(), 9.116)
    _check("cassette effect", b[1], -0.1272)
    _check("recommended hold", decode(x_rec)["hold_temp"], 29.54)
    _check("current titer", truth(np.zeros(4)), 7.436)
    _check("recommended titer", truth(x_rec), 8.376)
    _check("best titer", -best.fun, 9.442)
    assert list(plan.index[is_cp]) == [6, 12, 22, 26], "centre runs moved"
    assert result.active_main_effects == ["feed_rate"]
    assert result.active_quadratics == ["hold_temp^2"]
    assert result.active_interactions == ["hold_temp:shift_day"]

    return {"reps": reps, "plan": plan, "is_cp": is_cp, "C": C, "cassette": cassette, "b": b,
            "result": result, "terms": terms, "bs": bs, "x_rec": x_rec, "best": best,
            "pairs": pairs, "coded": coded}


if __name__ == "__main__":
    S = study()
    print("study reproduced; every checked value matches the chapter")
    print(f"true best recipe: {decode(S['best'].x)}")
