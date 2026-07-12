"""Validate the model inversion against the withheld ground truth.

For each candidate chromogen, invert the interaction PLS onto the reference goal (allowing settings
outside the coded [-1, 1] box, since a designed experiment establishes the factor effects and mild
extrapolation is a testable next-run hypothesis), unscale to real units, push those settings back
through the ground-truth simulator, and measure how close the resulting curve is to the reference.
Compare against each compound's shape floor (the best any amplitude can do). Run from figures/doe.
"""

import contextlib
import io

import numpy as np

from colour_case_study import (
    CONT,
    build_design,
    coded_to_real,
    curve_match_inversion,
    fit_coding,
    goal_projection,
    ground_truth_curve,
    invert_to_factors,
    shape_floor,
    simulate_curves,
)

NOISE = 0.03


def quiet(fn):
    with contextlib.redirect_stderr(io.StringIO()):
        return fn()


def main():
    design = build_design("i_optimal", 60)
    curves = simulate_curves(design)
    goal = ground_truth_curve("A", [0, 0, 0, 0])   # reference: A at centre, noiseless

    # Closeness is measured on the developed curve (t1 onward); t0 is near-zero, noise-dominated.
    dev = slice(1, None)
    print(f"reference = A at centre; peak ~ {goal.max():.2f}; measurement noise SD = {NOISE}")
    print("Closeness measured on the developed curve (t1 onward).")
    print("\nShape floor (best RMSE to A at any amplitude; set by drift alone):")
    for c in ["B", "C", "D", "E", "F"]:
        rmse, amp = shape_floor(c)
        print(f"  {c}: floor RMSE = {rmse:.4f}  ({rmse / NOISE:.1f}x noise, best amp {amp:.2f})")

    print("\nCurve-match inversion (coding-invariant, extrapolating) -> ground truth:")
    cm = curve_match_inversion(design, curves, "treatment")
    for c in ["B", "C", "D", "E", "F"]:
        cd = [float(cm.loc[c, f"{n}_coded"]) for n in CONT]
        rmse = float(np.sqrt(np.mean((ground_truth_curve(c, cd)[dev] - goal[dev]) ** 2)))
        real = coded_to_real(cd)
        outside = [n for i, n in enumerate(CONT) if abs(cd[i]) > 1 + 1e-9]
        rl = f"conc {real['concentration']:.1f}, co-solv {real['co_solvent']:.1f}, pH {real['pH']:.1f}, T {real['temperature']:.0f}"
        print(f"  {c}: RMSE {rmse:.4f} ({rmse / NOISE:.1f}x noise)  [{rl}]  "
              f"outside box: {outside if outside else 'none'}")

    print("\nCell-means score-match (in-range where possible) -> ground truth:")
    pls_c, di_c = quiet(lambda: fit_coding(design, curves, "cell_means"))
    tbl = invert_to_factors(pls_c, di_c, goal_projection(pls_c, di_c)["score"])
    for c in ["B", "C", "D", "E", "F"]:
        cd = [float(tbl.loc[c, f"{n}_coded"]) for n in CONT]
        rmse = float(np.sqrt(np.mean((ground_truth_curve(c, cd)[dev] - goal[dev]) ** 2)))
        inr = bool(np.all(np.abs(cd) <= 1 + 1e-9))
        print(f"  {c}: RMSE {rmse:.4f} ({rmse / NOISE:.1f}x noise)  in_range={inr}")


if __name__ == "__main__":
    main()
