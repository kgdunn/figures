"""Validate the CCD-variant constructions and print the omnibus-style comparison table
behind the FDS figures in this directory.

Run from this directory: ``python3 check_ccd_variants.py``. Every assertion must pass. The
first table is the k = 3 comparison the FDS figures visualise; the second reproduces the
run counts quoted in the central composite design post at k = 5 (with one centre run each),
showing where the "43 / 41 / 31 / 27" numbers come from. Requires numpy, pandas, scipy and
process_improve[expt].
"""

import numpy as np

from ccd_variants_designs import (
    ALPHA_ROTATABLE,
    FAMILY_ORDER,
    K,
    LABELS,
    build_designs,
    ccd_matrix,
    doehlert_matrix,
    evaluate,
)


def main() -> None:
    designs = build_designs()

    # --- structural assertions -------------------------------------------------------
    # At k = 3, one centre run each: CCD variants = 8 factorial + 6 axial + 1 = 15 runs;
    # Box-Behnken = 12 edge + 1 = 13; Doehlert = 12 shell + 1 = 13.
    expected_n = {"ccc": 15, "ccf": 15, "cci": 15, "bbd": 13, "doehlert": 13}
    for name, n in expected_n.items():
        assert len(designs[name]) == n, f"{name}: expected {n} runs, got {len(designs[name])}"

    # Every family is full rank on the 10-term second-order model at k = 3.
    for name in FAMILY_ORDER:
        r = evaluate(designs[name])
        assert r["full_rank"], f"{name} is not full rank ({r['rank']}/{r['p']})"

    # The inscribed CCD is the circumscribed one scaled into +/-1: same shape, alpha times
    # smaller. Its largest coded value is exactly 1, the circumscribed one's is alpha.
    assert abs(np.max(np.abs(designs["cci"])) - 1.0) < 1e-9
    assert abs(np.max(np.abs(designs["ccc"])) - ALPHA_ROTATABLE) < 1e-9
    assert np.allclose(designs["ccc"] / ALPHA_ROTATABLE, designs["cci"])

    # Doehlert shell points all sit on the unit sphere (uniform shell property).
    shell = designs["doehlert"][np.count_nonzero(designs["doehlert"], axis=1) > 0]
    assert np.allclose(np.linalg.norm(shell, axis=1), 1.0)

    print("All structural assertions passed.\n")

    # --- Table: k = 3 quality comparison (what the FDS figures show) -----------------
    print(f"Table: five design families at k = {K}, full second-order model (10 terms), cube region")
    header = f"{'design':32s} {'N':>3s} {'df':>3s} {'D-eff':>6s} {'avg SPV':>8s} {'max SPV':>8s}"
    print(header)
    print("-" * len(header))
    for name in FAMILY_ORDER:
        r = evaluate(designs[name])
        print(
            f"{LABELS[name]:32s} {r['N']:>3d} {r['residual_df']:>3d} "
            f"{r['d_eff']:>6.1f} {r['avg_spv']:>8.2f} {r['max_spv']:>8.2f}"
        )

    # --- Table: k = 5 run counts, tying to the post ----------------------------------
    # One centre run each. Circumscribed full cube: 2^5 + 2*5 + 1 = 43. Fractional (res-V
    # half-fraction cube): 2^4 + 2*5 + 1 = 27. Box-Behnken: 40 + 1 = 41. Doehlert: 5^2+5+1
    # = 31. These are the "43 / 41 / 31 / 27" numbers in the post.
    from process_improve.experiments import Factor, generate_design  # noqa: PLC0415

    k5 = [Factor(name=n, low=-1, high=1) for n in "ABCDE"]
    ccc_full = ccd_matrix(float((2**5) ** 0.25), k=5, n_centre=1)
    ccc_frac = generate_design(k5, "ccd", cube="fractional", alpha="rotatable", center_points=1).design
    bbd5 = generate_design(k5, "box_behnken", center_points=1).design
    doe5 = doehlert_matrix(k=5, n_centre=1)

    counts = {
        "CCD circumscribed, full cube": len(ccc_full),
        "CCD circumscribed, fractional cube": len(ccc_frac),
        "Box-Behnken": len(bbd5),
        "Doehlert": len(doe5),
    }
    print(f"\nTable: run counts at k = 5 (one centre run each) - the post's figures")
    for label, n in counts.items():
        print(f"  {label:38s} {n:>3d} runs")

    assert counts["CCD circumscribed, full cube"] == 43
    assert counts["CCD circumscribed, fractional cube"] == 27
    assert counts["Box-Behnken"] == 41
    assert counts["Doehlert"] == 31
    print("\nRun-count assertions (43 / 27 / 41 / 31) passed.")


if __name__ == "__main__":
    main()
