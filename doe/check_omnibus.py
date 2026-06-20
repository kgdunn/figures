"""Validate the six design constructions and print the two comparison tables that back
the omnibus subsection of ``judging-and-comparing-designs.rst``.

Run from this directory: ``python3 check_omnibus.py``. Every assertion must pass; the
printed numbers are the ones quoted in the book. Requires numpy and scipy.
"""

import numpy as np

from omnibus_designs import (
    LABELS,
    RSM_DESIGNS,
    alias_matrix,
    build_designs,
    evaluate,
    is_omars,
    main_quadratic_model,
    model_term_corr,
)


def main() -> None:
    designs = build_designs()

    # --- structural assertions -------------------------------------------------------
    expected_n = {"full": 34, "frac": 18, "ccd": 32, "bbd": 46, "dsd": 13, "omars": 25}
    for name, n in expected_n.items():
        assert len(designs[name]) == n, f"{name}: expected {n} runs, got {len(designs[name])}"

    # Both factorials are singular on the quadratic model: with centre points the five
    # quadratic columns are identical, so the 11-term model collapses to rank 7.
    for name in ("full", "frac"):
        m = main_quadratic_model(designs[name])
        assert np.linalg.matrix_rank(m) == 7, f"{name} should have rank 7"

    # The four response-surface designs are full rank on the 11-term model.
    for name in RSM_DESIGNS:
        m = main_quadratic_model(designs[name])
        assert np.linalg.matrix_rank(m) == 11, f"{name} should be full rank"

    # DSD and the larger OMARS both satisfy the OMARS orthogonality property; the OMARS is
    # genuinely distinct from a replicated DSD (more than 13 distinct rows).
    assert is_omars(designs["dsd"]), "DSD must be an OMARS design"
    assert is_omars(designs["omars"]), "OMARS construction must satisfy the OMARS property"
    distinct = len({tuple(np.round(r, 6)) for r in designs["omars"]})
    assert distinct > 13, f"OMARS should not be a replicated DSD (got {distinct} distinct rows)"

    # The 5-factor DSD has 13 runs and 2 residual degrees of freedom: not saturated, unlike
    # the 4-factor DSD elsewhere in the chapter.
    assert evaluate(designs["dsd"])["residual_df"] == 2

    # The two heatmap figures (heatmaps-four-designs.py) are locked to the omnibus table.
    # Alias map: worst |A| per design matches the "Maximum alias |A|" row (0, 0, 1.00, 1.09).
    expected_alias = {"bbd": 0.00, "ccd": 0.00, "omars": 1.00, "dsd": 1.09}
    for name, want in expected_alias.items():
        got = float(np.abs(alias_matrix(designs[name])).max())
        assert abs(got - want) < 0.005, f"{name}: alias |A| max {got:.3f} != {want}"
    # Correlation colour map (model_term_corr, 20 columns: main effects, quadratics,
    # interactions). The worst off-diagonal over the whole map:
    expected_corr = {"bbd": 0.15, "ccd": 0.75, "omars": 0.50, "dsd": 0.50}
    # ...and over the fitted-terms block only (main effects + quadratics, the first 10 columns),
    # which must equal the table's "Maximum |r|" row:
    expected_fitted = {"bbd": 0.15, "ccd": 0.75, "omars": 0.00, "dsd": 0.13}
    for name in RSM_DESIGNS:
        c = model_term_corr(designs[name])
        full = float(c[~np.eye(c.shape[0], dtype=bool)].max())
        fitted = c[:10, :10]
        fitted_max = float(fitted[~np.eye(10, dtype=bool)].max())
        assert abs(full - expected_corr[name]) < 0.005, f"{name}: |r| max {full:.3f}"
        assert abs(fitted_max - expected_fitted[name]) < 0.005, f"{name}: fitted |r| {fitted_max:.3f}"
        assert abs(fitted_max - evaluate(designs[name])["max_r"]) < 0.005, f"{name}: max_r mismatch"

    print("All structural assertions passed.\n")

    # --- Table A: can the design fit the model? --------------------------------------
    print("Table A: fitting the 5-factor main-effects-plus-quadratics model (11 terms)")
    print(f"{'design':26s} {'N':>3s}  {'fits?':>5s}  {'rank':>4s}  {'residual df':>11s}")
    for name in ("full", "frac", "ccd", "bbd", "dsd", "omars"):
        r = evaluate(designs[name])
        if r["fits"]:
            print(f"{LABELS[name]:26s} {r['N']:3d}  {'yes':>5s}  {11:>4d}  {r['residual_df']:>11d}")
        else:
            note = f"{r['reduced_df']} (reduced model)"
            print(f"{LABELS[name]:26s} {r['N']:3d}  {'no':>5s}  {r['rank']:>4d}  {note:>11s}")

    # --- Table B: the four RSM-capable designs head to head --------------------------
    print("\nTable B: quality metrics for the response-surface designs")
    header = (
        f"{'metric':34s}"
        + "".join(f"{LABELS[n].split('(')[0].strip():>14s}" for n in RSM_DESIGNS)
    )
    print(header)
    results = {n: evaluate(designs[n]) for n in RSM_DESIGNS}
    rows = [
        ("runs N", "N", "{:.0f}"),
        ("residual df", "residual_df", "{:.0f}"),
        ("power, main effect (delta=sigma)", "power_main", "{:.2f}"),
        ("power, quadratic (delta=sigma)", "power_quad", "{:.2f}"),
        ("avg prediction var (sigma^2)", "avg_pv", "{:.2f}"),
        ("max prediction var (sigma^2)", "max_pv", "{:.2f}"),
        ("avg scaled pred var (SPV)", "avg_spv", "{:.2f}"),
        ("max scaled pred var (SPV)", "max_spv", "{:.2f}"),
        ("A, summed coefficient variance", "A", "{:.2f}"),
        ("E-optimality, min eig(X'X)", "e_opt", "{:.2f}"),
        ("max |r| among model terms", "max_r", "{:.2f}"),
        ("max VIF", "max_vif", "{:.2f}"),
        ("max alias |A| (omitted 2fi)", "max_alias", "{:.2f}"),
        ("max alias on main effects", "max_alias_main", "{:.2f}"),
        ("information |X'X|^(1/p)", "information", "{:.2f}"),
        ("D-efficiency, per run (%)", "d_eff", "{:.1f}"),
    ]
    for label, key, fmt in rows:
        print(f"{label:34s}" + "".join(f"{fmt.format(results[n][key]):>14s}" for n in RSM_DESIGNS))


if __name__ == "__main__":
    main()
