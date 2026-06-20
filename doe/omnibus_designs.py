"""Shared construction and evaluation of the design families compared in the
design-quality subchapter (``judging-and-comparing-designs.rst``).

The designs and their quality metrics are built with Kevin Dunn's ``process_improve``
library (``pip install 'process-improve[expt]'``): ``generate_design`` constructs the
factorials, the Box-Behnken design and the definitive screening design, and
``evaluate_design`` scores D-efficiency, the variance inflation factors, the power and the
prediction variance integrated over the whole region (used for the FDS curves and the
average/maximum prediction-variance rows; v1.44.0+ exposes the configurable region sampling
and the dense FDS curve). One piece is computed by hand because the library does not expose
it on this model: the alias matrix of the omitted two-factor interactions. One design is
also built by hand: there is no OMARS generator in
the library, so the 25-run OMARS design is built as two permuted conference-matrix
foldovers. The central composite design uses ``process_improve``'s ``cube="fractional"``
option (v1.42.0+) to build its cube as a resolution-V half-fraction (the standard k=5 CCD)
rather than the full-factorial cube the library builds by default.

The model throughout is the main-effects-plus-quadratics model in five factors
(1 intercept + 5 linear + 5 pure quadratic = 11 terms, no two-factor interactions),
matching the chapter. It is passed to ``evaluate_design`` as an explicit patsy formula
so the library scores exactly this model: its built-in ``"quadratic"`` keyword would add
the ten two-factor interactions and make the small designs rank-deficient.

Run ``check_omnibus.py`` to validate the constructions and print the comparison tables;
run ``power-comparison-six-designs.py`` and ``fds-plot-six-designs.py`` to regenerate the
two figures.

Designs (all confined to the coded factor range [-1, 1] so the comparison is like for
like on a fixed experimental region):

- full factorial      2^5 with 2 centre runs            (34 runs, cannot fit quadratics)
- fractional factorial 2^(5-1) res V with 2 centre runs (18 runs, cannot fit quadratics)
- CCD, face-centred   res-V half-fraction cube + faces  (32 runs)
- Box-Behnken         all C(5,2) pairs + 6 centre       (46 runs)
- DSD                 order-6 conference foldover + c    (13 runs, smallest OMARS member)
- OMARS               two conference foldover blocks + c (25 runs)
"""

import numpy as np
import pandas as pd
from process_improve.experiments import Factor, evaluate_design, generate_design

K = 5  # number of factors
N_EVAL = 120_000  # uniform points for prediction-variance integration
EVAL_SEED = 1

FACTOR_NAMES = list("ABCDE")
FACTORS = [Factor(name=name, low=-1, high=1) for name in FACTOR_NAMES]

# The 11-term main-effects-plus-pure-quadratics model as an explicit patsy formula, so
# process_improve scores exactly the chapter's model rather than the full second-order
# model its "quadratic" keyword would assume.
MODEL_FORMULA = " + ".join(FACTOR_NAMES + [f"I({name}**2)" for name in FACTOR_NAMES])


def main_quadratic_model(design):
    """Return the 11-column model matrix [1 | x_i | x_i^2] for a 5-factor design."""
    design = np.asarray(design, float)
    n, k = design.shape
    cols = [np.ones(n)] + [design[:, i] for i in range(k)] + [design[:, i] ** 2 for i in range(k)]
    return np.column_stack(cols)


def _two_factor_interactions(design):
    """Return the 10-column matrix of two-factor interaction terms x_i * x_j."""
    design = np.asarray(design, float)
    return np.column_stack([design[:, i] * design[:, j] for i in range(K) for j in range(i + 1, K)])


def alias_matrix(design):
    """Alias (bias) matrix for the two-factor interactions left out of the model.

    With the fitted model matrix X1 (the 11 main-effect-plus-quadratic terms) and the
    omitted-term matrix X2 (the ten two-factor interactions), the least-squares estimates
    satisfy E[b1] = beta1 + A @ beta2 with A = (X1' X1)^-1 X1' X2. Each entry of A is the
    bias an omitted interaction imposes on a fitted coefficient. process_improve does not
    expose this, so it is computed directly here.
    """
    x1 = main_quadratic_model(design)
    x2 = _two_factor_interactions(design)
    return np.linalg.solve(x1.T @ x1, x1.T @ x2)


def secondorder_corr(design):
    """Absolute correlation among the fifteen second-order effects (the five pure quadratics
    and the ten two-factor interactions), each residualised against the intercept and the
    five main effects.

    Residualising removes the part of each second-order column explained by the intercept and
    the main effects, leaving the genuine entanglement between the second-order terms. This
    differs from ``evaluate``'s ``max_r`` (which is the correlation among the fitted linear and
    quadratic columns); it includes the omitted interactions, so it is the entanglement the
    correlation colour map shows. Columns that residualise to (near) zero contribute no
    correlation and stay at zero; the diagonal is one.
    """
    d = np.asarray(design, float)
    n = len(d)
    base = np.column_stack([np.ones(n)] + [d[:, i] for i in range(K)])
    quad = [d[:, i] ** 2 for i in range(K)]
    inter = [d[:, i] * d[:, j] for i in range(K) for j in range(i + 1, K)]
    second = np.column_stack(quad + inter)
    resid = second - base @ (np.linalg.pinv(base) @ second)
    keep = np.linalg.norm(resid, axis=0) > 1e-9
    corr = np.eye(second.shape[1])
    if keep.sum() > 1:
        idx = np.where(keep)[0]
        corr[np.ix_(idx, idx)] = np.corrcoef(resid[:, keep], rowvar=False)
    return np.abs(corr)


def _conference_order6():
    """Symmetric Paley conference matrix of order 6 (C @ C.T = 5 I).

    Built from the quadratic-residue character over GF(5). Conference matrices are the
    backbone of definitive screening designs: the foldover of a conference matrix makes
    every main effect orthogonal to every quadratic term.
    """
    q = 5
    residues = {(x * x) % q for x in range(1, q)}
    chi = [0] + [1 if a in residues else -1 for a in range(1, q)]
    c = np.zeros((6, 6))
    for i in range(1, 6):
        c[0, i] = c[i, 0] = 1
    for i in range(q):
        for j in range(q):
            if i != j:
                c[1 + i, 1 + j] = chi[(j - i) % q]
    return c


def _coded(design_result):
    """Strip the RunOrder column from a process_improve DesignResult, returning the
    (N x 5) coded factor array in A..E order."""
    df = design_result.design
    return np.asarray(df[FACTOR_NAMES], dtype=float)


def build_designs():
    """Construct all six design families. Returns a dict name -> (N x 5) coded array."""
    centre = np.zeros((1, K))

    # process_improve builds the factorials, Box-Behnken design and DSD directly. The
    # two-level factorials carry 2 centre runs so sigma^2 and a one-degree-of-freedom
    # curvature check are available even though the individual quadratics are not
    # estimable.
    full = _coded(generate_design(FACTORS, "full_factorial", center_points=2))
    frac = _coded(
        generate_design(FACTORS, "fractional_factorial", generators=["E=ABCD"], center_points=2)
    )
    bbd = _coded(generate_design(FACTORS, "box_behnken", center_points=6))
    dsd = _coded(generate_design(FACTORS, "dsd"))

    # Face-centred central composite design on a resolution-V half-fraction cube. The
    # standard five-factor CCD uses the 16-run res-V fraction (not the full 32-run cube),
    # which process_improve builds directly with cube="fractional" (v1.42.0+): a 16-run
    # res-V cube + 10 face (axial, alpha = 1) runs + 6 centre = 32 runs. Face-centred keeps
    # every run inside [-1, 1]; a rotatable CCD would place the axial runs at +/-2, i.e. on
    # a 2x wider range, which is not a like-for-like design on a fixed [-1, 1] region.
    ccd = _coded(
        generate_design(FACTORS, "ccd", cube="fractional", alpha="face_centered", center_points=6)
    )

    # OMARS: process_improve has no OMARS generator, so the 25-run design is built by
    # hand as two conference-matrix foldovers, the second with its columns permuted so
    # the design is genuinely distinct from a replicated DSD (23 distinct rows) while
    # keeping the OMARS property (main effects orthogonal to every second-order term).
    # Verified in check_omnibus.py.
    cm = _conference_order6()[:, :K]
    cm2 = cm[:, [2, 4, 1, 3, 0]]
    omars = np.vstack([cm, -cm, cm2, -cm2, centre])

    return {
        "full": full,
        "frac": frac,
        "ccd": ccd,
        "bbd": bbd,
        "dsd": dsd,
        "omars": omars,
    }


# Display metadata for the four response-surface-capable designs (the factorials cannot
# fit the quadratic model, so they are excluded from the quality comparison). Ordered
# largest-to-smallest by run count, matching the comparison table in the chapter.
RSM_DESIGNS = ["bbd", "ccd", "omars", "dsd"]
LABELS = {
    "full": "full factorial, 2^5 + 2c",
    "frac": "fractional, 2^(5-1) + 2c",
    "ccd": "CCD (face-centred)",
    "bbd": "Box-Behnken",
    "dsd": "DSD",
    "omars": "OMARS",
}


def _library_metrics(design):
    """D-efficiency, VIFs, power, residual df and the region prediction variance from
    process_improve on the 11-term model. A plain DataFrame of the coded factor columns
    is enough for evaluate_design. The prediction variance is integrated over the cube
    with the same sample size and seed used elsewhere in the chapter, and with the 2^5
    vertices added (include_vertices=True) so the worst-case (G) value at a corner is
    represented; fds_resolution=200 returns the dense FDS curve for the figure."""
    df = pd.DataFrame(np.asarray(design, float), columns=FACTOR_NAMES)
    return evaluate_design(
        df,
        model=MODEL_FORMULA,
        metric=["d_efficiency", "vif", "power", "degrees_of_freedom", "fds"],
        effect_size=1.0,  # delta = sigma
        sigma=1.0,
        n_samples=N_EVAL,
        random_seed=EVAL_SEED,
        include_vertices=True,
        fds_resolution=200,
    )


def evaluate(design):
    """Return every quality metric for one design on the 11-term quadratic model.

    D-efficiency, the variance inflation factors, power, the residual degrees of freedom
    and the prediction variance (integrated over the region, reported unscaled in sigma^2
    units and scaled by the run count, with the dense FDS curve for the figure) all come
    from process_improve's ``evaluate_design``. The summed coefficient variance A, the
    smallest eigenvalue, the pairwise correlations and the alias matrix are computed here.
    Power assumes an effect of one noise standard deviation (delta = sigma) at alpha = 0.05.
    """
    m = main_quadratic_model(design)
    n, p = m.shape
    rank = np.linalg.matrix_rank(m)
    if rank < p:
        # The quadratic model is not estimable: the five x_i^2 columns are identical at the
        # two factor levels, so they collapse to a single curvature indicator.
        return {"N": n, "fits": False, "rank": rank, "reduced_df": n - rank}

    xtx = m.T @ m
    xtx_inv = np.linalg.inv(xtx)

    terms = m[:, 1:]  # drop the intercept for the correlation summaries
    corr = np.corrcoef(terms, rowvar=False)
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]

    alias = alias_matrix(design)
    main_rows = alias[1 : 1 + K]  # alias of the five main effects with the interactions

    lib = _library_metrics(design)
    vif = lib["vif"]
    power = lib["power"]
    d_eff = lib["d_efficiency"]
    fds = lib["fds"]
    avg_pv = fds["average_prediction_variance"]
    max_pv = fds["max_prediction_variance"]

    return {
        "N": n,
        "fits": True,
        "residual_df": lib["degrees_of_freedom"]["residual"],
        "d_eff": d_eff,  # per-run D-efficiency (favours small N)
        "information": d_eff * n / 100.0,  # |X'X|^(1/p) = D-efficiency * N / 100
        "e_opt": float(np.linalg.eigvalsh(xtx).min()),  # E-optimality: min eigenvalue of X'X
        "A": float(np.trace(xtx_inv)),
        "avg_pv": float(avg_pv),
        "max_pv": float(max_pv),
        "avg_spv": float(avg_pv * n),
        "max_spv": float(max_pv * n),
        "max_r": float(np.abs(off_diag).max()),
        "mean_r": float(np.abs(off_diag).mean()),
        "max_vif": float(max(vif.values())),
        "mean_vif": float(np.mean(list(vif.values()))),
        "power_main": float(power["A"]),  # first linear coefficient
        "power_quad": float(power["I(A ** 2)"]),  # first quadratic coefficient
        "max_alias": float(np.abs(alias).max()),  # worst bias from an omitted interaction
        "max_alias_main": float(np.abs(main_rows).max()),  # bias on the main effects
        "alias_fro": float(np.linalg.norm(alias)),  # overall level of aliasing
        "curve": fds["curve"],  # dense FDS curve (fraction, prediction_variance, scaled) for the figure
    }


def is_omars(design):
    """True if every main effect is orthogonal to the intercept and to all second-order
    terms (pure quadratics and two-factor interactions): the defining OMARS property."""
    design = np.asarray(design, float)
    n, k = design.shape
    main = np.column_stack([design[:, i] for i in range(k)])
    second = np.column_stack(
        [design[:, i] ** 2 for i in range(k)]
        + [design[:, i] * design[:, j] for i in range(k) for j in range(i + 1, k)]
    )
    ones = np.ones((n, 1))
    cross = main.T @ main
    off = cross - np.diag(np.diag(cross))
    return (
        np.allclose(off, 0)
        and np.allclose(main.T @ ones, 0, atol=1e-9)
        and np.allclose(main.T @ second, 0, atol=1e-9)
    )
