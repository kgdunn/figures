"""Shared construction and evaluation of the six design families compared in the
design-quality subchapter (``judging-and-comparing-designs.rst``).

Everything needed to regenerate and validate the omnibus comparison lives here so a
reader can reproduce every number and both figures from scratch. The model throughout is
the main-effects-plus-quadratics model in five factors (1 intercept + 5 linear + 5 pure
quadratic = 11 terms, no two-factor interactions), matching the chapter.

Run ``check_omnibus.py`` to validate the constructions and print the comparison tables;
run ``power-comparison-six-designs.py`` and ``fds-plot-six-designs.py`` to regenerate the
two figures.

Designs (all confined to the coded factor range [-1, 1] so the comparison is like for
like on a fixed experimental region):

- full factorial      2^5 with 2 centre runs            (34 runs, cannot fit quadratics)
- fractional factorial 2^(5-1) res V with 2 centre runs (18 runs, cannot fit quadratics)
- CCD, face-centred   half-fraction cube + faces + 6 c  (32 runs)
- Box-Behnken         all C(5,2) pairs + 6 centre       (46 runs)
- DSD                 order-6 conference foldover + c    (13 runs, smallest OMARS member)
- OMARS               two conference foldover blocks + c (25 runs)
"""

import itertools

import numpy as np
from scipy import stats

K = 5  # number of factors
N_EVAL = 120_000  # uniform points for prediction-variance integration
EVAL_SEED = 1


def main_quadratic_model(design):
    """Return the 11-column model matrix [1 | x_i | x_i^2] for a 5-factor design."""
    design = np.asarray(design, float)
    n, k = design.shape
    cols = [np.ones(n)] + [design[:, i] for i in range(k)] + [design[:, i] ** 2 for i in range(k)]
    return np.column_stack(cols)


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


def build_designs():
    """Construct all six design families. Returns a dict name -> (N x 5) coded array."""
    centre = np.zeros((1, K))

    # Two-level factorials (cannot estimate individual quadratics; 2 centre runs added so
    # sigma^2 and a 1-degree-of-freedom curvature check are available).
    full = np.array(list(itertools.product([-1, 1], repeat=K)), float)
    half = np.array(list(itertools.product([-1, 1], repeat=K - 1)), float)
    half = np.column_stack([half, half[:, 0] * half[:, 1] * half[:, 2] * half[:, 3]])  # E = ABCD, res V

    # Face-centred central composite: res-V cube + face (axial at +/-1) + 6 centre runs.
    # Face-centred (alpha = 1) keeps every run inside [-1, 1]; a rotatable CCD would place
    # the axial runs at +/-2, i.e. on a 2x wider range, so it is not a like-for-like design
    # on a fixed [-1, 1] region.
    faces = np.array(
        [[s if i == m else 0 for i in range(K)] for m in range(K) for s in (-1, 1)], float
    )
    ccd = np.vstack([half, faces, np.tile(centre, (6, 1))])

    # Box-Behnken: every pair of factors gets a 2^2 with the others at 0, plus 6 centre.
    bbd_rows = [
        [a if i == p else (b if i == j else 0) for i in range(K)]
        for p in range(K)
        for j in range(p + 1, K)
        for a in (-1, 1)
        for b in (-1, 1)
    ]
    bbd = np.vstack([np.array(bbd_rows, float), np.tile(centre, (6, 1))])

    # Definitive screening design: foldover of the order-6 conference matrix (5 columns)
    # plus a centre run. The smallest OMARS member.
    cm = _conference_order6()[:, :K]
    dsd = np.vstack([cm, -cm, centre])

    # A larger OMARS: a second conference foldover block with the columns permuted, so the
    # design is genuinely distinct from a replicated DSD (23 distinct rows) while keeping
    # the OMARS property (main effects orthogonal to every second-order term). Verified in
    # check_omnibus.py.
    cm2 = cm[:, [2, 4, 1, 3, 0]]
    omars = np.vstack([cm, -cm, cm2, -cm2, centre])

    return {
        "full": np.vstack([full, np.tile(centre, (2, 1))]),
        "frac": np.vstack([half, np.tile(centre, (2, 1))]),
        "ccd": ccd,
        "bbd": bbd,
        "dsd": dsd,
        "omars": omars,
    }


# Display metadata for the four response-surface-capable designs (the factorials cannot
# fit the quadratic model, so they are excluded from the quality comparison).
RSM_DESIGNS = ["ccd", "bbd", "dsd", "omars"]
LABELS = {
    "full": "full factorial, 2^5 + 2c",
    "frac": "fractional, 2^(5-1) + 2c",
    "ccd": "CCD (face-centred)",
    "bbd": "Box-Behnken",
    "dsd": "DSD",
    "omars": "OMARS",
}


def _eval_points():
    """Uniform interior points augmented with all 2^5 cube vertices.

    The maximum prediction variance over [-1, 1]^5 sits at a vertex, which random interior
    sampling misses, so the vertices must be included explicitly (Goos and Nunez Ares,
    2025). The same augmented set anchors the right-hand tail of every FDS curve.
    """
    rng = np.random.default_rng(EVAL_SEED)
    interior = rng.uniform(-1, 1, size=(N_EVAL, K))
    vertices = np.array(list(itertools.product([-1, 1], repeat=K)), float)
    return interior, vertices


def evaluate(design, eval_interior=None, eval_vertices=None):
    """Return every quality metric for one design on the 11-term quadratic model.

    Prediction variances are reported unscaled (in sigma^2 units, ``avg_pv`` / ``max_pv``)
    and scaled by the run count (``avg_spv`` / ``max_spv``). Power assumes an effect of one
    noise standard deviation (delta = sigma) at alpha = 0.05.
    """
    if eval_interior is None:
        eval_interior, eval_vertices = _eval_points()
    m = main_quadratic_model(design)
    n, p = m.shape
    rank = np.linalg.matrix_rank(m)
    if rank < p:
        # The quadratic model is not estimable: the five x_i^2 columns are identical at the
        # two factor levels, so they collapse to a single curvature indicator.
        return {"N": n, "fits": False, "rank": rank, "reduced_df": n - rank}

    xtx = m.T @ m
    xtx_inv = np.linalg.inv(xtx)
    _, logdet = np.linalg.slogdet(xtx)
    d_eff = 100.0 * np.exp(logdet / p) / n  # per-run D-efficiency (favours small N)
    information = np.exp(logdet / p)  # |X'X|^(1/p): unscaled information content

    xi = main_quadratic_model(eval_interior)
    xv = main_quadratic_model(eval_vertices)
    pv_interior = np.einsum("ij,jk,ik->i", xi, xtx_inv, xi)
    pv_vertices = np.einsum("ij,jk,ik->i", xv, xtx_inv, xv)
    avg_pv = pv_interior.mean()
    max_pv = max(pv_interior.max(), pv_vertices.max())

    terms = m[:, 1:]  # drop the intercept for the correlation / VIF summaries
    corr = np.corrcoef(terms, rowvar=False)
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    vif = np.diag(np.linalg.inv(corr))

    nu = n - p
    f_crit = stats.f.ppf(0.95, 1, nu)

    def power(coef_index):
        lam = 1.0 / xtx_inv[coef_index, coef_index]  # delta = sigma
        return float(1 - stats.ncf.cdf(f_crit, 1, nu, lam))

    return {
        "N": n,
        "fits": True,
        "residual_df": nu,
        "d_eff": d_eff,
        "information": information,
        "A": float(np.trace(xtx_inv)),
        "avg_pv": float(avg_pv),
        "max_pv": float(max_pv),
        "avg_spv": float(avg_pv * n),
        "max_spv": float(max_pv * n),
        "max_r": float(np.abs(off_diag).max()),
        "mean_r": float(np.abs(off_diag).mean()),
        "max_vif": float(vif.max()),
        "mean_vif": float(vif.mean()),
        "power_main": power(1),  # first linear coefficient
        "power_quad": power(K + 1),  # first quadratic coefficient
        "pv_interior": pv_interior,  # retained for the FDS figure
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
