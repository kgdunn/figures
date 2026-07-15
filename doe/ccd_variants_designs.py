"""Shared construction and evaluation of the response-surface design families
contrasted in Igor Miranda Santana's DoE post (the central composite design and its
neighbours), built for the FDS comparison figures in this directory.

This mirrors ``omnibus_designs.py`` (which backs the five-factor omnibus subsection of
``judging-and-comparing-designs.rst``) but works at k = 3 factors, the smallest number at
which every family in the post exists: the Box-Behnken design needs at least three
factors, and Doehlert is only interesting from two upward. At k = 3 all five designs are
full rank on the complete second-order model (intercept + 3 linear + 3 two-factor
interactions + 3 pure quadratics = 10 terms), so they can be compared like for like.

The designs
-----------
Every design is expressed in coded units and evaluated on the same cube region [-1, 1]^3.

- ``ccc``  Central composite, circumscribed (rotatable). Factorial cube at +/-1, axial
           (star) runs at +/-alpha with alpha = (2^k)^(1/4) = 1.682 for k = 3. The classic
           textbook CCD; the star runs sit outside the cube.
- ``cci``  Central composite, inscribed. The circumscribed design divided through by alpha,
           so the star runs land on the +/-1 faces and the cube shrinks to +/-1/alpha. Keeps
           every run inside [-1, 1] at the cost of never sampling the corners.
- ``ccf``  Central composite, face-centred. Factorial cube at +/-1 and star runs at +/-1
           (alpha = 1). Three levels per factor and everything inside the cube.
- ``bbd``  Box-Behnken. Midpoints of the cube edges plus the centre; avoids the extreme
           corner combinations. Built with ``process_improve``'s ``generate_design``.
- ``doehlert``  Doehlert (uniform shell) design. The k(k+1) pairwise differences of a
           regular unit simplex, all on the unit sphere, plus the centre. The most
           economical of the five (13 runs at k = 3, tied with Box-Behnken here).

The central composite matrices are constructed directly (not through the library) because
the library's full-cube CCD only honours the named alphas "rotatable" / "orthogonal" /
"face_centered"; the varying-alpha figure needs arbitrary alpha, so all CCD variants are
built the same explicit way for consistency. Box-Behnken comes from ``process_improve``.
Prediction variance and the dense FDS curve come from ``process_improve``'s
``evaluate_design`` (v1.44.0+), exactly as in ``omnibus_designs.py``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from process_improve.experiments import Factor, evaluate_design, generate_design

K = 3  # number of factors
N_EVAL = 120_000  # uniform points for the prediction-variance integration
EVAL_SEED = 1
N_CENTRE = 1  # one centre run per design, so geometry (not replication) drives the contrast

FACTOR_NAMES = list("ABC")
FACTORS = [Factor(name=name, low=-1, high=1) for name in FACTOR_NAMES]

# Full second-order model: 3 linear + 3 two-factor interactions + 3 pure quadratics.
MODEL_FORMULA = " + ".join(
    FACTOR_NAMES
    + [f"{a}:{b}" for i, a in enumerate(FACTOR_NAMES) for b in FACTOR_NAMES[i + 1 :]]
    + [f"I({name}**2)" for name in FACTOR_NAMES]
)

# The rotatable axial distance, alpha = (number of factorial-cube runs)^(1/4) = (2^k)^(1/4).
ALPHA_ROTATABLE = float((2**K) ** 0.25)


def ccd_matrix(alpha: float, k: int = K, n_centre: int = N_CENTRE) -> np.ndarray:
    """Coded central composite design: 2^k factorial cube (+/-1), 2k axial runs (+/-alpha)
    and ``n_centre`` centre runs.

    Parameters
    ----------
    alpha : float
        Axial (star) distance. ``1`` gives the face-centred design, ``(2^k)^(1/4)`` the
        rotatable (circumscribed) design.
    k : int
        Number of factors.
    n_centre : int
        Number of centre runs.

    Returns
    -------
    numpy.ndarray
        The ``(2^k + 2k + n_centre)`` by ``k`` coded design matrix.
    """
    factorial = np.array(list(itertools.product([-1.0, 1.0], repeat=k)), dtype=float)
    axial = np.zeros((2 * k, k))
    for i in range(k):
        axial[2 * i, i] = -alpha
        axial[2 * i + 1, i] = alpha
    centre = np.zeros((n_centre, k))
    return np.vstack([factorial, axial, centre])


def doehlert_matrix(k: int = K, n_centre: int = N_CENTRE) -> np.ndarray:
    """Coded Doehlert (uniform shell) design.

    The non-centre points are the ``k(k+1)`` ordered pairwise differences ``v_i - v_j`` of a
    regular unit-edge simplex with ``k + 1`` vertices. Because a regular simplex is
    equidistant, every difference has unit length, so all shell points lie on the unit
    sphere. The result is the canonical Doehlert design (a hexagon plus centre at k = 2, a
    cuboctahedron plus centre at k = 3).

    Parameters
    ----------
    k : int
        Number of factors.
    n_centre : int
        Number of centre runs.

    Returns
    -------
    numpy.ndarray
        The ``(k^2 + k + n_centre)`` by ``k`` coded design matrix.
    """
    vertices = _unit_simplex(k)
    shell = np.array([vertices[i] - vertices[j] for i in range(k + 1) for j in range(k + 1) if i != j])
    # Guard the defining property: every shell point sits on the unit sphere.
    assert np.allclose(np.linalg.norm(shell, axis=1), 1.0), "Doehlert shell points must be unit length"
    centre = np.zeros((n_centre, k))
    return np.vstack([shell, centre])


def _unit_simplex(k: int) -> np.ndarray:
    """Vertices of a regular unit-edge simplex in ``k`` dimensions, vertex 0 at the origin.

    Built dimension by dimension: each new vertex shares the previous coordinates that keep
    it equidistant from those already placed, then takes a final coordinate that restores
    unit edge length. Returns a ``(k + 1)`` by ``k`` array.
    """
    vertices = np.zeros((k + 1, k))
    for i in range(1, k + 1):
        # Coordinates 0..i-2 place this vertex above the centroid of the previous ones.
        prev = vertices[:i, : i - 1]
        vertices[i, : i - 1] = prev.mean(axis=0) if i > 1 else 0.0
        # The remaining free coordinate (index i-1) restores unit distance to vertex 0.
        placed = np.linalg.norm(vertices[i, : i - 1] - vertices[0, : i - 1]) if i > 1 else 0.0
        vertices[i, i - 1] = float(np.sqrt(max(1.0 - placed**2, 0.0)))
    return vertices


def build_designs() -> dict[str, np.ndarray]:
    """Construct all five design families. Returns ``name -> (N x k)`` coded array."""
    ccc = ccd_matrix(ALPHA_ROTATABLE)
    cci = ccd_matrix(ALPHA_ROTATABLE) / ALPHA_ROTATABLE  # inscribe: scale everything into +/-1
    ccf = ccd_matrix(1.0)
    bbd = np.asarray(
        generate_design(FACTORS, "box_behnken", center_points=N_CENTRE).design[FACTOR_NAMES],
        dtype=float,
    )
    doehlert = doehlert_matrix()
    return {"ccc": ccc, "cci": cci, "ccf": ccf, "bbd": bbd, "doehlert": doehlert}


# Display order (largest exploration reach first) and labels for the figures/tables.
FAMILY_ORDER = ["ccc", "ccf", "cci", "bbd", "doehlert"]
LABELS = {
    "ccc": "CCD circumscribed (rotatable)",
    "ccf": "CCD face-centred",
    "cci": "CCD inscribed",
    "bbd": "Box-Behnken",
    "doehlert": "Doehlert",
}
STYLES = {
    "ccc": dict(color="#1f5fa8", lw=2.0),
    "ccf": dict(color="#2e8b57", lw=2.0, ls="-."),
    "cci": dict(color="#d68910", lw=2.0, ls=(0, (5, 1))),
    "bbd": dict(color="#8e44ad", lw=2.0, ls="--"),
    "doehlert": dict(color="#c0392b", lw=2.0, ls=":"),
}


def evaluate(design: np.ndarray, *, fds_resolution: int = 200) -> dict:
    """Quality metrics and the dense FDS curve for one design on the second-order model.

    D-efficiency, degrees of freedom and the region prediction variance (with the dense FDS
    curve) come from ``process_improve``'s ``evaluate_design``, integrated over the cube
    [-1, 1]^3 with the 2^k corner vertices added so the worst-case (G) value is represented.
    """
    m = _model_matrix(design)
    n, p = m.shape
    rank = int(np.linalg.matrix_rank(m))
    df = pd.DataFrame(np.asarray(design, float), columns=FACTOR_NAMES)
    lib = evaluate_design(
        df,
        model=MODEL_FORMULA,
        metric=["d_efficiency", "degrees_of_freedom", "fds"],
        n_samples=N_EVAL,
        random_seed=EVAL_SEED,
        include_vertices=True,
        region="cuboidal",
        fds_resolution=fds_resolution,
    )
    fds = lib["fds"]
    return {
        "N": n,
        "p": p,
        "rank": rank,
        "full_rank": rank == p,
        "residual_df": lib["degrees_of_freedom"]["residual"],
        "d_eff": float(lib["d_efficiency"]),
        "avg_pv": float(fds["average_prediction_variance"]),
        "max_pv": float(fds["max_prediction_variance"]),
        "avg_spv": float(fds["average_prediction_variance"] * n),
        "max_spv": float(fds["max_prediction_variance"] * n),
        "curve": fds["curve"],
    }


def _model_matrix(design: np.ndarray) -> np.ndarray:
    """Return the 10-column second-order model matrix [1 | x_i | x_i x_j | x_i^2]."""
    d = np.asarray(design, float)
    n, k = d.shape
    cols = [np.ones(n)]
    cols += [d[:, i] for i in range(k)]
    cols += [d[:, i] * d[:, j] for i in range(k) for j in range(i + 1, k)]
    cols += [d[:, i] ** 2 for i in range(k)]
    return np.column_stack(cols)


__all__ = [
    "ALPHA_ROTATABLE",
    "FAMILY_ORDER",
    "K",
    "LABELS",
    "MODEL_FORMULA",
    "STYLES",
    "build_designs",
    "ccd_matrix",
    "doehlert_matrix",
    "evaluate",
]
