"""Regenerate the literal data carried in omars-metric-choice.py, three factors.

The figure carries its numbers as literals because the exhaustive enumeration behind them
takes far too long to run at plot time. This module is where those literals come from, so
they can be re-derived rather than trusted.

Run it with no arguments to print ``POWER_C`` and ``ANCHORS`` exactly as the figure spells
them, ready to paste back:

    python omars_metric_choice_data.py

Run it with ``--verify`` to re-derive a sample of the six enumerated panels the naive way,
building the actual N x p model matrix for every candidate design with numpy and comparing
against the ``BEST`` and ``MAX_R`` literals in the figure. Nothing is shared between the
fast enumeration and the naive one except the definition of an OMARS foldover, so agreement
is real evidence. It takes a few minutes:

    python omars_metric_choice_data.py --verify

What is enumerated
------------------
A design is ``h`` real half-rows, their ``h`` mirror images and ``c`` runs at the centre, so
``N = 2h + c``. Every factor must vary and the main effects must be mutually orthogonal.
The enumeration walks a count per sign class of ``{-1, 0, 1}^k`` rather than a list of
rows, which is what makes it exhaustive rather than heuristic: for the second-order terms a
half-row and its negation are interchangeable, so 26 rows collapse to 13 classes at three
factors.

Why the power scoring is cheap
------------------------------
Main effects are orthogonal to everything in an OMARS design, so the main-effect entry of
``(X'X)^-1`` is ``1 / (2 n_j)`` with ``n_j`` the number of half-rows in which factor ``j``
is off zero. Everything else lives in the even block over the intercept, the quadratics and
the interactions, which is built from the class counts directly. No N x p matrix is ever
formed, so a run count takes seconds rather than hours.
"""

from __future__ import annotations

import itertools
import math
import sys

import numpy as np

K = 3
LEVELS = np.linspace(-1, 1, 7)          # the grid G is maximised over, per factor
CENTRES = (1, 2, 3)


# ---------------------------------------------------------------------------------------
# The design family
# ---------------------------------------------------------------------------------------
def build_classes(k: int) -> list[tuple[int, ...]]:
    """One representative half-row per sign class of {-1, 0, 1}^k, excluding the origin."""
    seen: set[tuple[int, ...]] = set()
    reps: list[tuple[int, ...]] = []
    for v in itertools.product((-1, 0, 1), repeat=k):
        if v in seen or not any(v):
            continue
        seen.add(v)
        seen.add(tuple(-x for x in v))
        reps.append(v)
    return reps


def second_order(v: tuple[int, ...], k: int) -> list[int]:
    """The k quadratics then the k(k-1)/2 interactions, evaluated on one half-row."""
    return ([v[i] * v[i] for i in range(k)]
            + [v[i] * v[j] for i, j in itertools.combinations(range(k), 2)])


def foldover(counts, reps, k: int, n_centre: int) -> np.ndarray:
    """Materialise [H; -H; 0] from a count per sign class."""
    rows = [reps[i] for i, c in enumerate(counts) for _ in range(c)]
    return np.array([list(r) for r in rows]
                    + [[-x for x in r] for r in rows]
                    + [[0] * k] * n_centre, dtype=float)


def walk(k: int, h: int, visit) -> None:
    """Call ``visit(counts)`` once per OMARS foldover with h half-rows.

    Depth-first over the compositions of h into the sign classes, pruned on the pairwise
    Gram sums that main-effect orthogonality forces to zero.
    """
    reps = build_classes(k)
    reps.sort(key=lambda v: -sum(1 for i, j in itertools.combinations(range(k), 2)
                                 if v[i] and v[j]))
    pairs = list(itertools.combinations(range(k), 2))
    contribs = [[v[i] * v[j] for i, j in pairs] for v in reps]
    n_pairs = len(pairs)
    suffix = [[0] * n_pairs for _ in range(len(reps) + 1)]
    for idx in range(len(reps) - 1, -1, -1):
        for q in range(n_pairs):
            suffix[idx][q] = max(suffix[idx + 1][q], abs(contribs[idx][q]))

    def dfs(idx, left, counts, gram):
        if left == 0:
            visit(counts + [0] * (len(reps) - len(counts)))
            return
        if idx == len(reps):
            return
        sm = suffix[idx]
        for q in range(n_pairs):
            if abs(gram[q]) > left * sm[q]:
                return
        contrib = contribs[idx]
        for c in range(left, -1, -1):
            dfs(idx + 1, left - c, counts + [c],
                [g + c * d for g, d in zip(gram, contrib)])

    dfs(0, h, [], [0] * n_pairs)


# ---------------------------------------------------------------------------------------
# Power under the full second-order model: the POWER_C literal
# ---------------------------------------------------------------------------------------
def power_frontier(k: int, n_runs: int, n_centre: int):
    """Smallest attainable (main effect, interaction, quadratic) coefficient variance.

    The three are minimised independently, so each is a frontier in its own right and one
    design need not attain all three. Returns ``None`` when no design of this size can fit
    the full second-order model.
    """
    if (n_runs - n_centre) % 2 or n_runs <= n_centre:
        return None
    h = (n_runs - n_centre) // 2
    if n_runs < 1 + 2 * k + k * (k - 1) // 2:
        return None
    # The even block sees at most h + 1 distinct rows against 1 + k(k+1)/2 columns, so the
    # full second-order model needs h >= k(k+1)/2 whatever the run count. Three centre runs
    # at N = 13 and k = 3 clears N >= p but leaves h = 5, one short.
    if h < k * (k + 1) // 2:
        return None

    reps = build_classes(k)
    reps.sort(key=lambda v: -sum(1 for i, j in itertools.combinations(range(k), 2)
                                 if v[i] and v[j]))
    so = [second_order(v, k) for v in reps]
    n_terms = len(so[0])
    cross = [[[s[a] * s[b] for b in range(n_terms)] for a in range(n_terms)] for s in so]
    support = [[1 if x else 0 for x in v] for v in reps]
    best = [math.inf, math.inf, math.inf]
    M = np.empty((n_terms + 1, n_terms + 1))

    def visit(counts):
        n = [0] * k
        for idx, c in enumerate(counts):
            if not c:
                continue
            s = support[idx]
            for a in range(k):
                if s[a]:
                    n[a] += c
        if min(n) < 1:
            return
        best[0] = min(best[0], 1.0 / (2 * min(n)))

        totals = [0] * n_terms
        gram = [[0] * n_terms for _ in range(n_terms)]
        for idx, c in enumerate(counts):
            if not c:
                continue
            si, ci = so[idx], cross[idx]
            for a in range(n_terms):
                totals[a] += c * si[a]
                ga, ca = gram[a], ci[a]
                for b in range(a, n_terms):
                    ga[b] += c * ca[b]
        M[0, 0] = n_runs
        for a in range(n_terms):
            M[0, a + 1] = M[a + 1, 0] = 2 * totals[a]
            for b in range(a, n_terms):
                M[a + 1, b + 1] = M[b + 1, a + 1] = 2 * gram[a][b]
        try:
            d = np.diag(np.linalg.inv(M))
        except np.linalg.LinAlgError:
            return
        # A rank-deficient even block does not always raise: numpy can return an inverse
        # with entries of order 1e15 instead. Every coefficient variance here is a small
        # rational, so anything past 1e6 is that failure, not a real design. Checking the
        # inverse costs nothing; np.linalg.cond would run an SVD on every candidate.
        if not np.all(np.isfinite(d)) or np.min(d[1:]) <= 0 or np.max(d[1:]) > 1e6:
            return
        best[1] = min(best[1], float(np.max(d[k + 1:])))       # interactions
        best[2] = min(best[2], float(np.max(d[1:k + 1])))      # quadratics

    walk(k, h, visit)
    return None if best[1] == math.inf else tuple(best)


# ---------------------------------------------------------------------------------------
# The six enumerated measures, scored naively for one design
# ---------------------------------------------------------------------------------------
def region_moments(k: int) -> np.ndarray:
    """Moment matrix of the cuboidal region, for the exact I criterion."""
    p = 2 * k + 1
    B = np.zeros((p, p))
    B[0, 0] = 1.0
    for i in range(k):
        B[1 + i, 1 + i] = 1 / 3
        B[0, 1 + k + i] = B[1 + k + i, 0] = 1 / 3
        for j in range(k):
            B[1 + k + i, 1 + k + j] = 1 / 5 if i == j else 1 / 9
    return B


def region_grid(k: int) -> np.ndarray:
    """Model matrix over a seven-level grid per factor, for the G criterion."""
    pts = np.array(list(itertools.product(LEVELS, repeat=k)))
    return np.column_stack([np.ones(len(pts)), pts, pts ** 2])


def six(design: np.ndarray, k: int, B: np.ndarray, F: np.ndarray):
    """(A/p, D, E, I, G, max |r|) for one design, main effects and quadratics."""
    n = len(design)
    X = np.column_stack([np.ones(n), design, design ** 2])
    M = X.T @ X
    p = M.shape[0]
    ev = np.linalg.eigvalsh(M)
    if ev.min() <= 1e-9:
        return None
    Minv = np.linalg.inv(M)
    A = np.trace(Minv) / p
    D = np.linalg.det(M) ** (1 / p)
    E = ev.min()
    I = np.trace(Minv @ B)
    G = np.max(((F @ Minv) * F).sum(axis=1))
    cols = [design[:, i] ** 2 for i in range(k)]
    cols += [design[:, i] * design[:, j] for i, j in itertools.combinations(range(k), 2)]
    Z = np.column_stack(cols)
    if np.any(Z.std(axis=0) < 1e-12):
        r = None                     # a constant second-order column has no correlation
    else:
        C = np.abs(np.corrcoef(Z, rowvar=False))
        r = float(C[~np.eye(len(C), dtype=bool)].max())
    return float(A), float(D), float(E), float(I), float(G), r


def best_six(k: int, n_runs: int, n_centre: int):
    """The naive enumeration: build every model matrix and take the best of each measure."""
    if (n_runs - n_centre) % 2 or n_runs <= n_centre:
        return None
    h = (n_runs - n_centre) // 2
    reps = build_classes(k)
    reps.sort(key=lambda v: -sum(1 for i, j in itertools.combinations(range(k), 2)
                                 if v[i] and v[j]))
    B, F = region_moments(k), region_grid(k)
    out = [math.inf, -math.inf, -math.inf, math.inf, math.inf, math.inf]

    def visit(counts):
        design = foldover(counts, reps, k, n_centre)
        if np.abs(design.T @ design - np.diag(np.diag(design.T @ design))).max() > 1e-9:
            return                   # main effects not mutually orthogonal
        got = six(design, k, B, F)
        if got is None:
            return
        A, D, E, I, G, r = got
        out[0] = min(out[0], A)
        out[1] = max(out[1], D)
        out[2] = max(out[2], E)
        out[3] = min(out[3], I)
        out[4] = min(out[4], G)
        if r is not None:
            out[5] = min(out[5], r)

    walk(k, h, visit)
    return out


# ---------------------------------------------------------------------------------------
# The two anchor designs: the ANCHORS literal
# ---------------------------------------------------------------------------------------
def anchor_values(k: int) -> dict:
    """Score the definitive screening and Box-Behnken designs built by process_improve."""
    from process_improve.experiments import Factor
    from process_improve.experiments.designs_response_surface import (
        dispatch_box_behnken,
        dispatch_dsd,
    )

    factors = [Factor(name=f"x{i + 1}", low=-1, high=1) for i in range(k)]
    B, F = region_moments(k), region_grid(k)
    p_full = 1 + 2 * k + k * (k - 1) // 2
    out = {}
    for key, (design, _) in (("bbd", dispatch_box_behnken(factors)),
                             ("dsd", dispatch_dsd(factors))):
        d = np.asarray(design, dtype=float)
        n = len(d)
        A, D, E, I, G, r = six(d, k, B, F)
        cols = [np.ones(n)] + [d[:, i] for i in range(k)] + [d[:, i] ** 2 for i in range(k)]
        cols += [d[:, i] * d[:, j] for i, j in itertools.combinations(range(k), 2)]
        X = np.column_stack(cols)
        if n - p_full > 0 and np.linalg.matrix_rank(X) == X.shape[1]:
            diag = np.diag(np.linalg.inv(X.T @ X))
            c = (float(np.max(diag[1:k + 1])),          # main effect
                 float(np.max(diag[2 * k + 1:])),       # interaction
                 float(np.max(diag[k + 1:2 * k + 1])))  # quadratic
        else:
            c = None                 # the full second-order model does not fit
        out[key] = {"n_runs": n, "A": A, "E": E, "D": D, "I": I, "G": G, "maxr": r, "c": c}
    return out


# ---------------------------------------------------------------------------------------
def print_literals() -> None:
    print("POWER_C = {")
    for centre in CENTRES:
        rows = []
        for n_runs in range(K * K + K + 1, 32):
            got = power_frontier(K, n_runs, centre)
            if got is not None:
                rows.append(f"{n_runs}: ({got[0]:.6f}, {got[1]:.6f}, {got[2]:.6f})")
        print(f"    {centre}: {{" + ",\n        ".join(rows) + "},")
    print("}")

    print("\nANCHORS = {")
    for key, v in anchor_values(K).items():
        c = "None" if v["c"] is None else \
            f'({v["c"][0]:.6f}, {v["c"][1]:.6f}, {v["c"][2]:.6f})'
        print(f'    "{key}": {{"n_runs": {v["n_runs"]}, "A": {v["A"]:.6f}, '
              f'"E": {v["E"]:.6f}, "D": {v["D"]:.6f},')
        print(f'            "I": {v["I"]:.6f}, "G": {v["G"]:.6f}, '
              f'"maxr": {v["maxr"]:.6f},')
        print(f'            "c": {c}}},')
    print("}")


def figure_literals() -> dict:
    """Read BEST and MAX_R out of the figure script.

    The figure is named with hyphens, so it cannot be imported; and importing it would
    draw the whole plot as a side effect. Parsing the assignments keeps this a read.
    """
    import ast                             # noqa: PLC0415
    import pathlib                         # noqa: PLC0415

    source = (pathlib.Path(__file__).parent / "omars-metric-choice.py").read_text("utf-8")
    return {node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(source).body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"BEST", "MAX_R"}}


def verify() -> int:
    """Re-derive a sample of the BEST and MAX_R cells naively and compare."""
    literals = figure_literals()
    cells = [(1, 9), (1, 11), (1, 13), (1, 15), (1, 17), (2, 10), (2, 12), (2, 14),
             (2, 16), (3, 11), (3, 13), (3, 15)]
    worst = 0.0
    for centre, n_runs in cells:
        mine = best_six(K, n_runs, centre)
        best = literals["BEST"][centre][n_runs]
        theirs = [best[0], best[1], best[2], best[3], best[4],
                  literals["MAX_R"][centre][n_runs]]
        diff = max(abs(a - b) for a, b in zip(mine, theirs))
        worst = max(worst, diff)
        flag = "ok " if diff < 5e-6 else "MISMATCH"
        print(f"{flag} c={centre} N={n_runs:2d}  "
              + "  ".join(f"{lab}={v:.6f}" for lab, v in zip("ADEIGr", mine)), flush=True)
    print(f"\nlargest absolute difference over {len(cells)} cells: {worst:.2e}")
    return 0 if worst < 5e-6 else 1


if __name__ == "__main__":
    sys.exit(verify() if "--verify" in sys.argv[1:] else print_literals())
