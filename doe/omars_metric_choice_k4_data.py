"""Regenerate the literal data carried in omars-metric-choice-k4.py, four factors.

The three-factor figure (omars-metric-choice.py) is exact at every run count: its data module
walks every OMARS foldover of every size. At four factors that walk is out of reach past ten
half-rows (22 million designs at h = 10, roughly 3.5 times more for every half-row after), so
this module does two things:

* **Exact, h <= 10.** Every OMARS foldover with up to ten half-rows is scored, which covers
  every run count up to 21 with one centre run, 22 with two and 23 with three.
* **Search, h >= 11.** Above that, an iterated local search over the same design family
  reports the best design it finds. Those points are a lower bound on what is attainable,
  not the frontier, and the figure draws them with open markers.

The search is checked where both exist: run with ``--check-search`` it searches h = 9 and
h = 10 as if they were out of reach, and compares what it finds with the exact values.

Run it with no arguments to print the literals the figure carries:

    python omars_metric_choice_k4_data.py                 # about an hour on four cores
    python omars_metric_choice_k4_data.py --check-k3      # scorer against the k = 3 figure
    python omars_metric_choice_k4_data.py --check-search  # search against exact, h = 9, 10

How it scores
-------------
A design is h half-rows, their mirror images and c centre runs, carried as a count per sign
class of {-1, 0, 1}^k (40 classes at four factors). Every measure is built from those counts
directly, many designs at a time: the main effects are orthogonal to everything, so X'X splits
into a diagonal main-effect block and an even block over the intercept, the quadratics and
(for power) the interactions. No N x p model matrix is formed. The same scorer, run at k = 3,
reproduces the three-factor figure's literals (``--check-k3``).
"""

# check-scripts: slow the full four-factor run takes about an hour on four cores
from __future__ import annotations

import itertools
import math
import multiprocessing
import sys

import numpy as np

LEVELS = np.linspace(-1, 1, 7)          # the grid G is maximised over, per factor
CENTRES = (1, 2, 3)
H_EXACT = 10                             # largest number of half-rows walked exhaustively
MAX_RUNS = 31
BATCH = 100_000


# ---------------------------------------------------------------------------------------
# The design family
# ---------------------------------------------------------------------------------------
class Family:
    """The sign classes of {-1, 0, 1}^k and every per-class quantity the scorer needs."""

    def __init__(self, k: int):
        self.k = k
        seen, reps = set(), []
        for v in itertools.product((-1, 0, 1), repeat=k):
            if v in seen or not any(v):
                continue
            seen.add(v)
            seen.add(tuple(-x for x in v))
            reps.append(v)
        pairs = list(itertools.combinations(range(k), 2))
        # Most-entangled classes first: the walk prunes earliest that way.
        reps.sort(key=lambda v: -sum(1 for i, j in pairs if v[i] and v[j]))
        self.reps = reps
        self.pairs = pairs
        R = np.array(reps, dtype=float)
        self.sq = R**2                                                # quadratics, n x k
        self.inter = np.column_stack([R[:, i] * R[:, j] for i, j in pairs])
        z = np.hstack([self.sq, self.inter])                          # second-order terms
        self.z = z
        self.zz = np.einsum("na,nb->nab", z, z).reshape(len(reps), -1)
        self.qq = np.einsum("na,nb->nab", self.sq, self.sq).reshape(len(reps), -1)
        self.contrib = self.inter.astype(int)                         # main-effect Gram
        # Region moments over [1, quadratics] for I, and the grid of squared levels for G:
        # every term in the even block and the main-effect variances depends on x only
        # through x squared, so the 7^k grid collapses to 4^k distinct points.
        b = np.full((k + 1, k + 1), 1 / 9)
        b[0, :] = b[:, 0] = 1 / 3
        b[0, 0] = 1.0
        np.fill_diagonal(b[1:, 1:], 1 / 5)
        self.b_even = b
        squares = np.unique(LEVELS**2)
        x2 = np.array(list(itertools.product(squares, repeat=k)))
        self.x2 = x2
        g = np.column_stack([np.ones(len(x2)), x2])
        self.gg = np.einsum("pa,pb->pab", g, g).reshape(len(x2), -1)

    def design(self, counts, n_centre: int) -> np.ndarray:
        """Materialise [H; -H; 0] from a count per sign class."""
        rows = [self.reps[i] for i, c in enumerate(counts) for _ in range(int(c))]
        return np.array([list(r) for r in rows] + [[-x for x in r] for r in rows]
                        + [[0] * self.k] * n_centre, dtype=float)


# ---------------------------------------------------------------------------------------
# Scoring, many designs at once
# ---------------------------------------------------------------------------------------
def score(fam: Family, C: np.ndarray, n_centre: int, full: bool) -> dict:
    """Every measure for a batch of designs given as class counts, one row per design.

    Returns arrays over the batch; an entry is NaN where the measure is not defined (a
    singular model, a factor that never moves, a constant second-order column).
    """
    k = fam.k
    C = np.asarray(C, dtype=float)
    h = C.sum(axis=1)
    N = 2 * h + n_centre
    S = C @ fam.sq                                       # half-rows with factor j off zero
    m = 2 * S                                            # main-effect block of X'X
    ok = (S > 0).all(axis=1)
    B = len(C)
    E = np.empty((B, k + 1, k + 1))
    E[:, 0, 0] = N
    E[:, 0, 1:] = E[:, 1:, 0] = m
    E[:, 1:, 1:] = 2 * (C @ fam.qq).reshape(B, k, k)
    ev = np.linalg.eigvalsh(E)
    m_safe = np.where(m > 0, m, 1.0)
    lam_min = np.minimum(ev[:, 0], np.where(ok, m.min(axis=1), 0.0))
    ok &= lam_min > 1e-9
    E_safe = np.where(ok[:, None, None], E, np.eye(k + 1))
    Einv = np.linalg.inv(E_safe)
    inv_m = 1 / m_safe
    p = 2 * k + 1
    out = {
        "A": (np.trace(Einv, axis1=1, axis2=2) + inv_m.sum(axis=1)) / p,
        "D": np.exp((np.log(np.linalg.det(E_safe)) + np.log(m_safe).sum(axis=1)) / p),
        "E": lam_min,
        "I": np.einsum("nab,ba->n", Einv, fam.b_even) + inv_m.sum(axis=1) / 3,
        "G": (Einv.reshape(B, -1) @ fam.gg.T + inv_m @ fam.x2.T).max(axis=1),
    }
    for key in out:
        out[key] = np.where(ok, out[key], np.nan)

    # Largest absolute correlation between two second-order columns.
    n_so = fam.z.shape[1]
    mom = 2 * (C @ fam.zz).reshape(B, n_so, n_so) / N[:, None, None]
    mean = 2 * (C @ fam.z) / N[:, None]
    cov = mom - mean[:, :, None] * mean[:, None, :]
    sd = np.sqrt(np.clip(np.einsum("naa->na", cov), 0, None))
    good = (sd > 1e-9).all(axis=1) & ok
    sd_safe = np.where(sd > 1e-9, sd, 1.0)
    corr = np.abs(cov / sd_safe[:, :, None] / sd_safe[:, None, :])
    corr[:, np.arange(n_so), np.arange(n_so)] = 0
    out["maxr"] = np.where(good, corr.max(axis=(1, 2)), np.nan)

    if full:
        # Full second-order model: even block over [1, quadratics, interactions].
        F = np.empty((B, n_so + 1, n_so + 1))
        F[:, 0, 0] = N
        F[:, 0, 1:] = F[:, 1:, 0] = 2 * (C @ fam.z)
        F[:, 1:, 1:] = 2 * (C @ fam.zz).reshape(B, n_so, n_so)
        evf = np.linalg.eigvalsh(F)
        okf = ok & (evf[:, 0] > 1e-9 * evf[:, -1])
        d = np.einsum("naa->na", np.linalg.inv(np.where(okf[:, None, None], F,
                                                         np.eye(n_so + 1))))
        okf &= np.isfinite(d).all(axis=1) & (d[:, 1:].min(axis=1) > 0) & (d.max(axis=1) < 1e6)
        out["c_main"] = np.where(okf, 1 / (2 * np.where(S.min(axis=1) > 0, S.min(axis=1), 1)),
                                 np.nan)
        out["c_int"] = np.where(okf, d[:, k + 1:].max(axis=1), np.nan)
        out["c_quad"] = np.where(okf, d[:, 1:k + 1].max(axis=1), np.nan)
    return out


# Which way is better, per measure.
SENSE = {"A": -1, "D": 1, "E": 1, "I": -1, "G": -1, "maxr": -1,
         "c_main": -1, "c_int": -1, "c_quad": -1}


class Best:
    """Running best of every measure, with the counts that attain it."""

    def __init__(self):
        self.value, self.counts = {}, {}

    def update(self, scores: dict, C: np.ndarray):
        for key, arr in scores.items():
            if not np.isfinite(arr).any():
                continue
            i = int(np.nanargmax(SENSE[key] * arr))
            v = float(arr[i])
            old = self.value.get(key)
            if old is None or SENSE[key] * v > SENSE[key] * old + 1e-12:
                self.value[key], self.counts[key] = v, np.array(C[i], dtype=int)


# ---------------------------------------------------------------------------------------
# Exact: walk every foldover with h half-rows
# ---------------------------------------------------------------------------------------
def walk(fam: Family, h: int, emit) -> None:
    """Call ``emit(counts)`` once per OMARS foldover with h half-rows (pruned DFS)."""
    contribs = [list(c) for c in fam.contrib]
    n_cls, n_pairs = len(contribs), len(fam.pairs)
    suffix = [[0] * n_pairs for _ in range(n_cls + 1)]
    for idx in range(n_cls - 1, -1, -1):
        for q in range(n_pairs):
            suffix[idx][q] = max(suffix[idx + 1][q], abs(contribs[idx][q]))
    counts = [0] * n_cls

    def dfs(idx, left, gram):
        if left == 0:
            if not any(gram):
                emit(counts)
            return
        if idx == n_cls:
            return
        sm = suffix[idx]
        for q in range(n_pairs):
            if abs(gram[q]) > left * sm[q]:
                return
        contrib = contribs[idx]
        for c in range(left, -1, -1):
            counts[idx] = c
            dfs(idx + 1, left - c, [g + c * d for g, d in zip(gram, contrib)])
        counts[idx] = 0

    dfs(0, h, [0] * n_pairs)


def exact_h(args):
    """Every measure's best over all foldovers with h half-rows, for each centre count."""
    k, h = args
    fam = Family(k)
    centres = [c for c in CENTRES if 2 * h + c <= MAX_RUNS]
    full = h >= k * (k + 1) // 2
    best = {c: Best() for c in centres}
    buf = np.zeros((BATCH, len(fam.reps)), dtype=np.int8)
    fill = [0]

    def flush():
        if fill[0]:
            for c in centres:
                best[c].update(score(fam, buf[:fill[0]], c, full), buf[:fill[0]])
            fill[0] = 0

    def emit(counts):
        buf[fill[0]] = counts
        fill[0] += 1
        if fill[0] == BATCH:
            flush()

    walk(fam, h, emit)
    flush()
    return h, {c: (b.value, {key: v.tolist() for key, v in b.counts.items()})
               for c, b in best.items()}


# ---------------------------------------------------------------------------------------
# Search: iterated local search over the same family, for h past the exact range
# ---------------------------------------------------------------------------------------
class Neighbourhood:
    """Moves that keep the main effects orthogonal: swap one or two half-rows for others
    with the same summed contribution to the main-effect Gram matrix."""

    def __init__(self, fam: Family):
        self.fam = fam
        n = len(fam.reps)
        key1 = {}
        for a in range(n):
            key1.setdefault(tuple(fam.contrib[a]), []).append(a)
        self.single = {a: [b for b in key1[tuple(fam.contrib[a])] if b != a] for a in range(n)}
        key2 = {}
        for a in range(n):
            for b in range(a, n):
                key2.setdefault(tuple(fam.contrib[a] + fam.contrib[b]), []).append((a, b))
        self.key2 = key2

    def moves(self, counts: np.ndarray) -> np.ndarray:
        """Every neighbour of ``counts``, as rows of class counts."""
        present = [a for a in range(len(counts)) if counts[a] > 0]
        out = []
        for a in present:
            for b in self.single[a]:
                nb = counts.copy()
                nb[a] -= 1
                nb[b] += 1
                out.append(nb)
        for i, a in enumerate(present):
            for b in present[i:]:
                if a == b and counts[a] < 2:
                    continue
                key = tuple(self.fam.contrib[a] + self.fam.contrib[b])
                for x, y in self.key2[key]:
                    if {x, y} == {a, b}:
                        continue
                    nb = counts.copy()
                    nb[a] -= 1
                    nb[b] -= 1
                    nb[x] += 1
                    nb[y] += 1
                    out.append(nb)
        return np.array(out, dtype=int) if out else np.zeros((0, len(counts)), dtype=int)


def random_design(fam: Family, h: int, rng) -> np.ndarray | None:
    """One random OMARS foldover with h half-rows: a randomised, pruned DFS."""
    n = len(fam.reps)
    order = rng.permutation(n)
    contribs = fam.contrib[order]
    n_pairs = contribs.shape[1]
    suffix = np.zeros((n + 1, n_pairs), dtype=int)
    for idx in range(n - 1, -1, -1):
        suffix[idx] = np.maximum(suffix[idx + 1], np.abs(contribs[idx]))
    counts = np.zeros(n, dtype=int)
    steps = [0]

    def dfs(idx, left, gram):
        steps[0] += 1
        if steps[0] > 20_000:
            return False
        if left == 0:
            return not gram.any()
        if idx == n or (np.abs(gram) > left * suffix[idx]).any():
            return False
        choices = list(range(min(left, 3) + 1))
        rng.shuffle(choices)
        if rng.random() < 0.6 and 0 in choices:          # keep designs spread over classes
            choices.remove(0)
            choices.append(0)
        for c in choices:
            counts[idx] = c
            if dfs(idx + 1, left - c, gram + c * contribs[idx]):
                return True
        counts[idx] = 0
        return False

    if not dfs(0, h, np.zeros(n_pairs, dtype=int)):
        return None
    out = np.zeros(n, dtype=int)
    out[order] = counts
    if (out @ fam.sq).min() < 1:
        return None
    return out


def search_hc(args):
    """Iterated local search for one (h, centre count), steering by each measure in turn
    and keeping the best of every measure over every design it scores."""
    k, h, n_centre, restarts, seed = args
    fam = Family(k)
    hood = Neighbourhood(fam)
    rng = np.random.default_rng(seed)
    full = h >= k * (k + 1) // 2 and 2 * h + n_centre > 1 + 2 * k + k * (k - 1) // 2
    best = Best()
    keys = [key for key in SENSE if full or not key.startswith("c_")]
    for key in keys:
        sense = SENSE[key]
        for _ in range(restarts):
            current = None
            while current is None:
                current = random_design(fam, h, rng)
            cur_score = score(fam, current[None], n_centre, full)
            best.update(cur_score, current[None])
            cur = sense * cur_score[key][0] if np.isfinite(cur_score[key][0]) else -math.inf
            for _ in range(200):
                nbs = hood.moves(current)
                if not len(nbs):
                    break
                sc = score(fam, nbs, n_centre, full)
                best.update(sc, nbs)
                vals = np.where(np.isfinite(sc[key]), sense * sc[key], -math.inf)
                i = int(np.argmax(vals))
                if vals[i] <= cur + 1e-12:
                    break
                current, cur = nbs[i], vals[i]
    return (h, n_centre), (best.value, {kk: v.tolist() for kk, v in best.counts.items()})


# ---------------------------------------------------------------------------------------
# Anchors: the DSD and the Box-Behnken design built by process_improve
# ---------------------------------------------------------------------------------------
def anchor_values(k: int) -> dict:
    from process_improve.experiments import Factor  # noqa: PLC0415
    from process_improve.experiments.designs_response_surface import (  # noqa: PLC0415
        dispatch_box_behnken,
        dispatch_dsd,
    )

    fam = Family(k)
    factors = [Factor(name=f"x{i + 1}", low=-1, high=1) for i in range(k)]
    out = {}
    for name, (design, _) in (("bbd", dispatch_box_behnken(factors)),
                              ("dsd", dispatch_dsd(factors))):
        d = np.asarray(design, dtype=float)
        n_centre = int((np.abs(d).sum(axis=1) == 0).sum())
        rest = d[np.abs(d).sum(axis=1) > 0]
        counts = np.zeros(len(fam.reps), dtype=int)
        index = {v: i for i, v in enumerate(fam.reps)}
        for row in rest:
            v = tuple(int(x) for x in row)
            counts[index[v] if v in index else index[tuple(-x for x in v)]] += 1
        counts //= 2                                     # each half-row appears with its mirror
        full = len(d) > 1 + 2 * k + k * (k - 1) // 2
        sc = score(fam, counts[None], n_centre, full)
        entry = {"n_runs": len(d)}
        entry.update({key: float(sc[key][0]) for key in ("A", "E", "D", "I", "G", "maxr")})
        entry["c"] = (tuple(float(sc[key][0]) for key in ("c_main", "c_int", "c_quad"))
                      if full and np.isfinite(sc["c_int"][0]) else None)
        out[name] = entry
    return out


# ---------------------------------------------------------------------------------------
def correlation_matrix(fam: Family, counts, n_centre: int) -> np.ndarray:
    d = fam.design(counts, n_centre)
    k = fam.k
    cols = [d[:, i] ** 2 for i in range(k)] + [d[:, i] * d[:, j] for i, j in fam.pairs]
    return np.abs(np.corrcoef(np.column_stack(cols), rowvar=False))


def run(k: int = 4, restarts: int = 150):
    with multiprocessing.Pool(4) as pool:
        exact = dict(pool.map(exact_h, [(k, h) for h in range(H_EXACT, k - 1, -1)]))
        jobs = [(k, h, c, restarts, 1000 * h + c) for c in CENTRES
                for h in range(H_EXACT + 1, (MAX_RUNS - c) // 2 + 1)]
        searched = dict(pool.map(search_hc, jobs))
    return exact, searched


def print_literals(k: int = 4) -> None:
    exact, searched = run(k)
    rows = {}            # (centre, N) -> (values, counts, exact?)
    for h, per_c in exact.items():
        for c, (vals, counts) in per_c.items():
            rows[(c, 2 * h + c)] = (vals, counts, True)
    for (h, c), (vals, counts) in searched.items():
        rows[(c, 2 * h + c)] = (vals, counts, False)
    fam = Family(k)

    def fmt(x):
        return "None" if x is None or not math.isfinite(x) else f"{x:.6f}"

    print("BEST = {")
    for c in CENTRES:
        items = sorted((n, v) for (cc, n), v in rows.items() if cc == c and "A" in v[0])
        body = ",\n        ".join(
            f"{n}: ({', '.join(fmt(v[0][key]) for key in ('A', 'D', 'E', 'I', 'G'))})"
            for n, v in items)
        print(f"    {c}: {{{body}}},")
    print("}\nMAX_R = {")
    for c in CENTRES:
        items = sorted((n, v) for (cc, n), v in rows.items() if cc == c and "maxr" in v[0])
        print(f"    {c}: {{" + ", ".join(f"{n}: {fmt(v[0]['maxr'])}" for n, v in items) + "},")
    print("}\nPOWER_C = {")
    for c in CENTRES:
        items = sorted((n, v) for (cc, n), v in rows.items() if cc == c and "c_int" in v[0])
        body = ",\n        ".join(
            f"{n}: ({', '.join(fmt(v[0][key]) for key in ('c_main', 'c_int', 'c_quad'))})"
            for n, v in items)
        print(f"    {c}: {{{body}}},")
    print("}\nEXACT = {")
    for c in CENTRES:
        ns = sorted(n for (cc, n), v in rows.items() if cc == c and v[2])
        print(f"    {c}: {max(ns)},")
    print("}\nMAX_R_DESIGNS = {")
    for (c, n), (vals, counts, _) in sorted(rows.items()):
        if "maxr" in counts:
            print(f"    ({c}, {n}): {counts['maxr']},")
    print("}\nANCHORS = " + repr(anchor_values(k)))
    print("\n# correlation matrices of the max |r| designs, for the insets")
    for (c, n), (vals, counts, _) in sorted(rows.items()):
        if "maxr" in counts:
            r = correlation_matrix(fam, counts["maxr"], c)
            print(f"# ({c}, {n}) max|r| = {vals['maxr']:.6f}")
            print("    (%d, %d): %s," % (c, n, np.round(r, 3).tolist()))


def check_k3() -> int:
    """The vectorised scorer, run exhaustively at k = 3, against the figure's literals."""
    import ast  # noqa: PLC0415
    import pathlib  # noqa: PLC0415

    src = (pathlib.Path(__file__).parent / "omars-metric-choice.py").read_text("utf-8")
    lit = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(src).body
           if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
           and node.targets[0].id in {"BEST", "MAX_R", "POWER_C"}}
    worst = 0.0
    for h in range(3, 16):
        _, per_c = exact_h((3, h))
        for c, (vals, _) in per_c.items():
            n = 2 * h + c
            if n in lit["BEST"].get(c, {}):
                mine = [vals[key] for key in ("A", "D", "E", "I", "G")]
                worst = max(worst, max(abs(a - b) for a, b in zip(mine, lit["BEST"][c][n])))
            if n in lit["MAX_R"].get(c, {}):
                worst = max(worst, abs(vals["maxr"] - lit["MAX_R"][c][n]))
            if n in lit["POWER_C"].get(c, {}):
                mine = [vals[key] for key in ("c_main", "c_int", "c_quad")]
                worst = max(worst, max(abs(a - b) for a, b in zip(mine, lit["POWER_C"][c][n])))
        print(f"h = {h:2d} checked, worst difference so far {worst:.2e}", flush=True)
    return 0 if worst < 5e-6 else 1


def check_search(k: int = 4, restarts: int = 150) -> int:
    """Search h = 9 and 10 as if out of reach, and compare with the exact walk."""
    with multiprocessing.Pool(4) as pool:
        exact = dict(pool.map(exact_h, [(k, 10), (k, 9)]))
        jobs = [(k, h, c, restarts, 7 * h + c) for h in (9, 10) for c in CENTRES
                if 2 * h + c <= MAX_RUNS]
        found = dict(pool.map(search_hc, jobs))
    hits = total = 0
    for (h, c), (vals, _) in sorted(found.items()):
        truth = exact[h][c][0]
        for key, v in sorted(vals.items()):
            total += 1
            gap = SENSE[key] * (truth[key] - v)
            hit = gap <= 1e-9 * max(1.0, abs(truth[key]))
            hits += hit
            print(f"h={h} c={c} {key:6s} exact {truth[key]:.6f} found {v:.6f} "
                  f"{'match' if hit else f'short by {gap:.2e}'}")
    print(f"\nsearch reached the exact best in {hits} of {total} cells")
    return 0


if __name__ == "__main__":
    if "--check-k3" in sys.argv[1:]:
        sys.exit(check_k3())
    if "--check-search" in sys.argv[1:]:
        sys.exit(check_search())
    print_literals()
