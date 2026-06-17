"""Validate the single-factor, three-level worked example and the four-design augmentation
comparison behind ``optimal-and-omars-designs.rst`` (the information-matrix worked example)
and ``judging-and-comparing-designs.rst`` (the augmentation table and the prediction-variance
discussion).

Run from this directory: ``python3 check_augmentation_example.py``. Every assertion must
pass; the printed numbers are the ones quoted in the book. Requires numpy only.

The model throughout is the one-factor quadratic ``y = b0 + b1 x + b2 x^2``, so each run at
coded level ``x`` expands to ``x_m = [1, x, x^2]``. Four designs on the coded range
[-1, +1] are compared:

- base                : x = -1, 0, +1                         (3 runs)
- base, all repeated  : the base design run a second time     (6 runs)
- base + two centres  : base plus two extra runs at x = 0     (5 runs)
- base + two extremes : base plus a run at x = -1 and x = +1  (5 runs)

For each design the script reports the raw optimality criteria (D = det(M),
A = trace(M^-1), E = smallest eigenvalue of M) and the unscaled prediction variance
d(x) = x_m' M^-1 x_m summarised over the region (G = max, I = average), matching the
augmentation table.
"""

import numpy as np

GRID = np.linspace(-1.0, 1.0, 20001)  # dense 1-D grid for the max/average prediction variance


def expand(x):
    """Quadratic model expansion x -> [1, x, x^2] for a vector of coded levels."""
    x = np.asarray(x, float)
    return np.column_stack([np.ones_like(x), x, x ** 2])


def designs():
    """The four one-factor designs, each as a vector of coded levels."""
    base = np.array([-1.0, 0.0, 1.0])
    return {
        "base {-1, 0, +1}": base,
        "base, all repeated": np.concatenate([base, base]),
        "base + two centres": np.concatenate([base, [0.0, 0.0]]),
        "base + two extremes": np.concatenate([base, [-1.0, 1.0]]),
    }


def evaluate(levels):
    """Information matrix and its scalar summaries for one one-factor design."""
    x_model = expand(levels)
    m = x_model.T @ x_model
    m_inv = np.linalg.inv(m)
    d = np.einsum("ij,jk,ik->i", expand(GRID), m_inv, expand(GRID))
    return {
        "N": len(levels),
        "M": m,
        "M_inv": m_inv,
        "D": float(np.linalg.det(m)),
        "A": float(np.trace(m_inv)),
        "E": float(np.linalg.eigvalsh(m)[0]),
        "G": float(d.max()),
        "I": float(d.mean()),
        "var": np.diag(m_inv),
    }


def main():
    evaluated = {name: evaluate(levels) for name, levels in designs().items()}

    # The information matrix and inverse of the base design (the worked example in
    # optimal-and-omars-designs.rst).
    base = evaluated["base {-1, 0, +1}"]
    assert np.allclose(base["M"], [[3, 0, 2], [0, 2, 0], [2, 0, 2]])
    assert np.allclose(base["M_inv"], [[1, 0, -1], [0, 0.5, 0], [-1, 0, 1.5]])
    assert np.allclose(base["var"], [1.0, 0.5, 1.5])  # Var(b0), Var(b1), Var(b2) in sigma^2

    # The two five-run designs share the same intercept information (M_00 = 5) but differ in
    # the intercept-quadratic cross-term: the centre-point design keeps it at M_02 = 2 and
    # drops Var(b2) from 1.5 to 0.83 sigma^2, while the extreme-point design raises it to
    # M_02 = 4. This is the de-correlation argument in the augmentation discussion.
    centre = evaluated["base + two centres"]
    extreme = evaluated["base + two extremes"]
    assert np.allclose(centre["M"], [[5, 0, 2], [0, 2, 0], [2, 0, 2]])
    assert np.allclose(extreme["M"], [[5, 0, 4], [0, 4, 0], [4, 0, 4]])
    assert round(centre["var"][2], 2) == 0.83
    assert round(extreme["var"][2], 2) == 1.25

    # The base design's prediction variance is the polynomial 1 - 1.5 x^2 + 1.5 x^4, with a
    # minimum of 0.625 sigma^2 at x = +/- 1/sqrt(2) and 5.2 sigma^2 at x = 1.5.
    def poly(x):
        return 1 - 1.5 * x ** 2 + 1.5 * x ** 4

    assert np.allclose(
        np.einsum("ij,jk,ik->i", expand(GRID), base["M_inv"], expand(GRID)), poly(GRID)
    )
    assert round(float(poly(1 / np.sqrt(2))), 3) == 0.625
    assert round(float(poly(1.5)), 1) == 5.2

    # The augmentation table, exactly as quoted in the book.
    expected = {
        # name                    N      D     A     E    G     I
        "base {-1, 0, +1}":    (3,  4.0, 3.00, 0.44, 1.0, 0.80),
        "base, all repeated":  (6, 32.0, 1.50, 0.88, 0.5, 0.40),
        "base + two centres":  (5, 12.0, 1.67, 1.00, 1.0, 0.44),
        "base + two extremes": (5, 16.0, 2.50, 0.47, 1.0, 0.67),
    }
    print(f"{'design':22s} {'N':>2s} {'D':>5s} {'A':>5s} {'E':>5s} {'G':>4s} {'I':>5s}")
    for name, (n, d_crit, a_crit, e_crit, g, i) in expected.items():
        r = evaluated[name]
        assert r["N"] == n
        assert round(r["D"], 1) == d_crit, f"{name}: D {r['D']} != {d_crit}"
        assert round(r["A"], 2) == a_crit, f"{name}: A {r['A']} != {a_crit}"
        assert round(r["E"], 2) == e_crit, f"{name}: E {r['E']} != {e_crit}"
        assert round(r["G"], 1) == g, f"{name}: G {r['G']} != {g}"
        assert round(r["I"], 2) == i, f"{name}: I {r['I']} != {i}"
        print(f"{name:22s} {n:2d} {d_crit:5.0f} {a_crit:5.2f} {e_crit:5.2f} {g:4.1f} {i:5.2f}")

    print("\nAll augmentation-example assertions passed.")


if __name__ == "__main__":
    main()
