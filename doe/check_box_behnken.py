"""Validate the Box-Behnken run-count table and the geometry claims in the Box-Behnken
subsection of ``response-surface-methods.rst``.

Run from this directory: ``python3 check_box_behnken.py``. Every assertion must pass.
Requires numpy only.

A Box-Behnken design (BBD) places no runs at the cube vertices: every non-centre run sits
at the midpoint of an edge, two factors at +/-1 and the rest at 0, so it lies on a sphere
of radius sqrt(2). For three, four, and five factors the design uses all C(k, 2) factor
pairs, giving 4 * C(k, 2) edge runs. For six and seven factors the classical designs
(Box and Behnken, 1960) choose the pairs through a balanced incomplete block design, so
they use fewer edge runs than the all-pairs count; those values are catalogue numbers,
not a formula, and are asserted directly.
"""

import itertools
from math import comb, sqrt

import numpy as np


def box_behnken_all_pairs(k):
    """All-pairs Box-Behnken edge runs for k factors (the construction for k = 3, 4, 5)."""
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        for a in (-1, 1):
            for b in (-1, 1):
                row = [0] * k
                row[i], row[j] = a, b
                rows.append(row)
    return np.array(rows, float)


# Published Box-Behnken catalogue: edge runs, typical centre runs, and the total.
TABLE = {
    3: (12, 3, 15),
    4: (24, 3, 27),
    5: (40, 6, 46),
    6: (48, 6, 54),
    7: (56, 6, 62),
}


def main():
    # Geometry, shown on the three-factor design.
    bbd3 = box_behnken_all_pairs(3)
    assert len(bbd3) == 12
    assert set(np.unique(bbd3)) <= {-1.0, 0.0, 1.0}, "only three levels are used"
    # No run sits at a cube vertex (all three factors at an extreme simultaneously).
    assert not any(np.all(np.abs(row) == 1) for row in bbd3), "a BBD has no corner runs"
    # Every edge run is at radius sqrt(2) from the centre.
    assert np.allclose(np.linalg.norm(bbd3, axis=1), sqrt(2))

    # Run-count table.
    print(f"{'factors k':>9s} {'edge':>5s} {'centre':>7s} {'total':>6s}")
    for k, (edge, centre, total) in TABLE.items():
        assert edge + centre == total
        if k <= 5:
            # Three to five factors: the BBD uses every factor pair.
            assert edge == 4 * comb(k, 2) == len(box_behnken_all_pairs(k))
        else:
            # Six and seven factors: the all-pairs count would be larger, so the classical
            # design must use a balanced incomplete block design with fewer pairs.
            assert edge < 4 * comb(k, 2)
        print(f"{k:>9d} {edge:>5d} {centre:>7d} {total:>6d}")

    print("\nAll Box-Behnken assertions passed.")


if __name__ == "__main__":
    main()
