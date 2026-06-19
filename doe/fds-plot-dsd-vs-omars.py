"""Generate the FDS (fraction-of-design-space) figure for the design-quality subchapter.

Compares a 4-factor definitive screening design (9 runs) against a 13-run OMARS
design on the main-effects-plus-quadratic model. The uniform sample is augmented
with the 2^4 cube vertices (the extreme corner points of the region), where the
worst-case prediction variance can sit and which random interior sampling misses.
Reproducible; run from this directory: it writes ``fds-plot-dsd-vs-omars.png``
alongside the script.
"""
import itertools

import numpy as np
import matplotlib.pyplot as plt
from process_improve.experiments import Factor, generate_design

rng = np.random.default_rng(1)


def main_quadratic_model(D):
    D = np.asarray(D, float)
    N, k = D.shape
    cols = [np.ones(N)] + [D[:, i] for i in range(k)] + [D[:, i] ** 2 for i in range(k)]
    return np.column_stack(cols)


def expand(P):
    cols = [np.ones(len(P))] + [P[:, i] for i in range(P.shape[1])] + \
           [P[:, i] ** 2 for i in range(P.shape[1])]
    return np.column_stack(cols)


# 4-factor DSD (9 runs) built with process_improve: a conference-matrix foldover plus a
# centre run. The library has no OMARS generator, so the 13-run OMARS design below is
# built by hand.
_dsd = generate_design([Factor(name=c, low=-1, high=1) for c in "ABCD"], design_type="dsd")
DSD = np.asarray(_dsd.design[["A", "B", "C", "D"]], float)
# 13-run OMARS design (balanced foldover member, 2 estimable interactions)
OM13 = np.array([
    [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, -1, -1], [1, -1, -1, 0], [1, 0, 1, -1],
    [1, 1, 0, 1], [0, 0, 0, -1], [0, 0, -1, 0], [0, -1, 1, 1], [-1, 1, 1, 0],
    [-1, 0, -1, 1], [-1, -1, 0, -1], [0, 0, 0, 0]], float)

# Augment the interior sample with the cube vertices: the worst-case (G) value can
# sit exactly at a corner, which interior sampling under-finds. Here it lifts the
# DSD's worst case from 8.98 to 9.00; the OMARS worst case is interior, so it is
# unchanged. The change is for methodological form, not substance.
vertices = np.array(list(itertools.product([-1, 1], repeat=4)), float)
pts = np.vstack([rng.uniform(-1, 1, size=(80000, 4)), vertices])
Xm = expand(pts)
fracs = np.linspace(0, 1, 200)

fig, ax = plt.subplots(figsize=(7.2, 4.6))
styles = {"DSD (9 runs)": dict(color="#1f5fa8", lw=2.0),
          "OMARS (13 runs)": dict(color="#c0392b", lw=2.0, ls="--")}
for name, D in [("DSD (9 runs)", DSD), ("OMARS (13 runs)", OM13)]:
    M = main_quadratic_model(D)
    N = len(D)
    Minv = np.linalg.inv(M.T @ M)
    spv = N * np.einsum("ij,jk,ik->i", Xm, Minv, Xm)
    curve = np.quantile(np.sort(spv), fracs)
    ax.plot(fracs, curve, label=name, **styles[name])

ax.set_xlabel("Fraction of design space (proportion of region with SPV at or below the curve)")
ax.set_ylabel("Scaled prediction variance, SPV")
ax.set_xlim(0, 1)
ax.set_ylim(0, 13)
ax.axvline(0.5, color="0.6", lw=0.8, ls=":")
ax.text(0.51, 0.6, "median", color="0.4", fontsize=8, rotation=90, va="bottom")
ax.legend(frameon=False, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fds-plot-dsd-vs-omars.png", dpi=300, facecolor="w", edgecolor="w",
            orientation="portrait", format=None, transparent=True)
print("saved figure")
