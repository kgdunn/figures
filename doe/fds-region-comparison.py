"""The experimental region decides the winner: the same five designs, two regions.

Both panels show the scaled prediction variance FDS for the circumscribed, face-centred
and inscribed CCDs, Box-Behnken and Doehlert at k = 3 on the full second-order model. Only
the region integrated over changes.

- left, the cube [-1, 1]^3: the corner-reaching designs (circumscribed and face-centred
  CCD, Box-Behnken) predict well everywhere, while the uniform-shell designs (inscribed CCD
  and Doehlert) never leave the unit sphere and so extrapolate at the corners; their curves
  shoot off the top of the axis.
- right, the unit ball (radius 1): the region those shell designs are actually built for.
  Now every design is well behaved. The Doehlert and inscribed CCD are competitive, and the
  circumscribed CCD looks wasteful in the middle, because it spends its star runs out at
  alpha = 1.682, well outside the ball it is being asked to predict inside.

The lesson (the same one the omnibus subsection makes): a design is only good or bad
relative to the region you intend to predict over. Doehlert and the inscribed CCD are
spherical-region designs; judging them on the cube is judging them off their home ground.

Reproducible; run from this directory: writes ``fds-region-comparison.png``.
"""

import matplotlib.pyplot as plt

from ccd_variants_designs import FAMILY_ORDER, LABELS, STYLES, ball_fds_curve, build_designs, evaluate

designs = build_designs()
cube = {name: evaluate(designs[name])["curve"] for name in FAMILY_ORDER}
ball = {name: ball_fds_curve(designs[name], radius=1.0) for name in FAMILY_ORDER}


def thin(curve, step=250):
    """Down-sample a dense FDS curve for a light vector line."""
    f = curve["fraction"]
    v = curve["scaled_prediction_variance"]
    idx = list(range(0, len(f), step)) + [len(f) - 1]
    return [f[i] for i in idx], [v[i] for i in idx]


fig, (ax_cube, ax_ball) = plt.subplots(1, 2, figsize=(11.0, 4.7), sharex=True)
for name in FAMILY_ORDER:
    n = len(designs[name])
    label = f"{LABELS[name]} [n={n}]"
    fc, vc = thin(cube[name], step=2)  # library curve is already ~200 points
    fb, vb = thin(ball[name])
    ax_cube.plot(fc, vc, label=label, **STYLES[name])
    ax_ball.plot(fb, vb, label=label, **STYLES[name])

ax_cube.set_title("Region = cube [-1, 1]$^3$ (corner-reaching designs win)")
ax_ball.set_title("Region = unit ball, radius 1 (shell designs are at home)")
for ax in (ax_cube, ax_ball):
    ax.set_xlabel("Fraction of design space (SPV at or below the curve)")
    ax.set_ylabel("Scaled prediction variance, SPV = N x'(X'X)$^{-1}$x")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 25)
    ax.grid(alpha=0.25)
ax_cube.legend(frameon=False, loc="upper left", fontsize=8.2)
ax_cube.annotate(
    "inscribed CCD and Doehlert\nrun off the top here",
    xy=(0.30, 16.0), fontsize=8.5, va="top", ha="left", color="#555555",
)

fig.suptitle("The same five designs judged over two regions (k = 3, second-order model)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("fds-region-comparison.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved fds-region-comparison.png")
