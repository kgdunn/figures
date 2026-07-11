"""Hotelling's T2 per run under four categorical codings of the same interaction model.

The six-level chromogen factor can be written with different contrast codings that all span the same
model space. A full-rank fit does not care which is used, but the 3-component, scaled PLS does: its
per-run Hotelling's T2 (leverage inside the model plane) depends on the coding. Under sum (effects)
coding the omitted level is carried as the negative sum of the other contrasts, the corner of the
contrast space farthest from the centre, so its runs read as high leverage; change which level is
omitted and the flag moves with it. Treatment (reference) and cell-means coding place no level at
that corner, so nothing crosses the 95% T2 limit. Regenerates ``colour-coding-diagnostics.png``.
"""

import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np

from colour_case_study import COMPOUND_LEVELS, build_design, fit_coding, simulate_curves

design = build_design("i_optimal", budget=60)
curves = simulate_curves(design)
compound = design.design["compound"].to_numpy()

palette = ["#1f5fa8", "#c0392b", "#2e8b57", "#8e44ad", "#d68910", "#17a2b8"]
colour_of = dict(zip(COMPOUND_LEVELS, palette))
xpos = {c: i for i, c in enumerate(COMPOUND_LEVELS)}
rng = np.random.default_rng(3)

panels = [
    ("Sum coding, F omitted", "sum", COMPOUND_LEVELS),
    ("Sum coding, A omitted", "sum", list(reversed(COMPOUND_LEVELS))),
    ("Treatment coding, A reference", "treatment", COMPOUND_LEVELS),
    ("Cell-means coding", "cell_means", COMPOUND_LEVELS),
]

fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True, sharey=True)
for ax, (title, coding, order) in zip(axes.ravel(), panels):
    with contextlib.redirect_stderr(io.StringIO()):
        pls, _ = fit_coding(design, curves, coding, order=order)
    t2 = pls.hotellings_t2_.iloc[:, -1].to_numpy()
    t2lim = float(pls.hotellings_t2_limit(0.95))
    for c in COMPOUND_LEVELS:
        m = compound == c
        jit = rng.uniform(-0.18, 0.18, m.sum())
        ax.scatter(xpos[c] + jit, t2[m], s=34, color=colour_of[c], edgecolor="w",
                   linewidth=0.4, zorder=3)
    ax.axhline(t2lim, color="0.4", ls="--", lw=1.1, zorder=2)
    ax.text(len(COMPOUND_LEVELS) - 0.5, t2lim, f" 95% = {t2lim:.1f}", color="0.35",
            fontsize=8, va="bottom", ha="right")
    n_over = int((t2 > t2lim).sum())
    flagged = sorted({c for c in COMPOUND_LEVELS if (t2[compound == c] > t2lim).any()})
    tag = f"{n_over} over limit ({', '.join(flagged)})" if flagged else "none over limit"
    ax.set_title(f"{title}\n{tag}", fontsize=9.5, loc="left")
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(["A (ref)" if c == "A" else c for c in COMPOUND_LEVELS])
    ax.grid(axis="y", alpha=0.2)

for ax in axes[:, 0]:
    ax.set_ylabel("Hotelling's $T^2$ (3 components)")
for ax in axes[1, :]:
    ax.set_xlabel("chromogen")
fig.suptitle("Diagnostic leverage depends on the categorical coding, not on the chemistry",
             fontsize=11, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("colour-coding-diagnostics.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved colour-coding-diagnostics.png")
