"""Two 2x2 heatmap panels for the four response-surface designs (BBD, CCD, OMARS, DSD).

Both figures visualise the entanglement that the omnibus table only summarises:

- ``alias-matrix-heatmaps-four-designs.png``: the absolute alias matrix |A|, where
  A = (X1' X1)^-1 X1' X2 maps the ten omitted two-factor interactions (columns) onto the
  eleven fitted terms (rows). The Box-Behnken and composite designs hold every entry at
  zero on this model; the OMARS and definitive screening designs keep the main-effect rows
  at zero but push the bias onto the quadratic rows (worst entry 1.00 and 1.09).

- ``correlation-colormap-four-designs.png``: the absolute correlation among the twenty
  model-effect columns, in three blocks separated by lines, the five main effects, the five
  pure quadratics, and the ten two-factor interactions. The main-effect and quadratic blocks
  are the fitted terms (their worst off-diagonal is the table's "Maximum |r|" row); the
  interaction block and its cross-blocks show the entanglement with the interactions the
  model omits.

Reproducible; run from this directory. The numbers are locked to the omnibus table by
asserts in ``check_omnibus.py``.
"""

import matplotlib.pyplot as plt
import numpy as np

from omnibus_designs import (
    LABELS,
    MODEL_TERM_BLOCKS,
    RSM_DESIGNS,
    alias_matrix,
    build_designs,
    model_term_corr,
)

CMAP = "Blues"  # 0 = white (no entanglement), dark = strong; matches the journal colour-map look

# Fitted-term (row) and omitted-interaction (column) labels for the alias matrix.
FITTED_LABELS = ["1", "A", "B", "C", "D", "E"] + [rf"${c}^2$" for c in "ABCDE"]
INTERACTION_LABELS = [f"{a}{b}" for i, a in enumerate("ABCDE") for b in "ABCDE"[i + 1:]]
# Twenty model-effect labels (main effects, quadratics, interactions) for the correlation map.
MODEL_LABELS = list("ABCDE") + [rf"${c}^2$" for c in "ABCDE"] + INTERACTION_LABELS


def worst_offdiagonal(square):
    """Largest absolute off-diagonal entry of a square matrix."""
    return float(np.abs(square)[~np.eye(square.shape[0], dtype=bool)].max())


def _panel(ax, matrix, row_labels, col_labels, title, vmax, annotate, separators,
           annotate_top=False):
    im = ax.imshow(matrix, cmap=CMAP, vmin=0.0, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=6.5, rotation=90)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6.5)
    ax.tick_params(length=0)
    for b in separators:  # block-separating lines between main effects, quadratics, interactions
        ax.axvline(b - 0.5, color="0.62", lw=0.9)
        ax.axhline(b - 0.5, color="0.62", lw=0.9)
    # Top-right corner is white for every design (main effects x interactions in the correlation
    # map; the intercept and main-effect rows in the alias map), so both maps place the annotation
    # there.
    y, va = (0.90, "top") if annotate_top else (0.04, "bottom")
    ax.text(0.97, y, annotate, transform=ax.transAxes, ha="right", va=va,
            fontsize=8, color="0.25")
    return im


def make_figure(matrices, row_labels, col_labels, vmax, annotate_fmt, cbar_label, outfile,
                separators=(), annotate_top=False):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 9.8))
    im = None
    for ax, name in zip(axes.flat, RSM_DESIGNS):
        m = matrices[name]
        worst = worst_offdiagonal(m) if row_labels is col_labels else float(np.abs(m).max())
        im = _panel(ax, m, row_labels, col_labels,
                    LABELS[name].split("(")[0].strip(), vmax, annotate_fmt.format(worst),
                    separators, annotate_top=annotate_top)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.94, bottom=0.10, hspace=0.30, wspace=0.25)
    cax = fig.add_axes([0.91, 0.12, 0.02, 0.76])
    fig.colorbar(im, cax=cax, label=cbar_label)
    fig.savefig(outfile, dpi=300, facecolor="w", edgecolor="w",
                orientation="portrait", format=None, transparent=True)
    print(f"saved {outfile}")


def main():
    designs = build_designs()

    alias = {name: np.abs(alias_matrix(designs[name])) for name in RSM_DESIGNS}
    alias_vmax = max(m.max() for m in alias.values())
    make_figure(alias, FITTED_LABELS, INTERACTION_LABELS, alias_vmax,
                annotate_fmt="max |A| = {:.2f}", cbar_label="Absolute alias coefficient, |A|",
                outfile="alias-matrix-heatmaps-four-designs.png", annotate_top=True)

    corr = {name: model_term_corr(designs[name]) for name in RSM_DESIGNS}
    make_figure(corr, MODEL_LABELS, MODEL_LABELS, 1.0,
                annotate_fmt="max |r| = {:.2f}",
                cbar_label="Absolute correlation among model-effect columns, |r|",
                outfile="correlation-colormap-four-designs.png",
                separators=MODEL_TERM_BLOCKS, annotate_top=True)


if __name__ == "__main__":
    main()
