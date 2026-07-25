"""Paste the geometric-PCA illustrations together in pairs.

The eight single-panel illustrations of the geometric interpretation of PCA
are drawn by hand, not by a script. The chapter shows them two at a time,
so this joins each pair side by side, leaving a transparent gap between
them, and writes the four combined images the book embeds:

- ``geometric-PCA-1-and-2-swarm-with-mean.png``
- ``geometric-PCA-3-and-4-centered-with-first-component.png``
- ``geometric-PCA-5-and-6-first-component-with-projections-and-second-component.png``
- ``geometric-PCA-7-and-8-second-component-and-both-components.png``

This replaces ``geometric-PCA-combine-figures.py``, which could not run:
it imported ``Image`` directly (Python 2 PIL), needed a
``transparent-pixel.png`` file that is not in this directory, filled the
gap by pasting that pixel one position at a time in a Python loop, and
asked for ``geometric-PCA-8-noth-components-with-plane.png``, misspelling
``both``. A new RGBA canvas is already transparent, so no filler is needed.

Usage
-----
    uv run --with pillow python geometric_pca_combine.py [output_dir]
"""

from __future__ import annotations

import pathlib
import sys

from PIL import Image

HERE = pathlib.Path(__file__).parent

PAIRS = [
    (
        "geometric-PCA-1-swarm-only.png",
        "geometric-PCA-2-swarm-with-mean.png",
        200,
        "geometric-PCA-1-and-2-swarm-with-mean.png",
    ),
    (
        "geometric-PCA-3-centered.png",
        "geometric-PCA-4-first-component.png",
        150,
        "geometric-PCA-3-and-4-centered-with-first-component.png",
    ),
    (
        "geometric-PCA-5-first-component-with-projections.png",
        "geometric-PCA-6-second-component.png",
        150,
        "geometric-PCA-5-and-6-first-component-with-projections-and-second-component.png",
    ),
    (
        "geometric-PCA-7-second-component-with-projections.png",
        "geometric-PCA-8-both-components-with-plane.png",
        150,
        "geometric-PCA-7-and-8-second-component-and-both-components.png",
    ),
]


def combine(left_name: str, right_name: str, gap: int, out_name: str,
            outdir: pathlib.Path) -> None:
    left = Image.open(HERE / left_name).convert("RGBA")
    right = Image.open(HERE / right_name).convert("RGBA")
    width = left.width + gap + right.width
    height = max(left.height, right.height)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    # Copy each panel's RGBA straight in, rather than compositing it over
    # the canvas: the panels already carry their own transparency, and
    # compositing would re-blend their antialiased edges.
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    canvas.save(outdir / out_name)
    print(f"wrote {outdir / out_name} ({width} by {height})")


def main(outdir: pathlib.Path) -> None:
    for left_name, right_name, gap, out_name in PAIRS:
        combine(left_name, right_name, gap, out_name, outdir)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
