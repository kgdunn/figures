"""Check that no label in the batch case-study figures is hidden, on a marker, or on another label.

Usage, from this directory::

    python check_labels.py batch-case-fmc-figures.py batch-case-dupont-figures.py batch-case-sbr-figures.py

Each script is imported with ``save`` replaced, so every figure is laid out and measured but nothing
is written to disk. One line is printed per violation; silence means every label is clear.

Two things make a label fail. Matplotlib does not clip an annotation to its axes, so a label is
invisible only when it runs off the *figure*. And it is unreadable when it intrudes into a marker's
disc, sits under a legend, or lands on another label. ``batch_case_common.place_labels`` is what
avoids all of these; this script is the check that it did.
"""
from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.transforms import Bbox

BATCH = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BATCH))

BITE = 2.0  # pixels a label may intrude into a marker or overlap another box before it is a collision


def discs(ax) -> list[tuple[np.ndarray, float]]:
    """Every plotted marker as (centre in pixels, radius in pixels); a marker's `s` is its area in pt^2."""
    out = []
    for coll in ax.collections:
        if not isinstance(coll, PathCollection):
            continue
        offsets = np.asarray(coll.get_offsets())
        if offsets.size == 0:
            continue
        centres = ax.transData.transform(offsets)
        sizes = np.asarray(coll.get_sizes())
        sizes = np.broadcast_to(sizes if sizes.size else np.array([36.0]), (len(centres),))
        out += [(c, float(np.sqrt(s)) / 2 * ax.figure.dpi / 72.0) for c, s in zip(centres, sizes, strict=True)]
    return out


def bite_into_disc(box: Bbox, centre: np.ndarray, radius: float) -> float:
    """How far the box reaches inside the disc, in pixels: positive means they genuinely overlap."""
    nearest = np.clip(centre, [box.x0, box.y0], [box.x1, box.y1])
    return radius - float(np.hypot(*(centre - nearest)))


def overlap(a: Bbox, b: Bbox) -> float:
    """The smaller side of the intersection of two boxes, in pixels; zero or less if they miss."""
    return min(min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0))


def audit_axes(ax, renderer, figure_box: Bbox) -> list[str]:
    labels = [t for t in ax.texts if t.get_text().strip() and t.get_visible()]
    if not labels:
        return []
    boxes = [Text.get_window_extent(t, renderer) for t in labels]  # glyphs only: not the leader line
    markers = discs(ax)
    legends = [lg.get_window_extent(renderer) for lg in ax.findobj(Legend)]
    problems = []
    for text, box in zip(labels, boxes, strict=True):
        name = repr(text.get_text())
        if not figure_box.contains(box.x0, box.y0) or not figure_box.contains(box.x1, box.y1):
            problems.append(f"{name} runs off the figure")
        for other, other_box in zip(labels, boxes, strict=True):
            if other is not text and overlap(box, other_box) > BITE:
                problems.append(f"{name} overlaps the label {other.get_text()!r}")
                break
        deepest = max((bite_into_disc(box, c, r) for c, r in markers), default=0.0)
        # A rotated string's axis-aligned box is far larger than its glyphs, so the disc test would
        # flag every diagonal caption. Rotated text is checked by eye, not here.
        if deepest > BITE and not text.get_rotation():
            problems.append(f"{name} sits {deepest:.0f} px inside a marker")
        if any(overlap(box, lg) > BITE for lg in legends):
            problems.append(f"{name} sits under a legend")
    return problems


def audit(fig, name: str) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = Bbox.from_extents(0, 0, *fig.canvas.get_width_height())
    for k, ax in enumerate(fig.axes):
        for problem in audit_axes(ax, renderer, figure_box):
            print(f"{name}  axes[{k}] ({ax.get_title() or 'untitled'}): {problem}")


def run(script: pathlib.Path) -> None:
    import batch_case_common

    def audited(fig, out_dir, figure_name):  # noqa: ARG001 -- out_dir is deliberately ignored
        batch_case_common.place_labels(fig)  # exactly what the real `save` does before writing
        audit(fig, figure_name)
        plt.close(fig)

    batch_case_common.save = audited
    spec = importlib.util.spec_from_file_location(script.stem.replace("-", "_"), script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.save = audited  # the script imported `save` by name, so rebind it there too
    takes = inspect.signature(module.main).parameters
    module.main(pathlib.Path("/dev/null"), *([None] * (len(takes) - 1)))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(f"===== {arg} =====")
        run(BATCH / arg)
