"""How the chapters of the PID book refer to each other, drawn as an arc diagram.

The figure is measured, not drawn by hand: every ``:ref:`` in a chapter that
points at a label defined in a different chapter is one link. The chapters sit
on a line in reading order. Arcs above the line point back to an earlier
chapter (the material a chapter builds on); arcs below point ahead to a later
chapter. Line width grows with the square root of the number of links, which is printed
on each arc.

Re-run it after a content change and the figure follows the book.

Usage
-----
    uv run --with matplotlib python chapter-links.py [book_root] [output.png]

``book_root`` defaults to the book checkout that holds this repository as its
``figures/`` link, so from the book's root this is enough:
``uv run --with matplotlib python figures/concepts/chapter-links.py``.
"""

import collections
import pathlib
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Short names for the chapter folders, in reading order (contents.rst decides the order).
SHORT_NAMES = {
    "data-visualization": "Visualization",
    "univariate-review": "Univariate\nstatistics",
    "process-monitoring": "Process\nmonitoring",
    "least-squares-modelling": "Least\nsquares",
    "design-analysis-experiments": "Designed\nexperiments",
    "latent-variable-modelling": "Latent\nvariables",
    "product-development-product-improvement": "Product\ndevelopment",
}
LABEL_RE = re.compile(r"^\.\. _([^:]+):\s*$", re.MULTILINE)
REF_RE = re.compile(r":ref:`(?:[^`<]*<)?([^`<>]+)>?`")
BACK, AHEAD = "#2a6fb0", "#d9822b"


def read_chapter_order(book_root: pathlib.Path) -> list[str]:
    """Return the chapter folders in the order of the numbered toctree in ``contents.rst``."""
    lines = (book_root / "contents.rst").read_text().splitlines()
    return [ln.strip().removesuffix("/index") for ln in lines if ln.strip().removesuffix("/index") in SHORT_NAMES]


def count_links(book_root: pathlib.Path, chapters: list[str]) -> collections.Counter:
    """Count ``:ref:`` links from each chapter to every other chapter."""
    texts = {ch: [f.read_text() for f in (book_root / ch).rglob("*.rst")] for ch in chapters}
    owner = {lab.lower(): ch for ch, docs in texts.items() for doc in docs for lab in LABEL_RE.findall(doc)}
    links = collections.Counter()
    for ch, docs in texts.items():
        for doc in docs:
            for target in REF_RE.findall(doc):
                dest = owner.get(target.strip().lower())
                if dest and dest != ch:
                    links[ch, dest] += 1
    return links


def plot_links(chapters: list[str], links: collections.Counter) -> plt.Figure:
    """Draw the arc diagram: back-references above the line, forward references below."""
    pos = {ch: i for i, ch in enumerate(chapters)}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for (src, dest), n in sorted(links.items(), key=lambda kv: kv[1]):  # thick arcs drawn last
        x0, x1 = pos[src], pos[dest]
        back = x1 < x0
        # arc3's control point sits at y = -rad * dx, so rad > 0 sends leftward arrows up, rightward down.
        rad = 0.55
        colour = BACK if back else AHEAD
        ax.add_patch(
            FancyArrowPatch(
                (x0, 0), (x1, 0), connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                mutation_scale=7 + 2 * n**0.5, lw=0.8 * n**0.5, color=colour, alpha=0.75,
                shrinkA=16, shrinkB=16, zorder=2,
            )
        )
        ax.text((x0 + x1) / 2, -rad * (x1 - x0) / 2, str(n),  # a quadratic bezier peaks at half its control ha="center",
                va="center", fontsize=7, color=colour, zorder=4,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none"})
    for ch, x in pos.items():
        ax.scatter(x, 0, s=900, color="white", edgecolor="#333333", lw=1.4, zorder=3)
        ax.text(x, 0, str(x + 1), ha="center", va="center", fontsize=11, weight="bold", zorder=4)
        ax.text(x, -1.3, SHORT_NAMES[ch], ha="center", va="top", fontsize=9)
    ax.text(-0.5, 1.6, "Refers back to an earlier chapter", color=BACK, fontsize=10, weight="bold")
    ax.text(-0.5, -1.0, "Refers ahead to a later chapter", color=AHEAD, fontsize=10, weight="bold")
    ax.set_xlim(-0.6, len(chapters) - 0.4)
    ax.set_ylim(-1.75, 1.75)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def main() -> None:
    """Measure the links in the book and save the figure next to this script."""
    here = pathlib.Path(__file__).absolute()  # absolute, not resolve: keep the book's figures/ symlink
    book_root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else here.parents[2]
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else here.with_suffix(".png")
    chapters = read_chapter_order(book_root)
    links = count_links(book_root, chapters)
    plot_links(chapters, links).savefig(out, dpi=200)
    print(f"{sum(links.values())} cross-chapter links over {len(links)} chapter pairs -> {out}")


if __name__ == "__main__":
    main()
