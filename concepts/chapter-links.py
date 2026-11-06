"""How the chapters of the PID book build on one another, drawn as a building.

Chapter 1 (visualization) is used everywhere, so it is the foundation slab under
the whole figure. The applications chapter draws on the methods before it, so it
is the roof. The method chapters are the floors in between, each sitting above the
chapters it needs.

The ribbons are measured, not drawn by hand: every ``:ref:`` in one chapter that
points at a label in another counts once, in either direction, and a ribbon is
drawn for each pair of chapters with at least ``MIN_LINKS`` of them. Its width
grows with the square root of that count. Re-run the script after a content
change and the ribbons follow the book; only the placement of the cards is fixed.

Two files are written next to this script:

- ``chapter-links.png``: on white, for the PDF.
- ``chapter-links.svg``: transparent, for the HTML book, where it is served from
  ``_static/``. Each card is a link to its chapter, relative to that folder, and
  lightens under the pointer or keyboard focus.

Usage
-----
    uv run --with matplotlib python chapter-links.py [book_root]

``book_root`` defaults to the book checkout that holds this repository as its
``figures/`` link, so from the book's root this is enough:
``uv run --with matplotlib python figures/concepts/chapter-links.py``.
"""

import collections
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

FOUNDATION, ROOF = "data-visualization", "product-development-product-improvement"
# Card per chapter, in reading order (this sets the numbers): title, what is inside, and the
# centre (x, y) of the card. The slabs (foundation and roof) span the full width.
CHAPTERS = {
    FOUNDATION: (
        "Visualizing process data",
        "Time-series, bar, box and scatter plots, and tables: used in every chapter",
        (5.0, 0.55),
    ),
    "univariate-review": ("Univariate data analysis", "Variability and confidence intervals", (4.4, 1.95)),
    "process-monitoring": ("Process monitoring", "Shewhart, CUSUM and EWMA charts", (2.0, 3.35)),
    "least-squares-modelling": ("Least squares modelling", "Regression, residuals, outliers", (8.0, 3.35)),
    "design-analysis-experiments": ("Design of experiments", "Factorials and response surfaces", (8.0, 4.75)),
    "latent-variable-modelling": ("Latent variable modelling", "PCA and PLS", (4.6, 4.75)),
    ROOF: (
        "Applications",
        (
            "Troubleshooting  ·  multivariate and batch monitoring  ·  soft sensors  ·  image analysis  ·  "
            "product development  ·  case studies"
        ),
        (5.0, 6.35),
    ),
}
MIN_LINKS = 5
CARD_W, CARD_H, SLAB_X = 3.0, 0.78, (0.2, 9.8)
LABEL_RE = re.compile(r"^\.\. _([^:]+):\s*$", re.MULTILINE)
REF_RE = re.compile(r":ref:`(?:[^`<]*<)?([^`<>]+)>?`")
# Dark and solid at the foundation, lighter towards the applications at the top.
CMAP = LinearSegmentedColormap.from_list("floors", ["#0b4f5c", "#1f7a99", "#3a6fc4", "#6a4fb8"])
Y_RANGE = (0.55, 6.35)
# Under the pointer or keyboard focus, the clear overlay on a card turns faintly white.
SVG_CSS = "a{cursor:pointer;outline:none} a:hover path,a:focus path{opacity:0.18 !important}"


def count_links(book_root: pathlib.Path) -> collections.Counter:
    """Count ``:ref:`` links between each pair of chapters, ignoring direction."""
    texts = {ch: [f.read_text() for f in (book_root / ch).rglob("*.rst")] for ch in CHAPTERS}
    owner = {lab.lower(): ch for ch, docs in texts.items() for doc in docs for lab in LABEL_RE.findall(doc)}
    links = collections.Counter()
    for ch, docs in texts.items():
        for doc in docs:
            for target in REF_RE.findall(doc):
                dest = owner.get(target.strip().lower())
                if dest and dest != ch:
                    links[frozenset((ch, dest))] += 1
    return links


def colour_at(y: float) -> tuple:
    """Colour of the building at height ``y``."""
    return CMAP((y - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]))


def draw_ribbon(ax: plt.Axes, start: tuple, end: tuple, width: float) -> None:
    """Draw a cubic Bezier ribbon that leaves and arrives vertically, shaded by height."""
    (x0, y0), (x1, y1) = start, end
    bend = (y1 - y0) / 2
    ctrl = np.array([(x0, y0), (x0, y0 + bend), (x1, y1 - bend), (x1, y1)])
    t = np.linspace(0, 1, 80)[:, None]
    weights = np.hstack([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t**2 * (1 - t), t**3])
    centre = weights @ ctrl
    tangent = np.gradient(centre, axis=0)
    normal = np.c_[-tangent[:, 1], tangent[:, 0]] / np.linalg.norm(tangent, axis=1)[:, None]
    outline = np.vstack([centre + normal * width / 2, (centre - normal * width / 2)[::-1]])
    patch = Polygon(outline, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(patch)
    # One vertical gradient, clipped to the ribbon, so its colour always matches the floor it passes.
    image = ax.imshow(
        np.linspace(0, 1, 256)[:, None],
        cmap=CMAP,
        extent=(0, 10, *Y_RANGE),
        origin="lower",
        aspect="auto",
        alpha=0.42,
        zorder=1,
    )
    image.set_clip_path(patch)


def draw_card(ax: plt.Axes, number: int, chapter: str, centre: tuple, slab: bool) -> None:
    """Draw one chapter card: a rounded box with a number badge, the title, a subtitle and a link."""
    title, subtitle, _ = CHAPTERS[chapter]
    x, y = centre
    left, width = (SLAB_X[0], SLAB_X[1] - SLAB_X[0]) if slab else (x - CARD_W / 2, CARD_W)
    box = {"xy": (left, y - CARD_H / 2), "width": width, "height": CARD_H, "boxstyle": "round,pad=0,rounding_size=0.14"}
    shadow = {**box, "xy": (left + 0.04, y - CARD_H / 2 - 0.05)}
    ax.add_patch(FancyBboxPatch(**shadow, fc="black", ec="none", alpha=0.12, zorder=2))
    ax.add_patch(FancyBboxPatch(**box, fc=colour_at(y), ec="none", zorder=3))
    badge_x = left + 0.38
    ax.scatter(badge_x, y, s=560, color="white", zorder=4)
    ax.text(badge_x, y, str(number), ha="center", va="center", fontsize=12, weight="bold", color=colour_at(y), zorder=5)
    size = (12, 9) if slab else (10.5, 8)
    ax.text(
        badge_x + 0.38,
        y + 0.13,
        title,
        ha="left",
        va="center",
        fontsize=size[0],
        weight="bold",
        color="white",
        zorder=5,
    )
    ax.text(
        badge_x + 0.38,
        y - 0.16,
        subtitle,
        ha="left",
        va="center",
        fontsize=size[1],
        color=0.12 * np.array(colour_at(y)) + 0.88,  # 88% white, solid: text alpha renders soft in browsers
        zorder=5,
    )
    # A clear copy of the card on top carries the link, so the whole card is one click target in the SVG.
    # Not alpha=0: matplotlib then writes fill: none, and an unfilled shape catches no clicks.
    ax.add_patch(FancyBboxPatch(**box, fc="white", ec="none", alpha=0.001, zorder=6, url=f"../{chapter}/index"))


def plot_building(links: collections.Counter) -> plt.Figure:
    """Draw the slabs, the floors and the ribbons between them."""
    plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    for pair, n in links.items():
        if n < MIN_LINKS or FOUNDATION in pair:  # the foundation sits under everything: no ribbons needed
            continue
        low, high = sorted(pair, key=lambda ch: CHAPTERS[ch][2][1])
        start, end = CHAPTERS[low][2], CHAPTERS[high][2]
        if high == ROOF:  # rise straight up into the roof slab
            end = (start[0], end[1])
        draw_ribbon(ax, start, end, width=0.045 * n**0.5)
    for number, (chapter, (_, _, centre)) in enumerate(CHAPTERS.items(), start=1):
        draw_card(ax, number, chapter, centre, slab=chapter in (FOUNDATION, ROOF))
    grey = "#8a8f98"  # readable on the light and the dark page alike
    ax.add_patch(FancyArrowPatch((10.25, 1.1), (10.25, 5.8), arrowstyle="-|>", mutation_scale=14, color=grey, lw=1.2))
    ax.text(
        10.45,
        3.45,
        "Each chapter builds on those below it",
        rotation=90,
        ha="center",
        va="center",
        fontsize=9,
        color=grey,
    )
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 6.9)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig


def finish_svg(svg: str) -> str:
    """Open each link in the whole page (not inside the embedding <object>), name it, and add the hover style."""
    titles = {f"../{ch}/index": f"Chapter {i}: {title}" for i, (ch, (title, _, _)) in enumerate(CHAPTERS.items(), 1)}

    def tag_link(match: re.Match) -> str:
        return f'<a xlink:href="{match[1]}" target="_top"><title>{titles[match[1]]}</title>'

    svg = re.sub(r'<a xlink:href="([^"]+)"[^>]*>', tag_link, svg)  # replaces matplotlib's target="_blank"
    return svg.replace("</style>", f"{SVG_CSS}</style>", 1)


def main() -> None:
    """Measure the links in the book and save the PNG and the SVG next to this script."""
    here = pathlib.Path(__file__).absolute()  # absolute, not resolve: keep the book's figures/ symlink
    book_root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else here.parents[2]
    links = count_links(book_root)
    fig = plot_building(links)
    fig.savefig(here.with_suffix(".png"), dpi=200, facecolor="white")
    svg_path = here.with_suffix(".svg")
    fig.savefig(svg_path, transparent=True, metadata={"Date": None})  # no date: re-runs give the same file
    svg_path.write_text(finish_svg(svg_path.read_text()))
    drawn = sum(1 for pair, n in links.items() if n >= MIN_LINKS and FOUNDATION not in pair)
    print(f"{sum(links.values())} cross-chapter links; {drawn} ribbons drawn -> {here.with_suffix('.png|.svg')}")


if __name__ == "__main__":
    main()
