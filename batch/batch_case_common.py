"""Shared style and plot helpers for the three batch case-study figure scripts.

Used by ``batch-case-dupont-figures.py``, ``batch-case-sbr-figures.py`` and
``batch-case-fmc-figures.py`` in this directory, which write the PNGs for the
batch case-study pages of the pid-book Applications chapter. This module writes
no figures itself.

The colours are the book's house colours (the same ones the adaptive
soft-sensor figures use), checked as a categorical set with the data-viz
palette validator: dark blue for the main series, orange for the batch under
discussion, aqua for a second highlighted batch, grey for everything else.
Purple and magenta extend the set to five for the one figure that needs a
colour per quality attribute; the five clear the validator's colour-vision
and normal-vision separation checks (worst pair 9.4 and 16.1 Delta E).
"""

from __future__ import annotations

import pathlib
import typing

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as path_effects
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

if typing.TYPE_CHECKING:
    from collections.abc import Callable

DARK_BLUE = "#1f3d7a"
ORANGE = "#c55a11"
AQUA = "#1baf7a"
PURPLE = "#6f42c1"
MAGENTA = "#b03a78"
GOLD = "#d4a017"  # the third disposition class of the FMC case; distinct from purple, blue, orange and aqua
GREY = "0.55"
PALE_GREY = "0.82"
BAND = "#e9edf4"  # alternate shading of the tag blocks in a contribution vector
PHASE_LINE = "0.55"  # the vertical phase-separator lines drawn over grey overlays


def phase_lines(ax, xs, *, colour: str = PHASE_LINE, zorder: float = 1.5, lw: float = 0.9, ls: str = "--") -> None:
    """Draw a vertical line at each sample in ``xs`` (the ends of the batch phases)."""
    for x in xs:
        ax.axvline(x, color=colour, lw=lw, ls=ls, zorder=zorder)
MARKER, HIGHLIGHT = 28, 46  # scatter areas (points squared) of a plain dot and of a highlighted batch
MARKER_CODED, HIGHLIGHT_CODED = 112, 170  # the same when the markers are shape-coded: twice the side
BUBBLE = 70  # the area given to the median batch when the marker area carries a value, not just identity
FIGSIZE_WIDE = (9.0, 3.6)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.grid": True,
        "axes.axisbelow": True,  # the grid goes behind the data, never over a bar or a marker
        "grid.alpha": 0.3,
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10.5,
        "axes.titleweight": "normal",
        # A legend sits over the grid and often over the data: give it a ground solid enough to read
        # through, and no edge, so it does not read as a panel of its own.
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.framealpha": 0.85,
        "legend.edgecolor": "none",
        "legend.fontsize": 9,
    }
)


AUTO_LABEL = "auto-placed"  # gid on the labels ``save`` positions once the layout is final
LABEL_GAP = 2.5  # points of clear space asked for between a marker's edge and its label
LABEL_REACH = (1.0, 3.6, 6.0)  # multiples of that gap to try; past the first, the label earns a leader line
#   the steps past the first are wide, so a leader line is long enough to read as one
_R2 = 0.7071  # the diagonal sides, as a unit vector
# The eight sides a label can take, each with the alignment that pushes the text away from its point.
_LABEL_SIDES = {
    (1.0, 0.0): ("left", "center"), (-1.0, 0.0): ("right", "center"),
    (0.0, 1.0): ("center", "bottom"), (0.0, -1.0): ("center", "top"),
    (_R2, _R2): ("left", "bottom"), (-_R2, _R2): ("right", "bottom"),
    (_R2, -_R2): ("left", "top"), (-_R2, -_R2): ("right", "top"),
}


def _drawn_discs(ax) -> tuple[np.ndarray, np.ndarray]:
    """Centre (pixels) and radius (pixels) of every marker drawn on the axes.

    A scatter's ``s`` is the marker area in points squared, so the radius is ``sqrt(s) / 2`` points.
    """
    centres, radii = [], []
    for collection in ax.collections:
        if not isinstance(collection, PathCollection):
            continue
        offsets = np.asarray(collection.get_offsets(), dtype=float)
        if offsets.size == 0:
            continue
        sizes = np.asarray(collection.get_sizes(), dtype=float)
        sizes = np.broadcast_to(sizes if sizes.size else np.array([36.0]), (len(offsets),))
        centres.append(ax.transData.transform(offsets))
        radii.append(np.sqrt(sizes) / 2 * ax.figure.dpi / 72.0)
    if not centres:
        return np.empty((0, 2)), np.empty(0)
    return np.vstack(centres), np.concatenate(radii)


def _facing(here: np.ndarray, box: Bbox) -> np.ndarray:
    """The point on a label's box that faces the point it names: where its leader line ends."""
    return np.clip(here, [box.x0, box.y0], [box.x1, box.y1])


def _bite(box: Bbox, centres: np.ndarray, radii: np.ndarray) -> float:
    """How far a text box reaches inside the nearest marker, in pixels; negative means it is clear."""
    if len(centres) == 0:
        return -np.inf
    nearest = np.clip(centres, [box.x0, box.y0], [box.x1, box.y1])
    return float(np.max(radii - np.hypot(*(centres - nearest).T)))


def _segment_bite(centres: np.ndarray, radii: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """How deep a leader line from ``start`` to ``end`` passes inside any marker, in pixels."""
    if len(centres) == 0:
        return -np.inf
    along = end - start
    step = np.clip(((centres - start) @ along) / max(float(along @ along), 1e-9), 0.0, 1.0)
    closest = start + step[:, None] * along
    return float(np.max(radii - np.hypot(*(centres - closest).T)))


def _text_box(text, renderer) -> Bbox:
    """The box the glyphs occupy. ``Annotation.get_window_extent`` unions in its leader line, which
    reaches back to the point being labelled, so it would report every label as covering its own marker."""
    return Text.get_window_extent(text, renderer)


def _overlap(a: Bbox, b: Bbox) -> float:
    """The smaller side of two boxes' intersection, in pixels; negative if they miss each other."""
    return min(min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0))


def _place(text, ax, renderer, centres, radii, taken: list[Bbox], page: Bbox) -> Bbox:
    """Move one label to the clearest spot near its point, and report the box it ended up in.

    Tried in order: the eight sides of its own marker, then the same eight further out, which turns on
    the leader line back to the point. A label with nowhere clear to go keeps a white outline instead,
    so it stays readable over whatever it lands on.
    """
    here = ax.transData.transform(text.xy)
    mine = np.hypot(*(centres - here).T) < 1.5 if len(centres) else np.zeros(0, dtype=bool)
    own = float(radii[mine].max()) if mine.any() else 0.0  # the label's own marker, not a passing neighbour
    others, other_radii = centres[~mine], radii[~mine]
    gap = own * 72.0 / ax.figure.dpi + LABEL_GAP  # the discs are measured in pixels, the offset is in points
    best = None
    for reach in LABEL_REACH:
        for (dx, dy), (ha, va) in _LABEL_SIDES.items():
            text.set_ha(ha)
            text.set_va(va)
            text.xyann = (dx * gap * reach, dy * gap * reach)
            box = _text_box(text, renderer)
            crowding = max([_bite(box, centres, radii), *(_overlap(box, other) for other in taken)])
            if reach > LABEL_REACH[0]:  # a leader line that strikes through a marker points at the wrong one
                crowding = max(crowding, _segment_bite(others, other_radii, here, _facing(here, box)))
            if not (page.contains(box.x0, box.y0) and page.contains(box.x1, box.y1)):
                crowding += 1e4  # running off the page is never the best side
            if best is None or crowding < best[0]:
                best = (crowding, ha, va, text.xyann, reach, box)
        if best[0] <= 0:
            break  # the closest ring that clears everything is the one to use
    crowding, ha, va, offset, reach, box = best
    text.set_ha(ha)
    text.set_va(va)
    text.xyann = offset
    leader = getattr(text, "leader", None)
    if leader is not None:  # far enough out that the reader has to be told which point it names
        leader.set_visible(reach > LABEL_REACH[0])
        if leader.get_visible():
            facing = _facing(here, box)
            along = (facing - here) / max(float(np.hypot(*(facing - here))), 1e-9)
            ends = ax.transData.inverted().transform([here + along * (own + 1.0), facing - along * 2.0])
            leader.set_data(ends[:, 0], ends[:, 1])
    # A halo only where it is earned: nowhere is clear, so the text has to survive what it lands on.
    text.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="white")] if crowding > 0 else [])
    return _text_box(text, renderer)


def place_labels(fig: Figure, rounds: int = 3) -> None:
    """Put every auto-placed label on the clearest side of its point.

    A fixed offset cannot do this: it is smaller than a highlighted marker's own radius, so the text
    starts inside its own disc, and ``tight_layout`` moves the axes after the label is written. So the
    sides are compared here, on the finished layout, against the markers, the legend, the hand-placed
    text and the other labels. Placing one label moves the obstacles the next one sees, so the pass is
    repeated: a few rounds are enough to settle, and the order the labels were added stops mattering.
    """
    for _ in range(rounds):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        page = Bbox.from_extents(0, 0, *fig.canvas.get_width_height())
        for ax in fig.axes:
            auto = [text for text in ax.texts if text.get_gid() == AUTO_LABEL]
            if not auto:
                continue
            centres, radii = _drawn_discs(ax)
            # Everything else on the axes is an obstacle: the legend, and any text placed by hand
            # (a limit annotation, an arrow's caption) that an auto label must not land on.
            fixed = [t for t in ax.texts if t.get_gid() != AUTO_LABEL and t.get_text().strip()]
            boxes = {id(text): _text_box(text, renderer) for text in auto}
            for text in auto:
                obstacles = [legend.get_window_extent(renderer) for legend in ax.findobj(Legend)]
                obstacles += [_text_box(other, renderer) for other in fixed]
                obstacles += [box for key, box in boxes.items() if key != id(text)]
                boxes[id(text)] = _place(text, ax, renderer, centres, radii, obstacles, page)


def save(fig: Figure, out_dir: pathlib.Path, name: str) -> None:
    """Write ``name.png`` into ``out_dir`` and close the figure."""
    place_labels(fig)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def overlay_panels(
    batches: dict,
    tags: list[str],
    highlight: dict[int, str],
    *,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    xlabel: str = "Sample [aligned time]",
    vlines: tuple[float, ...] = (),
) -> Figure:
    """One panel per tag: every batch in grey, the highlighted batches in colour, ``vlines`` at the phase ends."""
    nrows = int(np.ceil(len(tags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (4.5 * ncols, 2.6 * nrows), squeeze=False)
    for ax, tag in zip(axes.ravel(), tags, strict=False):
        phase_lines(ax, vlines)
        for batch_id, batch in batches.items():
            if batch_id not in highlight:
                ax.plot(batch[tag].to_numpy(), color=PALE_GREY, lw=0.7, zorder=1)
        for batch_id, colour in highlight.items():
            ax.plot(batches[batch_id][tag].to_numpy(), color=colour, lw=1.8, label=f"batch {batch_id}", zorder=3)
        ax.set_title(tag)
        ax.set_xlabel(xlabel)
    for ax in axes.ravel()[len(tags) :]:
        ax.set_visible(False)
    axes.ravel()[0].legend(loc="best")
    fig.tight_layout()
    return fig


def explained_per_component(model) -> np.ndarray:
    """Fraction of the variance of X explained by each component, for the axis labels of a score plot.

    PCA-type models report it as ``r2_per_component_``. PLS-type models report the R2 of Y
    there and the cumulative R2 of every X column in ``r2_per_variable_``, whose column mean
    is the cumulative R2 of X.
    """
    if type(model).__name__ in ("PLS", "BatchPLS"):
        return np.diff([0.0, *np.asarray(model.r2_per_variable_.mean(axis=0), dtype=float)])
    return np.asarray(model.r2_per_component_, dtype=float)


def compact_legend(ax, loc: str, fontsize: float = 8) -> None:
    """A legend in two columns when it has four or more entries (a 2 x 2 grid), so it is wider and less tall."""
    handles, _labels = ax.get_legend_handles_labels()
    ax.legend(loc=loc, ncol=2 if len(handles) >= 4 else 1, fontsize=fontsize, columnspacing=1.0, handletextpad=0.5)


def group_scatter(
    ax,
    x: pd.Series,
    y: pd.Series,
    highlight: dict[int, str],
    *,
    groups: pd.Series | None = None,
    group_styles: dict[str, tuple[str, str]] | None = None,
    highlight_marker: str = "o",
    size: float | None = None,
    highlight_size: float | None = None,
    areas: pd.Series | None = None,
) -> None:
    """Scatter the batches in ``x`` and ``y``, colour- and shape-coded by ``groups`` when given, with ``highlight`` on top.

    Without ``groups`` every batch is a dark-blue circle. With ``groups`` (a Series over the batch ids) and
    ``group_styles`` (group -> (colour, marker)) each group gets its own colour and marker and one legend
    entry, "classed <group>"; a highlighted batch is drawn larger in its highlight colour but keeps its
    group's marker, so its class stays readable. Shape-coded markers are drawn at twice the side of plain
    dots (``MARKER_CODED`` against ``MARKER``), because a triangle and a square need the size to be told apart.
    ``highlight_marker`` gives the highlighted batches a shape of their own when there are no ``groups``, so
    that a set carried across several figures is recognised by its shape as well as by its colour.
    ``areas`` (a Series over the batches, in points squared) gives each batch its own marker area, so that the
    area can carry a quantity; a highlighted batch then keeps its area and is marked by a heavier edge instead
    of by a larger one, because two meanings on one channel cannot both be read.
    """
    styles = group_styles or {}
    area = {"s": 0.78}  # a square fills its bounding box; scale it to the visible area of a circle of the same s
    if size is None:
        size = MARKER if groups is None else MARKER_CODED
    if highlight_size is None:
        highlight_size = HIGHLIGHT if groups is None else HIGHLIGHT_CODED
    def area_of(members: list, marker: str, when_plain: float) -> "float | pd.Series":
        """The scatter area for these batches: their own when ``areas`` is given, else the shared size."""
        scale = area.get(marker, 1.0)
        return (areas.loc[members] * scale) if areas is not None else (when_plain * scale)

    if groups is None:
        others = [b for b in x.index if b not in highlight]
        ax.scatter(x.loc[others], y.loc[others], s=area_of(others, "o", size), color=DARK_BLUE,
                   edgecolor="white", linewidth=1, zorder=3)
    else:
        for label, (colour, marker) in styles.items():
            members = [b for b in x.index if groups.get(b) == label and b not in highlight]
            ax.scatter(x.loc[members], y.loc[members], s=area_of(members, marker, size), color=colour, marker=marker,
                       edgecolor="white", linewidth=1, zorder=3, label=f"classed {label}")
    for batch_id, colour in highlight.items():
        marker = styles.get(groups.get(batch_id), (None, "o"))[1] if groups is not None else highlight_marker
        ax.scatter(x.loc[batch_id], y.loc[batch_id], s=area_of([batch_id], marker, highlight_size), color=colour,
                   marker=marker, edgecolor="white" if areas is None else "0.25",
                   linewidth=1 if areas is None else 1.4, zorder=4)


def annotate_batches(ax, x: pd.Series, y: pd.Series, batch_ids, *, fontsize: float = 8.5) -> None:
    """Name a few points on a scatter panel; ``save`` then gives each label the clearest side of its marker."""
    for batch_id in batch_ids:
        point = (float(x.loc[batch_id]), float(y.loc[batch_id]))
        text = ax.annotate(str(batch_id), point, xytext=(4, 4), textcoords="offset points",
                           fontsize=fontsize, zorder=6, gid=AUTO_LABEL)
        # The leader is a line of its own rather than an annotation's arrow, so `save` can set both of
        # its ends: it turns the line on only for a label it has to move far out, and prefers a
        # position whose leader reaches the marker without striking through another one.
        # `add_artist`, not `add_line`, leaves the axes' data limits alone.
        leader = Line2D([], [], color=GREY, lw=0.8, zorder=5.5, visible=False, transform=ax.transData)
        ax.add_artist(leader)
        text.leader = leader


def score_plot(
    model,
    *,
    pc_horiz: int = 1,
    pc_vert: int = 2,
    highlight: dict[int, str] | None = None,
    labels: list[int] | None = None,
    label_leader: dict[int, tuple[float, float]] | None = None,
    sizes: pd.Series | None = None,
    size_name: str = "",
    size_reference: tuple[float, ...] = (),
    size_of_reference: "Callable[[float], float] | None" = None,
    conf_level: float = 0.95,
    title: str = "",
    legend_loc: str = "upper right",
    groups: pd.Series | None = None,
    group_styles: dict[str, tuple[str, str]] | None = None,
    highlight_marker: str = "o",
    ax=None,
) -> Figure:
    """Scores on two components with the Hotelling's T2 ellipse; selected batches coloured and labelled.

    ``labels`` names the batches to write beside their markers; the side each one takes is chosen by
    ``save`` from the finished layout. ``label_leader`` maps a batch to an offset in points and draws a
    leader line to it, for a batch that sits so deep inside a cloud that no side of it is clear. ``groups`` (a Series over the batches) with ``group_styles``
    (group -> (colour, marker)) colour- and shape-codes the batches by a known classification; a highlighted
    batch keeps its group's marker in the highlight colour.

    ``sizes`` (a Series over the batches, for instance their SPE) makes the marker **area** proportional to
    that quantity, with the median batch drawn at ``BUBBLE``; ``size_name`` and ``size_reference`` then add
    one legend circle per reference value, without which an area cannot be read off the plot. When ``sizes``
    is a transform of the quantity the reader thinks in, such as the square of the SPE, ``size_of_reference``
    maps a reference value back onto the same scale, so that the circle labelled "SPE 20" is the size a batch
    with an SPE of 20 is drawn.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
    else:
        fig = ax.figure
    scores = model.scores_
    x, y = scores.iloc[:, pc_horiz - 1], scores.iloc[:, pc_vert - 1]
    ex, ey = model.ellipse_coordinates(score_horiz=pc_horiz, score_vert=pc_vert, conf_level=conf_level)
    ax.plot(ex, ey, color=GREY, lw=1, ls="--", label=f"{conf_level:.0%} confidence ellipse")
    highlight = highlight or {}
    # Area, not radius, carries the quantity: doubling the area means doubling the value it stands for.
    areas = sizes.reindex(scores.index) * (BUBBLE / sizes.median()) if sizes is not None else None
    group_scatter(ax, x, y, highlight, groups=groups, group_styles=group_styles,
                  highlight_marker=highlight_marker, areas=areas)
    label_leader = label_leader or {}
    for batch_id in labels or []:
        point = (x.loc[batch_id], y.loc[batch_id])
        if batch_id in label_leader:  # a leader line is a deliberate mark, so its offset is given, not chosen
            dx, dy = label_leader[batch_id]
            ax.annotate(str(batch_id), point, xytext=(dx, dy), textcoords="offset points", fontsize=8.5,
                        ha="right" if dx < 0 else "left", va="top" if dy < 0 else "bottom", zorder=6,
                        arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.8, "shrinkA": 2, "shrinkB": 3})
            continue
        annotate_batches(ax, x, y, [batch_id])
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(0, color=GREY, lw=0.8)
    r2 = explained_per_component(model)
    note = "$R^2_X$ " if type(model).__name__ in ("PLS", "BatchPLS") else ""  # a PLS also has an R2 of Y: say which
    ax.set_xlabel(f"$t_{pc_horiz}$ [{note}{r2[pc_horiz - 1]:.1%}]")
    ax.set_ylabel(f"$t_{pc_vert}$ [{note}{r2[pc_vert - 1]:.1%}]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    if size_reference:
        handles, texts = ax.get_legend_handles_labels()
        for value in size_reference:
            on_scale = size_of_reference(value) if size_of_reference is not None else value
            handles.append(Line2D([], [], ls="none", marker="o", color=GREY, markeredgecolor="white",
                                  markersize=(on_scale * BUBBLE / sizes.median()) ** 0.5))
            texts.append(f"{size_name} {value:g}" if size_name else f"{value:g}")
        ax.legend(handles, texts, loc=legend_loc, labelspacing=0.9)
    elif groups is not None:
        compact_legend(ax, legend_loc)
    else:
        ax.legend(loc=legend_loc)
    return fig


def influence_plot(
    model,
    *,
    highlight: dict[int, str] | None = None,
    labels: list[int] | None = None,
    sizes: pd.Series | None = None,
    conf_level: float = 0.95,
    title: str = "",
    groups: pd.Series | None = None,
    group_styles: dict[str, tuple[str, str]] | None = None,
    legend_loc: str = "upper left",
    ax=None,
) -> Figure:
    """Hotelling's T2 against SPE for every batch, with both limits drawn.

    Each batch is one dot: how far it sits *along* the model's components
    (Hotelling's T2, horizontal) against how far it sits *away* from them (SPE,
    vertical). The two limits split the plot into quadrants, which separates a
    batch that is extreme in a direction the model knows from one the model
    cannot describe. Plotting either statistic against the batch number instead
    would put the order the batches happen to appear in on an axis, which
    carries no information about the batch.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
    else:
        fig = ax.figure
    t2 = model.hotellings_t2_.iloc[:, -1]
    spe = model.spe_.iloc[:, -1]
    t2_limit = float(model.hotellings_t2_limit(conf_level=conf_level))
    spe_limit = float(model.spe_limit(conf_level=conf_level))
    highlight = highlight or {}

    x_max = max(float(t2.max()), t2_limit) * 1.14
    y_min, y_max = float(spe.min()) * 0.94, max(float(spe.max()), spe_limit) * 1.06
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)

    ax.axvline(t2_limit, color=GREY, ls="--", lw=1)
    ax.axhline(spe_limit, color=GREY, ls="--", lw=1)
    ax.text(t2_limit, 0.99, f" {conf_level:.0%} limit", transform=ax.get_xaxis_transform(), va="top", ha="left", fontsize=8.5, color=GREY)
    ax.text(0.995, spe_limit, f"{conf_level:.0%} limit", transform=ax.get_yaxis_transform(), va="bottom", ha="right", fontsize=8.5, color=GREY)

    # ``sizes`` carries the same quantity, on the same scale, as the score plot that precedes this one,
    # so a batch keeps its marker size across the pair and can be followed from one to the other.
    areas = sizes.reindex(t2.index) * (BUBBLE / sizes.median()) if sizes is not None else None
    group_scatter(ax, t2, spe, highlight, groups=groups, group_styles=group_styles, areas=areas,
                  size=None if groups is not None else 30, highlight_size=None if groups is not None else 52)
    if groups is not None:
        compact_legend(ax, legend_loc)
    annotate_batches(ax, t2, spe, labels or [])

    ax.set_xlabel("Hotelling's $T^2$")
    ax.set_ylabel("SPE")
    ax.set_title(title)
    return fig


def _tag_blocks(row: pd.Series) -> list[tuple[str, int, int]]:
    """Return (tag, start, stop) for each contiguous tag block of an unfolded row (positions are 0-based)."""
    tags = row.index.get_level_values("tag").to_numpy()
    blocks: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(tags) + 1):
        if i == len(tags) or tags[i] != tags[start]:
            blocks.append((str(tags[start]), start, i))
            start = i
    return blocks


def shade_alternate_tags(ax, n_tags: int) -> None:
    """Shade alternate tag positions on a one-bar-per-tag axis.

    The bands line up with those of :func:`contribution_vector`, which shades
    the same alternate blocks of the unfolded axis, so a summed-per-tag panel
    below one of those reads against the same striping.
    """
    for position in range(1, n_tags, 2):
        ax.axvspan(position - 0.5, position + 0.5, color=BAND, zorder=0, lw=0)
    ax.set_xlim(-0.5, n_tags - 0.5)
    ax.grid(False, axis="x")


def contribution_vector(row: pd.Series, *, ax, ylabel: str, colour: str = DARK_BLUE) -> None:
    """Draw one batch's contribution vector over the unfolded (tag, time) axis with shaded tag blocks."""
    values = row.to_numpy(dtype=float)
    positions = np.arange(len(values))
    for k, (tag, start, stop) in enumerate(_tag_blocks(row)):
        if k % 2 == 1:
            ax.axvspan(start - 0.5, stop - 0.5, color=BAND, zorder=0, lw=0)
        ax.text((start + stop - 1) / 2, 1.02, tag, ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
    ax.bar(positions, values, width=1.0, color=colour, lw=0, zorder=2)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlim(-0.5, len(values) - 0.5)
    ax.set_xticks([])
    ax.set_xlabel("Unfolded (tag, time) cells, one block per tag")
    ax.set_ylabel(ylabel)
    ax.grid(False, axis="x")


def label_bars(ax, values: np.ndarray, *, fmt: str = "{:.1f}", floor: float = 0.05, colour: str = DARK_BLUE) -> None:
    """Write each bar's value at its outer end, above a positive bar and below a negative one.

    A value smaller than ``floor`` is written as a bound rather than as the
    rounded ``0.0``, which next to a bar of no visible height would read as an
    exact zero. The axis limits are widened on whichever side carries bars, so
    a label never lands on the frame.
    """
    low, high = min(0.0, float(values.min())), max(0.0, float(values.max()))
    span = (high - low) or 1.0
    ax.set_ylim(low - (0.15 * span if low < 0 else 0.0), high + (0.15 * span if high > 0 else 0.0))
    for position, value in enumerate(values):
        above = value >= 0
        if 0 < abs(value) < floor:
            text = f"<{fmt.format(floor * 2)}" if above else f">-{fmt.format(floor * 2)}"
        else:
            text = fmt.format(value)
        ax.text(
            position,
            value + (0.02 * span if above else -0.02 * span),
            text,
            ha="center",
            va="bottom" if above else "top",
            fontsize=8,
            color=colour,
            zorder=3,
        )


def contribution_triptych(
    row: pd.Series,
    *,
    what: str,
    title: str,
    vlines: tuple[float, ...] = (),
    vline_colour: str = ORANGE,
    phase_names: tuple[str, ...] = (),
) -> Figure:
    """The full contribution vector, its sum per tag, and its sum per time sample, in three panels.

    ``vlines`` marks the phase ends on the per-sample panel, drawn in ``vline_colour`` on top of the bars.
    ``phase_names`` labels the regions those lines divide, so the reader does not have to count them.
    """
    by_tag = row.groupby(level="tag", sort=False).sum()
    by_time = row.groupby(level="sequence").sum()
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), gridspec_kw={"height_ratios": [1.3, 1, 1]})
    contribution_vector(row, ax=axes[0], ylabel=what)
    axes[0].set_title(title, pad=18)
    axes[1].bar(range(len(by_tag)), by_tag.to_numpy(), color=DARK_BLUE, width=0.6, zorder=2)
    axes[1].set_xticks(range(len(by_tag)), [str(tag) for tag in by_tag.index], rotation=20, ha="right")
    axes[1].axhline(0, color=GREY, lw=0.8)
    shade_alternate_tags(axes[1], len(by_tag))
    label_bars(axes[1], by_tag.to_numpy(dtype=float))
    axes[1].set_ylabel("Summed per tag")  # the quantity is named by the title and the top panel
    axes[2].bar(by_time.index.to_numpy(), by_time.to_numpy(), width=1.0, color=DARK_BLUE, lw=0, zorder=2)
    axes[2].axhline(0, color=GREY, lw=0.8)
    axes[2].set_xlabel("Sample [aligned time]")
    axes[2].set_ylabel("Summed per sample")
    fig.tight_layout(h_pad=1.6)
    phase_lines(axes[2], vlines, colour=vline_colour, zorder=4, lw=1.2, ls="-")
    if phase_names:
        edges = [float(by_time.index.min()), *vlines, float(by_time.index.max())]
        for name, left, right in zip(phase_names, edges[:-1], edges[1:], strict=True):
            inside = by_time[(by_time.index >= left) & (by_time.index <= right)]
            quarter = max(1, len(inside) // 4)
            # A centred name lands on the bars; put it at whichever end of its phase leaves more headroom.
            at_left = inside.iloc[:quarter].max() <= inside.iloc[-quarter:].max()
            inset = 0.02 * (right - left)
            axes[2].text(left + inset if at_left else right - inset, 0.96, name,
                         transform=axes[2].get_xaxis_transform(), ha="left" if at_left else "right",
                         va="top", fontsize=8.5, color=vline_colour)
    return fig


def tag_panels(
    grid: pd.DataFrame,
    *,
    ylabel: str,
    ncols: int = 5,
    colour: str = DARK_BLUE,
    second: pd.DataFrame | None = None,
    second_label: str = "",
    first_label: str = "",
    secondary: pd.DataFrame | None = None,
    secondary_label: str = "",
    vlines: tuple[float, ...] = (),
) -> Figure:
    """Small multiples: one panel per tag (row of ``grid``), the values over time; an optional second grid overlaid.

    ``secondary`` is a grid on the same rows drawn in orange against a second, right-hand axis from 0 to 1 (an
    R2 per cell, for instance). ``vlines`` draws a faint vertical line at each of those samples in every panel,
    to mark the phases of the batch.
    """
    tags = list(grid.index)
    nrows = int(np.ceil(len(tags) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.1 * nrows), sharex=True, sharey=True, squeeze=False)
    for k, (ax, tag) in enumerate(zip(axes.ravel(), tags, strict=False)):
        for x in vlines:
            ax.axvline(x, color=PALE_GREY, lw=0.9, zorder=0)
        ax.plot(grid.columns.to_numpy(), grid.loc[tag].to_numpy(), color=colour, lw=1.4, label=first_label, zorder=3)
        if second is not None:
            ax.plot(second.columns.to_numpy(), second.loc[tag].to_numpy(), color=ORANGE, lw=1.4, label=second_label)
        ax.axhline(0, color=GREY, lw=0.7)
        ax.set_title(str(tag))
        if secondary is not None:
            twin = ax.twinx()
            twin.plot(secondary.columns.to_numpy(), secondary.loc[tag].to_numpy(), color=ORANGE, lw=1.1, alpha=0.55, zorder=2)
            twin.set_ylim(0, 1)
            twin.grid(False)
            twin.spines["top"].set_visible(False)
            twin.tick_params(axis="y", colors=ORANGE, labelsize=8)
            if k % ncols == ncols - 1 or k == len(tags) - 1:
                twin.set_ylabel(secondary_label, color=ORANGE, fontsize=9)
            else:
                # Clearing only the labels leaves the tick marks behind, orange dashes with
                # no scale beside them. Read against the left axis, which runs negative, they
                # make a quantity that cannot be negative look as though it is. Drop the ticks.
                twin.set_yticks([])
    for ax in axes.ravel()[len(tags) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Sample")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    if second is not None:
        axes.ravel()[0].legend(loc="best")
    fig.tight_layout()
    return fig


def parity_plot(
    observed: pd.Series,
    predicted: pd.Series,
    *,
    highlight: dict[int, str],
    ax,
    title: str,
    groups: pd.Series | None = None,
    group_styles: dict[str, tuple[str, str]] | None = None,
    label_offsets: dict[int, tuple[float, float]] | None = None,
    errors: dict[str, float] | None = None,
    sd: float | None = None,
    band_from: str = "",
    band_multiple: float = 2.0,
) -> None:
    """Observed against fitted values with the y = x line; selected batches coloured and labelled.

    ``label_offsets`` gives a batch its own label offset in points, with the alignment following the
    signs, for a marker whose neighbours crowd both default positions.
    Any other highlighted batch is labelled on whichever side ``save`` finds clearest. ``errors``
    (name -> value, for
    instance ``{"RMSEE": 1.87, "RMSEP": 2.42}``) is listed in the legend, in the units of the axes,
    with each value also given in units of ``sd`` when that is passed: the scatter about the
    ``y = x`` line is what the reader is judging, so the number belongs on the same plot.
    ``band_from`` names one of those errors to shade as a band of ``band_multiple`` times it either
    side of the line, which turns the number into the distance the reader is looking at.
    """
    group_scatter(ax, observed, predicted, highlight, groups=groups, group_styles=group_styles)
    for batch_id in highlight:
        if batch_id in (label_offsets or {}):
            dx, dy = label_offsets[batch_id]
            ax.annotate(str(batch_id), (observed.loc[batch_id], predicted.loc[batch_id]), xytext=(dx, dy),
                        textcoords="offset points", ha="right" if dx < 0 else "left",
                        va="top" if dy < 0 else ("center" if dy == 0 else "bottom"), fontsize=8.5)
            continue
        annotate_batches(ax, observed, predicted, [batch_id])
    lo, hi = float(min(observed.min(), predicted.min())), float(max(observed.max(), predicted.max()))
    if band_from:
        # Below the markers and above the grid: the band is context for the scatter, not a mark of its own.
        half = band_multiple * (errors or {})[band_from]
        ax.fill_between([lo, hi], [lo - half, hi - half], [lo + half, hi + half], color=BAND, zorder=1, lw=0,
                        label=f"$\\pm${band_multiple:g} {band_from}")
        ax.set_ylim(lo - 1.15 * half, hi + 1.15 * half)   # the band sets the view, not the outermost batch
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=1, ls="--", label="y = x", zorder=2)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Fitted")
    ax.set_title(title)
    handles, texts = ax.get_legend_handles_labels()
    keep = [i for i, text in enumerate(texts) if not text.startswith("classed ")]  # named once, on the score plot
    handles, texts = [handles[i] for i in keep], [texts[i] for i in keep]
    for name, value in (errors or {}).items():
        handles.append(Line2D([], [], ls="none", marker="none"))          # a value, with no mark of its own
        texts.append(f"{name} {value:.3g}" + (f" ({value / sd:.2f} sd)" if sd else ""))
    ax.legend(handles, texts, loc="upper left", handlelength=1.4, handletextpad=0.6)


def online_chart(
    ax,
    result,
    statistic: str,
    *,
    colour: str,
    mean_trace: np.ndarray,
    conf_level: float,
    fault_at: int | None = None,
    fault_label: str = "fault begins",
    fault_label_top: float = 0.97,
    legend_loc: str = "upper left",
) -> None:
    """One batch's on-line Hotelling's T2 or SPE against the per-sample control limit.

    ``result`` is what ``BatchMonitor.monitor`` returns for the batch. The
    batch's statistic is drawn in ``colour`` over the mean of the reference
    batches at each sample and the dashed limit, with a marker on every
    sample above the limit. The vertical axis is scaled to the batch's own
    trace and the typical limit, with headroom for the legend, so the very
    wide limits of the first few samples (where the reference batches have
    hardly been observed) run off the top of the panel rather than squash
    the rest. ``fault_at`` draws a dotted vertical line at that sample with
    ``fault_label`` written along it from ``fault_label_top`` (axes fraction) downward, so the label
    can be placed below a legend.
    """
    if statistic == "t2":
        trace, limit, alarm, ylabel = result.hotellings_t2, result.t2_limit, result.t2_alarm, "Hotelling's $T^2$"
    elif statistic == "spe":
        trace, limit, alarm, ylabel = result.spe, result.spe_limit, result.spe_alarm, "SPE of the newest sample"
    else:
        raise ValueError(f"statistic must be 't2' or 'spe'; got {statistic!r}.")
    time = np.asarray(result.time)
    trace, limit, alarm = np.asarray(trace, dtype=float), np.asarray(limit, dtype=float), np.asarray(alarm, dtype=bool)

    ax.plot(time, np.asarray(mean_trace, dtype=float)[: len(time)], color=PALE_GREY, lw=1.6, zorder=1, label="reference-batch mean")
    ax.plot(time, limit, color=GREY, lw=1, ls="--", zorder=2, label=f"{conf_level:.0%} limit")
    ax.plot(time, trace, color=colour, lw=1.3, zorder=3)
    if alarm.any():
        ax.plot(time[alarm], trace[alarm], ls="none", marker="o", ms=3.4, color=colour, mec="white", mew=0.4, zorder=4, label="above the limit")
    if fault_at is not None:
        ax.axvline(fault_at, color=GREY, lw=1, ls=":", zorder=2)
        ax.text(fault_at, fault_label_top, fault_label, transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right",
                fontsize=8.5, color=GREY)
    ax.set_ylim(0, 1.35 * max(float(trace.max()), float(np.median(limit))))
    ax.set_xlim(0, float(time[-1]) + 1)
    ax.set_xlabel("Samples observed")
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)
