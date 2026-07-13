"""Promotional hero graphic for the mixed-level profile case study.

A single frame that tells the story: one target colour-development curve (the
incumbent, A, marked with a star), five candidate compounds whose curves track
it early and fan out at the tail because each compound's shape is fixed, four
process "knobs" (two easy, two hard to change), and a large arrow showing the
model run backwards from the goal to the settings.

The curves use the same rise-to-plateau-plus-late-drift ground truth as the
case study; each candidate is drawn at the amplitude that best matches A on the
developed part of the curve, so the fan-out is the shape difference that
amplitude cannot remove. Self-contained (numpy + matplotlib only).
Regenerates ``colour-hero.png``.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Wedge

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Liberation Serif", "DejaVu Serif"]


# ---- curve ground truth (same shape as the case study) -------------------
def ref_shape(t):
    r = 1.0 - np.exp(-t / 2.0)
    return r / (1.0 - np.exp(-9 / 2.0))


def tail_shape(t):
    tl = np.clip((t - 4) / 5.0, 0, None)
    return tl / ((9 - 4) / 5.0)


drift = {"A": 0.00, "B": 0.05, "C": 0.20, "D": 0.30, "E": 0.35, "F": -0.10}
colour = {"A": "#14181d", "B": "#1f5fa8", "C": "#2e8b57",
          "D": "#8e44ad", "E": "#d68910", "F": "#17a2b8"}

ti = np.arange(10)                      # 10 measured points
tf = np.linspace(0, 9, 240)            # smooth line
goal_i = ref_shape(ti)                 # A at amplitude 1 is the target
amp = {}
for c, d in drift.items():
    s = np.clip(ref_shape(ti) + d * tail_shape(ti), 0, None)
    amp[c] = float(s[1:] @ goal_i[1:] / (s[1:] @ s[1:]))   # best match on t1..9


def curve(c, t):
    return amp[c] * np.clip(ref_shape(t) + drift[c] * tail_shape(t), 0, None)


amax = max(curve(c, tf).max() for c in drift)

# ---- canvas (cropped tightly to the content, banner ratio) ---------------
W, H = 1600, 840
YLO, YHI = 92, 756
fig = plt.figure(figsize=(W / 100, (YHI - YLO) / 100), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(YLO, YHI); ax.axis("off")

grad = np.linspace(0, 1, 256).reshape(-1, 1)
ax.imshow(grad, extent=[0, W, YLO, YHI], aspect="auto", origin="lower",
          cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
              "bg", ["#eef2f8", "#fbfcfe"]), zorder=0)

# ---- plot region ----------------------------------------------------------
PX0, PX1, PYB, PYT = 120, 980, 210, 650
def X(t): return PX0 + t / 9.0 * (PX1 - PX0)
def Y(a): return PYB + a / (amax * 1.08) * (PYT - PYB)

ax.add_line(Line2D([PX0, PX1], [PYB, PYB], color="#c3cad4", lw=1.8, zorder=2))
ax.add_line(Line2D([PX0, PX0], [PYB, PYT + 24], color="#c3cad4", lw=1.8, zorder=2))
ax.text(PX0 - 6, PYT + 54, "colour depth", color="#5b6675", fontsize=18,
        ha="left", va="bottom", style="italic")
ax.text(PX1, PYB - 30, "time →", color="#5b6675", fontsize=18, ha="right", va="top",
        style="italic")

# candidate curves (analogs first, reference on top)
for c in ["E", "D", "C", "F", "B"]:
    ax.plot(X(tf), Y(curve(c, tf)), color=colour[c], lw=5.5, solid_capstyle="round",
            zorder=4, alpha=0.95)
    ax.plot(X(ti), Y(curve(c, ti)), "o", color=colour[c], ms=6, zorder=5)
ax.plot(X(tf), Y(curve("A", tf)), color=colour["A"], lw=8.5, solid_capstyle="round", zorder=6)
ax.plot(X(ti), Y(curve("A", ti)), "o", color=colour["A"], ms=7, zorder=7)

# goal star at the reference endpoint
gx, gy = X(9), Y(curve("A", ti)[-1])
ax.scatter([gx], [gy], s=2600, marker="*", color="#f4c542", edgecolor="#14181d",
           linewidth=1.8, zorder=8)
ax.annotate("the goal:\nmatch A's curve", xy=(gx, gy - 12), xytext=(X(7.05), Y(0.30 * amax)),
            color="#14181d", fontsize=18, fontweight="bold", ha="center", va="top",
            arrowprops=dict(arrowstyle="->", color="#14181d", lw=1.7,
                            connectionstyle="arc3,rad=0.15"), zorder=9)

ax.annotate("tails fan out:\nshape is baked in", xy=(X(8.3), Y(curve("E", np.array([8.3]))[0])),
            xytext=(X(5.2), Y(curve("E", np.array([9]))[0]) + 70), color="#5b6675",
            fontsize=18, ha="center", va="bottom",
            arrowprops=dict(arrowstyle="->", color="#9aa4b2", lw=1.8,
                            connectionstyle="arc3,rad=-0.2"), zorder=9)

# compact legend, upper-left, with clear space around the swatches
lx, ly = PX0 + 60, PYT - 6
ax.add_line(Line2D([lx, lx + 60], [ly, ly], color=colour["A"], lw=9,
                   solid_capstyle="round", zorder=6))
ax.text(lx + 84, ly, "target: the incumbent, A", color="#2b3340", fontsize=19,
        va="center", ha="left", zorder=6)
for i, c in enumerate(["B", "C", "D", "E", "F"]):
    ax.add_line(Line2D([lx + i * 15, lx + 10 + i * 15], [ly - 50, ly - 50],
                       color=colour[c], lw=8, solid_capstyle="round", zorder=6))
ax.text(lx + 84, ly - 50, "five candidate compounds", color="#2b3340", fontsize=19,
        va="center", ha="left", zorder=6)

# ---- knobs (four process factors) ----------------------------------------
KR = 66
knob_specs = [
    (1215, 500, "concentration", "easy",  135),
    (1445, 500, "co-solvent",    "hard",  -35),
    (1215, 268, "pH",            "easy",   60),
    (1445, 268, "temperature",   "hard",  200),
]
for kx, ky, name, tag, ang in knob_specs:
    ax.add_patch(Circle((kx, ky), KR + 9, facecolor="#ffffff", edgecolor="#dbe1ea",
                        lw=2.2, zorder=4))
    ax.add_patch(Wedge((kx, ky), KR, 215, 215 + 290, width=8, facecolor="#e6ebf2",
                       edgecolor="none", zorder=5))
    for a in np.linspace(215, 215 + 290, 11):
        r = np.deg2rad(a)
        ax.add_line(Line2D([kx + (KR - 13) * np.cos(r), kx + (KR - 3) * np.cos(r)],
                           [ky + (KR - 13) * np.sin(r), ky + (KR - 3) * np.sin(r)],
                           color="#c3cad4", lw=1.6, zorder=5))
    acc = "#c0392b" if tag == "hard" else "#1f5fa8"
    r = np.deg2rad(ang)
    ax.add_line(Line2D([kx, kx + (KR - 17) * np.cos(r)], [ky, ky + (KR - 17) * np.sin(r)],
                       color=acc, lw=6, solid_capstyle="round", zorder=6))
    ax.add_patch(Circle((kx, ky), 10, facecolor=acc, edgecolor="white", lw=1.6, zorder=7))
    ax.text(kx, ky - KR - 28, name, color="#2b3340", fontsize=20, fontweight="bold",
            ha="center", va="top", zorder=6)
    ax.text(kx, ky - KR - 54, tag + " to change", color="#8a94a3", fontsize=15,
            ha="center", va="top", zorder=6)

# ---- the inversion arrow: goal -> settings --------------------------------
arrow = FancyArrowPatch((gx + 24, gy + 6), (1330, 578),
                        connectionstyle="arc3,rad=-0.34", arrowstyle="-|>",
                        mutation_scale=40, lw=5, color="#c0392b", zorder=10)
ax.add_patch(arrow)
ax.text(1165, 704, "run the model backwards", color="#c0392b", fontsize=22,
        fontweight="bold", style="italic", ha="center", va="center", zorder=11)
ax.text(1165, 676, "goal  →  settings", color="#c0392b", fontsize=16,
        ha="center", va="center", zorder=11)

fig.savefig("colour-hero.png", dpi=300, facecolor="w", edgecolor="w")
print("saved colour-hero.png", f"{int(W * 3)}x{int((YHI - YLO) * 3)}")
