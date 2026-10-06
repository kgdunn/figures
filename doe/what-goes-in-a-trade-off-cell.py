"""What goes in a trade-off table cell: the fractional factorial column against the OMARS column.

A social-media graphic (4:5 portrait, 2160 x 2700 px) for the OMARS sections of the book. Four
factors on both sides: the left column shows the two-level designs at 8, 16 and 32 runs, the
right column the OMARS designs from the definitive screening design to the Box-Behnken design.
Every cell value is read from process-improve, so the graphic cannot drift from the book.

The layout is HTML, rendered to PNG by headless Chromium (what-goes-in-a-trade-off-cell.mjs,
which needs Node and Playwright). The Inter typeface is fetched once into ~/.cache; without it
the page falls back to the system sans.

Reproducible; run from this directory to write the PNG alongside it.
"""

import os
import pathlib
import subprocess
import tempfile
import urllib.request

from process_improve.experiments import get_omars_trade_off_table_entry, get_trade_off_table_entry
from process_improve.experiments.omars_trade_off import (
    box_behnken_runs,
    definitive_screening_runs,
    omars_anchor_entry,
)

HERE = pathlib.Path(__file__).parent
STEM = "what-goes-in-a-trade-off-cell"
INTER = "https://fonts.gstatic.com/s/inter/v20/UcC73FwrK3iLTeHuS_nVMrMxCp50SjIa1ZL7.woff2"
K = 4
N_PARAMS = 1 + 2 * K + K * (K - 1) // 2  # 15
N_INTERACTIONS = K * (K - 1) // 2  # 6
FRONTIER = K**2 + K + 1  # 21

# ---------------------------------------------------------------- data from the library
ff = {n: get_trade_off_table_entry(n_runs=n, n_factors=K, display=False) for n in (8, 16, 32)}
assert ff[8].roman == "IV" and ff[8].generators == ["D=ABC"]
assert ff[16].n_replicates == 1 and ff[32].n_replicates == 2

dsd, bbd = definitive_screening_runs(K), box_behnken_runs(K)
rows = []
for n in range(dsd, bbd + 1, 2):
    entry = omars_anchor_entry("bbd", K) if n == bbd else get_omars_trade_off_table_entry(n, K, display=False)
    h = (n - 1) // 2
    fit = N_INTERACTIONS if entry.capability == "full" else max(0, min(N_INTERACTIONS, h - K))
    rows.append({"n": n, "tag": entry.tag, "cap": entry.capability, "df": entry.error_df, "fit": fit})
assert [r["fit"] for r in rows] == [0, 1, 2, 3, 4, 5, 6, 6, 6, 6]
assert next(r for r in rows if r["cap"] == "full")["n"] == FRONTIER

# ---------------------------------------------------------------- layout
Y0, STEP = 404, 34  # y of 8 runs, pixels per run


def y(n):
    return Y0 + (n - 8) * STEP


CHECK = (
    '<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">'
    '<path d="M5 12.5l4.2 4.2L19 7" fill="none" stroke="currentColor" stroke-width="2.6" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

parts = []
add = parts.append

# Right panel: OMARS rows.
RX = 396
RW = 1032 - RX  # right panel width
BAND_W = 448
band_top, band_bot = y(15) - 31, y(19) + 31
add(f'<div class="band" style="left:{RX - 8}px;top:{band_top}px;width:{BAND_W}px;'
    f'height:{band_bot - band_top}px"></div>')
for r in rows:
    yy = y(r["n"])
    key = "bbd" if r["n"] == bbd else r["cap"]
    dots = "".join(f'<i class="dot{" on" if i < r["fit"] else ""}"></i>' for i in range(N_INTERACTIONS))
    add(f'<div class="num" style="left:{RX}px;top:{yy}px">{r["n"]}</div>')
    add(f'<div class="chk" style="left:{RX + 90}px;top:{yy}px">{CHECK}</div>')
    add(f'<div class="pill {key}" style="left:{RX + 124}px;top:{yy}px">{r["tag"]}</div>')
    add(f'<div class="df" style="left:{RX + 214}px;top:{yy}px">{r["df"]}</div>')
    add(f'<div class="dots" style="left:{RX + 286}px;top:{yy}px">{dots}</div>')
fy = y(FRONTIER)
add(f'<div class="frontier" style="left:{RX - 8}px;top:{fy - 30}px;width:{BAND_W}px"></div>')

# Right-panel annotations.
AX = RX + BAND_W + 12
add(f'<div class="note" style="left:{AX}px;top:{y(9)}px">definitive<br>screening design</div>')
add(f'<div class="note" style="left:{AX}px;top:{y(12)}px">each two more runs<br>make room for one<br>'
    'more interaction</div>')
add(f'<div class="note" style="left:{AX}px;top:{y(17)}px"><b>15 to 19 runs:</b><br>at least one run per<br>'
    f'parameter ({N_PARAMS}), yet<br>the full model<br>does not fit</div>')
add(f'<div class="note" style="left:{AX}px;top:{fy}px"><b class="vm">the frontier</b><br>'
    f'k² + k + 1 = {FRONTIER} runs</div>')
add(f'<div class="note" style="left:{AX}px;top:{y(bbd)}px">Box-Behnken design</div>')

# Legend for the model classes, in the free space under the OMARS rows.
LEG = (("satd", "Satd", "saturated: estimates,<br>no error df"),
       ("quad", "Quad", "main effects and<br>quadratics"),
       ("full", "Full", "plus all 6 interactions"))
for i, (key, tag, text) in enumerate(LEG):
    ly = y(bbd) + 92 + i * 50
    add(f'<div class="pill sm {key}" style="left:{RX + 124}px;top:{ly}px">{tag}</div>')
    add(f'<div class="legt" style="left:{RX + 214}px;top:{ly}px">{text}</div>')

# Left panel: fractional factorial rows.
LX = 48
add(f'<div class="num" style="left:{LX}px;top:{y(8) + 70}px">8</div>')
add(f'<div class="card" style="left:{LX + 56}px;top:{y(8) - 18}px">'
    '<div class="cap">4 designs fit this cell</div>'
    '<div class="group"><div class="chip muted"><span class="rom">III</span><span class="gen">D = AB'
    '<small>also D = AC or D = BC</small></span></div>'
    '<div class="chip pick"><span class="rom">IV</span><span class="gen">D = ABC'
    '<small>printed in the table</small></span></div></div></div>')
for n, note in ((16, "nothing aliased"), (32, "run twice")):
    add(f'<div class="num" style="left:{LX}px;top:{y(n)}px">{n}</div>')
    add(f'<div class="chip plain single" style="left:{LX + 56}px;top:{y(n)}px">'
        f'<span class="rom">2<sup>4</sup></span><span class="gen">full factorial<small>{note}</small>'
        '</span></div>')
add(f'<div class="aside" style="left:{LX + 56}px;top:{(y(16) + y(32)) // 2}px">Fractional factorials come<br>'
    'in powers of two: 8, 16 and<br>32 runs. The OMARS column<br>moves two runs at a time.</div>')

body = "\n".join(parts)

# The script check (tools/check_figure_scripts.py) sets FIGURE_SCRIPT_CHECK: it then runs
# everything above, every library call and assertion, but fetches no font and renders no PNG.
CHECK_RUN = bool(os.environ.get("FIGURE_SCRIPT_CHECK"))
font_file = pathlib.Path.home() / ".cache" / "pid-figures" / "inter-latin.woff2"
try:
    if CHECK_RUN:
        raise OSError("check run")
    if not font_file.exists():
        font_file.parent.mkdir(parents=True, exist_ok=True)
        font_file.write_bytes(urllib.request.urlopen(INTER, timeout=30).read())
    font_face = (f"@font-face {{ font-family: Inter; font-weight: 100 900; "
                 f"src: url({font_file.as_uri()}) format('woff2'); }}")
except OSError:
    font_face = ""  # fall back to the system sans

html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Trade-off table cells</title>
<style>
{font_face}
:root {{
  --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781; --hair: #e1e0d9;
  --satd: #E69F00; --quad: #56B4E9; --full: #0072B2; --bbd: #009E73; --vermilion: #D55E00;
}}
* {{ box-sizing: border-box; margin: 0; }}
html, body {{ width: 1080px; height: 1350px; background: var(--surface); }}
body {{ position: relative; font-family: Inter, system-ui, sans-serif; color: var(--ink);
  -webkit-font-smoothing: antialiased; overflow: hidden; }}
.abs, .num, .chk, .pill, .df, .dots, .note, .card, .chip.single, .aside, .band, .frontier
  {{ position: absolute; }}
h1 {{ position: absolute; left: 48px; top: 46px; font-size: 50px; line-height: 1.08; font-weight: 800;
  letter-spacing: -0.02em; width: 984px; }}
.sub {{ position: absolute; left: 48px; top: 112px; width: 960px; font-size: 25px; line-height: 1.38;
  color: var(--ink2); }}
.panel-h {{ position: absolute; top: 186px; font-size: 31px; font-weight: 750; letter-spacing: -0.01em; }}
.panel-s {{ position: absolute; top: 228px; font-size: 19px; line-height: 1.35; color: var(--ink2); }}
.rule {{ position: absolute; top: 178px; height: 3px; background: var(--ink); border-radius: 2px; }}
.colh {{ position: absolute; transform: translateY(-100%); font-size: 18px; line-height: 1.25; color: var(--ink2); font-weight: 700; }}
.num {{ width: 46px; text-align: right; font-size: 25px; font-weight: 650; transform: translateY(-50%);
  font-variant-numeric: tabular-nums; }}
.chk {{ transform: translate(-50%, -50%); color: var(--muted); }}
.pill {{ width: 84px; height: 40px; border-radius: 9px; transform: translateY(-50%);
  display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: 700; }}
.pill.satd {{ background: var(--satd); color: #4A2F00; }}
.pill.quad {{ background: var(--quad); color: #10334A; }}
.pill.full {{ background: var(--full); color: #fff; }}
.pill.sm {{ width: 70px; height: 32px; font-size: 16px; border-radius: 7px; }}
.pill.bbd  {{ background: var(--bbd); color: #fff; }}
.df {{ width: 44px; text-align: right; font-size: 22px; font-weight: 600; color: var(--ink2);
  transform: translateY(-50%); font-variant-numeric: tabular-nums; }}
.dots {{ display: flex; gap: 6px; transform: translateY(-50%); }}
.dot {{ width: 19px; height: 19px; border-radius: 50%; border: 2px solid #c3c2b7; background: var(--surface); }}
.dot.on {{ border-color: var(--full); background: var(--full); }}
.band {{ background: rgba(213, 94, 0, 0.07); border-radius: 12px; }}
.frontier {{ height: 58px; border: 3px solid var(--vermilion); border-radius: 12px; }}
.note {{ width: 176px; font-size: 18px; line-height: 1.3; color: var(--ink2); transform: translateY(-50%); }}
.note b {{ color: var(--ink); font-weight: 650; }}
.note b.vm {{ color: var(--ink); }}
.card {{ width: 264px; }}
.card .cap {{ font-size: 15px; color: var(--muted); font-weight: 500; margin: 0 0 7px 2px; }}
.chip {{ width: 264px; border-radius: 12px; padding: 10px 14px; display: flex; align-items: center; gap: 14px; }}
.group {{ position: relative; padding-left: 10px; }}
.group::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--ink); border-radius: 2px; }}
.group .chip {{ width: 254px; }}
.chip + .chip {{ margin-top: 8px; }}
.chip.single {{ transform: translateY(-50%); }}
.chip .rom {{ width: 52px; font-size: 30px; font-weight: 800; letter-spacing: 0.02em; text-align: center; }}
.chip .rom sup {{ font-size: 0.55em; }}
.chip .gen {{ font-size: 22px; font-weight: 650; display: flex; flex-direction: column; }}
.chip small {{ font-size: 15px; font-weight: 450; color: var(--ink2); margin-top: 2px; }}
.chip.muted {{ background: #f0efec; color: var(--muted); }}
.chip.muted small {{ color: var(--muted); }}
.chip.pick {{ background: #fff; box-shadow: 0 0 0 2.5px var(--ink) inset; }}
.chip.plain {{ background: #f0efec; }}
.aside {{ width: 264px; font-size: 18px; line-height: 1.38; color: var(--muted); transform: translateY(-50%);
  border-left: 3px solid var(--hair); padding-left: 14px; }}
.legt {{ position: absolute; transform: translateY(-50%); font-size: 16px; line-height: 1.25; color: var(--ink2); }}
.foot {{ position: absolute; left: 48px; right: 48px; top: 1292px; display: flex; justify-content: space-between;
  font-size: 15px; color: var(--muted); border-top: 1px solid var(--hair); padding-top: 14px; }}
.foot b {{ color: var(--ink2); font-weight: 600; }}
</style></head>
<body>
<h1>What goes in a trade-off table cell?</h1>
<div class="sub">Four factors. Each cell is one design, chosen from those that fit the budget.</div>

<div class="rule" style="left:48px;width:320px"></div>
<div class="panel-h" style="left:48px">Fractional factorial</div>
<div class="panel-s" style="left:48px;width:320px">The cell reports resolution.</div>
<div class="rule" style="left:396px;width:636px"></div>
<div class="panel-h" style="left:396px">OMARS</div>
<div class="panel-s" style="left:396px;width:636px">Main effects are clean in every row, so the cell reports<br>
the model that fits and its error degrees of freedom.</div>

<div class="colh" style="left:48px;top:372px;width:46px;text-align:right">runs</div>
<div class="colh" style="left:104px;top:372px">resolution, generator</div>
<div class="colh" style="left:396px;top:372px;width:46px;text-align:right">runs</div>
<div class="colh" style="left:450px;top:372px;width:72px;text-align:center">main<br>effects<br>clean</div>
<div class="colh" style="left:524px;top:372px;width:76px;text-align:center">model</div>
<div class="colh" style="left:604px;top:372px;width:50px;text-align:right">error<br>df</div>
<div class="colh" style="left:682px;top:372px;width:160px">interactions<br>that fit, of 6</div>

{body}

<div class="foot"><span>OMARS rows: foldovers with one centre run. Values from the process-improve
Python library.</span><b>learnche.org/pid</b></div>
</body></html>
"""

if CHECK_RUN:
    print("check run: page built, PNG not rendered")
else:
    with tempfile.TemporaryDirectory() as tmp:
        page = pathlib.Path(tmp) / f"{STEM}.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run(["node", str(HERE / f"{STEM}.mjs"), str(page), str(HERE / f"{STEM}.png")], check=True)
