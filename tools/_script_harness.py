"""Run one figure script the way ``python <script>`` would, but without side effects.

Invoked as a subprocess by ``check_figure_scripts.py``; not useful on its own.

Three things differ from a plain run, and each matters:

* ``savefig`` and ``show`` are no-ops, so a check leaves the committed PNGs alone.
  A script is checked for whether it still runs, not for what it draws.
* The script's own directory goes on ``sys.path``, which is what CPython does for
  ``python foo.py`` and what ``runpy.run_path`` does not. Several scripts here
  import a sibling module (``colour_case_study``, ``omnibus_designs``).
* A ``DeprecationWarning``, or any warning class ``process_improve`` defines,
  raised from a line of the script itself, is reported. That is the signal that
  the library has renamed something the script still uses, which is how five
  scripts in this repository went stale without anyone noticing.
"""

from __future__ import annotations

import contextlib
import json
import os
import runpy
import sys
import traceback
import warnings
from pathlib import Path

FAILURE_PREFIX = "@@FIGURE-SCRIPT-RESULT@@ "


def _silence_output() -> None:
    """Make plotting headless and stop every writer of image files."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    with contextlib.suppress(ImportError):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.figure
        import matplotlib.pyplot as plt

        matplotlib.figure.Figure.savefig = lambda self, *a, **k: None
        plt.savefig = lambda *a, **k: None
        plt.show = lambda *a, **k: None
    with contextlib.suppress(ImportError):
        import plotly.basedatatypes as basedatatypes
        import plotly.io as pio

        basedatatypes.BaseFigure.show = lambda self, *a, **k: None
        basedatatypes.BaseFigure.write_image = lambda self, *a, **k: None
        basedatatypes.BaseFigure.write_html = lambda self, *a, **k: None
        pio.show = lambda *a, **k: None
        pio.write_image = lambda *a, **k: None


def _is_library_category(category: type[Warning]) -> bool:
    module = getattr(category, "__module__", "") or ""
    return module == "process_improve" or module.startswith("process_improve.")


def main() -> int:
    script = Path(sys.argv[1]).resolve()
    _silence_output()

    # As CPython does for `python <script>`: the script's directory comes first.
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            runpy.run_path(str(script), run_name="__main__")
        except BaseException:  # noqa: BLE001 - any failure is the script's failure
            payload = {"status": "failed", "detail": traceback.format_exc()}
            print(FAILURE_PREFIX + json.dumps(payload))
            return 1

    attributed = [
        f"{w.category.__name__} at line {w.lineno}: {w.message}"
        for w in caught
        if Path(w.filename).resolve() == script
        and (issubclass(w.category, DeprecationWarning) or _is_library_category(w.category))
    ]
    payload = {"status": "warned" if attributed else "passed", "warnings": attributed}
    print(FAILURE_PREFIX + json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
