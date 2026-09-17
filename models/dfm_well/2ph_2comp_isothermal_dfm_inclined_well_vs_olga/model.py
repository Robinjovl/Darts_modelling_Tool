"""Two-phase two-component isothermal inclined DFM well benchmarked against OLGA.

Injection of pure gaseous CO2 at a constant mass rate into a 45-degree inclined
well containing water. The construction is shared with the vertical benchmark and
lives in ``models/dfm_well/_olga_benchmark_base.py``.
"""

import os
import sys

# The shared base module sits in the parent directory (models/dfm_well). Anchor the
# lookup on this file, because model.py is imported as a top-level module both by the
# test suite (which puts only this model directory on sys.path) and by main.py.
_DFM_WELL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DFM_WELL_DIR not in sys.path:
    sys.path.insert(0, _DFM_WELL_DIR)

from _olga_benchmark_base import OLGABenchmarkModel  # noqa: E402


class Model(OLGABenchmarkModel):
    """Inclined OLGA benchmark (no formulation variants)."""

    first_ts_seconds = 0.001
    reservoir_depth = 689.42911
    p_init_res = 75.48341
    inclination_angle = 45.0
