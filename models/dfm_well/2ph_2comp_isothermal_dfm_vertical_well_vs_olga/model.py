"""Two-phase two-component isothermal vertical DFM well benchmarked against OLGA.

Injection of pure gaseous CO2 at a constant mass rate into a vertical well
containing water. The construction is shared with the inclined benchmark and
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
    """Vertical OLGA benchmark.

    Beside the base model, the following test-suite variants are supported and
    selected with ``Model(formulation=...)``:

    * ``'tang_2019'`` — Tang et al. (2019) unified drift-flux closure in the pipe.
    * ``'bhagwat_ghajar_2014'`` — Bhagwat and Ghajar (2014) drift-flux closure.
    * ``'bai_2023'`` — Bai et al. (2023) CO2-specific drift-flux closure.
    * ``'ipr_volumetric'`` — same injector, but a volumetric-PI IPR with nonzero
      intercept and pressure offset.
    * ``'ipr_producer'`` — BHP-controlled DFM producer (wellhead pressure below the
      reservoir pressure) with a molar-PI IPR, exercising the hook's
      reservoir-upstream branch (total_rate < 0).
    """

    first_ts_seconds = 0.0001
    reservoir_depth = 975
    p_init_res = 102.66245
    inclination_angle = 0.0

    supported_formulations = (
        'tang_2019',
        'bhagwat_ghajar_2014',
        'bai_2023',
        'ipr_volumetric',
        'ipr_producer',
    )
