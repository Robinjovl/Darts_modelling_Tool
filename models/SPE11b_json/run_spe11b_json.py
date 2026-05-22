"""Driver: run the schema-first SPE11b JSON config in three injection segments.

The SPE11b benchmark injects CO2 in two wells with a staggered schedule:

* t = 0 .. 25 yr — well I1 only (3024 kg/day)
* t = 25 .. 50 yr — wells I1 and I2 (3024 kg/day each)
* t = 50 .. 90 yr — both wells shut

The schema-first JSON config (:file:`SPE11b.json`) declares the *initial*
controls (I1 on, I2 trickle to keep the well object connected); this driver
loads the JSON, builds the model, and steps through the three segments,
re-applying well controls between them via the standard
:meth:`Physics.set_well_controls` API.

Usage::

    python run_spe11b_json.py
    python run_spe11b_json.py --segment1 25 --segment2 25 --segment3 40

Each segment can be overridden in years from the CLI for fast smoke tests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from darts.engines import well_control_iface

from darts.api import ModelBuilder, ModelSpec
from darts.api.json_model import JsonModel
from darts.api.presets import resolve_section_presets


def _set_inj_mass_rate(
    model: JsonModel,
    well_name: str,
    rate_kg_per_day: float,
    inj_composition: list[float],
    inj_temp: float,
) -> None:
    """Set a mass-rate injector control on ``well_name``.

    :param model: live :class:`JsonModel` after ``init()``
    :type model: darts.api.json_model.JsonModel
    :param well_name: well name as it appears in ``self.reservoir.wells``
    :type well_name: str
    :param rate_kg_per_day: target mass rate (positive for injection)
    :type rate_kg_per_day: float
    :param inj_composition: injection composition vector of length nc-1
    :type inj_composition: list[float]
    :param inj_temp: injection temperature [K] (used only for thermal physics)
    :type inj_temp: float
    """
    for w in model.reservoir.wells:
        if w.name == well_name:
            model.physics.set_well_controls(
                wctrl=w.control,
                control_type=well_control_iface.MASS_RATE,
                is_inj=True,
                target=rate_kg_per_day,
                phase_name="V",
                inj_composition=inj_composition,
                inj_temp=inj_temp,
            )
            return
    raise KeyError(f"Well '{well_name}' not found in reservoir.wells")


def main() -> None:
    """Build the SPE11b JSON model, then run the three injection segments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        default=str(Path(__file__).resolve().parent / "SPE11b.json"),
        help="Path to the SPE11b JSON config",
    )
    parser.add_argument("--segment1", type=float, default=25.0, help="Years with I1 only")
    parser.add_argument(
        "--segment2", type=float, default=25.0, help="Years with both wells"
    )
    parser.add_argument(
        "--segment3", type=float, default=40.0, help="Years with both wells shut"
    )
    parser.add_argument(
        "--report-days",
        type=float,
        default=None,
        help="Report sub-step (days); defaults to one report per segment",
    )
    args = parser.parse_args()

    json_path = Path(args.json).resolve()
    with json_path.open() as f:
        spec_dict = json.load(f)
    spec_dict = resolve_section_presets(spec_dict)
    spec = ModelSpec.model_validate(spec_dict)

    model = JsonModel()
    ModelBuilder.apply(spec, model, base_path=str(json_path.parent))
    model.init(platform="cpu")

    out_spec = getattr(model, "_output_spec", None)
    folder = (out_spec.folder if out_spec and out_spec.folder else "output_json")
    precision = (out_spec.precision if out_spec and out_spec.precision else "d")
    os.makedirs(folder, exist_ok=True)
    model.set_output(output_folder=folder, precision=precision)

    inj_composition = [0.01]   # z_H2O in the injected stream (length nc-1)
    inj_temp = 283.15           # 10 °C
    inj_rate = 3024.0           # kg/day per well
    trickle = 1e-10             # placeholder rate when well is "off"

    days_per_year = 365.0
    segments = [
        ("segment1 — I1 only",
         args.segment1 * days_per_year,
         [("I1", inj_rate), ("I2", trickle)]),
        ("segment2 — I1 + I2",
         args.segment2 * days_per_year,
         [("I1", inj_rate), ("I2", inj_rate)]),
        ("segment3 — shut-in",
         args.segment3 * days_per_year,
         [("I1", trickle), ("I2", trickle)]),
    ]

    for label, span_days, controls in segments:
        for well_name, rate in controls:
            _set_inj_mass_rate(model, well_name, rate, inj_composition, inj_temp)
        print(f"\n=== {label}: running {span_days} days "
              f"(rates: {[(n, r) for n, r in controls]}) ===")
        if args.report_days is None or args.report_days >= span_days:
            model.run(days=span_days)
        else:
            end_time = model.physics.engine.t + span_days
            while model.physics.engine.t < end_time - 1e-9:
                dt = min(args.report_days, end_time - model.physics.engine.t)
                model.run(days=dt)

    model.print_stat()
    model.print_timers()


if __name__ == "__main__":
    sys.exit(main() or 0)
