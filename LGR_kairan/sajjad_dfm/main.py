from pathlib import Path

import numpy as np
import pandas as pd
from darts.engines import redirect_darts_output
from darts.pipes.save_results import save_dfm_well_props

from model import AquiferCO2InjectionConfig, Model


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output_injection_only"
REPORT_STEPS = [0.001] * 200


def write_initial_equilibrium_check(model: Model) -> None:
    cfg = model.config
    well = model.reservoir.get_well("I1")
    perforated_segment_local = int(well.perforations[-1][0])
    source_segment = int(model.wells["I1"].source_sinks["IsenthalpicInjection"].segment_idx)
    well_states = model.initial_well_state_table()
    bhp = well_states[perforated_segment_local]["pressure_bar"]
    bht = well_states[perforated_segment_local]["temperature_K"]

    rows = [{
        "reservoir_pressure_bar": cfg.p_init,
        "well_bhp_bar": bhp,
        "pressure_difference_bar": bhp - cfg.p_init,
        "reservoir_temperature_K": cfg.t_reservoir,
        "well_bht_K": bht,
        "temperature_difference_K": bht - cfg.t_reservoir,
        "perforated_segment_local": perforated_segment_local,
        "injection_source_segment_local": source_segment,
        "injector_cell_i": model.injector_cell()[0],
        "injector_cell_j": model.injector_cell()[1],
        "injector_cell_k": model.injector_cell()[2],
    }]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "initial_equilibrium_check.csv", index=False)
    pd.DataFrame(well_states).to_csv(OUTPUT_DIR / "initial_well_state.csv", index=False)

    if not np.isclose(bhp, cfg.p_init, atol=1e-8):
        raise RuntimeError(f"Initial I1 BHP {bhp} bar does not match reservoir pressure {cfg.p_init} bar.")
    if source_segment != 0:
        raise RuntimeError(f"CO2 injection source must be placed in segment 0, got segment {source_segment}.")


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    redirect_darts_output(str(OUTPUT_DIR / "run.log"))

    model = Model(AquiferCO2InjectionConfig())
    model.init(platform="cpu")
    model.set_output(output_folder=str(OUTPUT_DIR), save_initial=False)

    write_initial_equilibrium_check(model)

    output_props = model.physics.vars + model.output.properties
    model.output.output_to_vtk(ith_step=0, output_properties=output_props, engine=True)
    model.output.well_output_to_vtp(ith_step=0, output_properties=output_props)

    for ith_step, dt in enumerate(REPORT_STEPS, start=1):
        model.run(dt, save_well_data=True, save_well_data_after_run=False, save_reservoir_data=False)
        model.output.output_to_vtk(ith_step=ith_step, output_properties=output_props, engine=True)
        model.output.well_output_to_vtp(ith_step=ith_step, output_properties=output_props)

    time_data = model.output.store_well_time_data(save_output_files=True)
    pd.DataFrame(time_data).to_csv(OUTPUT_DIR / "well_time_data.csv", index=False)
    save_dfm_well_props("I1", model, include_phase_velocities=True)

    model.print_timers()
    model.print_stat()


if __name__ == "__main__":
    run()
