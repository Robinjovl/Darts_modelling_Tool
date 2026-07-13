from pathlib import Path

import numpy as np
import pandas as pd
from darts.engines import redirect_darts_output
from darts.pipes.save_results import save_dfm_well_props
from darts.tools.hdf5_tools import load_hdf5_to_dict

from model import AquiferCO2InjectionConfig, Model


CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output_injection_production"
REPORT_STEPS = [0.001] * 50


def write_initial_equilibrium_check(model: Model) -> None:
    cfg = model.config
    rows = []
    all_well_states = []
    for well_name, cell_func in (("I1", model.injector_cell), ("P1", model.producer_cell)):
        well = model.reservoir.get_well(well_name)
        perforated_segment_local = int(well.perforations[-1][0])
        source_segment = int(model.wells["I1"].source_sinks["IsenthalpicInjection"].segment_idx) if well_name == "I1" else 0
        well_states = model.initial_well_state_table(well_name)
        all_well_states.extend(well_states)
        bhp = well_states[perforated_segment_local]["pressure_bar"]
        bht = well_states[perforated_segment_local]["temperature_K"]
        well_cell = cell_func()
        rows.append({
            "well": well_name,
            "reservoir_pressure_bar": cfg.p_init,
            "well_bhp_bar": bhp,
            "pressure_difference_bar": bhp - cfg.p_init,
            "reservoir_temperature_K": cfg.t_reservoir,
            "well_bht_K": bht,
            "temperature_difference_K": bht - cfg.t_reservoir,
            "perforated_segment_local": perforated_segment_local,
            "source_segment_local": source_segment,
            "well_cell_i": well_cell[0],
            "well_cell_j": well_cell[1],
            "well_cell_k": well_cell[2],
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "initial_equilibrium_check.csv", index=False)
    pd.DataFrame(all_well_states).to_csv(OUTPUT_DIR / "initial_well_state.csv", index=False)

    for row in rows:
        if not np.isclose(row["well_bhp_bar"], cfg.p_init, atol=1e-8):
            raise RuntimeError(f"Initial {row['well']} BHP {row['well_bhp_bar']} bar does not match reservoir pressure {cfg.p_init} bar.")
    if rows[0]["source_segment_local"] != 0:
        raise RuntimeError(f"CO2 injection source must be placed in segment 0, got segment {rows[0]['source_segment_local']}.")


def _cell_position(cell_ids, cell_id):
    matches = np.where(np.asarray(cell_ids, dtype=int) == int(cell_id))[0]
    if matches.size != 1:
        raise RuntimeError(f"Expected one entry for cell id {cell_id}, found {matches.size}.")
    return int(matches[0])


def _source_rate_series(model: Model, time):
    source = model.wells["I1"].source_sinks["IsenthalpicInjection"]
    composition = source.inj_fluid_props["composition"]
    mw = np.asarray(model.physics.property_containers[0].Mw[: model.physics.property_containers[0].nc_fl])
    ramp = np.minimum(np.asarray(time, dtype=float) / source.ramp_up_period, 1.0) if source.ramp_up_period > 0 else 1.0
    molar_rate = ramp * source.target_rate
    return molar_rate, molar_rate * composition[0], molar_rate * float(np.dot(mw, composition[: mw.size]))


def _production_rate_series(model: Model, time, states, source_pos):
    pc = model.physics.property_containers[0]
    mw = np.asarray(pc.Mw[: pc.nc_fl])
    ramp_period = float(model.config.production_ramp_up_period)
    ramp = np.minimum(np.asarray(time, dtype=float) / ramp_period, 1.0) if ramp_period > 0 else 1.0
    molar_rate = np.zeros_like(np.asarray(time, dtype=float))
    co2_molar_rate = np.zeros_like(molar_rate)
    for i, state in enumerate(states[:, source_pos, :]):
        _, _, composition = pc.get_state(state)
        molar_rate[i] = -ramp[i] * model.config.target_production_mass_rate_kg_day / float(np.dot(mw, composition[: mw.size]))
        co2_molar_rate[i] = molar_rate[i] * composition[0]
    return molar_rate, co2_molar_rate, -ramp * model.config.target_production_mass_rate_kg_day


def save_dfm_well_time_data(model: Model, connection_time_data: dict) -> None:
    h5_data = load_hdf5_to_dict(model.well_filepath)
    dynamic = h5_data["dynamic"]
    time = np.asarray(dynamic["time"], dtype=float)
    cell_ids = np.asarray(dynamic["cell_id"], dtype=int)
    variable_names = [name.decode() if isinstance(name, bytes) else str(name) for name in dynamic["variable_names"]]
    states = np.asarray(dynamic["X"], dtype=float)
    p_idx = variable_names.index("pressure")
    t_idx = variable_names.index("temperature")

    data = {"time": time}
    for well_name in ("I1", "P1"):
        well = model.reservoir.get_well(well_name)
        source_segment_local = int(model.wells["I1"].source_sinks["IsenthalpicInjection"].segment_idx) if well_name == "I1" else 0
        perforated_segment_local = int(well.perforations[-1][0])
        wellhead_cell = int(well.well_head_idx)
        source_cell = wellhead_cell + source_segment_local
        perforated_cell = wellhead_cell + perforated_segment_local
        wh_pos = _cell_position(cell_ids, wellhead_cell)
        source_pos = _cell_position(cell_ids, source_cell)
        perf_pos = _cell_position(cell_ids, perforated_cell)
        if well_name == "I1":
            source_molar_rate, source_co2_molar_rate, source_mass_rate = _source_rate_series(model, time)
        else:
            source_molar_rate, source_co2_molar_rate, source_mass_rate = _production_rate_series(model, time, states, source_pos)
        data[f"well_{well_name}_WHP"] = states[:, wh_pos, p_idx]
        data[f"well_{well_name}_WHT"] = states[:, wh_pos, t_idx]
        data[f"well_{well_name}_true_BHP"] = states[:, perf_pos, p_idx]
        data[f"well_{well_name}_true_BHT"] = states[:, perf_pos, t_idx]
        data[f"well_{well_name}_source_segment_pressure"] = states[:, source_pos, p_idx]
        data[f"well_{well_name}_source_segment_temperature"] = states[:, source_pos, t_idx]
        data[f"well_{well_name}_source_total_molar_rate"] = source_molar_rate
        data[f"well_{well_name}_source_CO2_molar_rate"] = source_co2_molar_rate
        data[f"well_{well_name}_source_total_mass_rate"] = source_mass_rate
        data[f"well_{well_name}_perforated_segment_local"] = np.full(time.size, perforated_segment_local)
        data[f"well_{well_name}_source_segment_local"] = np.full(time.size, source_segment_local)
        for key in (f"well_{well_name}_mass_rate_CO2_by_sum_perfs", f"well_{well_name}_mass_rate_G_by_sum_perfs",
                    f"well_{well_name}_molar_rate_CO2_by_sum_perfs", f"well_{well_name}_molar_rate_G_by_sum_perfs"):
            if key in connection_time_data:
                data[key] = connection_time_data[key]
    pd.DataFrame(data).to_csv(OUTPUT_DIR / "dfm_well_time_data.csv", index=False)


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

    connection_time_data = model.output.store_well_time_data(save_output_files=False)
    pd.DataFrame(connection_time_data).to_csv(OUTPUT_DIR / "well_connection_time_data.csv", index=False)
    save_dfm_well_time_data(model, connection_time_data)
    save_dfm_well_props("I1", model, include_phase_velocities=True)
    save_dfm_well_props("P1", model, include_phase_velocities=True)

    model.print_timers()
    model.print_stat()


if __name__ == "__main__":
    run()
