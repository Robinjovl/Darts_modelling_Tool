from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from darts.engines import ms_well, redirect_darts_output
from darts.tools.hdf5_tools import load_hdf5_to_dict
from model import Model

matplotlib.use("Agg")
from matplotlib import pyplot as plt

def print_pipe_phase_props(model, well_name="P1"):
    pipe = model.wells.get(well_name)
    if pipe is None or not hasattr(pipe, "iter_phases_props"):
        print(f"{well_name} pipe phase properties are not available yet.")
        return

    well = next(
        (well for well in model.reservoir.wells if well.name == well_name),
        None,
    )
    if well is None:
        print(f"{well_name} reservoir well object is not available.")
        return

    n_vars = model.physics.n_vars
    p_idx = model.physics.vars.index("pressure")
    t_idx = (
        model.physics.vars.index("temperature")
        if "temperature" in model.physics.vars
        else None
    )
    start = int(well.well_head_idx)
    stop = start + int(pipe.geometry.num_segments)
    x = np.asarray(model.physics.engine.X, dtype=float).reshape(-1, n_vars)
    pipe_state = x[start:stop]
    pressure = pipe_state[:, p_idx]
    temperature = pipe_state[:, t_idx] if t_idx is not None else None

    xg_mass, xl_mass, sg, rhog, rhol, mug, mul = pipe.iter_phases_props
    sl = 1.0 - np.asarray(sg, dtype=float)
    props = {
        "p": pressure,
        "sG": sg,
        "sL": sl,
        "rhoG": rhog,
        "rhoL": rhol,
        "muG": mug,
        "muL": mul,
    }
    summary = []
    for name, values in props.items():
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            summary.append(f"{name}=nan")
        else:
            summary.append(f"{name}=[{finite.min():.4g},{finite.max():.4g}]")
    print(
        f"T={model.physics.engine.t:.6g} day {well_name} pipe phase props: "
        + ", ".join(summary)
    )
    bottom_idx = int(pipe.geometry.num_segments) - 1
    bottom_items = [
        f"p_top={pressure[0]:.4g} bar",
        f"p_bottom={pressure[bottom_idx]:.4g} bar",
    ]
    if temperature is not None:
        bottom_items.extend(
            [
                f"T_top={temperature[0]:.4g} K",
                f"T_bottom={temperature[bottom_idx]:.4g} K",
            ]
        )
    bottom_items.extend(
        [
            f"sG_bottom={np.asarray(sg, dtype=float)[bottom_idx]:.4g}",
            f"rhoG_bottom={np.asarray(rhog, dtype=float)[bottom_idx]:.4g}",
            f"rhoL_bottom={np.asarray(rhol, dtype=float)[bottom_idx]:.4g}",
        ]
    )
    print(f"{well_name} pipe top/bottom: " + ", ".join(bottom_items))

def _decode_names(names):
    return [name.decode() if isinstance(name, bytes) else str(name) for name in names]


def _cell_positions(cell_ids, target_ids):
    position_by_cell = {int(cell_id): idx for idx, cell_id in enumerate(cell_ids)}
    missing = [
        int(cell_id)
        for cell_id in target_ids
        if int(cell_id) not in position_by_cell
    ]
    if missing:
        raise ValueError(f"Missing requested well cell ids in well_data.h5: {missing}")

    return np.array(
        [position_by_cell[int(cell_id)] for cell_id in target_ids],
        dtype=int,
    )


def _well_segment_ids(well):
    return np.arange(
        int(well.well_head_idx),
        int(well.well_head_idx) + int(well.num_segments),
        dtype=int,
    )


def _perforated_segment_local_idx(well):
    if len(well.perforations) == 0:
        return int(well.num_segments) - 1

    segment_idx = int(well.perforations[-1][0]) + 1
    return min(max(segment_idx, 0), int(well.num_segments) - 1)


def _well_depths(model, well):
    pipe = model.wells.get(well.name)
    if pipe is None:
        return np.arange(int(well.num_segments), dtype=float)

    depths = np.asarray(pipe.geometry.TVD_segments, dtype=float)
    if depths.size != int(well.num_segments):
        return np.arange(int(well.num_segments), dtype=float)

    return depths


def _profile_steps(time):
    if len(time) <= 6:
        return np.arange(len(time), dtype=int)

    return np.unique(np.linspace(0, len(time) - 1, 6, dtype=int))


def _plot_true_bhp_bht(time_data, wells, figures_dir):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for well in wells:
        name = well.name
        axes[0].plot(
            time_data["time"],
            time_data[f"well_{name}_true_BHP"],
            label=f"{name} BHP",
        )
        axes[0].plot(
            time_data["time"],
            time_data[f"well_{name}_WHP"],
            linestyle="--",
            label=f"{name} WHP",
        )
        axes[1].plot(
            time_data["time"],
            time_data[f"well_{name}_true_BHT"] - 273.15,
            label=f"{name} BHT",
        )
        axes[1].plot(
            time_data["time"],
            time_data[f"well_{name}_WHT"] - 273.15,
            linestyle="--",
            label=f"{name} WHT",
        )

    axes[0].set_ylabel("Pressure, bar")
    axes[1].set_ylabel("Temperature, C")
    axes[1].set_xlabel("Time, days")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "dfm_true_bhp_bht_and_wellhead.png", dpi=200)
    plt.close(fig)


def _plot_well_profiles(
    well_name, time, depths, pressure, temperature, steps, figures_dir
):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)

    for step in steps:
        label = f"{time[step]:.4g} d"
        axes[0].plot(pressure[step], depths, label=label)
        axes[1].plot(temperature[step] - 273.15, depths, label=label)

    axes[0].invert_yaxis()
    axes[0].set_xlabel("Pressure, bar")
    axes[1].set_xlabel("Temperature, C")
    axes[0].set_ylabel("Depth, m")
    axes[0].set_title(f"{well_name} pressure profile")
    axes[1].set_title(f"{well_name} temperature profile")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(title="Time")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{well_name}_pressure_temperature_profiles.png", dpi=200)
    plt.close(fig)


def save_dfm_well_diagnostics(model, time_data):
    output_dir = Path(model.output.output_folder)
    diagnostics_dir = output_dir / "dfm_well_diagnostics"
    figures_dir = output_dir / "figures" / "dfm_well_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    h5_data = load_hdf5_to_dict(str(output_dir / model.output.well_filename))
    dynamic = h5_data["dynamic"]
    time = np.asarray(dynamic["time"], dtype=float)
    cell_ids = np.asarray(dynamic["cell_id"], dtype=int)
    variables = _decode_names(dynamic["variable_names"])
    states = np.asarray(dynamic["X"], dtype=float)

    pressure_idx = variables.index("pressure")
    temperature_idx = variables.index("temperature")

    time_data = pd.DataFrame(time_data)
    if "time" not in time_data.columns:
        time_data.insert(0, "time", time)

    profile_rows = []
    dfm_wells = [
        well for well in model.reservoir.wells if well.ms_type == ms_well.MS_Type.DFM
    ]
    profile_sample_steps = _profile_steps(time)

    for well in dfm_wells:
        segment_ids = _well_segment_ids(well)
        segment_positions = _cell_positions(cell_ids, segment_ids)
        perforated_local_idx = _perforated_segment_local_idx(well)
        wellhead_position = segment_positions[0]
        bottom_position = segment_positions[perforated_local_idx]
        depths = _well_depths(model, well)

        pressure = states[:, segment_positions, pressure_idx]
        temperature = states[:, segment_positions, temperature_idx]

        time_data[f"well_{well.name}_WHP"] = states[:, wellhead_position, pressure_idx]
        time_data[f"well_{well.name}_WHT"] = states[
            :,
            wellhead_position,
            temperature_idx,
        ]
        time_data[f"well_{well.name}_true_BHP"] = states[
            :,
            bottom_position,
            pressure_idx,
        ]
        time_data[f"well_{well.name}_true_BHT"] = states[
            :,
            bottom_position,
            temperature_idx,
        ]
        time_data[f"well_{well.name}_bottom_segment_global"] = segment_ids[
            perforated_local_idx
        ]
        time_data[f"well_{well.name}_bottom_depth_m"] = depths[perforated_local_idx]

        for step in range(len(time)):
            for local_idx, depth in enumerate(depths):
                profile_rows.append(
                    {
                        "time": time[step],
                        "well": well.name,
                        "segment_local": local_idx,
                        "segment_global": segment_ids[local_idx],
                        "depth_m": depth,
                        "pressure_bar": pressure[step, local_idx],
                        "temperature_K": temperature[step, local_idx],
                        "temperature_C": temperature[step, local_idx] - 273.15,
                    }
                )

        _plot_well_profiles(
            well.name,
            time,
            depths,
            pressure,
            temperature,
            profile_sample_steps,
            figures_dir,
        )

    time_data.to_csv(
        diagnostics_dir / "well_time_data_with_true_bhp_bht.csv",
        index=False,
    )
    pd.DataFrame(profile_rows).to_csv(
        diagnostics_dir / "dfm_well_profiles.csv",
        index=False,
    )
    _plot_true_bhp_bht(time_data, dfm_wells, figures_dir)


if __name__ == "__main__":
    case_dir = Path(__file__).resolve().parent
    redirect_darts_output(str(case_dir / "run.log"))

    model = Model()
    model.init()
    model.set_output(output_folder=str(case_dir / "output"), save_initial=False)
    model.output.save_data_to_h5(kind="well")

    report_steps = [0.001] * 10 + [0.01] * 9 + [0.1] * 9 + [1.0] * 49 + [1.0] * 100
    output_props = model.physics.vars + model.output.properties
    model.output.output_to_vtk(
        ith_step=0,
        output_properties=output_props,
        engine=True,
    )
    model.output.well_output_to_vtp(
        ith_step=0,
        output_properties=output_props,
    )

    for ith_step, dt in enumerate(report_steps):
        if 0.03 < model.physics.engine.t < 0.05:
            model.data_ts.dt_max = 0.05
        elif 1 < model.physics.engine.t < 3:
            model.data_ts.dt_max = 0.1

        model.run(
            dt,
            save_well_data=True,
            save_well_data_after_run=False,
            save_reservoir_data=False,
        )
        print_pipe_phase_props(model, "P1")
        print_pipe_phase_props(model, "I1")

        model.output.output_to_vtk(
            ith_step=ith_step + 1,
            output_properties=output_props,
            engine=True,
        )
        model.output.well_output_to_vtp(
            ith_step=ith_step + 1,
            output_properties=output_props,
        )

    time_data = model.output.store_well_time_data(save_output_files=False)
    save_dfm_well_diagnostics(model, time_data)

    model.print_timers()
    model.print_stat()
