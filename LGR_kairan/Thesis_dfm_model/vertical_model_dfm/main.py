import os
import sys
import importlib.util

print("=" * 80)
print("exe =", sys.executable)
print("cwd =", os.getcwd())
print("sys.path[:10] =")
for p in sys.path[:10]:
    print("  ", p)

print("darts spec =", importlib.util.find_spec("darts"))
print("darts.engines spec =", importlib.util.find_spec("darts.engines"))
print("=" * 80)

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from darts.engines import ms_well, redirect_darts_output, well_control_iface
from stage2_dfm_thermosiphon_model import Model, Stage2DFMThermosiphonConfig

from darts.pipes.save_results import save_dfm_well_props
from darts.pipes.viz.plot_heat_map_contourf import plot_heat_map_contourf
from darts.pipes.viz.plot_heat_map_pcolormesh import plot_heat_map_pcolormesh
from darts.tools.hdf5_tools import load_hdf5_to_dict

# from darts.engines import value_vector

matplotlib.use("Agg")
from matplotlib import pyplot as plt

CASE_DIR = Path(__file__).resolve().parent
STAGE1_STATE_PATH = (
    CASE_DIR / "output_precondition_dynamic_mu" / "final_reservoir_state_for_dfm.npz"
)
OUTPUT_DIR = CASE_DIR / "output_co2_producer"

DFM_WELL_NAMES = ("I1", "P1")

# Switch this to True only when you want to run the simulation again.
RUN_SIMULATION = False

# Recompute after model output property names change; set False to reuse existing
# dfm_well_props_*.pkl files.
RECOMPUTE_DFM_WELL_PROPS = True
INCLUDE_PHASE_VELOCITIES = True


REPORT_STEPS = [0.001] * 10 + [0.01] *9 + [0.1] *9


# Producer WHP ramp-down control. This is applied only in run_simulation().
ENABLE_PRODUCER_WHP_RAMP_DOWN = False
PRODUCER_WHP_RAMP_START_DAY = 0.0
PRODUCER_WHP_RAMP_DURATION_DAY = 10.0 / (24 * 60)  # 10 minutes
PRODUCER_WHP_RAMP_START_BAR = None  # None uses stage2_config.producer_head_pressure.
PRODUCER_WHP_RAMP_END_BAR = 45.0

# Water-filled producer startup control. Vent first with a WHP lower than the
# initial WHP, then switch to the same producer WHP used by model_rate.
ENABLE_PRODUCER_VENT_THEN_HOLD = False
PRODUCER_VENT_DURATION_DAY = 0.1
PRODUCER_VENT_WHP_BAR = 71.203
PRODUCER_POST_VENT_WHP_BAR = 100


def print_initial_pipe_state(model, well_name):
    pipe = model.wells[well_name]
    pc = model.physics.property_containers[0]
    n_vars = model.physics.n_vars

    states = np.asarray(
        pipe.initial_conditions.initial_conditions_vector,
        dtype=float,
    ).reshape(-1, n_vars)

    print(f"\n{well_name} initial pipe state")
    for i, state in enumerate(states):
        pc.evaluate(state)
        print(
            f"seg={i:02d}, "
            f"p={state[0]:.3f} bar, "
            f"T={pc.temperature:.2f} K, "
            f"sG={pc.sat[0]:.4f}, "
            f"sL={pc.sat[1]:.4f}, "
            f"rhoG={pc.dens[0]:.3f}, "
            f"rhoL={pc.dens[1]:.3f}"
        )


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

    xG_mass, xL_mass, sg, sl, rhog, rhol, mug, mul = pipe.iter_phases_props
    # xG_mass, xL_mass, sg, rhog, rhol, mug, mul = pipe.iter_phases_props
    # sl = 1 -sg
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
        int(cell_id) for cell_id in target_ids if int(cell_id) not in position_by_cell
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


def _face_value(values, phase_saturation=None, face_idx=0):
    values = np.asarray(values, dtype=float)
    if values.size <= face_idx + 1:
        return values[face_idx]

    if phase_saturation is None:
        return 0.5 * (values[face_idx] + values[face_idx + 1])

    phase_saturation = np.asarray(phase_saturation, dtype=float)
    if phase_saturation[face_idx] == 0:
        return values[face_idx + 1]
    if phase_saturation[face_idx + 1] == 0:
        return values[face_idx]
    return 0.5 * (values[face_idx] + values[face_idx + 1])


def _phase_mw_from_mass_fractions(segment_df, model, phase_name, face_idx=0):
    pc = model.physics.property_containers[0]
    inv_mw = 0.0
    found = False
    for comp_idx, comp_name in enumerate(pc.components_name[: pc.nc_fl]):
        col = f"x{comp_name}_in_{phase_name}_mass"
        if col not in segment_df:
            continue
        mass_fraction = _face_value(segment_df[col].to_numpy(), face_idx=face_idx)
        inv_mw += mass_fraction / pc.Mw[comp_idx]
        found = True

    if not found or inv_mw <= 0:
        return pc.Mw[0]
    return 1.0 / inv_mw


def _add_time_segment_index_to_dfm_props(props_df, time, well, depths):
    n_steps = len(time)
    n_segments = int(well.num_segments)
    expected_rows = n_steps * n_segments
    if len(props_df) != expected_rows:
        raise ValueError(
            f"dfm_well_props_{well.name} has {len(props_df)} rows, "
            f"expected {expected_rows}."
        )

    segment_local = np.tile(np.arange(n_segments, dtype=int), n_steps)
    segment_global = int(well.well_head_idx) + segment_local
    indexed = props_df.copy()
    indexed.insert(0, "depth_m", np.tile(depths, n_steps))
    indexed.insert(0, "segment_global", segment_global)
    indexed.insert(0, "segment_local", segment_local)
    indexed.insert(0, "time", np.repeat(time, n_segments))
    return indexed


def _save_indexed_dfm_well_props(
    model,
    well,
    time,
    depths,
    diagnostics_dir,
    recompute_dfm_well_props=False,
):
    props_path = Path(model.output_folder) / f"dfm_well_props_{well.name}.pkl"
    if recompute_dfm_well_props or not props_path.exists():
        with np.errstate(divide="ignore", invalid="ignore"):
            save_dfm_well_props(
                well.name,
                model,
                include_phase_velocities=INCLUDE_PHASE_VELOCITIES,
            )

    props_df = pd.read_pickle(props_path)
    for source_col, alias_col in (("miuG", "muG"), ("miuL", "muL")):
        if source_col in props_df and alias_col not in props_df:
            props_df[alias_col] = props_df[source_col]

    for velocity_col in ("vG", "vL"):
        if velocity_col in props_df:
            props_df[velocity_col] = np.nan_to_num(
                props_df[velocity_col].to_numpy(dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
    props_df.to_pickle(props_path)

    indexed = _add_time_segment_index_to_dfm_props(props_df, time, well, depths)
    indexed.to_csv(
        diagnostics_dir / f"dfm_well_props_{well.name}_with_index.csv",
        index=False,
    )
    indexed.to_pickle(diagnostics_dir / f"dfm_well_props_{well.name}_with_index.pkl")
    return indexed


def _add_velocity_based_wellhead_rates(time_data, props_df, model, well):
    missing_velocity_cols = [
        velocity_col
        for velocity_col in ("vG", "vL")
        if velocity_col not in props_df.columns
    ]
    if missing_velocity_cols:
        print(
            f"Skipped velocity-based wellhead rates for {well.name}: "
            f"missing columns {missing_velocity_cols}. "
            "Set INCLUDE_PHASE_VELOCITIES = True and "
            "RECOMPUTE_DFM_WELL_PROPS = True to regenerate them."
        )
        return

    pipe_area = model.wells[well.name].geometry.pipe_internal_A
    pc = model.physics.property_containers[0]
    time = np.asarray(time_data["time"], dtype=float)

    gas_mass_rate = np.full(time.size, np.nan)
    liquid_mass_rate = np.full(time.size, np.nan)
    gas_molar_rate = np.full(time.size, np.nan)
    liquid_molar_rate = np.full(time.size, np.nan)

    for step_idx, step_time in enumerate(time):
        segment_df = props_df[props_df["time"] == step_time].sort_values(
            "segment_local"
        )
        if segment_df.empty:
            continue

        sG = segment_df["sG"].to_numpy(dtype=float)
        rhoG = segment_df["rhoG"].to_numpy(dtype=float)
        rhoL = segment_df["rhoL"].to_numpy(dtype=float)
        vG = segment_df["vG"].to_numpy(dtype=float)
        vL = segment_df["vL"].to_numpy(dtype=float)
        if not np.isfinite(vG[0]):
            vG[0] = 0.0
        if not np.isfinite(vL[0]):
            vL[0] = 0.0

        sG_face = _face_value(sG, face_idx=0)
        rhoG_face = _face_value(rhoG, phase_saturation=sG, face_idx=0)
        rhoL_face = _face_value(rhoL, phase_saturation=1.0 - sG, face_idx=0)

        gas_mass_rate[step_idx] = vG[0] * pipe_area * sG_face * rhoG_face
        liquid_mass_rate[step_idx] = vL[0] * pipe_area * (1.0 - sG_face) * rhoL_face

        gas_mw = _phase_mw_from_mass_fractions(segment_df, model, "G", face_idx=0)
        liquid_mw = _phase_mw_from_mass_fractions(segment_df, model, "L", face_idx=0)
        gas_molar_rate[step_idx] = gas_mass_rate[step_idx] / gas_mw
        liquid_molar_rate[step_idx] = liquid_mass_rate[step_idx] / liquid_mw

    tag = f"well_{well.name}"
    time_data[f"{tag}_mass_rate_G_at_wh_from_velocity"] = gas_mass_rate
    time_data[f"{tag}_mass_rate_L_at_wh_from_velocity"] = liquid_mass_rate
    time_data[f"{tag}_molar_rate_G_at_wh_from_velocity"] = gas_molar_rate
    time_data[f"{tag}_molar_rate_L_at_wh_from_velocity"] = liquid_molar_rate

    for phase_name in pc.phases_name[: pc.np_fl]:
        for rate_kind in ("mass", "molar"):
            original_key = f"{tag}_{rate_kind}_rate_{phase_name}_at_wh"
            velocity_key = f"{original_key}_from_velocity"
            if original_key in time_data and velocity_key in time_data:
                time_data[f"{original_key}_minus_velocity"] = (
                    time_data[original_key] - time_data[velocity_key]
                )


def _plot_wellhead_rate_comparison(time_data, wells, figures_dir):
    time = time_data["time"]
    for rate_kind, unit in (("mass", "kg/day"), ("molar", "kmol/day")):
        for phase_name in ("G", "L"):
            fig, axes = plt.subplots(len(wells), 1, figsize=(9, 3.4 * len(wells)))
            axes = np.atleast_1d(axes)
            has_data = False
            for ax, well in zip(axes, wells, strict=False):
                tag = f"well_{well.name}_{rate_kind}_rate_{phase_name}"
                series = {
                    "at WH, operator": f"{tag}_at_wh",
                    "at WH, velocity": f"{tag}_at_wh_from_velocity",
                    "sum perforations": f"{tag}_by_sum_perfs",
                }
                for label, key in series.items():
                    if key not in time_data:
                        continue
                    ax.plot(time, time_data[key], label=label)
                    has_data = True
                ax.set_title(f"{well.name} {phase_name} {rate_kind} rate")
                ax.set_ylabel(unit)
                ax.grid(True, alpha=0.3)
                ax.legend()

            axes[-1].set_xlabel("Time, days")
            fig.tight_layout()
            if has_data:
                fig.savefig(
                    figures_dir
                    / f"dfm_wellhead_{phase_name}_{rate_kind}_rate_comparison.png",
                    dpi=200,
                )
            plt.close(fig)


def _plot_velocity_pcolormesh_no_mask(well_name, model, save_as="pdf"):
    output_dir = Path(model.output_folder) / f"heat_maps_pcolormesh_{well_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    props_path = Path(model.output_folder) / f"dfm_well_props_{well_name}.pkl"
    if not props_path.exists():
        print(
            f"Skipped no-mask velocity heat maps for {well_name}: {props_path} not found."
        )
        return

    props_df = pd.read_pickle(props_path)
    if not {"vG", "vL"}.issubset(props_df.columns):
        print(
            f"Skipped no-mask velocity heat maps for {well_name}: vG/vL columns not found."
        )
        return

    well_geometry = model.wells[well_name].geometry
    num_segments = int(well_geometry.num_segments)
    num_interfaces = int(well_geometry.num_interfaces)
    num_ts = len(props_df) // num_segments
    if num_ts == 0 or num_interfaces == 0:
        print(f"Skipped no-mask velocity heat maps for {well_name}: no velocity data.")
        return

    h5_well_dict = load_hdf5_to_dict(model.well_filepath)
    simulated_time = (
        np.asarray(h5_well_dict["dynamic"]["time"], dtype=float)[:num_ts] * 24 * 60 * 60
    )
    num_ts = min(num_ts, simulated_time.size)
    y_interfaces = np.asarray(well_geometry.z_interfaces, dtype=float)[:num_interfaces]

    for velocity_col, title, label, file_name in (
        (
            "vG",
            "Gas velocity profile along the wellbore over time",
            "Gas velocity [m/s]",
            "gas_velocity_no_mask",
        ),
        (
            "vL",
            "Liquid velocity profile along the wellbore over time",
            "Liquid velocity [m/s]",
            "liquid_velocity_no_mask",
        ),
    ):
        velocity_matrix = np.zeros((num_interfaces, num_ts))
        for ts_counter in range(num_ts):
            start = ts_counter * num_segments
            stop = (ts_counter + 1) * num_segments
            velocity = props_df[velocity_col].iloc[start:stop].to_numpy(dtype=float)
            velocity_matrix[:, ts_counter] = velocity[:num_interfaces] / (24 * 60 * 60)

        velocity_matrix = np.nan_to_num(
            velocity_matrix,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.pcolormesh(
            simulated_time,
            y_interfaces,
            velocity_matrix,
            cmap="jet",
            shading="auto",
        )
        ax.set_xlabel("Simulated time [second]", fontsize=14)
        ax.set_ylabel("Interface MD [meter]", fontsize=14)
        ax.tick_params(axis="both", labelsize=14)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=14, fontweight="bold")
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(label, fontsize=14)
        cbar.ax.tick_params(labelsize=14)
        plt.tight_layout()
        fig.savefig(output_dir / f"{file_name}.{save_as}")
        plt.close(fig)


def _plot_dfm_well_heat_maps(model, well_names=("I1",)):
    available_wells = {well.name for well in model.reservoir.wells}
    for well_name in well_names:
        if well_name not in available_wells:
            continue
        try:
            plot_heat_map_pcolormesh(well_name, model, show_plot=False)
        except PermissionError as exc:
            print(
                f"Skipped pcolormesh heat maps for {well_name}: {exc}. "
                "Close any open heat-map PDF files and rerun postprocess to "
                "regenerate them."
            )
        _plot_velocity_pcolormesh_no_mask(well_name, model)
        try:
            plot_heat_map_contourf(
                well_name,
                model,
                y_axis_tick_interval=250,
                show_plot=False,
            )
        except PermissionError as exc:
            print(
                f"Skipped contourf heat maps for {well_name}: {exc}. "
                "Close any open heat-map PDF files and rerun postprocess to "
                "regenerate them."
            )
        except ValueError as exc:
            contourf_messages = (
                "Contour levels must be increasing",
                "lower_level and upper_level cannot be NaN",
            )
            if not any(message in str(exc) for message in contourf_messages):
                raise
            print(
                f"Skipped contourf heat maps for {well_name}: {exc}. "
                "This usually happens when one plotted variable is constant "
                "or contains only NaN values; "
                "pcolormesh heat maps were still saved."
            )


def save_dfm_well_diagnostics(
    model,
    time_data,
    recompute_dfm_well_props=False,
):
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
        props_df = _save_indexed_dfm_well_props(
            model,
            well,
            time,
            depths,
            diagnostics_dir,
            recompute_dfm_well_props=recompute_dfm_well_props,
        )

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

        _add_velocity_based_wellhead_rates(time_data, props_df, model, well)

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
    _plot_wellhead_rate_comparison(time_data, dfm_wells, figures_dir)
    _plot_dfm_well_heat_maps(model, well_names=DFM_WELL_NAMES)


def producer_whp_ramp_target(model, time_day):
    if not ENABLE_PRODUCER_WHP_RAMP_DOWN:
        return None

    if PRODUCER_WHP_RAMP_START_BAR is None:
        p_start = float(model.stage2_config.producer_head_pressure)
    else:
        p_start = float(PRODUCER_WHP_RAMP_START_BAR)
    p_end = float(PRODUCER_WHP_RAMP_END_BAR)

    ramp_start = float(PRODUCER_WHP_RAMP_START_DAY)
    ramp_duration = float(PRODUCER_WHP_RAMP_DURATION_DAY)
    if ramp_duration <= 0.0 or time_day >= ramp_start + ramp_duration:
        return p_end
    if time_day <= ramp_start:
        return p_start

    fraction = (time_day - ramp_start) / ramp_duration
    return p_start + fraction * (p_end - p_start)


def producer_vent_then_hold_whp_target(time_day):
    if not ENABLE_PRODUCER_VENT_THEN_HOLD:
        return None

    if time_day < PRODUCER_VENT_DURATION_DAY:
        return PRODUCER_VENT_WHP_BAR
    return PRODUCER_POST_VENT_WHP_BAR


def producer_whp_target(model, time_day):
    vent_target = producer_vent_then_hold_whp_target(time_day)
    if vent_target is not None:
        return vent_target
    return producer_whp_ramp_target(model, time_day)


def apply_producer_whp_control(model):
    target = producer_whp_target(model, float(model.physics.engine.t))
    if target is None:
        return

    producer = next(
        (well for well in model.reservoir.wells if well.name == "P1"),
        None,
    )
    if producer is None:
        raise RuntimeError("P1 well was not found for WHP control.")

    model.physics.set_well_controls(
        wctrl=producer.control,
        control_type=well_control_iface.BHP,
        is_inj=False,
        target=target,
    )
    print(
        "P1 WHP target: "
        f"{target:.4g} bar at t={model.physics.engine.t:.6g} day"
    )


def build_model():
    config = Stage2DFMThermosiphonConfig(stage1_state_path=STAGE1_STATE_PATH)
    model = Model(config)
    model.init(platform="cpu")
    model.set_output(output_folder=str(OUTPUT_DIR), save_initial=False)
    print_initial_pipe_state(model, "I1")
    print_initial_pipe_state(model, "P1")
    return model


def run_simulation():
    redirect_darts_output(str(CASE_DIR / "run.log"))

    model = build_model()
    apply_producer_whp_control(model)

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

    for ith_step, dt in enumerate(REPORT_STEPS):
        if model.physics.engine.t < 0.1:
            model.data_ts.dt_max = 2e-4
        elif model.physics.engine.t < 0.5:
            model.data_ts.dt_max = 1e-3
        else:
            model.data_ts.dt_max = 1e-2

        apply_producer_whp_control(model)

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

    model.print_timers()
    model.print_stat()


def postprocess_existing_results():
    model = build_model()
    well_data_path = OUTPUT_DIR / model.output.well_filename

    if not well_data_path.exists():
        raise FileNotFoundError(
            f"{well_data_path} does not exist. Set RUN_SIMULATION = True first "
            "or point OUTPUT_DIR to an existing simulation output folder."
        )

    time_data = model.output.store_well_time_data(save_output_files=False)
    save_dfm_well_diagnostics(
        model,
        time_data,
        recompute_dfm_well_props=RECOMPUTE_DFM_WELL_PROPS,
    )


if RUN_SIMULATION:
    run_simulation()
else:
    postprocess_existing_results()
