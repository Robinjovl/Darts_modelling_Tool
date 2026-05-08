import argparse
import os
from pathlib import Path

import pandas as pd

from Auxiliary_functions import cal_average_true_reservoir_pressure
from darts.engines import redirect_darts_output
from drawing import (
    get_physics_field,
    plot_average_res_pressure,
    plot_well_time_data_2,
    plot_xy_plane,
    plot_xz_section,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def make_cfg_lgr():
    return {
        "burden": {
            "poro_burden": 1e-5,
            "perm_burden": 1e-9,
            "rcond_over": 149.54,
            "hcap_over": 2347.29,
            "rcond_under": 149.54,
            "hcap_under": 2347.29,
        },
        "reservoir": {
            "nx": 60,
            "ny": 60,
            "nz": 9,
            "poro": 0.2,
            "permx": 800.0,
            "permy": 800.0,
            "permz": 80.0,
            "rcond": 2.1 * 86.4,
            "hcap": 2200.0,
            "dx": 30,
            "dy": 30,
            "dz": 10,
        },
        "lgrs": {
            "lgr0": {
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid": {
                    "i_range": [42, 42],
                    "j_range": [30, 30],
                    "k_range": [1, 7],
                    "refine": [5, 5, 1],
                    "dx_vec": [6, 6, 6, 6, 6],
                    "dy_vec": [6, 6, 6, 6, 6],
                    "tag": "inj",
                },
            },
            "lgr1": {
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid": {
                    "i_range": [19, 19],
                    "j_range": [30, 30],
                    "k_range": [1, 7],
                    "refine": [5, 5, 1],
                    "dx_vec": [6, 6, 6, 6, 6],
                    "dy_vec": [6, 6, 6, 6, 6],
                    "tag": "prod",
                },
            },
        },
        "wells": {
            "I1": {"lgr": "lgr0", "k_from": 1, "k_to": 7},
            "P1": {"lgr": "lgr1", "k_from": 1, "k_to": 7},
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the pure CO2 LGR Egg model with injector and producer open from t=0."
    )
    parser.add_argument("--nt", type=int, default=50, help="Number of report steps.")
    parser.add_argument("--dt", type=float, default=365.0, help="Report step size in days.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "output_lgr_pure_co2"),
        help="Output directory.",
    )
    parser.add_argument("--platform", type=str, default="cpu", help="DARTS platform.")
    parser.add_argument(
        "--no-vtk",
        action="store_true",
        help="Skip final VTK export.",
    )
    return parser.parse_args()


def unique_names(names):
    seen = set()
    out = []
    for name in names:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def get_output_property_names(model):
    names = list(model.physics.vars)
    container = model.physics.property_containers[0]
    names.extend(container.output_props.keys())
    return unique_names(names)


def get_primary_field(primary_fields, candidates, required=True):
    for name in candidates:
        for key, values in primary_fields.items():
            if key.lower() == name.lower():
                return key, values
    if required:
        raise KeyError(
            f"None of {candidates} found in primary variables: {list(primary_fields.keys())}"
        )
    return None, None


def get_property_field(property_array, candidates, required=True):
    for name in candidates:
        for key, values in property_array.items():
            if key.lower() == name.lower():
                return key, values[0, :]
    if required:
        raise KeyError(
            f"None of {candidates} found in property outputs: {list(property_array.keys())}"
        )
    return None, None


def plot_primary_xy_xz(
    model,
    values,
    basename,
    title,
    section_dir,
    xy_depth,
    zmin,
    zmax,
    vmin=None,
    vmax=None,
):
    plot_xz_section(
        model,
        values,
        use_lgr=True,
        zmin=zmin,
        zmax=zmax,
        savepath=os.path.join(section_dir, f"{basename}_xz.png"),
        title=title,
        logscale=False,
        vmin=vmin,
        vmax=vmax,
        edgecolor="none",
        linewidth=0.0,
    )
    plot_xy_plane(
        model,
        values,
        depth=xy_depth,
        use_lgr=True,
        savepath=os.path.join(section_dir, f"{basename}_xy_middle.png"),
        title=title,
        logscale=False,
        vmin=vmin,
        vmax=vmax,
        edgecolor="none",
        linewidth=0.0,
    )


def plot_property_xy_xz(
    model,
    values,
    property_name,
    tag,
    section_dir,
    xy_depth,
    zmin,
    zmax,
    vmin=None,
    vmax=None,
):
    plot_xz_section(
        model,
        values,
        use_lgr=True,
        zmin=zmin,
        zmax=zmax,
        savepath=os.path.join(section_dir, f"section_{property_name}_{tag}_xz.png"),
        title=f"{property_name} ({tag})",
        logscale=False,
        vmin=vmin,
        vmax=vmax,
        edgecolor="none",
        linewidth=0.0,
    )
    plot_xy_plane(
        model,
        values,
        depth=xy_depth,
        use_lgr=True,
        savepath=os.path.join(section_dir, f"section_{property_name}_{tag}_xy_middle.png"),
        title=f"{property_name} ({tag})",
        logscale=False,
        vmin=vmin,
        vmax=vmax,
        edgecolor="none",
        linewidth=0.0,
    )


def output_properties(model, output_property_names):
    timesteps, property_array = model.output.output_properties(
        output_properties=output_property_names,
        engine=True,
    )
    model.output.append_properties_to_reservoir(float(model.physics.engine.t), property_array)
    return timesteps, property_array


def plot_state(model, tag, section_dir, xy_depth, zmin, zmax, temperature_limits=None):
    primary = get_physics_field(model)
    pressure_key, pressure = get_primary_field(primary, ["pressure", "p"])
    temperature_key, temperature = get_primary_field(primary, ["temperature", "temp", "T"])

    plot_primary_xy_xz(
        model,
        pressure,
        f"section_{pressure_key}_{tag}",
        f"{pressure_key} ({tag})",
        section_dir,
        xy_depth,
        zmin,
        zmax,
    )
    plot_primary_xy_xz(
        model,
        temperature,
        f"section_{temperature_key}_{tag}",
        f"{temperature_key} ({tag})",
        section_dir,
        xy_depth,
        zmin,
        zmax,
        vmin=None if temperature_limits is None else temperature_limits[0],
        vmax=None if temperature_limits is None else temperature_limits[1],
    )
    return primary


def plot_selected_properties(
    model,
    property_array,
    tag,
    section_dir,
    xy_depth,
    zmin,
    zmax,
):
    rho_name, rho = get_property_field(property_array, ["rho_CO2_rich"], required=False)
    mu_name, mu = get_property_field(property_array, ["miu_CO2_rich"], required=False)
    sat_name, sat = get_property_field(property_array, ["sat_CO2_rich"], required=False)
    enthalpy_name, enthalpy = get_property_field(property_array, ["enth_CO2_rich"], required=False)

    if rho_name is not None:
        plot_property_xy_xz(model, rho, rho_name, tag, section_dir, xy_depth, zmin, zmax)
    if mu_name is not None:
        plot_property_xy_xz(model, mu, mu_name, tag, section_dir, xy_depth, zmin, zmax)
    if sat_name is not None:
        plot_property_xy_xz(model, sat, sat_name, tag, section_dir, xy_depth, zmin, zmax, 0.0, 1.0)
    if enthalpy_name is not None:
        plot_property_xy_xz(model, enthalpy, enthalpy_name, tag, section_dir, xy_depth, zmin, zmax)


def plot_permeability(model, section_dir, xy_depth, zmin, zmax):
    plot_xz_section(
        model,
        model.reservoir.kx,
        use_lgr=True,
        zmin=zmin,
        zmax=zmax,
        savepath=os.path.join(section_dir, "section_kx_xz.png"),
        title="permeability kx",
        logscale=True,
        edgecolor="none",
        linewidth=0.0,
    )
    plot_xy_plane(
        model,
        model.reservoir.kx,
        depth=xy_depth,
        use_lgr=True,
        savepath=os.path.join(section_dir, "section_kx_xy_middle.png"),
        title="permeability kx",
        logscale=True,
        edgecolor="none",
        linewidth=0.0,
    )


def main():
    args = parse_args()

    from Homo_model_pure_co2.lgr_homo_single_phase_egg import Model

    output_dir = os.path.abspath(args.output_dir)
    figures_dir = os.path.join(output_dir, "figures")
    section_dir = os.path.join(figures_dir, "sections")
    well_dir = os.path.join(figures_dir, "well_time_plots")

    os.makedirs(section_dir, exist_ok=True)
    os.makedirs(well_dir, exist_ok=True)

    cfg = make_cfg_lgr()
    model = Model(cfg)

    redirect_darts_output(os.path.join(output_dir, "run.log"))
    model.init(platform=args.platform)
    model.set_output(output_folder=output_dir)

    # Explicitly set controls before the first timestep: this pure-CO2 script is a doublet from t=0.
    model.set_well_controls()

    plot_zmin = 1990.0
    plot_zmax = 2080.0
    plot_xy_depth = 2035.0
    temperature_limits = (35.0 + 273.15, 90.0 + 273.15)

    output_property_names = get_output_property_names(model)
    avg_p_time = [float(model.physics.engine.t)]
    avg_p_value = [cal_average_true_reservoir_pressure(model)]

    plot_permeability(model, section_dir, plot_xy_depth, plot_zmin, plot_zmax)
    plot_state(
        model,
        "initial",
        section_dir,
        plot_xy_depth,
        plot_zmin,
        plot_zmax,
    )

    timesteps, property_array = output_properties(model, output_property_names)
    model.output.save_property_array(timesteps, property_array)
    plot_selected_properties(
        model,
        property_array,
        "initial",
        section_dir,
        plot_xy_depth,
        plot_zmin,
        plot_zmax,
    )

    plot_days = {
        365,
        2 * 365,
        5 * 365,
        10 * 365,
        15 * 365,
        20 * 365,
        30 * 365,
        40 * 365,
        50 * 365,
    }
    saved_property_days = set()

    for _ in range(args.nt):
        model.set_well_controls()
        model.run(args.dt)

        t_end = float(model.physics.engine.t)
        report_day = int(round(t_end))
        report_tag = f"{report_day}d"
        avg_p_time.append(t_end)
        avg_p_value.append(cal_average_true_reservoir_pressure(model))

        if report_day in plot_days:
            plot_state(
                model,
                report_tag,
                section_dir,
                plot_xy_depth,
                plot_zmin,
                plot_zmax,
                temperature_limits=temperature_limits,
            )
            timesteps, property_array = output_properties(model, output_property_names)
            model.output.save_property_array(timesteps, property_array)
            saved_property_days.add(report_day)
            plot_selected_properties(
                model,
                property_array,
                report_tag,
                section_dir,
                plot_xy_depth,
                plot_zmin,
                plot_zmax,
            )

        print(f"Finished report step at t={t_end:.2f} days")
        model.print_timers()
        model.print_stat()

    final_day = int(round(float(model.physics.engine.t)))
    if final_day not in saved_property_days:
        timesteps, property_array = output_properties(model, output_property_names)
        model.output.save_property_array(timesteps, property_array)

    time_data_dict = model.output.store_well_time_data(save_output_files=True)
    time_data_df = pd.DataFrame.from_dict(time_data_dict)
    plot_well_time_data_2(
        model,
        time_data_df,
        save_output_files=well_dir,
        component="CO2_rich",
    )
    plot_average_res_pressure(avg_p_time, avg_p_value, save_dir=well_dir)

    if not args.no_vtk:
        model.output.output_to_vtk(output_properties=output_property_names)


if __name__ == "__main__":
    main()
