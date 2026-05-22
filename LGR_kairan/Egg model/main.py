import os
from copy import deepcopy
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from Heter_model_aquifer.lgr_aquifer import Model

from darts.engines import value_vector, redirect_darts_output
from darts.physics.base.operators_base import PropertyOperators as props
from drawing import get_physics_field, plot_xy_plane, plot_xz_section, plot_well_time_data_2, plot_average_res_pressure
from Auxiliary_functions import LGRInterfaceTransAnalyzer, FineEffectiveTransAnalyzer, cal_average_true_reservoir_pressure
from Auxiliary_functions import AquiferPhaseFineEffectiveTransAnalyzer, AquiferPhaseLGRInterfaceTransAnalyzer
from plot_diff_and_profile import make_difference_maps_batch, make_profiles_batch, get_property_key


def make_cfg_lgr():
    cfg = {
        "burden":{
            "poro_burden" : 1e-5,
            "perm_burden" : 1e-9,
            "rcond_over":149.54, "hcap_over":2347.29,
            "rcond_under" : 149.54, "hcap_under":2347.29,
        },

        "reservoir":{
            "nx":60,
            "ny":60,
            "nz":9,
            "poro": 0.2,
            "permx": 800.0,
            "permy": 800.0,
            "permz": 80.0,
            "rcond": 2.1 * 86.4,
            "hcap": 2200.0,
            "dx" : 30,
            "dy" : 30,
            "dz" : 10,
        },

        "lgrs":{
            "lgr0":{
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid" :{"i_range": [42, 42],
                                                "j_range": [30, 30],
                                                "k_range": [1, 7],
                                                "refine": [5, 5, 1],
                                                "dx_vec": [6,6,6,6,6],
                                                "dy_vec": [6,6,6,6,6],
                                                "tag" : "inj"

                }
            },
            "lgr1":{
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid" :{"i_range": [19, 19],
                                                "j_range": [30, 30],
                                                "k_range": [1, 7],
                                                "refine": [5, 5, 1],
                                                "dx_vec": [6,6,6,6,6],
                                                "dy_vec": [6,6,6,6,6],
                                                "tag" : "prod"

                }
            }
        },
        "wells":{
           "I1": {"lgr": "lgr0", "k_from": 1, "k_to": 7},
           "P1": {"lgr": "lgr1", "k_from": 1, "k_to": 7},
        }
    }
    return cfg


def make_injector_only_cfg(cfg):
    """
    Keep the same reservoir and LGR layout, but remove producer wells for the
    preparatory injection-only stage.
    """
    inj_cfg = deepcopy(cfg)
    inj_cfg["wells"] = {
        name: well_cfg
        for name, well_cfg in cfg["wells"].items()
        if name.startswith("I")
    }
    return inj_cfg


def get_padded_limits(values, pad_fraction=0.05):
    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None, None

    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * pad_fraction, 1.0)
    else:
        pad = (vmax - vmin) * pad_fraction
    return vmin - pad, vmax + pad


def get_primary_field(primary_fields, candidates):
    for name in candidates:
        for key, values in primary_fields.items():
            if key.lower() == name.lower():
                return key, values
    raise KeyError(f"None of {candidates} found in primary variables: {list(primary_fields.keys())}")


def plot_primary_xy_xz(model, values, basename, title, section_dir, xy_depth, zmin, zmax,
                       use_lgr=True, vmin=None, vmax=None):
    plot_xz_section(
        model, values, use_lgr=use_lgr, zmin=zmin, zmax=zmax,
        savepath=os.path.join(section_dir, f"{basename}_xz.png"),
        title=title, logscale=False, vmin=vmin, vmax=vmax,
        edgecolor="none", linewidth=0.0,
    )
    plot_xy_plane(
        model, values, depth=xy_depth, use_lgr=use_lgr,
        savepath=os.path.join(section_dir, f"{basename}_xy_middle.png"),
        title=title, logscale=False, vmin=vmin, vmax=vmax,
        edgecolor="none", linewidth=0.0,
    )


def cal_average_reservoir_pressure(model, use_lgr=True):
    if use_lgr:
        return cal_average_true_reservoir_pressure(model)

    n_vars = len(model.physics.vars)
    n_res_blocks = model.reservoir.mesh.n_res_blocks
    states = np.asarray(model.physics.engine.X, dtype=float).reshape((-1, n_vars))[:n_res_blocks]

    nx, ny = int(model.reservoir.nx), int(model.reservoir.ny)
    k_index = np.arange(n_res_blocks) // (nx * ny)
    reservoir_mask = (k_index >= 1) & (k_index < 8)

    pressure = states[reservoir_mask, 0]
    poro = np.asarray(model.reservoir.mesh.poro, dtype=float)[:n_res_blocks][reservoir_mask]
    volume = np.asarray(model.reservoir.mesh.volume, dtype=float)[:n_res_blocks][reservoir_mask]
    pore_volume = poro * volume
    return float(np.sum(pressure * pore_volume) / np.sum(pore_volume))


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--perm-file", type=str, required=True,
#                         help="Permeability include file name, e.g. PERM1_ECL.INC")
#     parser.add_argument("--output-dir", type=str, required=True,
#                         help="Output directory for this run")
#     parser.add_argument("--use-lgr", action="store_true",
#                         help="Use LGR model")
#     parser.add_argument("--nt", type=int, default=50)
#     parser.add_argument("--dt", type=float, default=365.0)
#     return parser.parse_args()



if __name__ == '__main__':

    # args = parse_args()
    # Perm_file_name = args.perm_file
    # output_dir = args.output_dir
    # Nt = args.nt
    # Dt = args.dt
    # USE_LGR = args.use_lgr

    Perm_file_name = "PERM66_ECL.INC"
    Nt=5
    Dt = 365.0
    INJECTION_ONLY_STEPS = 3
    USE_LGR = True
    Fine = False
    lgr_output_dir = "output_perm66_correct_depth"
    fine_output_dir = "output_fine_VLAq_3yr_inj_perm66"
    coarse_output_dir = "output_coarse_VLAq_3yr_inj_perm66"
    output_dir = lgr_output_dir if USE_LGR else fine_output_dir if Fine else coarse_output_dir
    Refine = (5, 5, 1)
    RHO_PROP = "rhoG"
    RHO_AQ_PROP = "rhoAq"
    MU_PROP = "muG"
    MU_AQ_PROP = "muAq"
    SAT_PROP = "satG"
    PROP_OUTPUTS = [RHO_PROP, RHO_AQ_PROP, MU_PROP, MU_AQ_PROP, "temperature", SAT_PROP, "pressure", "CO2"]
    PLOT_ZMIN = 1990
    PLOT_ZMAX = 2080
    PLOT_XY_DEPTH = 2035

    FIG_DIR = os.path.join(output_dir, "figures")
    SECTION_DIR = os.path.join(FIG_DIR, "sections")
    WELL_DIR = os.path.join(FIG_DIR, "well_time_plots")

    PROP_FILE = os.path.join(output_dir, "property_array.h5")
    FIG = os.path.join(output_dir, "figures_extra")

    os.makedirs(SECTION_DIR, exist_ok=True)
    os.makedirs(WELL_DIR, exist_ok=True)

    cfg = make_cfg_lgr()
    if USE_LGR:
        stage1_output_dir = f"{output_dir}_stage1_inj_only"
        os.makedirs(stage1_output_dir, exist_ok=True)
        stage1_log_path = os.path.join(stage1_output_dir, "run.log")

        cfg_inj_only = make_injector_only_cfg(cfg)
        redirect_darts_output(stage1_log_path)
        injection_model = Model(cfg_inj_only,perm_file_name=Perm_file_name)
        injection_model.init(platform="cpu")
        print("raw T data:", injection_model.lgr_fc_debug_df)
        injection_model.set_output(output_folder=stage1_output_dir)
        injection_model.run(
            INJECTION_ONLY_STEPS * Dt,
            save_reservoir_data=True,
        )
        injection_model.output.store_well_time_data(save_output_files=True)
        restart_file = os.path.join(stage1_output_dir, "reservoir_solution.h5")

        stage2_log_path = os.path.join(output_dir, "run.log")
        redirect_darts_output(stage2_log_path)
        darts_model = Model(cfg,perm_file_name=Perm_file_name)
        darts_model.init(platform="cpu", restart=True)
        darts_model.set_output(output_folder=output_dir)
        darts_model.load_restart_data(restart_file, ts_idx=-1)
        del injection_model
    elif Fine:
        stage1_output_dir = f"{output_dir}_stage1_inj_only"
        os.makedirs(stage1_output_dir, exist_ok=True)
        stage1_log_path = os.path.join(stage1_output_dir, "run.log")

        redirect_darts_output(stage1_log_path)
        injection_model = Model(include_producer=False, perm_file_name=Perm_file_name)
        injection_model.init(platform="cpu")
        injection_model.set_output(output_folder=stage1_output_dir)
        injection_model.run(
            INJECTION_ONLY_STEPS * Dt,
            save_reservoir_data=True,
        )
        injection_model.output.store_well_time_data(save_output_files=True)
        restart_file = os.path.join(stage1_output_dir, "reservoir_solution.h5")

        run_log_path = os.path.join(output_dir, "run.log")
        redirect_darts_output(run_log_path)
        darts_model = Model(include_producer=True, perm_file_name=Perm_file_name)
        darts_model.init(platform="cpu", restart=True)
        darts_model.set_output(output_folder=output_dir)
        darts_model.load_restart_data(restart_file, ts_idx=-1)
        del injection_model
    else:
        stage1_output_dir = f"{output_dir}_stage1_inj_only"
        os.makedirs(stage1_output_dir, exist_ok=True)
        stage1_log_path = os.path.join(stage1_output_dir, "run.log")

        redirect_darts_output(stage1_log_path)
        injection_model = Model(include_producer=False)
        injection_model.init(platform="cpu")
        injection_model.set_output(output_folder=stage1_output_dir)
        injection_model.run(
            INJECTION_ONLY_STEPS * Dt,
            save_reservoir_data=True,
        )
        injection_model.output.store_well_time_data(save_output_files=True)
        restart_file = os.path.join(stage1_output_dir, "reservoir_solution.h5")

        run_log_path = os.path.join(output_dir, "run.log")
        redirect_darts_output(run_log_path)
        darts_model = Model(include_producer=True)
        darts_model.init(platform="cpu", restart=True)
        darts_model.set_output(output_folder=output_dir)
        darts_model.load_restart_data(restart_file, ts_idx=-1)
        del injection_model

    # initial_co2_kmol, initial_co2_mass_kg = calculate_total_co2_in_reservoir_single_phase(darts_model)

    # print(f"Initial CO2 in reservoir: {initial_co2_kmol:.6e} kmol")
    # print(f"Initial CO2 mass in reservoir: {initial_co2_mass_kg:.6e} kg")

    """check well index and perforation index"""
    # for w in darts_model.reservoir.wells:
    #     print(f"\nWell: {w.name}")
    #     for i, perf in enumerate(w.perforations):
    #         print(f"  perf {i}: {perf}")

    # for w in darts_model.reservoir.wells:
    #     print(f"\nWell: {w.name}")
    #     total_wi = 0.0
    #     for i, perf in enumerate(w.perforations):
    #         print(f"  perf {i}: {perf}")
    #         total_wi += perf[2]
    #     print(f"  total WI = {total_wi}")


    avg_p_value = []
    avg_p_time = []
     #plot initial condition
    prim0 = get_physics_field(darts_model)
    pressure_key, pressure0 = get_primary_field(prim0, ["pressure", "p"])
    temperature_key, temperature0 = get_primary_field(prim0, ["temperature", "temp", "T"])
    bulk_co2_key, bulk_co2_0 = get_primary_field(prim0, ["CO2"])
    avg_p_time.append(float(darts_model.physics.engine.t))
    avg_p_value.append(cal_average_reservoir_pressure(darts_model, use_lgr=USE_LGR))

    #Permeability
    if USE_LGR:
            plot_xz_section(
            darts_model, darts_model.reservoir.kx, use_lgr=USE_LGR, zmin=PLOT_ZMIN, zmax=PLOT_ZMAX,
            savepath=os.path.join(SECTION_DIR, f"section_kx_xz.png"),
            title=f"permeability kx", logscale=True,
        )
            plot_xy_plane(
                darts_model, darts_model.reservoir.kx, depth=PLOT_XY_DEPTH, use_lgr=USE_LGR,
                savepath=os.path.join(SECTION_DIR, f"section_kx_xy_middle.png"),
                title=f"permeability kx", logscale=True,
            )

    else:
        plot_xz_section(
            darts_model, darts_model.reservoir.global_data["permx"].reshape(-1, order="F"), use_lgr=USE_LGR, zmin=PLOT_ZMIN, zmax=PLOT_ZMAX,
            savepath=os.path.join(SECTION_DIR, f"section_kx_xz.png"),
            title=f"permeability kx", logscale=True, edgecolor="none", linewidth=0.0
        )
        plot_xy_plane(
            darts_model, darts_model.reservoir.global_data["permx"].reshape(-1, order="F"), depth=PLOT_XY_DEPTH, use_lgr=USE_LGR,
            savepath=os.path.join(SECTION_DIR, f"section_kx_xy_middle.png"),
            title=f"permeability kx ", logscale=True,
        )


    Tmin = 25 + 273.15
    Tmax = 50 + 273.15
    t0_vmin, t0_vmax = get_padded_limits(temperature0)

    plot_primary_xy_xz(
        darts_model, pressure0, "section_P_initial", f"{pressure_key} (initial)",
        SECTION_DIR, PLOT_XY_DEPTH, PLOT_ZMIN, PLOT_ZMAX, use_lgr=USE_LGR,
    )
    plot_primary_xy_xz(
        darts_model, temperature0, "section_temperature_initial", f"{temperature_key} (initial)",
        SECTION_DIR, PLOT_XY_DEPTH, PLOT_ZMIN, PLOT_ZMAX, use_lgr=USE_LGR,
        vmin=t0_vmin, vmax=t0_vmax,
    )
    plot_primary_xy_xz(
        darts_model, bulk_co2_0, "section_bulk_CO2_initial", f"{bulk_co2_key} bulk molar fraction (initial)",
        SECTION_DIR, PLOT_XY_DEPTH, PLOT_ZMIN, PLOT_ZMAX, use_lgr=USE_LGR,
        vmin=0.0, vmax=1.0,
    )

    _, initial_property_array = darts_model.output.output_properties(
        output_properties=PROP_OUTPUTS,
        engine=True,
    )
    darts_model.output.append_properties_to_reservoir(
        float(darts_model.physics.engine.t),
        initial_property_array,
    )
    rho0 = initial_property_array[RHO_PROP][0, :]
    mu0 = initial_property_array[MU_PROP][0, :]
    satg0 = initial_property_array[SAT_PROP][0, :]

    plot_xz_section(
        darts_model, rho0, use_lgr=USE_LGR, zmin=PLOT_ZMIN, zmax=PLOT_ZMAX,
        savepath=os.path.join(SECTION_DIR, f"section_{RHO_PROP}_initial_xz_injector.png"),
        title=f"{RHO_PROP} (initial)", logscale=False,
        edgecolor="none", linewidth=0.0
    )

    plot_xy_plane(
        darts_model, rho0, depth=PLOT_XY_DEPTH, use_lgr=USE_LGR,
        savepath=os.path.join(SECTION_DIR, f"section_{RHO_PROP}_initial_xy_middle.png"),
        title=f"{RHO_PROP} (initial)", logscale=False,
        edgecolor="none", linewidth=0.0
    )

    plot_xz_section(
        darts_model, mu0, use_lgr=USE_LGR, zmin=PLOT_ZMIN, zmax=PLOT_ZMAX,
        savepath=os.path.join(SECTION_DIR, f"section_{MU_PROP}_initial_xz_injector.png"),
        title=f"{MU_PROP} (initial)", logscale=False,
        edgecolor="none", linewidth=0.0
    )

    plot_xy_plane(
        darts_model, mu0, depth=PLOT_XY_DEPTH, use_lgr=USE_LGR,
        savepath=os.path.join(SECTION_DIR, f"section_{MU_PROP}_initial_xy_middle.png"),
        title=f"{MU_PROP} (initial)", logscale=False,
        edgecolor="none", linewidth=0.0
    )

    plot_xz_section(
        darts_model, satg0, use_lgr=USE_LGR, zmin=PLOT_ZMIN, zmax=PLOT_ZMAX,
        savepath=os.path.join(SECTION_DIR, f"section_{SAT_PROP}_initial_xz_injector.png"),
        title=f"{SAT_PROP} (initial)", logscale=False, vmin=0.0, vmax=1.0,
        edgecolor="none", linewidth=0.0
    )

    plot_xy_plane(
        darts_model, satg0, depth=PLOT_XY_DEPTH, use_lgr=USE_LGR,
        savepath=os.path.join(SECTION_DIR, f"section_{SAT_PROP}_initial_xy_middle.png"),
        title=f"{SAT_PROP} (initial)", logscale=False, vmin=0.0, vmax=1.0,
        edgecolor="none", linewidth=0.0
    )




    target_days = {365, 2 * 365, 5 * 365, 6 * 365, 10 * 365, 11 * 365, 15 * 365, 16 * 365, 19 * 365, 20 * 365}
    plot_days = {365, 2 * 365, 5 * 365, 10 * 365, 15 * 365, 20 * 365, 30 * 365, 40 * 365, 50 * 365}
    run_steps = max(Nt - INJECTION_ONLY_STEPS, 0)
    if True:
        for t in range(run_steps):
            darts_model.run(Dt)
            darts_model.print_timers()
            darts_model.print_stat()

            t_end = float(darts_model.physics.engine.t)
            report_day = int(round(t_end))
            avg_p_time.append(t_end)
            avg_p_value.append(cal_average_reservoir_pressure(darts_model, use_lgr=USE_LGR))

            if report_day in plot_days:
                prim = get_physics_field(darts_model)
                pressure_key, pressure = get_primary_field(prim, ["pressure", "p"])
                temperature_key, temperature = get_primary_field(prim, ["temperature", "temp", "T"])
                bulk_co2_key, bulk_co2 = get_primary_field(prim, ["CO2"])
                report_tag = f"{report_day}d"
                print("vars:", list(prim.keys()))

                plot_primary_xy_xz(
                    darts_model, pressure, f"section_P_{report_tag}", pressure_key,
                    SECTION_DIR, PLOT_XY_DEPTH, PLOT_ZMIN, PLOT_ZMAX, use_lgr=USE_LGR,
                )
                plot_primary_xy_xz(
                    darts_model, temperature, f"section_temperature_{report_tag}", temperature_key,
                    SECTION_DIR, PLOT_XY_DEPTH, PLOT_ZMIN, PLOT_ZMAX, use_lgr=USE_LGR,
                    vmin=Tmin, vmax=Tmax,
                )
                plot_primary_xy_xz(
                    darts_model, bulk_co2, f"section_bulk_CO2_{report_tag}", f"{bulk_co2_key} bulk molar fraction",
                    SECTION_DIR, PLOT_XY_DEPTH, PLOT_ZMIN, PLOT_ZMAX, use_lgr=USE_LGR,
                    vmin=0.0, vmax=1.0,
                )

            if report_day in target_days:
                timesteps, property_array = darts_model.output.output_properties(
                    output_properties=PROP_OUTPUTS,
                    engine=True
                )
                darts_model.output.save_property_array(timesteps, property_array)

            if report_day in plot_days:
                output_props = darts_model.physics.vars + darts_model.output.properties
                timesteps, property_array = darts_model.output.output_properties(
                    output_properties=PROP_OUTPUTS, engine=True
                    )
                darts_model.output.save_property_array(timesteps, property_array)

                mu_co2 = property_array[MU_PROP][0,:]
                rho_co2 = property_array[RHO_PROP][0,:]
                satg = property_array[SAT_PROP][0, :]


                plot_xy_plane(
                    darts_model, mu_co2, depth=PLOT_XY_DEPTH, use_lgr=USE_LGR,
                    savepath=os.path.join(SECTION_DIR, f"section_{MU_PROP}_xy_{t_end}_middle.png"),
                    title=MU_PROP, logscale=False, edgecolor="none", linewidth=0.0
                )
                plot_xy_plane(
                    darts_model, rho_co2, depth=PLOT_XY_DEPTH, use_lgr=USE_LGR,
                    savepath=os.path.join(SECTION_DIR, f"section_{RHO_PROP}_xy_{t_end}_middle.png"),
                    title=RHO_PROP, logscale=False, edgecolor="none", linewidth=0.0
                )

                plot_xz_section(
                    darts_model, mu_co2, use_lgr=USE_LGR, zmin=PLOT_ZMIN, zmax=PLOT_ZMAX,
                    savepath=os.path.join(SECTION_DIR, f"section_{MU_PROP}_xz_{t_end}_injector.png"),
                    title=MU_PROP, logscale=False, edgecolor="none", linewidth=0.0
                )

                plot_xy_plane(
                    darts_model, temperature, depth=PLOT_XY_DEPTH, use_lgr=USE_LGR,
                    savepath=os.path.join(SECTION_DIR, f"section_temperature_xy_{t_end}_middle.png"),
                    title="temperature", logscale=False, vmin=Tmin, vmax=Tmax, edgecolor="none", linewidth=0.0
                )

                plot_xz_section(
                    darts_model, temperature, use_lgr=USE_LGR, zmin=PLOT_ZMIN, zmax=PLOT_ZMAX,
                    savepath=os.path.join(SECTION_DIR, f"section_temperature_xz_{t_end}_injector.png"),
                    title="temperature", vmin=Tmin, vmax=Tmax, edgecolor="none", linewidth=0.0,
                    logscale=False
                )

                plot_xy_plane(
                    darts_model, satg, depth=PLOT_XY_DEPTH, use_lgr=USE_LGR,
                    savepath=os.path.join(SECTION_DIR, f"section_{SAT_PROP}_xy_{t_end}_middle.png"),
                    title=SAT_PROP, logscale=False, vmin=0.0, vmax=1.0, edgecolor="none", linewidth=0.0
                )

                plot_xz_section(
                    darts_model, satg, use_lgr=USE_LGR, zmin=PLOT_ZMIN, zmax=PLOT_ZMAX,
                    savepath=os.path.join(SECTION_DIR, f"section_{SAT_PROP}_xz_{t_end}_injector.png"),
                    title=SAT_PROP, logscale=False, vmin=0.0, vmax=1.0, edgecolor="none", linewidth=0.0
                )

        # compute well time data

        time_data_dict = darts_model.output.store_well_time_data(save_output_files=True)
        time_data_df = pd.DataFrame.from_dict(time_data_dict)
        print(type(darts_model.output.reservoir))
        print(darts_model.output.reservoir.output_to_vtk.__qualname__)
        print(darts_model.output.reservoir.output_to_vtk.__module__)
        darts_model.output.output_to_vtk(output_properties=PROP_OUTPUTS)

        plot_well_time_data_2(darts_model, time_data_df,save_output_files=WELL_DIR)
        plot_average_res_pressure(avg_p_time, avg_p_value, save_dir=WELL_DIR)


        """material balance check: calculate total CO2 mass in reservoir"""
        # final_co2_kmol, final_co2_mass_kg = calculate_total_co2_in_reservoir_single_phase(darts_model)

        # print(f"Final CO2 in reservoir: {final_co2_kmol:.6e} kmol")
        # print(f"Final CO2 mass in reservoir: {final_co2_mass_kg:.6e} kg")

        # print(f"Delta CO2 in reservoir: {final_co2_kmol - initial_co2_kmol:.6e} kmol")
        # print(f"Delta CO2 mass in reservoir: {final_co2_mass_kg - initial_co2_mass_kg:.6e} kg")
        '''tramsmissibility analysis'''
        if USE_LGR:
            analyzer = AquiferPhaseLGRInterfaceTransAnalyzer(darts_model)
            df = analyzer.phase_effective_trans_per_layer_interfaces(
                mobility_mode="interface_avg"
            )

            print(df)
            df.to_excel(
                os.path.join(output_dir, "lgr_interface_summary.xlsx"),
                index=False
            )


        if Fine:
            analyzer = AquiferPhaseFineEffectiveTransAnalyzer(
                darts_model=darts_model,
                nx=300,
                ny=300,
                nz=9,
                patch_size=5,
                patch_center_1b=(208, 148),
                reservoir_k0_range=range(1, 8),
                n_nb_cols=5,
                lgrs=cfg["lgrs"],
                parent_refine=(5, 5, 1),
            )

            df = analyzer.phase_effective_trans_per_layer_interfaces_for_lgrs(
                mobility_mode="interface_avg"
            )

            print(df)
            df.to_excel(
                os.path.join(output_dir, "fine_effective_trans_summary.xlsx"),
                index=False
            )

        # plotting
        # -----------------------------------------------------
        # 2) load property arrays
        # -----------------------------------------------------
        time_vector, prop = darts_model.output.load_property_array(PROP_FILE)

        # choose actual keys saved in your file
        # common candidates:
        rho_key = get_property_key(prop, [RHO_PROP, "rhoG"])
        temp_key = get_property_key(prop, ["temperature", "temp", "T"])
        pres_key = get_property_key(prop, ["pressure", "p"])

        rho_arr = prop[rho_key]
        temp_arr = prop[temp_key]
        pres_arr = prop[pres_key]

        # -----------------------------------------------------
        # 3) difference maps
        # -----------------------------------------------------
        year_pairs = [(2, 5), (5, 6), (10, 11), (15, 16), (19, 20)]

        make_difference_maps_batch(
            model=darts_model,
            time_vector=time_vector,
            prop_array=rho_arr,
            out_dir=os.path.join(FIG, "rho_diff_maps"),
            prop_name=rho_key,
            year_pairs=year_pairs,
            depth=PLOT_XY_DEPTH,
            use_lgr=USE_LGR,
            cmap="coolwarm",
            symmetric=True,
            colorbar_label="Density difference [kg/m$^3$]",
        )

        make_difference_maps_batch(
            model=darts_model,
            time_vector=time_vector,
            prop_array=temp_arr,
            out_dir=os.path.join(FIG, "temperature_diff_maps"),
            prop_name=temp_key,
            year_pairs=year_pairs,
            depth=PLOT_XY_DEPTH,
            use_lgr=USE_LGR,
            cmap="coolwarm",
            symmetric=True,
            colorbar_label="Temperature difference [K]",
        )

        # -----------------------------------------------------
        # 4) 1D profiles between wells
        # -----------------------------------------------------
        years = (2, 5, 10, 15, 20,30, 40, 50)

        make_profiles_batch(
            model=darts_model,
            time_vector=time_vector,
            prop_array=temp_arr,
            out_dir=os.path.join(FIG, "profiles"),
            prop_name=temp_key,
            years=years,
            depth=PLOT_XY_DEPTH,
            use_lgr=USE_LGR,
            ylabel="Temperature [K]",
        )

        make_profiles_batch(
            model=darts_model,
            time_vector=time_vector,
            prop_array=rho_arr,
            out_dir=os.path.join(FIG, "profiles"),
            prop_name=rho_key,
            years=years,
            depth=PLOT_XY_DEPTH,
            use_lgr=USE_LGR,
            ylabel="Density [kg/m$^3$]",
        )

        make_profiles_batch(
            model=darts_model,
            time_vector=time_vector,
            prop_array=pres_arr,
            out_dir=os.path.join(FIG, "profiles"),
            prop_name=pres_key,
            years=years,
            depth=PLOT_XY_DEPTH,
            use_lgr=USE_LGR,
            ylabel="Pressure [bar]",
        )

        print("All extra figures have been generated.")
