import os
import numpy as np
import pandas as pd
import argparse

from Homo_model.lgr_homo_single_phase_egg import Model
from darts.engines import value_vector, redirect_darts_output
from darts.physics.base.operators_base import PropertyOperators as props
from drawing import get_physics_field, plot_xy_plane, plot_xz_section, plot_well_time_data_2, plot_average_res_pressure
from Auxiliary_functions import LGRInterfaceTransAnalyzer, FineEffectiveTransAnalyzer, cal_average_true_reservoir_pressure
from plot_diff_and_profile import make_difference_maps_batch, make_profiles_batch, get_property_key

def make_cfg_lgr():
    cfg = {
        "burden":{
            "over_thickness":500,
            "under_thickness":500,
            "poro_burden" : 1e-5,
            "perm_burden" : 1e-9,
            "rcond_over":149.54, "hcap_over":2347.29,
            "rcond_under" : 149.54, "hcap_under":2347.29,
        },

        "reservoir":{
            "nx":60,
            "ny":60,
            "nz":9,
            # "nz": 3,
            "poro": 0.2,
            "perm": 100,
            "dx" : 30,
            "dy" : 30,
            "dz" : 10,
            # "dz" : 70,
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
    Perm_file_name = "PERM8_ECL.INC"
    output_dir = "S-07_single_phase"
    Nt = 50
    Dt = 365.0
    USE_LGR = True
    Fine = False
    PLOT_PHASE = "CO2_rich"
    RHO_PROP = f"rho_{PLOT_PHASE}"
    MU_PROP = f"miu_{PLOT_PHASE}"
    ENTH_PROP = f"enth_{PLOT_PHASE}"

    FIG_DIR = os.path.join(output_dir, "figures")
    SECTION_DIR = os.path.join(FIG_DIR, "sections")
    WELL_DIR = os.path.join(FIG_DIR, "well_time_plots")

    PROP_FILE = os.path.join(output_dir, "property_array.h5")
    FIG = os.path.join(output_dir, "figures_extra")

    os.makedirs(SECTION_DIR, exist_ok=True)
    os.makedirs(WELL_DIR, exist_ok=True)

    if USE_LGR:
        cfg = make_cfg_lgr()
        # darts_model = Model(cfg, perm_file_name=Perm_file_name)
        darts_model = Model(cfg)
    else:
        # darts_model = Model(perm_file_name=Perm_file_name)
        darts_model = Model()

    redirect_darts_output(os.path.join(output_dir, "run.log"))
    darts_model.init(platform="cpu")

    darts_model.set_output(output_folder=output_dir)

    # initial_co2_kmol, initial_co2_mass_kg = calculate_total_co2_in_reservoir_single_phase(darts_model)

    # print(f"Initial CO2 in reservoir: {initial_co2_kmol:.6e} kmol")
    # print(f"Initial CO2 mass in reservoir: {initial_co2_mass_kg:.6e} kg")

    """check well index and perforation index"""
    # for w in darts_model.reservoir.wells:
    #     print(f"\nWell: {w.name}")
    #     for i, perf in enumerate(w.perforations):
    #         print(f"  perf {i}: {perf}")

    for w in darts_model.reservoir.wells:
        print(f"\nWell: {w.name}")
        total_wi = 0.0
        for i, perf in enumerate(w.perforations):
            print(f"  perf {i}: {perf}")
            total_wi += perf[2]
        print(f"  total WI = {total_wi}")


    avg_p_value = []
    avg_p_time = []
     #plot initial condition
    prim0 = get_physics_field(darts_model)
    # avg_p_time.append(float(darts_model.physics.engine.t))
    # avg_p_value.append(cal_average_true_reservoir_pressure(darts_model))

    prim0 = get_physics_field(darts_model)

    #Permeability
    if USE_LGR:
            plot_xz_section(
            darts_model, darts_model.reservoir.kx, use_lgr=USE_LGR, zmin=490, zmax=580,
            savepath=os.path.join(SECTION_DIR, f"section_kx_xz.png"),
            title=f"permeability kx", logscale=True,
        )
            plot_xy_plane(
                darts_model, darts_model.reservoir.kx, depth=535, use_lgr=USE_LGR,
                savepath=os.path.join(SECTION_DIR, f"section_kx_xy_middle.png"),
                title=f"permeability kx", logscale=True,
            )

    else:
        plot_xz_section(
            darts_model, darts_model.reservoir.global_data["permx"].reshape(-1, order="F"), use_lgr=USE_LGR, zmin=490, zmax=580,
            savepath=os.path.join(SECTION_DIR, f"section_kx_xz.png"),
            title=f"permeability kx", logscale=True, edgecolor="none", linewidth=0.0
        )
        plot_xy_plane(
            darts_model, darts_model.reservoir.global_data["permx"].reshape(-1, order="F"), depth=535, use_lgr=USE_LGR,
            savepath=os.path.join(SECTION_DIR, f"section_kx_xy_middle.png"),
            title=f"permeability kx ", logscale=True,
        )
        plot_xz_section(
            darts_model, darts_model.reservoir.global_data["permy"].reshape(-1, order="F"), use_lgr=USE_LGR, zmin=490, zmax=580,
            savepath=os.path.join(SECTION_DIR, f"section_ky_xz.png"),
            title=f"permeability ky", logscale=True, edgecolor="none", linewidth=0.0
        )
        plot_xy_plane(
            darts_model, darts_model.reservoir.global_data["permy"].reshape(-1, order="F"), depth=535, use_lgr=USE_LGR,
            savepath=os.path.join(SECTION_DIR, f"section_ky_xy_middle.png"),
            title=f"permeability ky ", logscale=True,
        )

    # pressure
    for k in prim0.keys():
        kk = k.lower()
        if ("pressure" in kk) or (kk == "p"):
            plot_xz_section(
                darts_model, prim0[k], use_lgr=USE_LGR, zmin=490, zmax=580,
                savepath=os.path.join(SECTION_DIR, f"section_P_XZ_injector.png"),
                title=f"{k} (initial)", logscale=False, edgecolor="none", linewidth=0.0
            )


            plot_xy_plane(
                darts_model, prim0[k], depth=535, use_lgr=USE_LGR,
                savepath=os.path.join(SECTION_DIR, f"section_P_xy_middle.png"),
                title=f"{k} (initial)", logscale=False, edgecolor="none", linewidth=0.0
            )
            break


    Tmin = 10+ 273.15
    Tmax = 35+ 273.15

    # Single-phase PT formulation: temperature and thermophysical outputs are used for plotting.
    _, initial_property_array = darts_model.output.output_properties(
        output_properties=["temperature", ENTH_PROP, RHO_PROP, MU_PROP],
        engine=True,
    )
    temperature0 = initial_property_array["temperature"][0, :]
    enthalpy0 = initial_property_array[ENTH_PROP][0, :]
    rho0 = initial_property_array[RHO_PROP][0, :]
    mu0 = initial_property_array[MU_PROP][0, :]
    t0_vmin, t0_vmax = get_padded_limits(temperature0)

    plot_xz_section(
        darts_model, temperature0, use_lgr=USE_LGR, zmin=490, zmax=580,
        savepath=os.path.join(SECTION_DIR, f"section_temperature_initial_xz_injector.png"),
        title="temperature (initial)", vmin=t0_vmin, vmax=t0_vmax,
        logscale=False, edgecolor="none", linewidth=0.0
    )

    plot_xy_plane(
        darts_model, temperature0, depth=535, use_lgr=USE_LGR,
        savepath=os.path.join(SECTION_DIR, f"section_temperature_initial_xy_middle.png"),
        title="temperature (initial)", logscale=False, vmin=t0_vmin, vmax=t0_vmax,
        edgecolor="none", linewidth=0.0
    )

    plot_xz_section(
        darts_model, enthalpy0, use_lgr=USE_LGR, zmin=490, zmax=580,
        savepath=os.path.join(SECTION_DIR, f"section_enthalpy_initial_xz_injector.png"),
        title="enthalpy (initial)",
        logscale=False, edgecolor="none", linewidth=0.0
    )

    plot_xy_plane(
        darts_model, enthalpy0, depth=535, use_lgr=USE_LGR,
        savepath=os.path.join(SECTION_DIR, f"section_enthalpy_initial_xy_middle.png"),
        title="enthalpy (initial)", logscale=False,
        edgecolor="none", linewidth=0.0
    )

    plot_xz_section(
        darts_model, rho0, use_lgr=USE_LGR, zmin=490, zmax=580,
        savepath=os.path.join(SECTION_DIR, f"section_{RHO_PROP}_initial_xz_injector.png"),
        title=f"{RHO_PROP} (initial)", logscale=False,
        edgecolor="none", linewidth=0.0
    )

    plot_xy_plane(
        darts_model, rho0, depth=535, use_lgr=USE_LGR,
        savepath=os.path.join(SECTION_DIR, f"section_{RHO_PROP}_initial_xy_middle.png"),
        title=f"{RHO_PROP} (initial)", logscale=False,
        edgecolor="none", linewidth=0.0
    )

    plot_xz_section(
        darts_model, mu0, use_lgr=USE_LGR, zmin=490, zmax=580,
        savepath=os.path.join(SECTION_DIR, f"section_{MU_PROP}_initial_xz_injector.png"),
        title=f"{MU_PROP} (initial)", logscale=False,
        edgecolor="none", linewidth=0.0
    )

    plot_xy_plane(
        darts_model, mu0, depth=535, use_lgr=USE_LGR,
        savepath=os.path.join(SECTION_DIR, f"section_{MU_PROP}_initial_xy_middle.png"),
        title=f"{MU_PROP} (initial)", logscale=False,
        edgecolor="none", linewidth=0.0
    )

    target_days = {365, 2*365, 5*365, 6*365, 10*365, 11*365, 15*365, 16*365, 20*365, 21*365, 30*365, 31*365, 40*365, 41*365, 49*365, 50*365}
    if True:
        for t in range(Nt):
            darts_model.run(Dt)


            darts_model.print_timers()
            darts_model.print_stat()

            t_end = float(darts_model.physics.engine.t)
            # avg_p_time.append(t_end)
            # avg_p_value.append(cal_average_true_reservoir_pressure(darts_model))

            if int(round(t_end)) in target_days:
                timesteps, property_array = darts_model.output.output_properties(
                    output_properties=[RHO_PROP, "temperature", "pressure"],
                    engine=True
                )
                darts_model.output.save_property_array(timesteps, property_array)

            if t_end == Dt or t_end ==Dt*5 or t_end == Dt*10 or t_end == Dt*15 or t_end == Dt*20 or t_end == Dt*25 or t_end == Dt*30  or t_end == Dt*40 or t_end == Dt*50:
                output_props = darts_model.physics.vars + darts_model.output.properties
                timesteps, property_array = darts_model.output.output_properties(
                    output_properties = [RHO_PROP, MU_PROP, "temperature", ENTH_PROP],engine=True
                    )
                darts_model.output.save_property_array(timesteps, property_array)

                mu_co2 = property_array[MU_PROP][0,:]
                rho_co2 = property_array[RHO_PROP][0,:]
                temperature = property_array["temperature"][0,:]

                plot_xy_plane(
                    darts_model, mu_co2, depth=535, use_lgr=USE_LGR,
                    savepath=os.path.join(SECTION_DIR, f"section_{MU_PROP}_xy_{t_end}_middle.png"),
                    title=MU_PROP, logscale=False, edgecolor="none", linewidth=0.0
                )
                plot_xy_plane(
                    darts_model, rho_co2, depth=535, use_lgr=USE_LGR,
                    savepath=os.path.join(SECTION_DIR, f"section_{RHO_PROP}_xy_{t_end}_middle.png"),
                    title=RHO_PROP, logscale=False, edgecolor="none", linewidth=0.0
                )

                plot_xz_section(
                    darts_model, mu_co2, use_lgr=USE_LGR, zmin=490, zmax=580,
                    savepath=os.path.join(SECTION_DIR, f"section_{MU_PROP}_xz_{t_end}_injector.png"),
                    title=MU_PROP, logscale=False, edgecolor="none", linewidth=0.0
                )

                plot_xy_plane(
                    darts_model, temperature, depth=535, use_lgr=USE_LGR,
                    savepath=os.path.join(SECTION_DIR, f"section_temperature_xy_{t_end}_middle.png"),
                    title="temperature", logscale=False, vmin=Tmin, vmax=Tmax, edgecolor="none", linewidth=0.0
                )

                plot_xz_section(
                    darts_model, temperature, use_lgr=USE_LGR, zmin=490, zmax=580,
                    savepath=os.path.join(SECTION_DIR, f"section_temperature_xz_{t_end}_injector.png"),
                    title="temperature", vmin=Tmin, vmax=Tmax, edgecolor="none", linewidth=0.0,
                    logscale=False
                )



                prim = get_physics_field(darts_model)
                print("vars:", list(prim.keys()))
                for k in prim.keys():
                    if "pressure" in k.lower() or k.lower() == "p":
                        plot_xz_section(darts_model, prim[k],use_lgr=USE_LGR ,zmin=490, zmax=580,
                                        savepath=os.path.join(SECTION_DIR, f"section_P_xz_{t_end}_injector.png"),
                                        title=k,
                                        logscale=False, edgecolor="none", linewidth=0.0)

                        plot_xy_plane(
                                    darts_model, prim[k], depth=535, use_lgr=USE_LGR,
                                    savepath=os.path.join(SECTION_DIR, f"section_P_xy_{t_end}_middle.png"),
                                    title=k, logscale=False, edgecolor="none", linewidth=0.0
                                )
                        break


                for k in prim.keys():
                    if "co2" in k.lower() :
                        plot_xz_section(darts_model, prim[k] - 1e-8, use_lgr=USE_LGR, zmin=490, zmax=580,
                                        savepath=os.path.join(SECTION_DIR, f"section_CO2_xz_{t_end}.png"),
                                        title="CO2_delta",
                                        logscale=False,
                                        vmin=0, vmax=1)
                        plot_xy_plane(
                                    darts_model, prim[k] - 1e-8, depth=535, use_lgr=USE_LGR,
                                    savepath=os.path.join(SECTION_DIR, f"section_CO2_xy_{t_end}_middle.png"),
                                    title="CO2_delta", logscale=False, vmin=0, vmax=1
                                )
                        break

        # compute well time data

        time_data_dict = darts_model.output.store_well_time_data(save_output_files=True)
        time_data_df = pd.DataFrame.from_dict(time_data_dict)
        print(type(darts_model.output.reservoir))
        print(darts_model.output.reservoir.output_to_vtk.__qualname__)
        print(darts_model.output.reservoir.output_to_vtk.__module__)
        darts_model.output.output_to_vtk()

        plot_well_time_data_2(darts_model, time_data_df,save_output_files=WELL_DIR)
        # plot_average_res_pressure(avg_p_time, avg_p_value, save_dir=WELL_DIR)


        """material balance check: calculate total CO2 mass in reservoir"""
        # final_co2_kmol, final_co2_mass_kg = calculate_total_co2_in_reservoir_single_phase(darts_model)

        # print(f"Final CO2 in reservoir: {final_co2_kmol:.6e} kmol")
        # print(f"Final CO2 mass in reservoir: {final_co2_mass_kg:.6e} kg")

        # print(f"Delta CO2 in reservoir: {final_co2_kmol - initial_co2_kmol:.6e} kmol")
        # print(f"Delta CO2 mass in reservoir: {final_co2_mass_kg - initial_co2_mass_kg:.6e} kg")
        '''tramsmissibility analysis'''
        if USE_LGR:
            analyzer = LGRInterfaceTransAnalyzer(darts_model)
            all_res = analyzer.summarize_all_lgrs()
            all_faces = analyzer.summarize_face
            print(all_faces)

            print(all_res["summary_df"])
            all_res["summary_df"].to_excel(
                os.path.join(output_dir, "lgr_interface_summary.xlsx"),
                index=False
            )


        if Fine:
            analyzer = FineEffectiveTransAnalyzer(
                darts_model=darts_model,
                nx=300,
                ny=300,
                nz=9,
                patch_size=5,
                patch_center_1b=(208, 148),
                reservoir_k0_range=range(1, 8),
                n_nb_cols= 5,
            )

            res_avg = analyzer.effective_trans_all_faces(
                mobility_mode="cell_a",
                ref_mu_mode="interface_avg",
                agg_mode="average",
            )
            print(res_avg["left"]["per_layer_df"])
            print(res_avg["up"]["per_layer_df"])

            print(res_avg["summary_df"])
            res_avg["summary_df"].to_excel(
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
        rho_key = get_property_key(prop, [RHO_PROP, "rhoG", "rho_LCO2", "rho_V"])
        temp_key = get_property_key(prop, ["temperature", "temp", "T"])
        pres_key = get_property_key(prop, ["pressure", "p"])

        rho_arr = prop[rho_key]
        temp_arr = prop[temp_key]
        pres_arr = prop[pres_key]

        # -----------------------------------------------------
        # 3) difference maps
        # -----------------------------------------------------
        year_pairs = [(1, 2), (5, 6), (10, 11), (15, 16), (20, 21), (30, 31), (40, 41), (49, 50)]

        make_difference_maps_batch(
            model=darts_model,
            time_vector=time_vector,
            prop_array=rho_arr,
            out_dir=os.path.join(FIG, "rho_diff_maps"),
            prop_name=rho_key,
            year_pairs=year_pairs,
            depth=535.0,
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
            depth=535.0,
            use_lgr=USE_LGR,
            cmap="coolwarm",
            symmetric=True,
            colorbar_label="Temperature difference [K]",
        )

        # -----------------------------------------------------
        # 4) 1D profiles between wells
        # -----------------------------------------------------
        years = (1, 5, 10, 15, 20, 30, 40, 50)

        make_profiles_batch(
            model=darts_model,
            time_vector=time_vector,
            prop_array=temp_arr,
            out_dir=os.path.join(FIG, "profiles"),
            prop_name=temp_key,
            years=years,
            depth=535.0,
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
            depth=535.0,
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
            depth=535.0,
            use_lgr=USE_LGR,
            ylabel="Pressure [bar]",
        )

        print("All extra figures have been generated.")
