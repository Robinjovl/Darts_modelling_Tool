import os
from copy import deepcopy
import numpy as np
import pandas as pd
from lgr_model_CMG import Model
from darts.engines import value_vector, redirect_darts_output
from darts.physics.base.operators_base import PropertyOperators as props
from output_cmg import get_physics_field, plot_xy_saturation_cmg, plot_xz_section, plot_well_time_data_2, plot_xy_plane
from darts.engines import well_control_iface


VTK_OUTPUT_PROPERTIES = [
    "pressure",
    "temperature",
    "CO2",
    "satG",
    "XCO2",
    "rhoG",
    "rhoAq",
    "muG",
    "muAq",
]
VTK_PROPERTY_RENAMES = {"CO2": "Z_CO2"}
VTK_STATIC_PROPERTIES = {
    "depth": lambda model: np.asarray(model.reservoir.mesh.depth, dtype=float),
    "permx": lambda model: np.asarray(model.reservoir.kx, dtype=float),
    "permy": lambda model: np.asarray(model.reservoir.ky, dtype=float),
    "permz": lambda model: np.asarray(model.reservoir.kz, dtype=float),
}

def make_cfg_lgr():
    cfg = {
        "grid":{
            "nx" : 81,
            "ny" : 81,
            "dx" :100,
            "dy" :100,
            "nz_res" : 40,
            "dz_res" : 5.0,
            "reservoir_top" : 2000,
        },
        "burden":{
            "over_thickness":2000,
            "under_thickness":2000,
            "dz_ratio" : 2,
            "poro_burden" : 1e-5,
            "perm_burden" : 1e-9,

        },
        "rock":{
            "perm_x":50, "perm_y" :50, "perm_z" : 50,
            "poro" : 0.1,
            "rcond_res" : 181.44, "hcap_res":2650.0,
            "rcond_over":149.54, "hcap_over":2347.29,
            "rcond_under" : 149.54, "hcap_under":2347.29,
        },

        "lgrs":{
            "lgr0":{
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid" :{"i_range": [41, 41],
                                                "j_range": [41, 41],
                                                "k_range": [1, 40],
                                                "refine": [7, 7, 1],
                                                "tag" : "inj"

                }
            },
            "lgr1":{
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid" :{"i_range": [36, 36],
                                                "j_range": [46, 46],
                                                "k_range": [1, 40],
                                                "refine": [7, 7, 1],
                                                "tag" : "prod"

                }
            },
            "lgr2":{
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid" :{"i_range": [46, 46],
                                                "j_range": [46, 46],
                                                "k_range": [1, 40],
                                                "refine": [7, 7, 1],
                                                "tag" : "prod"

                }
            },
            "lgr3":{
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid" :{"i_range": [36, 36],
                                                "j_range": [36, 36],
                                                "k_range": [1, 40],
                                                "refine": [7, 7, 1],
                                                "tag" : "prod"

                }
            },
            "lgr4":{
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid" :{"i_range": [46, 46],
                                                "j_range": [36, 36],
                                                "k_range": [1, 40],
                                                "refine": [7, 7, 1],
                                                "tag" : "prod"

                }
            }
        },

        "wells":{
           "I1": {"lgr": "lgr0", "k_from": 1, "k_to": 40},
           "P1": {"lgr": "lgr1", "k_from": 1, "k_to": 20},
           "P2": {"lgr": "lgr2", "k_from": 1, "k_to": 20},
           "P3": {"lgr": "lgr3", "k_from": 1, "k_to": 20},
           "P4": {"lgr": "lgr4", "k_from": 1, "k_to": 20},
        },

        "water_inj": {
            "W1": {"k_from": 1, "k_to": 40, "i0": 29, "j0": 53},
            "W2": {"k_from": 1, "k_to": 40, "i0": 53, "j0": 53},
            "W3": {"k_from": 1, "k_to": 40, "i0": 29, "j0": 29},
            "W4": {"k_from": 1, "k_to": 40, "i0": 53, "j0": 29}
        }
    }
    return cfg


def make_injection_only_cfg(cfg):
    """
    Keep the same LGR/reservoir mesh for restart compatibility, but only add
    the CO2 injector during the pre-production stage.
    """
    inj_cfg = deepcopy(cfg)
    inj_cfg["wells"] = {
        name: well_cfg
        for name, well_cfg in cfg["wells"].items()
        if name.startswith("I")
    }
    inj_cfg["water_inj"] = {}
    return inj_cfg

def true_reservoir_geometry(model):
    n_res_blocks = model.reservoir.mesh.n_res_blocks

    depth = np.asarray(model.reservoir.mesh.depth, dtype=float)[:n_res_blocks]
    poro = np.asarray(model.reservoir.mesh.poro, dtype=float)[:n_res_blocks]
    volume = np.asarray(model.reservoir.mesh.volume, dtype=float)[:n_res_blocks]

    reservoir_top = float(model.cfg["grid"]["reservoir_top"])
    reservoir_bottom = reservoir_top + float(model.cfg["grid"]["nz_res"]) * float(model.cfg["grid"]["dz_res"])
    reservoir_mask = (depth >= reservoir_top) & (depth < reservoir_bottom)

    if not np.any(reservoir_mask):
        raise ValueError("No valid true reservoir cells found")

    return reservoir_mask, poro, volume


def report_true_reservoir_volume(model, label="true reservoir"):
    reservoir_mask, poro, volume = true_reservoir_geometry(model)
    bulk_volume = float(np.sum(volume[reservoir_mask]))
    pore_volume = float(np.sum(poro[reservoir_mask] * volume[reservoir_mask]))
    n_blocks = int(np.count_nonzero(reservoir_mask))
    print(f"{label} bulk volume: {bulk_volume:.6e} m3")
    print(f"{label} pore volume: {pore_volume:.6e} m3")
    print(f"{label} block count: {n_blocks}")
    print(f"{label} max block volume: {float(np.max(volume[reservoir_mask])):.6e} m3")
    return bulk_volume, pore_volume


def cal_average_true_reservoir_pressure(model):
    n_vars = len(model.physics.vars)
    n_res_blocks = model.reservoir.mesh.n_res_blocks
    states = np.asarray(model.physics.engine.X, dtype=float).reshape((-1, n_vars))[:n_res_blocks]

    reservoir_mask, poro, volume = true_reservoir_geometry(model)

    pressure = states[reservoir_mask, 0]
    pore_volume = poro[reservoir_mask] * volume[reservoir_mask]
    if np.sum(pore_volume) <= 0:
        raise ValueError("No valid true reservoir cells found for average pressure calculation")

    return float(np.sum(pressure * pore_volume) / np.sum(pore_volume))


def cal_average_res_pre(model):
    return cal_average_true_reservoir_pressure(model)


def append_average_pressure_record(records, model, stage):
    records.append(
        {
            "stage": stage,
            "time_days": float(model.physics.engine.t),
            "time_years": float(model.physics.engine.t) / 365.0,
            "average_reservoir_pressure_bar": cal_average_res_pre(model),
        }
    )


def save_average_pressure_history(records, output_folder):
    if not records:
        return None
    path = os.path.join(output_folder, "average_reservoir_pressure.csv")
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"Average reservoir pressure history saved to: {path}")
    return path


def set_cmg_injection_controls(model):
    for w in model.reservoir.wells:
        if "I" not in w.name:
            continue
        model.physics.set_well_controls(
            wctrl=w.control,
            control_type=well_control_iface.MASS_RATE,
            is_inj=True,
            target=3.3264e7,
            inj_composition=[1.0 - model.zero],
            phase_name="CO2_rich",
            inj_temp=314.15,
        )
        model.physics.set_well_controls(
            wctrl=w.constraint,
            control_type=well_control_iface.BHP,
            is_inj=True,
            target=306.90,
            inj_composition=[1.0 - model.zero],
            inj_temp=314.15,
        )


def set_cmg_producer_zero_rate_controls(model):
    for w in model.reservoir.wells:
        if "P" not in w.name:
            continue
        model.physics.set_well_controls(
            wctrl=w.control,
            control_type=well_control_iface.MASS_RATE,
            is_inj=False,
            target=0.0,
        )


def set_cmg_producer_open_rate_controls(model):
    for w in model.reservoir.wells:
        if "P" not in w.name:
            continue
        model.physics.set_well_controls(
            wctrl=w.control,
            control_type=well_control_iface.MASS_RATE,
            is_inj=False,
            target=8.316e6,
        )
        model.physics.set_well_controls(
            wctrl=w.constraint,
            control_type=well_control_iface.BHP,
            is_inj=False,
            target=102.30,
        )


def set_cmg_water_pressure_controls(model, average_pressure, initial_average_pressure):
    for w in model.reservoir.wells:
        if "W" not in w.name:
            continue
        if average_pressure < initial_average_pressure:
            model.physics.set_well_controls(
                wctrl=w.control,
                control_type=well_control_iface.BHP,
                is_inj=True,
                target=initial_average_pressure,
                inj_composition=[model.zero],
                inj_temp=288.15,
            )
        else:
            model.physics.set_well_controls(
                wctrl=w.control,
                control_type=well_control_iface.MASS_RATE,
                is_inj=True,
                target=0,
                inj_composition=[model.zero],
                inj_temp=288.15,
            )


def set_cmg_stage2_controls(model, average_pressure, initial_average_pressure):
    set_cmg_injection_controls(model)
    set_cmg_producer_open_rate_controls(model)
    set_cmg_water_pressure_controls(model, average_pressure, initial_average_pressure)


def set_cmg_single_stage_controls(model, average_pressure, initial_average_pressure, time_days):
    set_cmg_injection_controls(model)
    if time_days < INJECTION_ONLY_DAYS:
        set_cmg_producer_zero_rate_controls(model)
    else:
        set_cmg_producer_open_rate_controls(model)
    set_cmg_water_pressure_controls(model, average_pressure, initial_average_pressure)


def generate_vtk_output(model, output_properties=None, output_directory=None):
    if output_properties is None:
        output_properties = list(
            dict.fromkeys(model.physics.vars + VTK_OUTPUT_PROPERTIES)
        )

    if output_directory is None:
        output_directory = os.path.join(model.output.output_folder, "vtk_files")

    os.makedirs(output_directory, exist_ok=True)
    timesteps, property_array = model.output.output_properties(
        output_properties=output_properties,
    )
    for old_name, new_name in VTK_PROPERTY_RENAMES.items():
        if old_name in property_array:
            property_array[new_name] = property_array.pop(old_name)
    for prop_name, prop_getter in VTK_STATIC_PROPERTIES.items():
        values = prop_getter(model)[:model.reservoir.mesh.n_res_blocks]
        property_array[prop_name] = np.tile(values, (len(timesteps), 1))

    model.output.output_to_vtk(
        output_directory=output_directory,
        output_data=[timesteps, property_array],
    )
    print(f"VTK files generated in: {output_directory}")


def plot_primary_sections(model, section_dir, tag, title_suffix="", use_lgr=True):
    os.makedirs(section_dir, exist_ok=True)
    prim = get_physics_field(model)
    reservoir_top = float(model.cfg["grid"]["reservoir_top"])
    reservoir_bottom = reservoir_top + float(model.cfg["grid"]["nz_res"]) * float(model.cfg["grid"]["dz_res"])

    for k in prim.keys():
        kk = k.lower()
        if ("pressure" in kk) or (kk == "p"):
            plot_xz_section(
                model, prim[k], use_lgr=use_lgr, zmin=reservoir_top, zmax=reservoir_bottom,
                savepath=os.path.join(section_dir, f"section_P_step_{tag}.png"),
                title=f"{k}{title_suffix}", logscale=False
            )
            break

    for k in prim.keys():
        if "co2" in k.lower():
            plot_xz_section(
                model, prim[k] - 1e-8, use_lgr=use_lgr, zmin=reservoir_top, zmax=reservoir_bottom,
                savepath=os.path.join(section_dir, f"section_CO2_step_{tag}.png"),
                title=f"CO2_delta{title_suffix}", logscale=False, vmin=0, vmax=1
            )
            break

    for k in prim.keys():
        kk = k.lower()
        if ("temp" in kk) or ("temperature" in kk):
            plot_xz_section(
                model, prim[k], use_lgr=use_lgr, zmin=reservoir_top, zmax=reservoir_bottom,
                savepath=os.path.join(section_dir, f"section_T_step_{tag}.png"),
                title=f"{k}{title_suffix}", logscale=False
            )
            break


def initialize_two_stage_restart_model(cfg, average_pressure_records):
    stage1_output_dir = f"{output_dir}_stage1_inj_only"
    os.makedirs(stage1_output_dir, exist_ok=True)
    redirect_darts_output(os.path.join(stage1_output_dir, "run.log"))

    injection_model = Model(make_injection_only_cfg(cfg))
    injection_model.init(platform="gpu")
    injection_model.set_output(output_folder=stage1_output_dir)
    stage1_section_dir = os.path.join(stage1_output_dir, "figures", "sections")
    initial_average_pressure = cal_average_res_pre(injection_model)
    append_average_pressure_record(average_pressure_records, injection_model, "stage1_injection_only")
    plot_primary_sections(
        injection_model,
        stage1_section_dir,
        f"{float(injection_model.physics.engine.t)}",
        title_suffix=" (stage 1 initial)",
        use_lgr=USE_LGR,
    )
    for _ in range(INJECTION_ONLY_STEPS):
        injection_model.run(Dt, save_reservoir_data=True)
        append_average_pressure_record(average_pressure_records, injection_model, "stage1_injection_only")
    injection_model.output.store_well_time_data(save_output_files=True)
    plot_primary_sections(
        injection_model,
        stage1_section_dir,
        f"{float(injection_model.physics.engine.t)}",
        title_suffix=" (stage 1 end)",
        use_lgr=USE_LGR,
    )
    generate_vtk_output(
        injection_model,
        output_directory=os.path.join(stage1_output_dir, "vtk_files"),
    )
    restart_file = os.path.join(stage1_output_dir, "reservoir_solution.h5")

    redirect_darts_output(os.path.join(output_dir, "run.log"))
    model = Model(cfg)
    model.init(platform="gpu", restart=True)
    model.set_output(output_folder=output_dir)
    model.load_restart_data(restart_file, ts_idx=-1)
    return model, initial_average_pressure


def initialize_single_stage_model(cfg):
    redirect_darts_output(os.path.join(output_dir, "run.log"))
    model = Model(cfg)
    model.init(platform="gpu")
    model.set_output(output_folder=output_dir)
    return model


USE_LGR=True
RUN_MODE = "single_stage_scheduled"  # "two_stage_restart" or "single_stage_scheduled"
Nt = 100
Dt = 365.0/2
INJECTION_ONLY_YEARS = 3.5
INJECTION_ONLY_DAYS = INJECTION_ONLY_YEARS * 365.0
INJECTION_ONLY_STEPS = int(round(INJECTION_ONLY_DAYS / Dt))
STAGE2_STEPS = max(Nt - INJECTION_ONLY_STEPS, 0)
SECTION_OUTPUT_STEP_MULTIPLIERS = {8, 10, 20, 40, 60, 80, 100}
output_dir = r"output_total_rate_control_single_stage"
FIG_DIR = os.path.join(output_dir, "figures")
SECTION_DIR = os.path.join(FIG_DIR, "sections")
WELL_DIR = os.path.join(FIG_DIR, "well_time_plots") # lowercase is better
os.makedirs(SECTION_DIR, exist_ok=True)
os.makedirs(WELL_DIR, exist_ok=True)

if __name__ == '__main__':
    average_pressure_records = []
    if USE_LGR:
        cfg = make_cfg_lgr()
        if RUN_MODE == "two_stage_restart":
            darts_model, initial_average_pressure = initialize_two_stage_restart_model(
                cfg,
                average_pressure_records,
            )
        elif RUN_MODE == "single_stage_scheduled":
            darts_model = initialize_single_stage_model(cfg)
            initial_average_pressure = cal_average_res_pre(darts_model)
        else:
            raise ValueError(
                f"Unknown RUN_MODE {RUN_MODE!r}. Use 'two_stage_restart' or 'single_stage_scheduled'."
            )
    else:
        darts_model = Model()
        redirect_darts_output('run.log')
        darts_model.init(platform="gpu") # cpu or GPU, PLATFORM
        darts_model.set_output(output_folder=output_dir)
        initial_average_pressure = cal_average_res_pre(darts_model)

    report_true_reservoir_volume(darts_model)
    print(f"Initial average reservoir pressure: {initial_average_pressure:.6f} bar")
    initial_stage_label = (
        "stage2_production"
        if RUN_MODE == "two_stage_restart"
        else "single_stage_injection_only"
    )
    append_average_pressure_record(average_pressure_records, darts_model, initial_stage_label)

    #plot initial condition
    prim0 = get_physics_field(darts_model)
    t0 = float(darts_model.physics.engine.t)

    # pressure
    for k in prim0.keys():
        kk = k.lower()
        if ("pressure" in kk) or (kk == "p"):
            plot_xz_section(
                darts_model, prim0[k], use_lgr=USE_LGR, zmin=2000, zmax=2200,
                savepath=os.path.join(SECTION_DIR, f"section_P_step_{t0}.png"),
                title=f"{k} (initial)", logscale=False
            )
            break

    # CO2 (if exists in prim)
    for k in prim0.keys():
        if "co2" in k.lower():
            plot_xz_section(
                darts_model, prim0[k] - 1e-8, use_lgr=USE_LGR, zmin=2000, zmax=2200,
                savepath=os.path.join(SECTION_DIR, f"section_CO2_step_{t0}.png"),
                title="CO2_delta (initial)", logscale=False, vmin=0, vmax=1
            )
            break

    # temperature
    for k in prim0.keys():
        kk = k.lower()
        if ("temp" in kk) or ("temperature" in kk):
            plot_xz_section(
                darts_model, prim0[k], use_lgr=USE_LGR, zmin=2000, zmax=2200,
                savepath=os.path.join(SECTION_DIR, f"section_T_step_{t0}.png"),
                title=f"{k} (initial)", logscale=False
            )

            break


    if True:
        reported_section_steps = set()
        n_run_steps = STAGE2_STEPS if RUN_MODE == "two_stage_restart" else Nt
        for t in range(n_run_steps):
            ave_pre = cal_average_res_pre(darts_model)
            if RUN_MODE == "two_stage_restart":
                set_cmg_stage2_controls(darts_model, ave_pre, initial_average_pressure)
                stage_label = "stage2_production"
            else:
                set_cmg_single_stage_controls(
                    darts_model,
                    ave_pre,
                    initial_average_pressure,
                    float(darts_model.physics.engine.t),
                )
                stage_label = (
                    "single_stage_injection_only"
                    if float(darts_model.physics.engine.t) < INJECTION_ONLY_DAYS
                    else "single_stage_production"
                )
            print(
                f"Time {darts_model.physics.engine.t:.2f} days, "
                f"mode={RUN_MODE}, stage={stage_label}, "
                f"Average reservoir pressure: {ave_pre:.2f} bar"
            )
            darts_model.run(Dt)
            pc = darts_model.physics.property_containers[0]

            t_end = float(darts_model.physics.engine.t)
            append_average_pressure_record(average_pressure_records, darts_model, stage_label)
            step_multiplier = int(round(t_end / Dt))

            if (
                step_multiplier in SECTION_OUTPUT_STEP_MULTIPLIERS
                and step_multiplier not in reported_section_steps
            ):
                reported_section_steps.add(step_multiplier)
                output_props = darts_model.physics.vars + darts_model.output.properties
                timesteps, property_array = darts_model.output.output_properties(
                    output_properties=VTK_OUTPUT_PROPERTIES, engine=True
                    )
                darts_model.output.save_property_array(timesteps, property_array)

                satG = property_array["satG"][0,:]
                muG = property_array["muG"][0,:]
                muAQ = property_array["muAq"][0,:]
                # XCO2 = property_array["XCO2"][0,:]
                rhoG = property_array["rhoG"][0,:]
                rhoAq = property_array["rhoAq"][0,:]
                # plot_xy_plane(
                #     darts_model, satG, depth=2002.5, use_lgr=USE_LGR,
                #     savepath=os.path.join(SECTION_DIR, f"section_satG_xy_step_{t_end}.png"),
                #     title=f"satG at step {t_end}", logscale=False, vmin=0, vmax=1
                # )
                plot_xy_saturation_cmg(
                            darts_model,
                            values=satG,
                            depth=2002.5,
                            use_lgr=USE_LGR,
                            savepath=os.path.join(SECTION_DIR, f"section_CO2_sat_xy_step_{t_end}_cmg.png"),
                            title="CO2 Saturation",
                            vmin=0.009,
                            vmax=0.695
                        )
                plot_xy_plane(
                    darts_model, satG, depth=2097.5, use_lgr=USE_LGR,
                    savepath=os.path.join(SECTION_DIR, f"section_satG_xy_step_{t_end} at bottom of Pro.png"),
                    title=f"satG at step {t_end}", logscale=False, vmin=0, vmax=1
                )
                plot_xy_plane(
                    darts_model, muG, depth=2097.5, use_lgr=USE_LGR,
                    savepath=os.path.join(SECTION_DIR, f"section_muG_xy_step_{t_end} at bottom of Producer.png"),
                    title=f"muG at step {t_end}", logscale=False
                )
                plot_xy_plane(
                    darts_model, muAQ, depth=2097.5, use_lgr=USE_LGR,
                    savepath=os.path.join(SECTION_DIR, f"section_muAq_xy_step_{t_end} at bottom of Producer.png"),
                    title=f"muAq at step {t_end}", logscale=False
                )
                plot_xy_plane(
                    darts_model, rhoAq, depth=2002.5, use_lgr=USE_LGR,
                    savepath=os.path.join(SECTION_DIR, f"section_rhoAq_xy_step_{t_end}.png"),
                    title=f"rhoAq at step {t_end}", logscale=False
                )

                prim = get_physics_field(darts_model)
                # pressure
                for k in prim.keys():
                    kk = k.lower()
                    if ("pressure" in kk) or (kk == "p"):
                        plot_xz_section(
                            darts_model, prim[k], use_lgr=USE_LGR, zmin=2000, zmax=2200,
                            savepath=os.path.join(SECTION_DIR, f"section_P_step_{t_end}.png"),
                            title=k, logscale=False
                        )
                        plot_xy_plane(
                            darts_model, prim[k], depth=2002.5, use_lgr=USE_LGR,
                            savepath=os.path.join(SECTION_DIR, f"section_P_xy_step_{t_end}.png"),
                            title=k, logscale=False
                        )
                        break

                # CO2
                for k in prim.keys():
                    kk = k.lower()
                    if "co2" in kk:
                        plot_xz_section(
                            darts_model, prim[k] - 1e-8, use_lgr=USE_LGR, zmin=2000, zmax=2200,
                            savepath=os.path.join(SECTION_DIR, f"section_CO2_step_{t_end}.png"),
                            title="CO2_delta", logscale=False, vmin=0, vmax=1
                        )
                        # plot_xy_plane(
                        #     darts_model, prim[k] - 1e-8, depth=2002.5, use_lgr=USE_LGR,
                        #     savepath=os.path.join(SECTION_DIR, f"section_CO2_xy_step_{t_end}.png"),
                        #     title="CO2_delta", logscale=False, vmin=0, vmax=1
                        # )

                        break


                # temperature
                for k in prim.keys():
                    kk = k.lower()
                    if ("temp" in kk) or ("temperature" in kk):
                        plot_xz_section(
                            darts_model, prim[k], use_lgr=USE_LGR, zmin=2000, zmax=2200,
                            savepath=os.path.join(SECTION_DIR, f"section_T_step_{t_end}.png"),
                            title=k, logscale=False
                        )
                        plot_xy_plane(
                            darts_model, prim[k], depth=2002.5, use_lgr=USE_LGR,
                            savepath=os.path.join(SECTION_DIR, f"section_T_xy_step_{t_end}.png"),
                            title=k, logscale=False
                        )
                        break
        time_data_dict = darts_model.output.store_well_time_data(save_output_files=True)
        time_data_df = pd.DataFrame.from_dict(time_data_dict)
        plot_well_time_data_2(darts_model, time_data_df, save_output_files=WELL_DIR)
        generate_vtk_output(darts_model)
        save_average_pressure_history(average_pressure_records, output_dir)



        n_vars = darts_model.physics.n_vars
        n_res_blocks = darts_model.reservoir.mesh.n_res_blocks

        X = np.array(darts_model.physics.engine.X, copy=False).reshape((-1, n_vars))
        volumes = np.array(darts_model.reservoir.mesh.volume, copy=False)[:n_res_blocks]
        poro = np.array(darts_model.reservoir.mesh.poro, copy=False)[:n_res_blocks]
        reservoir_mask, _, _ = true_reservoir_geometry(darts_model)

        pc = darts_model.physics.property_containers[0]

        co2_idx = pc.components_name.index("CO2")
        total_co2_kmol_sequestered = 0.0

        for i in np.flatnonzero(reservoir_mask):
            state = X[i, :]
            pc.evaluate(state)

            co2_cell_kmol_per_m3 = 0.0
            for ph in range(pc.np_fl):   # fluid phases only
                co2_cell_kmol_per_m3 += pc.sat[ph] * pc.dens_m[ph] * pc.x[ph, co2_idx]

            total_co2_kmol_sequestered += co2_cell_kmol_per_m3 * poro[i] * volumes[i]

        total_co2_mass_kg = total_co2_kmol_sequestered * 44.01

        print(f"Total CO2 in reservoir: {total_co2_kmol_sequestered:.6e} kmol")
        print(f"Total CO2 mass in reservoir: {total_co2_mass_kg:.6e} kg")
