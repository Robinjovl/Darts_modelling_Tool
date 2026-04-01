import os
import numpy as np
import pandas as pd
from lgr_model import Model
from darts.engines import value_vector, redirect_darts_output
from darts.physics.base.operators_base import PropertyOperators as props
from output import get_physics_field, plot_xy_saturation_cmg, plot_xz_section, plot_well_time_data_2, plot_xy_plane
from darts.engines import well_control_iface

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

# def cal_average_res_pre(m):
#     n_res = m.reservoir.mesh.n_res_blocks
#     n_vars = len(m.physics.vars)
#     X = np.asarray(m.physics.engine.X, dtype=float)
#     P = X.reshape((-1, n_vars))[:n_res, 0]
#     return float(P.mean())

def cal_average_res_pre(m):
    n_res = m.reservoir.mesh.n_res_blocks
    n_vars = len(m.physics.vars)

    X = np.asarray(m.physics.engine.X, dtype=float)
    P = X.reshape((-1, n_vars))[:n_res, 0]

    poro = np.array(m.reservoir.mesh.poro, copy=False)[:n_res]
    volume = np.array(m.reservoir.mesh.volume, copy=False)[:n_res]

    pv = poro * volume
    return float(np.sum(P * pv) / np.sum(pv))

USE_LGR=True
Nt = 100
Dt = 366/2
output_dir = r"output_lgr"
FIG_DIR = os.path.join(output_dir, "figures")
SECTION_DIR = os.path.join(FIG_DIR, "sections")
WELL_DIR = os.path.join(FIG_DIR, "well_time_plots") # lowercase is better
os.makedirs(SECTION_DIR, exist_ok=True)
os.makedirs(WELL_DIR, exist_ok=True)

if __name__ == '__main__':
    if USE_LGR:
        cfg = make_cfg_lgr()
        darts_model = Model(cfg)
    else:
        darts_model = Model()

    # darts_model.params.linear_type = darts_model.params.linear_solver_t.cpu_superlu
    redirect_darts_output('run.log')
    darts_model.init(platform="gpu") # cpu or GPU, PLATFORM
    darts_model.set_output(output_folder="output_lgr")

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
        for t in range(Nt):
            if t < 7:
                for i, w in enumerate(darts_model.reservoir.wells):
                    if "I" in w.name:
                        # injector: MASS_RATE with BHP constraint
                        darts_model.physics.set_well_controls(
                            wctrl=w.control,
                            control_type=well_control_iface.MASS_RATE,
                            # control_type=well_control_iface.VOLUMETRIC_RATE,
                            is_inj=True,
                            # target=4.32e4,
                            target=3.3264e7,
                            inj_composition=[1.0 - darts_model.zero],
                            phase_name="CO2_rich",
                            inj_temp=314.15
                        )
                        darts_model.physics.set_well_controls(
                            wctrl=w.constraint,
                            control_type=well_control_iface.BHP,
                            is_inj=True,
                            target=306.90,
                            inj_composition=[1.0 - darts_model.zero],
                            inj_temp=314.15
                        )
                    if "P" in w.name:
                        # producers: closed
                        darts_model.physics.set_well_controls(
                            wctrl=w.control,
                            control_type=well_control_iface.MASS_RATE,
                            is_inj=False,
                            target=0,
                            phase_name="aqueous"
                        )

            else:
                for i, w in enumerate(darts_model.reservoir.wells):
                    if "I" in w.name:
                        # injector: MASS_RATE with BHP constraint
                        darts_model.physics.set_well_controls(
                            wctrl=w.control,
                            control_type=well_control_iface.MASS_RATE,
                            # control_type=well_control_iface.VOLUMETRIC_RATE,
                            is_inj=True,
                            # target=4.32e4,
                            target=3.3264e7,
                            phase_name="CO2_rich",
                            inj_composition=[1.0 - darts_model.zero],
                            inj_temp=314.15
                        )
                        darts_model.physics.set_well_controls(
                            wctrl=w.constraint,
                            control_type=well_control_iface.BHP,
                            is_inj=True,
                            target=306.90,
                            inj_composition=[1.0 - darts_model.zero],
                            inj_temp=314.15
                        )
                    if "P" in w.name:
                        # producers: MASS_RATE with BHP constraint
                        darts_model.physics.set_well_controls(
                            wctrl=w.control,
                            control_type=well_control_iface.MASS_RATE,
                            is_inj=False,
                            target=8.316e6,
                            phase_name="CO2_rich"
                        )
                        darts_model.physics.set_well_controls(
                            wctrl=w.constraint,
                            control_type=well_control_iface.BHP,
                            is_inj=False,
                            target=102.30
                        )
            ave_pre = cal_average_res_pre(darts_model)
            print(f"Time {darts_model.physics.engine.t:.2f} days, Average reservoir pressure: {ave_pre:.2f} bar")
            darts_model.run(Dt)
            pc = darts_model.physics.property_containers[0]
    
            if ave_pre < 205.1:
                for i, w in enumerate(darts_model.reservoir.wells):
                    if "W" in w.name:
                        darts_model.physics.set_well_controls(
                            wctrl=w.control,
                            control_type=well_control_iface.BHP,
                            is_inj=True,
                            target=205.1,
                            inj_composition=[darts_model.zero],
                            inj_temp=288.15
                        )
            else:
                for i, w in enumerate(darts_model.reservoir.wells):
                    if "W" in w.name:
                        darts_model.physics.set_well_controls(
                            wctrl=w.control,
                            control_type=well_control_iface.MASS_RATE,
                            is_inj=True,
                            target=0,
                            phase_name="aqueous",
                            inj_composition=[darts_model.zero],
                            inj_temp=288.15
                        )

            t_end = float(darts_model.physics.engine.t)

            if t_end == Dt or t_end ==Dt*7 or t_end == Dt*10 or t_end == Dt*20 or t_end == Dt*40 or t_end == Dt*60 or t_end == Dt*80 or t_end == Dt*100:                
                output_props = darts_model.physics.vars + darts_model.output.properties
                timesteps, property_array = darts_model.output.output_properties(
                    output_properties = ["satG", "XCO2", "rhoG", "rhoAq", "muG", "muAq"],engine=True
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

        

        n_vars = darts_model.physics.n_vars
        n_res_blocks = darts_model.reservoir.mesh.n_res_blocks

        X = np.array(darts_model.physics.engine.X, copy=False).reshape((-1, n_vars))
        volumes = np.array(darts_model.reservoir.mesh.volume, copy=False)[:n_res_blocks]
        poro = np.array(darts_model.reservoir.mesh.poro, copy=False)[:n_res_blocks]

        pc = darts_model.physics.property_containers[0]

        co2_idx = pc.components_name.index("CO2")
        total_co2_kmol_sequestered = 0.0

        for i in range(n_res_blocks):
            state = X[i, :]
            pc.evaluate(state)

            co2_cell_kmol_per_m3 = 0.0
            for ph in range(pc.np_fl):   # fluid phases only
                co2_cell_kmol_per_m3 += pc.sat[ph] * pc.dens_m[ph] * pc.x[ph, co2_idx]

            total_co2_kmol_sequestered += co2_cell_kmol_per_m3 * poro[i] * volumes[i]

        total_co2_mass_kg = total_co2_kmol_sequestered * 44.01

        print(f"Total CO2 in reservoir: {total_co2_kmol_sequestered:.6e} kmol")
        print(f"Total CO2 mass in reservoir: {total_co2_mass_kg:.6e} kg")