import os
import numpy as np
import pandas as pd
from lgr_heter_model import Model
from darts.engines import value_vector, redirect_darts_output
from darts.physics.base.operators_base import PropertyOperators as props
from output import get_physics_field, plot_xy_plane, plot_xz_section, plot_well_time_data_2

def make_cfg_lgr():
    cfg = {
        "burden":{
            "over_thickness":2000,
            "under_thickness":2000,
            "poro_burden" : 0.0001,
            "perm_burden" : 1e-6,
            "rcond_over":149.54, "hcap_over":2347.29,
            "rcond_under" : 149.54, "hcap_under":2347.29,
        },

        "lgrs":{
            "lgr0":{
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid" :{"i_range": [30, 30],
                                                "j_range": [30, 30],
                                                "k_range": [1, 7],
                                                "refine": [5, 5, 1],
                                                "tag" : "inj"

                }
            },
            "lgr1":{
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid" :{"i_range": [14, 14],
                                                "j_range": [46, 46],
                                                "k_range": [1, 7],
                                                "refine": [5, 5, 1],
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


USE_LGR=True

output_dir = r".\egg_model_output"
FIG_DIR = os.path.join(output_dir, "figures")
SECTION_DIR = os.path.join(FIG_DIR, "sections")
WELL_DIR = os.path.join(FIG_DIR, "well_time_plots")
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
    darts_model.init()
    darts_model.set_output(output_folder='egg_model_output')
    
     #plot initial condition
    prim0 = get_physics_field(darts_model)

    #Permeability
    plot_xz_section(
        darts_model, darts_model.reservoir.kx, use_lgr=USE_LGR, zmin=0, zmax=90,
        savepath=os.path.join(SECTION_DIR, f"section_kx_xz.png"),
        title=f"permeability kx (initial)", logscale=True
    )
    plot_xy_plane(
        darts_model, darts_model.reservoir.kx, depth=15, use_lgr=USE_LGR,
        savepath=os.path.join(SECTION_DIR, f"section_kx_xy.png"),
        title=f"permeability kx (initial)", logscale=True
    )

    # pressure
    for k in prim0.keys():
        kk = k.lower()
        if ("pressure" in kk) or (kk == "p"):
            plot_xz_section(
                darts_model, prim0[k], use_lgr=USE_LGR, zmin=0, zmax=90,
                savepath=os.path.join(SECTION_DIR, f"section_P_XZ.png"),
                title=f"{k} (initial)", logscale=False
            )
            plot_xy_plane(
                darts_model, prim0[k], depth=15, use_lgr=USE_LGR,
                savepath=os.path.join(SECTION_DIR, f"section_P_xy.png"),
                title=f"{k} (initial)", logscale=False
            )
            break

    # CO2 
    for k in prim0.keys():
        if "co2" in k.lower():
            plot_xz_section(
                darts_model, prim0[k] - 1e-8, use_lgr=USE_LGR, zmin=0, zmax=90,
                savepath=os.path.join(SECTION_DIR, f"section_CO2_xz.png"),
                title="CO2_delta (initial)", logscale=False, vmin=0, vmax=1
            )
            plot_xy_plane(
                darts_model, prim0[k] - 1e-8, depth=15, use_lgr=USE_LGR,
                savepath=os.path.join(SECTION_DIR, f"section_CO2_xy.png"),
                title="CO2_delta (initial)", logscale=False, vmin=0, vmax=1
            )
            break

    # temperature
    for k in prim0.keys():
        kk = k.lower()
        if ("temp" in kk) or ("temperature" in kk):
            plot_xz_section(
                darts_model, prim0[k], use_lgr=USE_LGR, zmin=0, zmax=90,
                savepath=os.path.join(SECTION_DIR, f"section_T_xz.png"),
                title=f"{k} (initial)", logscale=False
            )
            plot_xy_plane(
                darts_model, prim0[k], depth=15, use_lgr=USE_LGR,
                savepath=os.path.join(SECTION_DIR, f"section_T_xy.png"),
                title=f"{k} (initial)", logscale=False
            )
            break


    if True:
        darts_model.run(30)
        # darts_model.reservoir.wells[0].control = n.physics.new_bhp_inj(100, 3*[n.zero])
        # darts_model.run_python(300, restart_dt=1e-3)
        darts_model.print_timers()
        darts_model.print_stat()
        # save_dict = {
        #     "X": np.array(darts_model.physics.engine.X, copy=True),
        #     "vars": [str(v) for v in darts_model.physics.vars],
        #     "n_cells": int(len(darts_model.reservoir.poro)),
        #     "n_vars": len(darts_model.physics.vars),
        #     }

        # np.save("solution_final.npy", save_dict)
        output_props = darts_model.physics.vars + darts_model.output.properties
        timesteps, property_array = darts_model.output.output_properties(output_properties = ["satG", "XCO2", "rhoG", "rhoAq"],engine=True)
        darts_model.output.save_property_array(timesteps, property_array)
        
        satG = property_array["satG"][0,:]
        XCO2 = property_array["XCO2"][0,:]
        rhoG = property_array["rhoG"][0,:]
        rhoAq = property_array["rhoAq"][0,:]
        plot_xy_plane(
            darts_model, satG, depth=15, use_lgr=USE_LGR,
            savepath=os.path.join(SECTION_DIR, f"section_satG_xy.png"),
            title=f"satG", logscale=False, vmin=0, vmax=1
        )
        plot_xy_plane(
            darts_model, XCO2, depth=15, use_lgr=USE_LGR,
            savepath=os.path.join(SECTION_DIR, f"section_XCO2_xy_step.png"),
            title=f"XCO2", logscale=False, vmin=0, vmax=1
        )
        plot_xy_plane(
            darts_model, rhoG, depth=15, use_lgr=USE_LGR,
            savepath=os.path.join(SECTION_DIR, f"section_rhoG_xy_step.png"),
            title=f"rhoG", logscale=False
        )
        plot_xy_plane(
            darts_model, rhoAq, depth=15, use_lgr=USE_LGR,
            savepath=os.path.join(SECTION_DIR, f"section_rhoAq_xy_step.png"),
            title=f"rhoAq", logscale=False
        )
        plot_xz_section(
            darts_model, satG, use_lgr=USE_LGR, zmin=0, zmax=90,
            savepath=os.path.join(SECTION_DIR, f"section_satG_xz.png"),
            title=f"satG", logscale=False, vmin=0, vmax=1
        )

        prim = get_physics_field(darts_model)
        print("vars:", list(prim.keys()))
        for k in prim.keys():
            if "pressure" in k.lower() or k.lower() == "p":
                plot_xz_section(darts_model, prim[k],use_lgr=USE_LGR ,zmin=0, zmax=90,
                                savepath=os.path.join(SECTION_DIR, "section_P_xz.png"),
                                title=k,
                                logscale=False)
                plot_xy_plane(
                            darts_model, prim[k], depth=15, use_lgr=USE_LGR,
                            savepath=os.path.join(SECTION_DIR, f"section_P_xy.png"),
                            title=k, logscale=False
                        )
                break

     
        for k in prim.keys():
            if "co2" in k.lower() :  
                plot_xz_section(darts_model, prim[k] - 1e-8, use_lgr=USE_LGR, zmin=0, zmax=90,
                                savepath=os.path.join(SECTION_DIR, "section_CO2_xz.png"),
                                title="CO2_delta",
                                logscale=False,
                                vmin=0, vmax=1)   
                plot_xy_plane(
                            darts_model, prim[k] - 1e-8, depth=15, use_lgr=USE_LGR,
                            savepath=os.path.join(SECTION_DIR, f"section_CO2_xy.png"),
                            title="CO2_delta", logscale=False, vmin=0, vmax=1
                        )
                break
        
        for k in prim.keys():
            if "temp" in k.lower() or "temperature" in k.lower():
                plot_xz_section(darts_model, prim[k], use_lgr=USE_LGR, zmin=0, zmax=90,
                                savepath=os.path.join(SECTION_DIR, "coarse_section_T.png"),
                                title=k,
                                logscale=False)
                plot_xy_plane(
                            darts_model, prim[k], depth=15, use_lgr=USE_LGR,
                            savepath=os.path.join(SECTION_DIR, f"section_T_xy.png"),
                            title=k, logscale=False
                        )
                break

        # compute well time data
        time_data_dict = darts_model.output.store_well_time_data(save_output_files=True)
        time_data_df = pd.DataFrame.from_dict(time_data_dict)

        plot_well_time_data_2(darts_model, time_data_df,save_output_files=WELL_DIR)
    else:
        # darts_model.load_restart_data()
        darts_model.load_restart_data('output/solution.h5')
        time_data = pd.read_pickle("darts_time_data.pkl")




