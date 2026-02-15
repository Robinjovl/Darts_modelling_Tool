import os
import numpy as np
import pandas as pd
from model_consK import Model
from darts.engines import value_vector, redirect_darts_output
from darts.physics.base.operators_base import PropertyOperators as props
from output import get_physics_field, plot_xz_section, plot_well_time_data_2

def make_cfg_lgr():
    cfg = {
        "grid":{
            "nx" : 80,
            "ny" : 80,
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
            "poro_burden" : 0.0001,
            "perm_burden" : 1e-6,

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
                "lgr_coords_in_parent_grid" :{"i_range": [20, 20],
                                                "j_range": [40,40],
                                                "k_range": [1, 40],
                                                "refine": [3, 3, 1],
                                                "tag" : "inj"

                }
            },
            "lgr1":{
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid" :{"i_range": [60, 60],
                                                "j_range": [40, 40],
                                                "k_range": [1, 40],
                                                "refine": [3, 3, 1],
                                                "tag" : "prod"

                }
            }
        },
        "wells":{
           "I1": {"lgr": "lgr0", "k_from": 1, "k_to": 40},
           "P1": {"lgr": "lgr1", "k_from": 1, "k_to": 40},
        }
    }
    return cfg


USE_LGR=True
output_dir = r".\output"
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
    darts_model.set_output()


    if True:
        darts_model.run(50)
        # darts_model.reservoir.wells[0].control = n.physics.new_bhp_inj(100, 3*[n.zero])
        # darts_model.run_python(300, restart_dt=1e-3)
        darts_model.print_timers()
        darts_model.print_stat()
        save_dict = {
            "X": np.array(darts_model.physics.engine.X, copy=True),
            "vars": [str(v) for v in darts_model.physics.vars],
            "n_cells": int(len(darts_model.reservoir.poro)),
            "n_vars": len(darts_model.physics.vars),
            }

        np.save("solution_final.npy", save_dict)

        prim = get_physics_field(darts_model)
        print("vars:", list(prim.keys()))
        for k in prim.keys():
            if "pressure" in k.lower() or k.lower() == "p":
                plot_xz_section(darts_model, prim[k],
                                savepath=os.path.join(SECTION_DIR, "coarse_section_P.png"),
                                title=k,
                                logscale=False)
                break

     
        for k in prim.keys():
            if "co2" in k.lower() :  
                plot_xz_section(darts_model, prim[k] - 1e-8,
                                savepath=os.path.join(SECTION_DIR, "coarse_section_CO2.png"),
                                title="CO2_delta",
                                logscale=False,
                                vmin=0, vmax=1)   
                break
        
        for k in prim.keys():
            if "temp" in k.lower() or "temperature" in k.lower():
                plot_xz_section(darts_model, prim[k],
                                savepath=os.path.join(SECTION_DIR, "coarse_section_T.png"),
                                title=k,
                                logscale=False)
                break

        # compute well time data
        time_data_dict = darts_model.output.store_well_time_data(save_output_files=True)
        time_data_df = pd.DataFrame.from_dict(time_data_dict)

        plot_well_time_data_2(darts_model, time_data_df,save_output_files=WELL_DIR)
    else:
        # darts_model.load_restart_data()
        darts_model.load_restart_data('output/solution.h5')
        time_data = pd.read_pickle("darts_time_data.pkl")

    # if True:
    #     Xn = np.array(darts_model.physics.engine.X, copy=False)
    #     nc = darts_model.physics.nc + darts_model.physics.thermal
    #     nb = darts_model.reservoir.mesh.n_res_blocks

    #     plt.figure(num=1, figsize=(12, 8), dpi=100)
    #     for i in range(nc if nc < 3 else 3):
    #         plt.subplot(330 + (i + 1))
    #         plt.plot(Xn[i:nb*nc:nc])
    #     plt.savefig('out.png')
    # else:
    #     #plot_sol(n)
    #     n.print_and_plot('sim_data')





