import os
import numpy as np
import pandas as pd
from model_consK import Model
from darts.engines import value_vector, redirect_darts_output
from darts.physics.base.operators_base import PropertyOperators as props
from output import get_physics_field, plot_xz_section, plot_well_time_data_2, plot_xy_plane
from darts.engines import well_control_iface

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
                "lgr_coords_in_parent_grid" :{"i_range": [40, 40],
                                                "j_range": [40,40],
                                                "k_range": [1, 40],
                                                "refine": [3, 3, 1],
                                                "tag" : "inj"

                }
            },
            "lgr1":{
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid" :{"i_range": [35, 35],
                                                "j_range": [45, 45],
                                                "k_range": [1, 40],
                                                "refine": [3, 3, 1],
                                                "tag" : "prod"

                }
            },
            "lgr2":{
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid" :{"i_range": [45, 45],
                                                "j_range": [45, 45],
                                                "k_range": [1, 40],
                                                "refine": [3, 3, 1],
                                                "tag" : "prod"

                }
            },
            "lgr3":{
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid" :{"i_range": [35, 35],
                                                "j_range": [35, 35],
                                                "k_range": [1, 40],
                                                "refine": [3, 3, 1],
                                                "tag" : "prod"

                }
            },
            "lgr4":{
                "parent_grid_name": "global",
                "lgr_coords_in_parent_grid" :{"i_range": [45, 45],
                                                "j_range": [35, 35],
                                                "k_range": [1, 40],
                                                "refine": [3, 3, 1],
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

        # "water_inj": {
        #     "W1": {"k_from": 1, "k_to": 40, "i0": 25, "j0": 45},
        #     "W2": {"k_from": 1, "k_to": 40, "i0": 25, "j0": 35},
        #     "W3": {"k_from": 1, "k_to": 40, "i0": 55, "j0": 35},
        #     "W4": {"k_from": 1, "k_to": 40, "i0": 55, "j0": 45}
        # }
    }
    return cfg


USE_LGR=True
Nt = 60
Dt = 366/2
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
            plot_xy_plane(
                darts_model, prim0[k], depth=2002.5, use_lgr=USE_LGR,
                savepath=os.path.join(SECTION_DIR, f"section_P_xy_step_{t0}.png"),
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
            plot_xy_plane(
                darts_model, prim0[k] - 1e-8, depth=2002.5, use_lgr=USE_LGR,
                savepath=os.path.join(SECTION_DIR, f"section_CO2_xy_step_{t0}.png"),
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
            plot_xy_plane(
                darts_model, prim0[k], depth=2002.5, use_lgr=USE_LGR,
                savepath=os.path.join(SECTION_DIR, f"section_T_xy_step_{t0}.png"),
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
                            is_inj=True,
                            target=3.3264e7,
                            inj_composition=[1.0 - darts_model.zero],
                            phase_name="CO2_rich",
                            inj_temp=296.15
                        )
                        darts_model.physics.set_well_controls(
                            wctrl=w.constraint,
                            control_type=well_control_iface.BHP,
                            is_inj=True,
                            target=306.90,
                            inj_composition=[1.0 - darts_model.zero],
                            inj_temp=296.15
                        )
                    else:
                        # producers: closed
                        darts_model.physics.set_well_controls(
                            wctrl=w.control,
                            control_type=well_control_iface.MASS_RATE,
                            is_inj=False,
                            target=0,
                            phase_name="aqueous"
                        )
            else:
                # darts_model.reservoir.add_well("P1")
                # for k in range(darts_model.nz_over, darts_model.nz_over + darts_model.nz_res):
                #     darts_model.reservoir.add_perforation("P1", res_cell_idx=(60,40,k))
                for i, w in enumerate(darts_model.reservoir.wells):
                    if "I" in w.name:
                        # injector: MASS_RATE with BHP constraint
                        darts_model.physics.set_well_controls(
                            wctrl=w.control,
                            control_type=well_control_iface.MASS_RATE,
                            is_inj=True,
                            target=3.3264e7,
                            phase_name="CO2_rich",
                            inj_composition=[1.0 - darts_model.zero],
                            inj_temp=296.15
                        )
                        darts_model.physics.set_well_controls(
                            wctrl=w.constraint,
                            control_type=well_control_iface.BHP,
                            is_inj=True,
                            target=306.90,
                            inj_composition=[1.0 - darts_model.zero],
                            inj_temp=296.15
                        )
                    else:
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
            darts_model.run(Dt)
            t_end = float(darts_model.physics.engine.t)
            if t_end ==Dt*7 or t_end == Dt*20 or t_end == Dt*40 or t_end == Dt*60:
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
                        plot_xy_plane(
                            darts_model, prim[k] - 1e-8, depth=2002.5, use_lgr=USE_LGR,
                            savepath=os.path.join(SECTION_DIR, f"section_CO2_xy_step_{t_end}.png"),
                            title="CO2_delta", logscale=False, vmin=0, vmax=1
                        )
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
                     
    






