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

Nt = 50
Dt = 365
output_dir = r".\output_lgr_heter"
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
    darts_model.init(platform="cpu")
    darts_model.set_output(output_folder='output_lgr_heter')
    
    # for name in darts_model.lgr_meta["lgr_orders"]:
    #     off = darts_model.lgr_meta["lgr_offsets"][name]
    #     nloc = darts_model.level1[name].n
    #     print(name, "dx unique:", np.unique(darts_model.reservoir.dx[off:off+nloc]))
    #     print(name, "dy unique:", np.unique(darts_model.reservoir.dy[off:off+nloc]))
    #     print(name, "x unique first layer:", np.unique(np.round(darts_model.reservoir.cell_center_x[off:off+25], 6)))
    #     print(name, "y unique first layer:", np.unique(np.round(darts_model.reservoir.cell_center_y[off:off+25], 6)))

     #plot initial condition
    prim0 = get_physics_field(darts_model)

    #Permeability
    if USE_LGR:
            plot_xz_section(
            darts_model, darts_model.reservoir.kx, use_lgr=USE_LGR, zmin=1990, zmax=2080,
            savepath=os.path.join(SECTION_DIR, f"section_kx_xz.png"),
            title=f"permeability kx", logscale=True
        )
            plot_xy_plane(
                darts_model, darts_model.reservoir.kx, depth=2005, use_lgr=USE_LGR,
                savepath=os.path.join(SECTION_DIR, f"section_kx_xy.png"),
                title=f"permeability kx", logscale=True
            )

    else:
        plot_xz_section(
            darts_model, darts_model.reservoir.global_data["permx"].reshape(-1, order="F"), use_lgr=USE_LGR, zmin=1990, zmax=2080,
            savepath=os.path.join(SECTION_DIR, f"section_kx_xz.png"),
            title=f"permeability kx", logscale=True
        )
        plot_xy_plane(
            darts_model, darts_model.reservoir.global_data["permx"].reshape(-1, order="F"), depth=2005, use_lgr=USE_LGR,
            savepath=os.path.join(SECTION_DIR, f"section_kx_xy.png"),
            title=f"permeability kx ", logscale=True
        )

    # pressure
    for k in prim0.keys():
        kk = k.lower()
        if ("pressure" in kk) or (kk == "p"):
            plot_xz_section(
                darts_model, prim0[k], use_lgr=USE_LGR, zmin=1990, zmax=2080,
                savepath=os.path.join(SECTION_DIR, f"section_P_XZ_injector.png"),
                title=f"{k} (initial)", logscale=False, edgecolor="none", linewidth=0.0
            )
            plot_xz_section(
                darts_model, prim0[k], use_lgr=USE_LGR, zmin=1990, zmax=2080,
                y0=1365,savepath=os.path.join(SECTION_DIR, f"section_P_xz_producer.png"),
                title=f"{k} (initial)", logscale=False, edgecolor="none", linewidth=0.0
            )

            plot_xy_plane(
                darts_model, prim0[k], depth=2035, use_lgr=USE_LGR,
                savepath=os.path.join(SECTION_DIR, f"section_P_xy_middle.png"),
                title=f"{k} (initial)", logscale=False, edgecolor="none", linewidth=0.0
            )
            break

    # CO2 
    # for k in prim0.keys():
    #     if "co2" in k.lower():
    #         plot_xz_section(
    #             darts_model, prim0[k] - 1e-8, use_lgr=USE_LGR, zmin=1990, zmax=2080,
    #             savepath=os.path.join(SECTION_DIR, f"section_CO2_xz.png"),
    #             title="CO2_delta (initial)", logscale=False, vmin=0, vmax=1
    #         )
    #         plot_xy_plane(
    #             darts_model, prim0[k] - 1e-8, depth=2035, use_lgr=USE_LGR,
    #             savepath=os.path.join(SECTION_DIR, f"section_CO2_xy_middle.png"),
    #             title="CO2_delta (initial)", logscale=False, vmin=0, vmax=1
    #         )
    #         break

    # temperature
    for k in prim0.keys():
        kk = k.lower()
        if ("temp" in kk) or ("temperature" in kk):
            plot_xz_section(
                darts_model, prim0[k], use_lgr=USE_LGR, zmin=1990, zmax=2080,
                savepath=os.path.join(SECTION_DIR, f"section_T_xz_injector.png"),
                title=f"{k} (initial)", logscale=False, edgecolor="none", linewidth=0.0
            )
            plot_xz_section(
                darts_model, prim0[k], use_lgr=USE_LGR, zmin=1990, zmax=2080,
                y0=1365,savepath=os.path.join(SECTION_DIR, f"section_T_xz_producer.png"),
                title=f"{k} (initial)", logscale=False, edgecolor="none", linewidth=0.0
            )
            plot_xy_plane(
                darts_model, prim0[k], depth=2035, use_lgr=USE_LGR,
                savepath=os.path.join(SECTION_DIR, f"section_T_xy_middle.png"),
                title=f"{k} (initial)", logscale=False, edgecolor="none", linewidth=0.0
            )
            break

    Tmin = 68 + 273.15
    Tmax = 80+ 273.15
    if True:
        for t in range(Nt):
            darts_model.run(Dt)

            darts_model.print_timers()
            darts_model.print_stat()
            
            t_end = float(darts_model.physics.engine.t)

            if t_end == Dt or t_end ==Dt*5 or t_end == Dt*10 or t_end == Dt*15 or t_end == Dt*20 or t_end == Dt*25 or t_end == Dt*30  or t_end == Dt*40 or t_end == Dt*50:        
                output_props = darts_model.physics.vars + darts_model.output.properties
                timesteps, property_array = darts_model.output.output_properties(
                    output_properties = ["satG","rhoG", "muG"],engine=True
                    )
                darts_model.output.save_property_array(timesteps, property_array)

                satG = property_array["satG"][0,:]
                muG = property_array["muG"][0,:]
                rhoG = property_array["rhoG"][0,:]
        
                plot_xy_plane(
                    darts_model, muG, depth=2035, use_lgr=USE_LGR,
                    savepath=os.path.join(SECTION_DIR, f"section_muG_xy_{t_end}_middle.png"),
                    title=f"muG", logscale=False, edgecolor="none", linewidth=0.0
                )
                plot_xy_plane(
                    darts_model, rhoG, depth=2035, use_lgr=USE_LGR,
                    savepath=os.path.join(SECTION_DIR, f"section_rhoG_xy_{t_end}_middle.png"),
                    title=f"rhoG", logscale=False, edgecolor="none", linewidth=0.0
                )

                plot_xz_section(
                    darts_model, muG, use_lgr=USE_LGR, zmin=1990, zmax=2080,
                    savepath=os.path.join(SECTION_DIR, f"section_muG_xz_{t_end}_injector.png"),
                    title=f"muG", logscale=False, edgecolor="none", linewidth=0.0
                )

                plot_xz_section(
                    darts_model, muG, use_lgr=USE_LGR, zmin=1990, zmax=2080,
                    y0=1365,savepath=os.path.join(SECTION_DIR, f"section_muG_xz_{t_end}_producer.png"),
                    title=f"muG", logscale=False, edgecolor="none", linewidth=0.0
                )

                prim = get_physics_field(darts_model)
                print("vars:", list(prim.keys()))
                for k in prim.keys():
                    if "pressure" in k.lower() or k.lower() == "p":
                        plot_xz_section(darts_model, prim[k],use_lgr=USE_LGR ,zmin=1990, zmax=2080,
                                        savepath=os.path.join(SECTION_DIR, f"section_P_xz_{t_end}_injector.png"),
                                        title=k,
                                        logscale=False, edgecolor="none", linewidth=0.0)
                        plot_xz_section(darts_model, prim[k], use_lgr=USE_LGR, zmin=1990, zmax=2080,
                                        y0=1365,savepath=os.path.join(SECTION_DIR, f"section_P_xz_{t_end}_producer.png"),
                                        title=k,
                                        logscale=False, edgecolor="none", linewidth=0.0)
                        plot_xy_plane(
                                    darts_model, prim[k], depth=2035, use_lgr=USE_LGR,
                                    savepath=os.path.join(SECTION_DIR, f"section_P_xy_{t_end}_middle.png"),
                                    title=k, logscale=False, edgecolor="none", linewidth=0.0
                                )
                        break

            
                # for k in prim.keys():
                #     if "co2" in k.lower() :  
                #         plot_xz_section(darts_model, prim[k] - 1e-8, use_lgr=USE_LGR, zmin=1990, zmax=2080,
                #                         savepath=os.path.join(SECTION_DIR, f"section_CO2_xz_{t_end}.png"),
                #                         title="CO2_delta",
                #                         logscale=False,
                #                         vmin=0, vmax=1)   
                #         plot_xy_plane(
                #                     darts_model, prim[k] - 1e-8, depth=2035, use_lgr=USE_LGR,
                #                     savepath=os.path.join(SECTION_DIR, f"section_CO2_xy_{t_end}_middle.png"),
                #                     title="CO2_delta", logscale=False, vmin=0, vmax=1
                #                 )
                #         break
                
                for k in prim.keys():
                    if "temp" in k.lower() or "temperature" in k.lower():
                        plot_xz_section(darts_model, prim[k], use_lgr=USE_LGR, zmin=1990, zmax=2080,
                                        savepath=os.path.join(SECTION_DIR, f"coarse_section_T_{t_end}_injector.png"),
                                        title=k,vmin=Tmin, vmax=Tmax, edgecolor="none", linewidth=0.0,
                                        logscale=False)
                        
                        plot_xz_section(darts_model, prim[k], use_lgr=USE_LGR, zmin=1990, zmax=2080,
                                        y0=1365,savepath=os.path.join(SECTION_DIR, f"coarse_section_T_{t_end}_producer.png"),
                                        title=k,vmin=Tmin, vmax=Tmax, edgecolor="none", linewidth=0.0,
                                        logscale=False)
                        plot_xy_plane(
                                    darts_model, prim[k], depth=2035, use_lgr=USE_LGR,
                                    savepath=os.path.join(SECTION_DIR, f"section_T_xy_{t_end}_middle.png"),
                                    title=k, logscale=False,vmin=Tmin, vmax=Tmax, edgecolor="none", linewidth=0.0
                                )
                        break

        # compute well time data
        time_data_dict = darts_model.output.store_well_time_data(save_output_files=True)
        time_data_df = pd.DataFrame.from_dict(time_data_dict)

        plot_well_time_data_2(darts_model, time_data_df,save_output_files=WELL_DIR)





